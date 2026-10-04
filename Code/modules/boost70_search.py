from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor_ml_extract"
if not VENDOR.exists():
    VENDOR = ROOT / "vendor_sklearn"
sys.path.insert(0, str(VENDOR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC, SVR

import sant_orf_v4 as v4


OUT = ROOT / "results_boost70"
OUT.mkdir(exist_ok=True)
CUT_LOW, CUT_HIGH = map(float, v4.YIELD_CUTS)


def physical_features(data: pd.DataFrame) -> pd.DataFrame:
    x = data[v4.CORE_FEATURES].copy()
    eps = 1e-4
    ratio = x["ratio_g_ml"].clip(lower=eps)
    bp = x["bp_c"].replace(0, np.nan)
    x["temperature_to_bp"] = x["temperature_c"] / bp
    x["temperature_minus_bp"] = x["temperature_c"] - x["bp_c"]
    x["log_extraction_time"] = np.log1p(x["time_h"].clip(lower=0))
    x["log_pressure"] = np.log1p(x["pressure_bar"].clip(lower=0))
    x["solvent_loading_ml_g"] = (1.0 / ratio).clip(upper=250)
    x["polarity_density"] = x["dipole_moment"] * x["density_g_ml"]
    x["volatility_density"] = x["bp_c"] * x["density_g_ml"]
    x["extraction_thermal_severity"] = (
        x["temperature_c"] * np.log1p(x["time_h"].clip(lower=0)) / 100.0
    )
    x["drying_thermal_severity"] = (
        x["dry_temperature_c"]
        * np.log1p(x["dry_time_h"].clip(lower=0))
        / 100.0
    )
    x["drying_minus_extraction_temp"] = (
        x["dry_temperature_c"] - x["temperature_c"]
    )
    x["pressure_temperature"] = (
        np.log1p(x["pressure_bar"].clip(lower=0)) * x["temperature_c"] / 100.0
    )
    return x.replace([np.inf, -np.inf], np.nan)


STABLE_COLUMNS = [
    "dry_time_h",
    "pressure_bar",
    "temperature_c",
    "dry_temperature_c",
    "bp_c",
    "time_h",
    "temperature_to_bp",
    "extraction_thermal_severity",
    "drying_thermal_severity",
    "pressure_temperature",
]


def matrix_for(data: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "core":
        return data[v4.CORE_FEATURES].copy()
    engineered = physical_features(data)
    if feature_set == "stable":
        return engineered[STABLE_COLUMNS].copy()
    if feature_set == "physics":
        return engineered
    raise ValueError(feature_set)


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    task: str
    feature_set: str
    scaled: bool
    weight_mode: str
    params: tuple


CANDIDATES = [
    Candidate("lda_core", "lda", "classifier", "core", True, "none", ()),
    Candidate("lda_physics", "lda", "classifier", "physics", True, "none", ()),
    Candidate("logit_physics", "logit", "classifier", "physics", True, "source_class", (0.20,)),
    Candidate("logit_stable", "logit", "classifier", "stable", True, "source_class", (0.35,)),
    Candidate("svc_rbf_physics", "svc", "classifier", "physics", True, "source_class", (2.0,)),
    Candidate("extra_cls_leaf1", "extra_cls", "classifier", "physics", False, "source", (1, 0.7)),
    Candidate("extra_cls_leaf3", "extra_cls", "classifier", "physics", False, "source_class", (3, 1.0)),
    Candidate("rf_cls_leaf2", "rf_cls", "classifier", "physics", False, "source_class", (2, 0.8)),
    Candidate("hist_cls", "hist_cls", "classifier", "physics", False, "source_class", (7, 5.0)),
    Candidate("ridge_yield", "ridge", "regressor", "physics", True, "source", (3.0,)),
    Candidate("huber_yield", "huber", "regressor", "stable", True, "source", (1.7, 0.1)),
    Candidate("svr_yield", "svr", "regressor", "physics", True, "source", (10.0, 1.0)),
    Candidate("extra_reg_leaf2", "extra_reg", "regressor", "physics", False, "source", (2, 0.8)),
    Candidate("extra_reg_leaf4", "extra_reg", "regressor", "physics", False, "source", (4, 1.0)),
    Candidate("rf_reg_leaf2", "rf_reg", "regressor", "physics", False, "source", (2, 0.8)),
    Candidate("hist_reg", "hist_reg", "regressor", "physics", False, "source", (9, 5.0)),
    Candidate("gbr_huber", "gbr", "regressor", "stable", False, "source", (1, 0.04)),
]


def source_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(map(str, groups))
    w = np.asarray([1.0 / counts[str(g)] for g in groups], dtype=float)
    return w / np.mean(w)


def fit_weights(groups: np.ndarray, target: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    weights = source_weights(groups)
    if mode == "source_class":
        counts = np.bincount(target.astype(int))
        class_factor = np.ones(max(len(counts), int(np.max(target)) + 1), dtype=float)
        nonzero = counts > 0
        class_factor[nonzero] = len(target) / (np.sum(nonzero) * counts[nonzero])
        weights *= class_factor[target.astype(int)]
    return weights / np.mean(weights)


def build_model(candidate: Candidate, seed: int):
    p = candidate.params
    if candidate.family == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if candidate.family == "logit":
        return LogisticRegression(C=p[0], max_iter=3000, solver="lbfgs", random_state=seed)
    if candidate.family == "svc":
        return SVC(C=p[0], kernel="rbf", gamma="scale", random_state=seed)
    if candidate.family == "extra_cls":
        return ExtraTreesClassifier(
            n_estimators=220,
            min_samples_leaf=p[0],
            max_features=p[1],
            class_weight=None,
            random_state=seed,
            n_jobs=1,
        )
    if candidate.family == "rf_cls":
        return RandomForestClassifier(
            n_estimators=260,
            min_samples_leaf=p[0],
            max_features=p[1],
            class_weight=None,
            random_state=seed,
            n_jobs=1,
        )
    if candidate.family == "hist_cls":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.055,
            max_leaf_nodes=p[0],
            l2_regularization=p[1],
            random_state=seed,
        )
    if candidate.family == "ridge":
        return Ridge(alpha=p[0], random_state=seed)
    if candidate.family == "huber":
        return HuberRegressor(epsilon=p[0], alpha=p[1], max_iter=2000)
    if candidate.family == "svr":
        return SVR(C=p[0], epsilon=p[1], kernel="rbf", gamma="scale")
    if candidate.family == "extra_reg":
        return ExtraTreesRegressor(
            n_estimators=220,
            min_samples_leaf=p[0],
            max_features=p[1],
            random_state=seed,
            n_jobs=1,
        )
    if candidate.family == "rf_reg":
        return RandomForestRegressor(
            n_estimators=260,
            min_samples_leaf=p[0],
            max_features=p[1],
            random_state=seed,
            n_jobs=1,
        )
    if candidate.family == "hist_reg":
        return HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.055,
            max_leaf_nodes=p[0],
            l2_regularization=p[1],
            loss="squared_error",
            random_state=seed,
        )
    if candidate.family == "gbr":
        return GradientBoostingRegressor(
            loss="huber",
            n_estimators=220,
            learning_rate=p[1],
            max_depth=p[0],
            min_samples_leaf=3,
            random_state=seed,
        )
    raise ValueError(candidate)


def target_for_mode(y_class: np.ndarray, mode: str) -> np.ndarray:
    if mode == "multi":
        return y_class.astype(int)
    if mode == "high":
        return (y_class == 2).astype(int)
    if mode == "lowmid":
        return (y_class == 1).astype(int)
    raise ValueError(mode)


def class_from_yield(predicted_yield: np.ndarray, mode: str) -> np.ndarray:
    if mode == "multi":
        return np.digitize(predicted_yield, [CUT_LOW, CUT_HIGH]).astype(int)
    if mode == "high":
        return (predicted_yield >= CUT_HIGH).astype(int)
    if mode == "lowmid":
        return (predicted_yield >= CUT_LOW).astype(int)
    raise ValueError(mode)


def fit_predict(
    candidate: Candidate,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    y_class_train: np.ndarray,
    yield_train: np.ndarray,
    groups_train: np.ndarray,
    mode: str,
    seed: int,
) -> np.ndarray:
    x_train = matrix_for(train_data, candidate.feature_set)
    x_test = matrix_for(test_data, candidate.feature_set)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_matrix = imputer.fit_transform(x_train)
    test_matrix = imputer.transform(x_test)
    if candidate.scaled:
        scaler = RobustScaler().fit(train_matrix)
        train_matrix = scaler.transform(train_matrix)
        test_matrix = scaler.transform(test_matrix)
    target = target_for_mode(y_class_train, mode)
    if candidate.task == "classifier" and len(np.unique(target)) < 2:
        return np.full(len(test_data), int(target[0]), dtype=int)
    model = build_model(candidate, seed)
    if candidate.task == "classifier":
        weights = fit_weights(groups_train, target, candidate.weight_mode)
        if weights is None or candidate.family == "lda":
            model.fit(train_matrix, target)
        else:
            model.fit(train_matrix, target, sample_weight=weights)
        return model.predict(test_matrix).astype(int)
    weights = source_weights(groups_train) if candidate.weight_mode != "none" else None
    if weights is None:
        model.fit(train_matrix, yield_train)
    else:
        try:
            model.fit(train_matrix, yield_train, sample_weight=weights)
        except TypeError:
            model.fit(train_matrix, yield_train)
    return class_from_yield(np.asarray(model.predict(test_matrix), dtype=float), mode)


def macro_score(true: np.ndarray, pred: np.ndarray, mode: str) -> tuple[float, float, float]:
    labels = [0, 1, 2] if mode == "multi" else [0, 1]
    accuracy = float(accuracy_score(true, pred))
    macro = float(f1_score(true, pred, labels=labels, average="macro", zero_division=0))
    balanced = float(balanced_accuracy_score(true, pred))
    return accuracy, macro, balanced


def composite_score(true: np.ndarray, pred: np.ndarray, mode: str) -> float:
    accuracy, macro, balanced = macro_score(true, pred, mode)
    if mode == "multi":
        return 0.62 * accuracy + 0.23 * macro + 0.15 * balanced
    return 0.65 * accuracy + 0.35 * balanced


def group_splits(y: np.ndarray, groups: np.ndarray, repeats: int, n_splits: int, seed: int):
    fold = 0
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat
        )
        for train, test in splitter.split(np.zeros(len(y)), y, groups=groups):
            assert not set(groups[train]) & set(groups[test])
            yield fold, train, test
            fold += 1


def exploratory_fixed_cv(data: pd.DataFrame, repeats: int = 2) -> pd.DataFrame:
    y_class = v4.target_classes(data["yield_pct"].to_numpy(float))
    yield_value = data["yield_pct"].to_numpy(float)
    groups = data["source_id"].astype(str).to_numpy()
    rows = []
    for mode in ("multi", "high"):
        target = target_for_mode(y_class, mode)
        for fold, train, test in group_splits(target, groups, repeats, 5, 20260820):
            for candidate_id, candidate in enumerate(CANDIDATES):
                try:
                    pred = fit_predict(
                        candidate,
                        data.iloc[train],
                        data.iloc[test],
                        y_class[train],
                        yield_value[train],
                        groups[train],
                        mode,
                        11000 + 100 * fold + candidate_id,
                    )
                    accuracy, macro, balanced = macro_score(target[test], pred, mode)
                    error = ""
                except Exception as exc:
                    accuracy = macro = balanced = np.nan
                    error = f"{type(exc).__name__}: {exc}"
                rows.append(
                    {
                        "mode": mode,
                        "fold": fold,
                        "candidate": candidate.name,
                        "family": candidate.family,
                        "accuracy": accuracy,
                        "macro_f1": macro,
                        "balanced_accuracy": balanced,
                        "error": error,
                    }
                )
            print(f"exploratory {mode:5s} fold={fold:02d} complete", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "exploratory_candidate_detail.csv", index=False)
    summary = (
        result.groupby(["mode", "candidate", "family"], as_index=False)
        .agg(
            folds=("accuracy", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
        )
        .sort_values(["mode", "accuracy_mean"], ascending=[True, False])
    )
    summary.to_csv(OUT / "exploratory_candidate_summary.csv", index=False)
    return result


def select_distinct(ranked: list[tuple[float, Candidate]], limit: int = 3):
    selected = []
    families = set()
    for score, candidate in ranked:
        if candidate.family in families:
            continue
        selected.append((score, candidate))
        families.add(candidate.family)
        if len(selected) == limit:
            break
    return selected


def inner_rank(
    data: pd.DataFrame,
    y_class: np.ndarray,
    yield_value: np.ndarray,
    groups: np.ndarray,
    mode: str,
    seed: int,
) -> list[tuple[float, Candidate]]:
    target = target_for_mode(y_class, mode)
    predictions = {candidate.name: np.full(len(data), -1, dtype=int) for candidate in CANDIDATES}
    filled = np.zeros(len(data), dtype=bool)
    for inner_fold, train, validation in group_splits(target, groups, 1, 3, seed):
        for candidate_id, candidate in enumerate(CANDIDATES):
            try:
                predictions[candidate.name][validation] = fit_predict(
                    candidate,
                    data.iloc[train],
                    data.iloc[validation],
                    y_class[train],
                    yield_value[train],
                    groups[train],
                    mode,
                    seed + 100 * inner_fold + candidate_id,
                )
            except Exception:
                pass
        filled[validation] = True
    ranked = []
    for candidate in CANDIDATES:
        valid = filled & (predictions[candidate.name] >= 0)
        if np.sum(valid) < max(12, len(data) // 2):
            score = -1.0
        else:
            score = composite_score(target[valid], predictions[candidate.name][valid], mode)
        ranked.append((score, candidate))
    return sorted(ranked, key=lambda item: item[0], reverse=True)


def weighted_vote(predictions: list[np.ndarray], scores: list[float], classes: int):
    votes = np.zeros((len(predictions[0]), classes), dtype=float)
    weights = np.maximum(np.asarray(scores, dtype=float), 0.01)
    for pred, weight in zip(predictions, weights):
        votes[np.arange(len(pred)), pred.astype(int)] += weight
    return np.argmax(votes, axis=1).astype(int)


def fit_ranked_ensemble(
    selected,
    train_data,
    test_data,
    y_class_train,
    yield_train,
    groups_train,
    mode,
    seed,
):
    predictions = []
    scores = []
    for candidate_id, (score, candidate) in enumerate(selected):
        predictions.append(
            fit_predict(
                candidate,
                train_data,
                test_data,
                y_class_train,
                yield_train,
                groups_train,
                mode,
                seed + candidate_id,
            )
        )
        scores.append(score)
    classes = 3 if mode == "multi" else 2
    return predictions[0], weighted_vote(predictions, scores, classes)


def nested_evaluation(data: pd.DataFrame, repeats: int = 3) -> pd.DataFrame:
    y_class = v4.target_classes(data["yield_pct"].to_numpy(float))
    yield_value = data["yield_pct"].to_numpy(float)
    groups = data["source_id"].astype(str).to_numpy()
    rows = []
    prediction_rows = []
    for fold, train, test in group_splits(y_class, groups, repeats, 5, 20260821):
        train_data = data.iloc[train].reset_index(drop=True)
        test_data = data.iloc[test].reset_index(drop=True)
        y_train = y_class[train]
        yield_train = yield_value[train]
        groups_train = groups[train]

        multi = select_distinct(
            inner_rank(train_data, y_train, yield_train, groups_train, "multi", 21000 + fold)
        )
        high = select_distinct(
            inner_rank(train_data, y_train, yield_train, groups_train, "high", 31000 + fold)
        )
        low_mask = y_train < 2
        lowmid = select_distinct(
            inner_rank(
                train_data.loc[low_mask].reset_index(drop=True),
                y_train[low_mask],
                yield_train[low_mask],
                groups_train[low_mask],
                "lowmid",
                41000 + fold,
            )
        )

        multi_top1, multi_vote = fit_ranked_ensemble(
            multi, train_data, test_data, y_train, yield_train, groups_train, "multi", 51000 + fold
        )
        high_top1, high_vote = fit_ranked_ensemble(
            high, train_data, test_data, y_train, yield_train, groups_train, "high", 61000 + fold
        )
        low_top1, low_vote = fit_ranked_ensemble(
            lowmid,
            train_data.loc[low_mask].reset_index(drop=True),
            test_data,
            y_train[low_mask],
            yield_train[low_mask],
            groups_train[low_mask],
            "lowmid",
            71000 + fold,
        )
        hierarchical_top1 = np.where(high_top1 == 1, 2, low_top1)
        hierarchical_vote = np.where(high_vote == 1, 2, low_vote)

        predictions_by_variant = {
            "nested_top1": multi_top1,
            "nested_family_vote": multi_vote,
            "hierarchical_top1": hierarchical_top1,
            "hierarchical_family_vote": hierarchical_vote,
        }
        for variant, pred in predictions_by_variant.items():
            accuracy, macro, balanced = macro_score(y_class[test], pred, "multi")
            rows.append(
                {
                    "task": "three_class",
                    "variant": variant,
                    "fold": fold,
                    "accuracy": accuracy,
                    "macro_f1": macro,
                    "balanced_accuracy": balanced,
                    "ordinal_mae": float(np.mean(np.abs(y_class[test] - pred))),
                    "test_n": len(test),
                    "test_sources": "|".join(sorted(np.unique(groups[test]))),
                    "multi_selected": "|".join(c.name for _, c in multi),
                    "high_selected": "|".join(c.name for _, c in high),
                    "lowmid_selected": "|".join(c.name for _, c in lowmid),
                }
            )
        for variant, pred in {
            "high_screen_top1": high_top1,
            "high_screen_family_vote": high_vote,
        }.items():
            true_high = (y_class[test] == 2).astype(int)
            accuracy, macro, balanced = macro_score(true_high, pred, "high")
            rows.append(
                {
                    "task": "high_yield_screen",
                    "variant": variant,
                    "fold": fold,
                    "accuracy": accuracy,
                    "macro_f1": macro,
                    "balanced_accuracy": balanced,
                    "ordinal_mae": np.nan,
                    "test_n": len(test),
                    "test_sources": "|".join(sorted(np.unique(groups[test]))),
                    "multi_selected": "|".join(c.name for _, c in multi),
                    "high_selected": "|".join(c.name for _, c in high),
                    "lowmid_selected": "|".join(c.name for _, c in lowmid),
                }
            )
        for local_index, global_index in enumerate(test):
            prediction_rows.append(
                {
                    "fold": fold,
                    "row_index": int(global_index),
                    "source_id": groups[global_index],
                    "yield_pct": yield_value[global_index],
                    "true_class": int(y_class[global_index]),
                    "nested_family_vote": int(multi_vote[local_index]),
                    "hierarchical_family_vote": int(hierarchical_vote[local_index]),
                    "true_high": int(y_class[global_index] == 2),
                    "predicted_high": int(high_vote[local_index]),
                }
            )
        pd.DataFrame(rows).to_csv(OUT / "nested_detail_checkpoint.csv", index=False)
        pd.DataFrame(prediction_rows).to_csv(OUT / "nested_predictions_checkpoint.csv", index=False)
        print(
            f"nested fold={fold:02d} multi={accuracy_score(y_class[test], multi_vote):.3f} "
            f"hier={accuracy_score(y_class[test], hierarchical_vote):.3f} "
            f"high={accuracy_score((y_class[test] == 2).astype(int), high_vote):.3f}",
            flush=True,
        )
    result = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    result.to_csv(OUT / "nested_detail.csv", index=False)
    predictions.to_csv(OUT / "nested_predictions.csv", index=False)
    summary = (
        result.groupby(["task", "variant"], as_index=False)
        .agg(
            folds=("accuracy", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            ordinal_mae_mean=("ordinal_mae", "mean"),
        )
        .sort_values(["task", "accuracy_mean"], ascending=[True, False])
    )
    summary["accuracy_ci95_fold_dispersion"] = 1.96 * summary["accuracy_std"] / np.sqrt(summary["folds"])
    summary.to_csv(OUT / "nested_summary.csv", index=False)
    source_summary = []
    for source, frame in predictions.groupby("source_id"):
        source_summary.append(
            {
                "source_id": source,
                "rows_across_repeats": len(frame),
                "three_class_accuracy": float(
                    accuracy_score(frame["true_class"], frame["nested_family_vote"])
                ),
                "hierarchical_accuracy": float(
                    accuracy_score(frame["true_class"], frame["hierarchical_family_vote"])
                ),
                "high_screen_accuracy": float(
                    accuracy_score(frame["true_high"], frame["predicted_high"])
                ),
                "mean_yield": float(frame["yield_pct"].mean()),
                "dominant_class": int(frame["true_class"].mode().iloc[0]),
            }
        )
    pd.DataFrame(source_summary).sort_values("three_class_accuracy").to_csv(
        OUT / "nested_source_diagnostics.csv", index=False
    )
    return result


def main():
    _, data = v4.load_datasets()
    profile = {
        "rows": int(len(data)),
        "sources": int(data["source_id"].nunique()),
        "class_counts": dict(Counter(map(int, v4.target_classes(data["yield_pct"].to_numpy(float))))),
        "target_three_class_accuracy": 0.70,
        "target_high_yield_screen_accuracy": 0.70,
        "leakage_guard": "all outer and inner splits keep source_id disjoint",
        "fixed_class_cuts": [CUT_LOW, CUT_HIGH],
    }
    (OUT / "search_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    exploratory_fixed_cv(data, repeats=2)
    nested_evaluation(data, repeats=3)
    print("\n=== EXPLORATORY ===")
    print(pd.read_csv(OUT / "exploratory_candidate_summary.csv").groupby("mode").head(8).to_string(index=False))
    print("\n=== NESTED ===")
    print(pd.read_csv(OUT / "nested_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
