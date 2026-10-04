from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

import multifeedstock_v12 as v12
import multifeedstock_v12_calibration as v12cal


OUT = Path(__file__).resolve().parent / "results_v13_hierarchical"
OUT.mkdir(parents=True, exist_ok=True)
DEVELOPMENT_SEEDS = [15101, 15102, 15103]
ELIGIBLE_MIN_N = 8
CONDITION_COLUMNS = [
    "ratio_g_ml", "time_h", "temperature_c", "dry_temperature_c", "dry_time_h",
    "particle_size_mid_um", "density_g_ml", "bp_c", "dipole_moment",
    "sample_weight_g", "solvent_volume_ml",
]
V13_NUMERIC = list(dict.fromkeys(v12.BATCH_NUMERIC + ["sample_weight_g", "solvent_volume_ml"]))


@dataclass(frozen=True)
class RegressorSpec:
    name: str
    target: str
    family: str
    categorical: str
    max_depth: int
    min_leaf: int


REGRESSORS = [
    RegressorSpec("delta_extra_process", "delta", "extra", "none", 6, 2),
    RegressorSpec("delta_extra_material", "delta", "extra", "class", 6, 2),
    RegressorSpec("delta_gb_process", "delta", "gb", "none", 2, 3),
    RegressorSpec("direct_extra_process", "yield", "extra", "none", 7, 2),
]
SELECTIONS = ["random_maximin", "center_maximin", "kennard_stone", "delta_guided"]


def condition_matrix(frame: pd.DataFrame) -> np.ndarray:
    matrix = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(
        frame[CONDITION_COLUMNS]
    )
    matrix = RobustScaler().fit_transform(matrix)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def calibration_indices(
    frame: pd.DataFrame,
    k: int,
    seed: int,
    strategy: str,
    guidance: np.ndarray | None = None,
) -> np.ndarray:
    matrix = condition_matrix(frame)
    n = len(frame)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    if strategy == "random_maximin":
        selected = [int(rng.integers(n))]
    elif strategy == "center_maximin":
        center = np.nanmedian(matrix, axis=0)
        selected = [int(np.argmin(np.sqrt(np.square(matrix - center).mean(axis=1))))]
    elif strategy == "kennard_stone":
        distance = np.sqrt(np.square(matrix[:, None, :] - matrix[None, :, :]).mean(axis=2))
        first, second = np.unravel_index(np.argmax(distance), distance.shape)
        selected = [int(first)]
        if k > 1 and second != first:
            selected.append(int(second))
    elif strategy == "delta_guided":
        if guidance is None or len(guidance) != n:
            raise ValueError("delta_guided requires one guidance score per row")
        order = np.argsort(guidance)
        selected = []
        for position in (0, len(order) - 1, len(order) // 2):
            candidate = int(order[position])
            if candidate not in selected:
                selected.append(candidate)
    else:
        raise ValueError(strategy)
    while len(selected) < k:
        distance = np.sqrt(
            np.square(matrix[:, None, :] - matrix[selected][None, :, :]).mean(axis=2)
        )
        nearest = distance.min(axis=1)
        nearest[selected] = -np.inf
        selected.append(int(np.argmax(nearest)))
    return frame.index.to_numpy()[np.asarray(selected, int)]


def make_regressor(spec: RegressorSpec, seed: int) -> Pipeline:
    categorical = v12.MATERIAL_CLASS if spec.categorical == "class" else []
    transformers = [
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]), V13_NUMERIC)
    ]
    if categorical:
        transformers.append((
            "cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical
        ))
    prep = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
    if spec.family == "extra":
        model = ExtraTreesRegressor(
            n_estimators=160, max_depth=spec.max_depth, min_samples_leaf=spec.min_leaf,
            max_features=0.8, random_state=seed, n_jobs=1,
        )
    elif spec.family == "gb":
        model = GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.035, max_depth=spec.max_depth,
            min_samples_leaf=spec.min_leaf, loss="huber", random_state=seed,
        )
    else:
        raise ValueError(spec.family)
    return Pipeline([("prep", prep), ("model", model)])


def source_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    sizes = frame.groupby("source_id")["source_id"].transform("size").to_numpy(float)
    return 1.0 / sizes


def delta_target(frame: pd.DataFrame) -> np.ndarray:
    means = frame.groupby("source_id")["yield_pct"].transform("mean")
    return (frame["yield_pct"] - means).to_numpy(float)


def local_residual_correction(
    source_frame: pd.DataFrame,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    cal_residual: np.ndarray,
) -> np.ndarray:
    matrix = condition_matrix(source_frame)
    position = {idx: pos for pos, idx in enumerate(source_frame.index.to_numpy())}
    cal_pos = np.asarray([position[int(idx)] for idx in cal_idx], int)
    test_pos = np.asarray([position[int(idx)] for idx in test_idx], int)
    distances = np.sqrt(
        np.square(matrix[test_pos, None, :] - matrix[cal_pos][None, :, :]).mean(axis=2)
    )
    positive = distances[distances > 1e-9]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    bandwidth = max(bandwidth, 0.25)
    weights = np.exp(-0.5 * np.square(distances / bandwidth))
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights @ cal_residual


def calibration_knn_probability(
    source_frame: pd.DataFrame,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    y_cal: np.ndarray,
) -> np.ndarray:
    matrix = condition_matrix(source_frame)
    position = {idx: pos for pos, idx in enumerate(source_frame.index.to_numpy())}
    cal_pos = np.asarray([position[int(idx)] for idx in cal_idx], int)
    test_pos = np.asarray([position[int(idx)] for idx in test_idx], int)
    distances = np.sqrt(
        np.square(matrix[test_pos, None, :] - matrix[cal_pos][None, :, :]).mean(axis=2)
    )
    positive = distances[distances > 1e-9]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    bandwidth = max(bandwidth, 0.20)
    weights = np.exp(-0.5 * np.square(distances / bandwidth))
    one_hot = np.eye(3)[y_cal.astype(int)]
    probability = weights @ one_hot + 0.10
    return probability / probability.sum(axis=1, keepdims=True)


def regression_probability(yhat: np.ndarray, scale: float) -> np.ndarray:
    low = norm.cdf((v12.ABSOLUTE_CUTS[0] - yhat) / scale)
    up_to_mid = norm.cdf((v12.ABSOLUTE_CUTS[1] - yhat) / scale)
    probability = np.column_stack([low, up_to_mid - low, 1.0 - up_to_mid])
    probability = np.clip(probability, 1e-9, None)
    return probability / probability.sum(axis=1, keepdims=True)


def primitive_predictions(data: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    y_class = data["absolute_class"].to_numpy(int)
    sizes = data.groupby("source_id").size()
    eligible = sorted(sizes[sizes >= ELIGIBLE_MIN_N].index.astype(str))
    rows: list[dict] = []
    for repeat, seed in enumerate(DEVELOPMENT_SEEDS):
        for selection in SELECTIONS:
            for source_index, source in enumerate(eligible):
                source_mask = data["source_id"].astype(str).eq(source)
                source_frame = data[source_mask]
                train_idx = data.index[~source_mask].to_numpy()
                train_frame = data.loc[train_idx]
                weights = source_balanced_weights(train_frame)
                guidance = None
                if selection == "delta_guided":
                    guidance_spec = next(
                        spec for spec in REGRESSORS if spec.name == "delta_extra_material"
                    )
                    guidance_model = make_regressor(
                        guidance_spec, seed * 20000 + source_index
                    )
                    guidance_model.fit(
                        train_frame, delta_target(train_frame), model__sample_weight=weights
                    )
                    guidance = guidance_model.predict(source_frame)
                cal_idx = calibration_indices(
                    source_frame, k, seed + source_index * 101, selection, guidance
                )
                test_idx = source_frame.index.difference(cal_idx).to_numpy()

                classifier_candidate = next(
                    item for item in v12.CANDIDATES if item.name == "extra_feedstock_batch"
                )
                classifier_train = np.concatenate([train_idx, cal_idx])
                classifier = v12.make_model(
                    classifier_candidate, seed * 1000 + source_index
                )
                classifier.fit(data.loc[classifier_train], y_class[classifier_train])
                class_probability = v12cal.aligned_probability(
                    classifier, data.loc[test_idx]
                )
                class_probability = v12cal.prior_shift(
                    class_probability, y_class[classifier_train], y_class[cal_idx], 1.0
                )
                cal_y = data.loc[cal_idx, "yield_pct"].to_numpy(float)
                cal_classes = y_class[cal_idx]
                knn_probability = calibration_knn_probability(
                    source_frame, cal_idx, test_idx, cal_classes
                )
                cal_counts = np.bincount(cal_classes, minlength=3)

                model_predictions: dict[str, dict[str, np.ndarray]] = {}
                for model_index, spec in enumerate(REGRESSORS):
                    model = make_regressor(
                        spec, seed * 10000 + source_index * 100 + model_index
                    )
                    target = (
                        delta_target(train_frame)
                        if spec.target == "delta"
                        else train_frame["yield_pct"].to_numpy(float)
                    )
                    model.fit(train_frame, target, model__sample_weight=weights)
                    cal_raw = model.predict(data.loc[cal_idx])
                    test_raw = model.predict(data.loc[test_idx])
                    if spec.target == "delta":
                        mean_baseline = float(np.mean(cal_y - cal_raw))
                        median_baseline = float(np.median(cal_y - cal_raw))
                        cal_mean_pred = mean_baseline + cal_raw
                        cal_median_pred = median_baseline + cal_raw
                        base_mean = mean_baseline + test_raw
                        base_median = median_baseline + test_raw
                    else:
                        mean_shift = float(np.mean(cal_y - cal_raw))
                        median_shift = float(np.median(cal_y - cal_raw))
                        cal_mean_pred = mean_shift + cal_raw
                        cal_median_pred = median_shift + cal_raw
                        base_mean = mean_shift + test_raw
                        base_median = median_shift + test_raw
                    local_mean = local_residual_correction(
                        source_frame, cal_idx, test_idx, cal_y - cal_mean_pred
                    )
                    local_median = local_residual_correction(
                        source_frame, cal_idx, test_idx, cal_y - cal_median_pred
                    )
                    model_predictions[spec.name] = {
                        "mean_l0": base_mean,
                        "mean_l05": base_mean + 0.5 * local_mean,
                        "mean_l1": base_mean + local_mean,
                        "median_l0": base_median,
                        "median_l05": base_median + 0.5 * local_median,
                        "median_l1": base_median + local_median,
                    }

                for position, idx in enumerate(test_idx):
                    row = {
                        "repeat": repeat,
                        "seed": seed,
                        "selection": selection,
                        "source_id": source,
                        "feedstock_name": data.loc[idx, "feedstock_name"],
                        "row_id": data.loc[idx, "row_id"],
                        "actual_class": int(y_class[idx]),
                        "actual_yield": float(data.loc[idx, "yield_pct"]),
                        "pclass_0": float(class_probability[position, 0]),
                        "pclass_1": float(class_probability[position, 1]),
                        "pclass_2": float(class_probability[position, 2]),
                        "pknn_0": float(knn_probability[position, 0]),
                        "pknn_1": float(knn_probability[position, 1]),
                        "pknn_2": float(knn_probability[position, 2]),
                        "cal_class_nunique": int(np.count_nonzero(cal_counts)),
                        "cal_low_n": int(cal_counts[0]),
                        "cal_mid_n": int(cal_counts[1]),
                        "cal_high_n": int(cal_counts[2]),
                        "cal_yield_mean": float(np.mean(cal_y)),
                        "cal_yield_range": float(np.ptp(cal_y)),
                    }
                    for model_name, variants in model_predictions.items():
                        for variant, values in variants.items():
                            row[f"{model_name}__{variant}"] = float(values[position])
                    rows.append(row)
    return pd.DataFrame(rows)


def candidate_grid(primitives: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    yhat_columns = [column for column in primitives if "__" in column]
    for (repeat, selection), frame in primitives.groupby(["repeat", "selection"]):
        truth = frame["actual_class"].to_numpy(int)
        sources = frame["source_id"].astype(str).to_numpy()
        feedstocks = frame["feedstock_name"].astype(str).to_numpy()
        pclass = frame[["pclass_0", "pclass_1", "pclass_2"]].to_numpy(float)
        pknn = frame[["pknn_0", "pknn_1", "pknn_2"]].to_numpy(float)
        candidates: list[tuple[str, np.ndarray]] = [
            ("v12_classifier", pclass),
            ("calibration_knn", pknn),
            ("class_knn_equal", 0.5 * pclass + 0.5 * pknn),
            ("class_knn_class75", 0.75 * pclass + 0.25 * pknn),
            ("class_knn_knn75", 0.25 * pclass + 0.75 * pknn),
        ]
        for column in yhat_columns:
            yhat = frame[column].to_numpy(float)
            for scale in (0.8, 1.2, 1.8):
                preg = regression_probability(yhat, scale)
                candidates.append((f"{column}__s{scale:g}__w1", preg))
                for weight in (0.5, 0.75):
                    blended = weight * preg + (1.0 - weight) * pclass
                    candidates.append((f"{column}__s{scale:g}__w{weight:g}", blended))
                candidates.append((
                    f"{column}__s{scale:g}__tri_50_25_25",
                    0.50 * pclass + 0.25 * preg + 0.25 * pknn,
                ))
                candidates.append((
                    f"{column}__s{scale:g}__tri_40_30_30",
                    0.40 * pclass + 0.30 * preg + 0.30 * pknn,
                ))
                candidates.append((
                    f"{column}__s{scale:g}__tri_25_50_25",
                    0.25 * pclass + 0.50 * preg + 0.25 * pknn,
                ))
                tri_probability = 0.25 * pclass + 0.50 * preg + 0.25 * pknn
                dominant_calibration = frame[
                    ["cal_low_n", "cal_mid_n", "cal_high_n"]
                ].to_numpy(int).max(axis=1) >= 4
                adaptive_probability = np.where(
                    dominant_calibration[:, None], pclass, tri_probability
                )
                candidates.append((
                    f"{column}__s{scale:g}__adaptive_dom4",
                    adaptive_probability,
                ))
        for candidate, probability in candidates:
            pred = np.argmax(probability, axis=1)
            source_accuracies = [
                float(np.mean(truth[sources == src] == pred[sources == src]))
                for src in np.unique(sources)
            ]
            metric_rows.append({
                "repeat": int(repeat), "selection": selection, "candidate": candidate,
                "n_test": len(frame), "worst_source_accuracy": min(source_accuracies),
                **v12.metrics(truth, pred, sources, feedstocks),
            })
            if candidate in {
                "v12_classifier", "calibration_knn", "class_knn_equal",
                "class_knn_class75", "class_knn_knn75",
            }:
                continue
            for (_, row), predicted in zip(frame.iterrows(), pred):
                prediction_rows.append({
                    "repeat": int(repeat), "selection": selection, "candidate": candidate,
                    "source_id": row["source_id"], "feedstock_name": row["feedstock_name"],
                    "row_id": row["row_id"], "actual_class": int(row["actual_class"]),
                    "predicted_class": int(predicted), "actual_yield": row["actual_yield"],
                })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = metrics.groupby(["selection", "candidate"], as_index=False).agg(
        repeats=("repeat", "size"), accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"), balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        publication_balanced_accuracy_mean=("publication_balanced_accuracy", "mean"),
        feedstock_balanced_accuracy_mean=("feedstock_balanced_accuracy", "mean"),
        worst_source_accuracy_mean=("worst_source_accuracy", "mean"),
        ordinal_mae_mean=("ordinal_mae", "mean"),
    )
    summary["selection_score"] = (
        0.30 * summary["accuracy_mean"]
        + 0.25 * summary["macro_f1_mean"]
        + 0.20 * summary["balanced_accuracy_mean"]
        + 0.15 * summary["publication_balanced_accuracy_mean"]
        + 0.10 * summary["worst_source_accuracy_mean"]
    )
    return summary.sort_values("selection_score", ascending=False)


def main() -> None:
    primitive_path = OUT / "development_primitives.csv"
    if os.environ.get("V13_REGRID_ONLY") == "1" and primitive_path.exists():
        primitives = pd.read_csv(primitive_path)
    else:
        data = v12.load_all()
        primitives = primitive_predictions(data, k=5)
    metrics, predictions = candidate_grid(primitives)
    summary = summarize(metrics)
    primitives.to_csv(OUT / "development_primitives.csv", index=False)
    metrics.to_csv(OUT / "development_metrics.csv", index=False)
    predictions.to_csv(OUT / "development_predictions.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv", index=False)
    best = summary.iloc[0].to_dict()
    baseline = summary[summary["candidate"].eq("v12_classifier")].sort_values(
        "selection_score", ascending=False
    ).iloc[0].to_dict()
    decision = {
        "phase": "development",
        "seeds": DEVELOPMENT_SEEDS,
        "k_calibration": 5,
        "best_candidate": best,
        "same_seed_v12_baseline": baseline,
        "selection_rule": "Composite score includes accuracy, Macro F1, balanced accuracy, publication balance, and worst-source accuracy.",
        "leakage_boundary": "Calibration rows are selected without outcomes. Target-source test outcomes are never used for fitting, baseline estimation, or selection.",
    }
    (OUT / "development_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(summary.head(30).to_string(index=False), flush=True)
    print("\nDECISION", flush=True)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
