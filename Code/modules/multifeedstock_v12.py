from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor_ml_extract"
if not VENDOR.exists():
    VENDOR = ROOT / "vendor_sklearn"
sys.path.insert(0, str(VENDOR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import boost70_search as boost
import sant_orf_v4 as v4


OUT = ROOT / "results_v12_multifeedstock"
OUT.mkdir(exist_ok=True)
PRIMARY = ROOT / "training_primary_v10_1.csv"
ABSOLUTE_CUTS = np.array([11.096667, 14.403333], dtype=float)

SOLVENT = {
    "n_hexane": {"density_g_ml": 0.655, "bp_c": 69.0, "dipole_moment": 0.0},
    "chloroform": {"density_g_ml": 1.489, "bp_c": 61.2, "dipole_moment": 1.04},
}


ARGEMONE = [
    (10, 3, 120, 46.88), (5, 3, 80, 39.71), (15, 3, 160, 50.39),
    (5, 5, 120, 44.26), (15, 5, 120, 48.08), (10, 3, 120, 45.83),
    (10, 3, 120, 45.93), (10, 5, 160, 50.25), (5, 1, 120, 39.46),
    (10, 1, 160, 44.20), (5, 3, 160, 49.95), (15, 3, 80, 43.98),
    (10, 1, 80, 36.18), (10, 5, 80, 41.12), (10, 3, 120, 46.78),
    (15, 1, 120, 40.82), (10, 3, 120, 45.73),
]

BUTTER_CODED = [
    (1, 1, 0, 13.1102), (0, 0, 0, 17.8780), (0, 1, 1, 13.5602),
    (0, -1, 1, 12.6760), (1, -1, 0, 9.6400), (0, 1, -1, 8.8878),
    (-1, 0, -1, 9.0790), (0, 0, 0, 12.1744), (0, 0, 0, 9.9718),
    (-1, 0, 1, 11.4024), (0, 0, 0, 9.4003), (-1, 1, 0, 12.5876),
    (1, 0, 1, 8.1188), (1, 0, -1, 9.2843), (-1, -1, 0, 14.1445),
    (0, -1, 1, 11.3265), (0, 0, -1, 10.0145),
]

PAPAYA = [
    (50, 50, 225, 17.00), (60, 50, 225, 17.33), (50, 60, 225, 15.87),
    (60, 60, 225, 17.45), (50, 55, 200, 16.10), (60, 55, 200, 15.03),
    (50, 55, 250, 17.40), (60, 55, 250, 20.50), (55, 50, 200, 17.64),
    (55, 60, 200, 20.40), (55, 50, 250, 23.67), (55, 60, 250, 20.51),
    (55, 55, 225, 21.07), (55, 55, 225, 21.07), (55, 55, 225, 21.07),
    (55, 55, 225, 21.07), (55, 55, 225, 22.50),
]

PALM_KERNEL = [
    (30, 150, 50, 30.92), (50, 150, 50, 36.86), (30, 200, 50, 35.97),
    (50, 200, 50, 35.72), (30, 175, 40, 31.56), (50, 175, 40, 37.47),
    (30, 175, 60, 37.60), (50, 175, 60, 37.47), (40, 150, 40, 33.08),
    (40, 200, 40, 36.31), (40, 150, 60, 34.84), (40, 200, 60, 34.62),
    (40, 175, 50, 33.07), (40, 175, 50, 37.47), (40, 175, 50, 37.47),
    (40, 175, 50, 37.50), (40, 175, 50, 32.99),
]


def external_row(
    source_id: str,
    run: int,
    feedstock_name: str,
    feedstock_family: str,
    plant_part: str,
    sample_g: float,
    solvent_ml: float,
    time_h: float,
    yield_pct: float,
    solvent: str,
    temperature_c: float,
    doi: str,
    source_url: str,
    quality_tier: str,
    dry_temperature_c: float = np.nan,
    dry_time_h: float = np.nan,
    particle_size_mid_um: float = np.nan,
) -> dict:
    props = SOLVENT[solvent]
    return {
        "row_id": f"{source_id}_{run:02d}",
        **props,
        "pressure_bar": 1.0,
        "ratio_g_ml": sample_g / solvent_ml,
        "time_h": time_h,
        "temperature_c": temperature_c,
        "dry_temperature_c": dry_temperature_c,
        "dry_time_h": dry_time_h,
        "particle_size_mid_um": particle_size_mid_um,
        "yield_pct": yield_pct,
        "source_id": source_id,
        "publication_cluster_id": source_id,
        "dataset_origin": "v12_multifeedstock_external",
        "doi": doi,
        "source_url": source_url,
        "method": "Soxhlet",
        "method_domain": "primary_soxhlet",
        "solvent": solvent,
        "feedstock_name": feedstock_name,
        "feedstock_family": feedstock_family,
        "plant_part": plant_part,
        "prior_processing": "native_seed",
        "quality_tier": quality_tier,
        "sample_weight_g": sample_g,
        "solvent_volume_ml": solvent_ml,
    }


def build_external() -> pd.DataFrame:
    rows: list[dict] = []
    for i, (sample, time_h, volume, value) in enumerate(ARGEMONE, 1):
        rows.append(external_row(
            "Gebreegziabher2022_Argemone", i, "argemone_mexicana_seed",
            "nonedible_oilseed", "seed", sample, volume, time_h, value,
            "chloroform", 61.2, "10.1016/j.rechem.2022.100570",
            "https://www.sciencedirect.com/science/article/pii/S2211715622002892",
            "gold",
        ))
    for i, (sv_code, sw_code, et_code, value) in enumerate(BUTTER_CODED, 1):
        volume = {-1: 180, 0: 200, 1: 220}[sv_code]
        sample = {-1: 40, 0: 50, 1: 60}[sw_code]
        minutes = {-1: 50, 0: 55, 1: 60}[et_code]
        rows.append(external_row(
            "Adepoju2019_ButterFruit", i, "dacryodes_edulis_seed",
            "fruit_seed", "seed", sample, volume, minutes / 60.0, value,
            "n_hexane", 69.0, "10.4314/njtd.v16i2.2",
            "https://journal.njtd.com.ng/index.php/njtd/article/view/268",
            "silver", dry_time_h=120.0,
        ))
    for i, (sample, minutes, volume, value) in enumerate(PAPAYA, 1):
        rows.append(external_row(
            "Amabogha2023_Papaya", i, "carica_papaya_seed",
            "fruit_seed", "seed", sample, volume, minutes / 60.0, value,
            "n_hexane", 70.0, "10.30574/gjeta.2023.15.3.0093",
            "https://gjeta.com/sites/default/files/GJETA-2023-0093.pdf",
            "gold", dry_temperature_c=60.0, particle_size_mid_um=425.0,
        ))
    for i, (sample, volume, minutes, value) in enumerate(PALM_KERNEL, 1):
        rows.append(external_row(
            "Ijaola2021_PalmKernel", i, "elaeis_guineensis_kernel",
            "oil_palm_kernel", "kernel", sample, volume, minutes / 60.0, value,
            "n_hexane", 69.0, "",
            "https://www.discoveryjournals.org/engineering/current_issue/2021/v18/n50/A11.pdf",
            "silver", dry_time_h=48.0,
        ))
    return pd.DataFrame(rows)


def load_all() -> pd.DataFrame:
    primary = pd.read_csv(PRIMARY)
    primary["publication_cluster_id"] = primary["source_id"].astype(str)
    primary["feedstock_name"] = "spent_coffee_grounds"
    primary["feedstock_family"] = "spent_biomass_residue"
    primary["plant_part"] = "brewed_ground_residue"
    primary["prior_processing"] = "roasted_and_brewed"
    primary["sample_weight_g"] = np.nan
    primary["solvent_volume_ml"] = np.nan
    primary["quality_tier"] = primary.get("quality_tier", "legacy_screened")
    external = build_external()
    all_data = pd.concat([primary, external], ignore_index=True, sort=False)
    for column in v4.CORE_FEATURES + ["particle_size_mid_um", "yield_pct"]:
        all_data[column] = pd.to_numeric(all_data[column], errors="coerce")
    all_data["source_id"] = all_data["source_id"].astype(str)
    return add_targets_and_batch_features(all_data)


def add_targets_and_batch_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    engineered = boost.physical_features(data)
    for column in engineered.columns:
        data[column] = engineered[column]
    data["absolute_class"] = np.digitize(data["yield_pct"], ABSOLUTE_CUTS, right=True)
    within_rank = data.groupby("source_id")["yield_pct"].rank(method="average", pct=True)
    data["relative_class"] = np.digitize(within_rank, [1 / 3, 2 / 3], right=True)
    relative_inputs = [
        "ratio_g_ml", "time_h", "temperature_c", "dry_temperature_c",
        "dry_time_h", "particle_size_mid_um", "density_g_ml", "bp_c",
        "dipole_moment",
    ]
    for column in relative_inputs:
        data[f"batchrank_{column}"] = data.groupby("source_id")[column].rank(
            method="average", pct=True, na_option="keep"
        )
    data["batch_solvent_loading_rank"] = data.groupby("source_id")["ratio_g_ml"].rank(
        method="average", pct=True, ascending=False, na_option="keep"
    )
    data["source_batch_n"] = data.groupby("source_id")["source_id"].transform("size")
    return data


BASE_NUMERIC = list(dict.fromkeys(v4.CORE_FEATURES + ["particle_size_mid_um"] + list(boost.physical_features(
    pd.DataFrame({c: [np.nan] for c in v4.CORE_FEATURES})
).columns)))
BATCH_NUMERIC = BASE_NUMERIC + [
    c for c in [
        "batchrank_ratio_g_ml", "batchrank_time_h", "batchrank_temperature_c",
        "batchrank_dry_temperature_c", "batchrank_dry_time_h",
        "batchrank_particle_size_mid_um", "batchrank_density_g_ml", "batchrank_bp_c",
        "batchrank_dipole_moment", "batch_solvent_loading_rank",
    ]
]
MATERIAL_CLASS = ["feedstock_family", "plant_part", "prior_processing"]
MATERIAL_FULL = ["feedstock_name"] + MATERIAL_CLASS


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    numeric: str
    categorical: str
    max_depth: int = 6
    min_leaf: int = 2


CANDIDATES = [
    Candidate("logit_process", "logit", "base", "none"),
    Candidate("extra_process", "extra", "base", "none"),
    Candidate("extra_material_class", "extra", "base", "class"),
    Candidate("extra_feedstock_name", "extra", "base", "full"),
    Candidate("extra_material_batch", "extra", "batch", "class"),
    Candidate("extra_feedstock_batch", "extra", "batch", "full"),
    Candidate("gb_material_batch", "gb", "batch", "class", max_depth=2, min_leaf=3),
]


def columns_for(candidate: Candidate) -> tuple[list[str], list[str]]:
    numeric = BASE_NUMERIC if candidate.numeric == "base" else BATCH_NUMERIC
    categorical = {
        "none": [], "class": MATERIAL_CLASS, "full": MATERIAL_FULL,
    }[candidate.categorical]
    return numeric, categorical


def make_model(candidate: Candidate, seed: int) -> Pipeline:
    numeric, categorical = columns_for(candidate)
    transformers = [
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", RobustScaler() if candidate.family == "logit" else "passthrough"),
        ]), numeric)
    ]
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    prep = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
    if candidate.family == "logit":
        model = LogisticRegression(C=0.5, max_iter=4000, class_weight="balanced", random_state=seed)
    elif candidate.family == "extra":
        model = ExtraTreesClassifier(
            n_estimators=120, max_depth=candidate.max_depth, min_samples_leaf=candidate.min_leaf,
            max_features=0.8, class_weight="balanced", random_state=seed, n_jobs=1,
        )
    elif candidate.family == "rf":
        model = RandomForestClassifier(
            n_estimators=120, max_depth=candidate.max_depth, min_samples_leaf=candidate.min_leaf,
            max_features=0.8, class_weight="balanced_subsample", random_state=seed, n_jobs=1,
        )
    elif candidate.family == "gb":
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.045, max_depth=candidate.max_depth,
            min_samples_leaf=candidate.min_leaf, random_state=seed,
        )
    else:
        raise ValueError(candidate.family)
    return Pipeline([("prep", prep), ("model", model)])


def metrics(truth: np.ndarray, pred: np.ndarray, sources: np.ndarray, feedstocks: np.ndarray) -> dict:
    source_values = [np.mean(truth[sources == s] == pred[sources == s]) for s in np.unique(sources)]
    feedstock_values = [
        np.mean(truth[feedstocks == f] == pred[feedstocks == f]) for f in np.unique(feedstocks)
    ]
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "publication_balanced_accuracy": float(np.mean(source_values)),
        "feedstock_balanced_accuracy": float(np.mean(feedstock_values)),
        "ordinal_mae": float(np.mean(np.abs(truth - pred))),
    }


def repeated_group_splits(data: pd.DataFrame, target: np.ndarray, seeds: list[int]):
    groups = data["source_id"].astype(str).to_numpy()
    for repeat, seed in enumerate(seeds):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(data)), target, groups)):
            yield repeat, fold, train, test


def evaluate_candidates(data: pd.DataFrame, target_column: str, seeds: list[int], phase: str) -> pd.DataFrame:
    y = data[target_column].to_numpy(int)
    sources = data["source_id"].astype(str).to_numpy()
    feedstocks = data["feedstock_name"].astype(str).to_numpy()
    rows = []
    for repeat, fold, train, test in repeated_group_splits(data, y, seeds):
        assert not set(sources[train]) & set(sources[test])
        for ci, candidate in enumerate(CANDIDATES):
            model = make_model(candidate, seeds[repeat] * 1000 + fold * 100 + ci)
            model.fit(data.iloc[train], y[train])
            pred = model.predict(data.iloc[test]).astype(int)
            rows.append({
                "phase": phase, "target": target_column, "repeat": repeat, "fold": fold,
                "candidate": candidate.name, "train_n": len(train), "test_n": len(test),
                **metrics(y[test], pred, sources[test], feedstocks[test]),
            })
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    summary = detail.groupby(["phase", "target", "candidate"], as_index=False).agg(
        folds=("fold", "size"), accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"), balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        publication_balanced_accuracy_mean=("publication_balanced_accuracy", "mean"),
        feedstock_balanced_accuracy_mean=("feedstock_balanced_accuracy", "mean"),
        ordinal_mae_mean=("ordinal_mae", "mean"),
    )
    summary["selection_score"] = (
        0.35 * summary["accuracy_mean"] + 0.25 * summary["macro_f1_mean"]
        + 0.20 * summary["balanced_accuracy_mean"]
        + 0.10 * summary["publication_balanced_accuracy_mean"]
        + 0.10 * summary["feedstock_balanced_accuracy_mean"]
    )
    return summary.sort_values(["target", "selection_score"], ascending=[True, False])


def locked_confirmation(
    data: pd.DataFrame, target_column: str, candidate: Candidate, seeds: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = data[target_column].to_numpy(int)
    sources = data["source_id"].astype(str).to_numpy()
    feedstocks = data["feedstock_name"].astype(str).to_numpy()
    rows, predictions = [], []
    for repeat, fold, train, test in repeated_group_splits(data, y, seeds):
        model = make_model(candidate, seeds[repeat] * 1000 + fold)
        model.fit(data.iloc[train], y[train])
        pred = model.predict(data.iloc[test]).astype(int)
        rows.append({
            "phase": "locked_confirmation", "target": target_column, "repeat": repeat,
            "fold": fold, "candidate": candidate.name, **metrics(
                y[test], pred, sources[test], feedstocks[test]
            ),
        })
        for idx, actual, predicted in zip(test, y[test], pred):
            predictions.append({
                "target": target_column, "repeat": repeat, "fold": fold,
                "candidate": candidate.name, "row_id": data.iloc[idx]["row_id"],
                "source_id": sources[idx], "feedstock_name": feedstocks[idx],
                "actual_class": int(actual), "predicted_class": int(predicted),
            })
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def leave_one_domain_out(
    data: pd.DataFrame, target_column: str, candidate: Candidate, domain_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = data[target_column].to_numpy(int)
    sources = data["source_id"].astype(str).to_numpy()
    feedstocks = data["feedstock_name"].astype(str).to_numpy()
    rows, predictions = [], []
    for fold, domain in enumerate(sorted(data[domain_column].astype(str).unique())):
        test = np.flatnonzero(data[domain_column].astype(str).to_numpy() == domain)
        train = np.flatnonzero(data[domain_column].astype(str).to_numpy() != domain)
        if len(np.unique(y[train])) < 2:
            continue
        model = make_model(candidate, 126000 + fold)
        model.fit(data.iloc[train], y[train])
        pred = model.predict(data.iloc[test]).astype(int)
        rows.append({
            "target": target_column, "domain_kind": domain_column, "held_out_domain": domain,
            "candidate": candidate.name, "train_n": len(train), "test_n": len(test),
            **metrics(y[test], pred, sources[test], feedstocks[test]),
        })
        for idx, actual, predicted in zip(test, y[test], pred):
            predictions.append({
                "target": target_column, "domain_kind": domain_column, "held_out_domain": domain,
                "row_id": data.iloc[idx]["row_id"], "source_id": sources[idx],
                "feedstock_name": feedstocks[idx], "actual_class": int(actual),
                "predicted_class": int(predicted),
            })
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def majority_baselines(data: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    for target_column in targets:
        y = data[target_column].to_numpy(int)
        majority = int(np.bincount(y, minlength=3).argmax())
        pred = np.full(len(y), majority)
        rows.append({
            "target": target_column, "majority_class": majority,
            **metrics(y, pred, data["source_id"].astype(str).to_numpy(), data["feedstock_name"].astype(str).to_numpy()),
        })
    return pd.DataFrame(rows)


def full_model_audit(data: pd.DataFrame, target_column: str, candidate: Candidate) -> tuple[pd.DataFrame, str]:
    y = data[target_column].to_numpy(int)
    model = make_model(candidate, 129001)
    model.fit(data, y)
    names = model.named_steps["prep"].get_feature_names_out()
    estimator = model.named_steps["model"]
    importance = getattr(estimator, "feature_importances_", None)
    if importance is None:
        importance = np.mean(np.abs(estimator.coef_), axis=0)
    importance_table = pd.DataFrame({"feature": names, "importance": importance}).sort_values(
        "importance", ascending=False
    )
    transformed = model.named_steps["prep"].transform(data)
    tree = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=129002
    ).fit(transformed, y)
    rules = export_text(tree, feature_names=list(names), decimals=3)
    return importance_table, rules


def source_quality(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, frame in data.groupby("source_id"):
        condition_columns = ["ratio_g_ml", "time_h", "temperature_c"]
        center = frame.copy()
        for column in condition_columns:
            if center[column].notna().any():
                median = center[column].median()
                center = center[np.isclose(center[column], median, equal_nan=False)]
        center_range = float(center["yield_pct"].max() - center["yield_pct"].min()) if len(center) > 1 else np.nan
        rows.append({
            "source_id": source, "feedstock_name": frame["feedstock_name"].iloc[0],
            "n_rows": len(frame), "yield_min": frame["yield_pct"].min(),
            "yield_max": frame["yield_pct"].max(), "center_like_n": len(center),
            "center_like_yield_range": center_range,
            "quality_tier": frame["quality_tier"].iloc[0],
        })
    return pd.DataFrame(rows)


def main() -> None:
    data = load_all()
    data.to_csv(OUT / "multifeedstock_training_data.csv", index=False)
    build_external().to_csv(OUT / "external_rows_verified.csv", index=False)
    source_quality(data).to_csv(OUT / "source_quality_audit.csv", index=False)

    inventory = {
        "total_rows": int(len(data)), "publication_sources": int(data["source_id"].nunique()),
        "feedstocks": int(data["feedstock_name"].nunique()),
        "scg_rows": int(data["feedstock_name"].eq("spent_coffee_grounds").sum()),
        "external_feedstock_rows": int(data["feedstock_name"].ne("spent_coffee_grounds").sum()),
        "absolute_class_counts": {str(k): int(v) for k, v in data["absolute_class"].value_counts().sort_index().items()},
        "claim_boundary": (
            "Absolute classes retain the locked SCG thresholds. High absolute accuracy can be inflated by "
            "between-feedstock oil-content differences; Macro F1, publication holdout, and leave-one-feedstock-out are mandatory."
        ),
        "batch_feature_boundary": (
            "Batch-rank predictors use only the unlabeled condition matrix of a submitted batch, never test yields."
        ),
    }
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    target_data = {
        "absolute_class": data,
        "relative_class": data[data["source_batch_n"] >= 3].reset_index(drop=True),
    }
    majority = pd.concat(
        [majority_baselines(frame, [target]) for target, frame in target_data.items()], ignore_index=True
    )
    majority.to_csv(OUT / "majority_baselines.csv", index=False)

    development_parts, locks, confirmation_parts, prediction_parts = [], {}, [], []
    dev_seeds = [12001, 12002]
    confirm_seeds = [12101, 12102, 12103]
    for target, frame in target_data.items():
        detail = evaluate_candidates(frame, target, dev_seeds, "development")
        development_parts.append(detail)
        summary = summarize(detail)
        winner_name = str(summary.iloc[0]["candidate"])
        winner = next(c for c in CANDIDATES if c.name == winner_name)
        locks[target] = {
            "selected_candidate": winner_name,
            "development_metrics": summary.iloc[0].to_dict(),
            "selection_rule": "35% accuracy + 25% Macro F1 + 20% balanced accuracy + 10% publication-balanced + 10% feedstock-balanced accuracy",
        }
        confirmed, predictions = locked_confirmation(frame, target, winner, confirm_seeds)
        confirmation_parts.append(confirmed)
        prediction_parts.append(predictions)
        importance, rules = full_model_audit(frame, target, winner)
        importance.to_csv(OUT / f"{target}_feature_importance.csv", index=False)
        (OUT / f"{target}_depth4_rules.txt").write_text(rules, encoding="utf-8")

        lofo, lofo_pred = leave_one_domain_out(frame, target, winner, "feedstock_name")
        loso, loso_pred = leave_one_domain_out(frame, target, winner, "source_id")
        lofo.to_csv(OUT / f"{target}_leave_one_feedstock_out.csv", index=False)
        lofo_pred.to_csv(OUT / f"{target}_leave_one_feedstock_out_predictions.csv", index=False)
        loso.to_csv(OUT / f"{target}_leave_one_source_out.csv", index=False)
        loso_pred.to_csv(OUT / f"{target}_leave_one_source_out_predictions.csv", index=False)

    development = pd.concat(development_parts, ignore_index=True)
    development.to_csv(OUT / "development_details.csv", index=False)
    summarize(development).to_csv(OUT / "development_summary.csv", index=False)
    (OUT / "development_locks.json").write_text(json.dumps(locks, indent=2), encoding="utf-8")

    confirmation = pd.concat(confirmation_parts, ignore_index=True)
    confirmation.to_csv(OUT / "locked_confirmation_details.csv", index=False)
    summarize(confirmation).to_csv(OUT / "locked_confirmation_summary.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_csv(OUT / "locked_confirmation_predictions.csv", index=False)

    print(json.dumps(inventory, indent=2), flush=True)
    print("\nDEVELOPMENT", flush=True)
    print(summarize(development).to_string(index=False), flush=True)
    print("\nLOCKS", flush=True)
    print(json.dumps(locks, indent=2), flush=True)
    print("\nCONFIRMATION", flush=True)
    print(summarize(confirmation).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
