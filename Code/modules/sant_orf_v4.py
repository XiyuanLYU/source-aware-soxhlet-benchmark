from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor_ml_extract"
if not VENDOR.exists():
    VENDOR = ROOT / "vendor_sklearn"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import RobustScaler


ORIGINAL_DATA = ROOT / "source" / "three_bin_data_original.xlsx"
NEW_DATA_FILES = [
    ROOT / "new_data_validated_2025.csv",
    ROOT / "new_data_validated_2026_bbd.csv",
]
OUT = ROOT / "results_v4"
OUT.mkdir(exist_ok=True)

YIELD_CUTS = np.array([11.096667, 14.403333], dtype=float)

CORE_FEATURES = [
    "density_g_ml",
    "bp_c",
    "dipole_moment",
    "pressure_bar",
    "ratio_g_ml",
    "time_h",
    "temperature_c",
    "dry_temperature_c",
    "dry_time_h",
]
EXTENDED_FEATURES = CORE_FEATURES + ["particle_size_mid_um"]

FEATURE_LABELS = {
    "density_g_ml": "Solvent density",
    "bp_c": "Solvent boiling point",
    "dipole_moment": "Solvent dipole moment",
    "pressure_bar": "Pressure",
    "ratio_g_ml": "SCG-to-solvent ratio",
    "time_h": "Extraction time",
    "temperature_c": "Extraction temperature",
    "dry_temperature_c": "Drying temperature",
    "dry_time_h": "Drying time",
    "particle_size_mid_um": "Particle size",
}


def entropy(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p = np.bincount(np.asarray(y, int), minlength=3).astype(float)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def inverse_source_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(map(str, groups))
    w = np.array([1.0 / counts[str(g)] for g in groups], dtype=float)
    return w / np.mean(w)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    original = pd.read_excel(ORIGINAL_DATA, sheet_name="Sheet1")
    original = original.rename(
        columns={
            "density (g/ml)": "density_g_ml",
            "BP": "bp_c",
            "Dipole moment": "dipole_moment",
            "pressure(bar)": "pressure_bar",
            "ratio": "ratio_g_ml",
            "time": "time_h",
            "temperature": "temperature_c",
            "dry temperature": "dry_temperature_c",
            "dry time": "dry_time_h",
            "yield": "yield_pct",
            "reference in article": "source_id",
        }
    )
    original["source_id"] = "OriginalRef_" + original["source_id"].astype(str)
    original["particle_size_mid_um"] = np.nan
    original["dataset_origin"] = "original_99"
    original["doi"] = np.nan
    original["source_url"] = np.nan
    original["yield_sd_pct"] = np.nan
    original["n_replicates"] = np.nan
    original["method"] = np.nan

    new_frames = []
    for path in NEW_DATA_FILES:
        frame = pd.read_csv(path)
        if "dataset_origin" not in frame.columns:
            if "2025" in path.name:
                frame["dataset_origin"] = "Krinski_2025"
            else:
                frame["dataset_origin"] = "Alhanif_2026"
        new_frames.append(frame)
    new = pd.concat(new_frames, ignore_index=True)

    keep = list(
        dict.fromkeys(
            CORE_FEATURES
            + [
                "particle_size_mid_um",
                "yield_pct",
                "source_id",
                "dataset_origin",
                "doi",
                "source_url",
                "yield_sd_pct",
                "n_replicates",
                "method",
            ]
        )
    )
    for frame in (original, new):
        for column in keep:
            if column not in frame:
                frame[column] = np.nan
    original = original[keep].copy()
    new = new[keep].copy()
    for column in CORE_FEATURES + ["particle_size_mid_um", "yield_pct"]:
        original[column] = pd.to_numeric(original[column], errors="coerce")
        new[column] = pd.to_numeric(new[column], errors="coerce")
    augmented = pd.concat([original, new], ignore_index=True)
    return original, augmented


def target_classes(yield_pct: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(yield_pct, float), YIELD_CUTS, right=True).astype(int)


def best_two_cuts(values: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    valid = np.isfinite(values)
    v = values[valid]
    labels = y[valid]
    if len(v) < 12 or np.unique(v).size < 3:
        if len(v) == 0:
            return np.array([0.0, 1.0]), 0.0
        q = np.quantile(v, [1 / 3, 2 / 3])
        if q[0] == q[1]:
            q = np.array([np.min(v), np.max(v)])
        return np.asarray(q, float), 0.0

    candidates = np.unique(np.quantile(v, np.linspace(0.10, 0.90, 9)))
    base = entropy(labels)
    minimum_bin = max(3, int(round(0.07 * len(v))))
    best_score = -np.inf
    best_cuts = np.asarray(np.quantile(v, [1 / 3, 2 / 3]), float)
    for left, right in combinations(candidates, 2):
        if right <= left:
            continue
        z = np.digitize(v, [left, right], right=True)
        sizes = np.bincount(z, minlength=3)
        if np.min(sizes) < minimum_bin:
            continue
        conditional = sum(
            (sizes[k] / len(v)) * entropy(labels[z == k]) for k in range(3)
        )
        information_gain = base - conditional
        bin_means = np.array(
            [np.mean(labels[z == k]) if sizes[k] else 0.0 for k in range(3)]
        )
        ordered_separation = float(np.average((bin_means - np.mean(labels)) ** 2, weights=sizes))
        imbalance = float(np.std(sizes / sizes.sum()))
        score = information_gain + 0.12 * ordered_separation - 0.08 * imbalance
        if score > best_score:
            best_score = score
            best_cuts = np.array([left, right], dtype=float)
    return best_cuts, float(max(best_score, 0.0))


class StableAdaptiveBinner:
    def __init__(self, n_bootstrap: int = 12, random_state: int = 0):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        self.columns_ = list(X.columns)
        unique_groups = np.unique(groups.astype(str))
        self.cuts_ = {}
        self.gain_ = {}
        self.cut_stability_ = {}
        self.coverage_ = {}
        for column in self.columns_:
            values = X[column].to_numpy(float)
            boot_cuts = []
            boot_gains = []
            base_cuts, base_gain = best_two_cuts(values, y)
            boot_cuts.append(base_cuts)
            boot_gains.append(base_gain)
            for _ in range(self.n_bootstrap):
                sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                indices = np.concatenate(
                    [np.flatnonzero(groups.astype(str) == g) for g in sampled_groups]
                )
                cuts, gain = best_two_cuts(values[indices], y[indices])
                boot_cuts.append(cuts)
                boot_gains.append(gain)
            cut_matrix = np.vstack(boot_cuts)
            final_cuts = np.nanmedian(cut_matrix, axis=0)
            if final_cuts[0] > final_cuts[1]:
                final_cuts = final_cuts[::-1]
            observed = values[np.isfinite(values)]
            scale = max(float(np.ptp(observed)) if len(observed) else 1.0, 1e-9)
            cut_iqr = np.nanmean(
                np.quantile(cut_matrix, 0.75, axis=0)
                - np.quantile(cut_matrix, 0.25, axis=0)
            )
            self.cuts_[column] = final_cuts
            self.gain_[column] = float(np.mean(boot_gains))
            self.cut_stability_[column] = float(np.clip(1.0 - cut_iqr / scale, 0.0, 1.0))
            self.coverage_[column] = float(np.mean(np.isfinite(values)))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = np.full((len(X), len(self.columns_)), -1, dtype=int)
        for j, column in enumerate(self.columns_):
            values = X[column].to_numpy(float)
            valid = np.isfinite(values)
            Z[valid, j] = np.digitize(values[valid], self.cuts_[column], right=True)
        return Z

    def base_scores(self) -> np.ndarray:
        return np.array(
            [
                self.gain_[c]
                * (0.50 + 0.50 * self.cut_stability_[c])
                * math.sqrt(max(self.coverage_[c], 0.0))
                for c in self.columns_
            ],
            dtype=float,
        )


def neighborhood_dependency(
    Z: np.ndarray,
    y: np.ndarray,
    row_weights: np.ndarray,
    k: int = 7,
    beta: float = 0.70,
) -> float:
    if Z.ndim == 1:
        Z = Z[:, None]
    n = len(Z)
    if n <= 2 or Z.shape[1] == 0:
        return 0.0
    memberships = np.zeros(n, dtype=float)
    ordinal_memberships = np.zeros(n, dtype=float)
    for i in range(n):
        valid = (Z[i][None, :] >= 0) & (Z >= 0)
        denominator = valid.sum(axis=1)
        differences = np.abs(Z - Z[i][None, :]) / 2.0
        distance = np.divide(
            (differences * valid).sum(axis=1),
            denominator,
            out=np.ones(n, dtype=float),
            where=denominator > 0,
        )
        distance[i] = np.inf
        neighbor_idx = np.argsort(distance)[: min(k, n - 1)]
        finite = np.isfinite(distance[neighbor_idx])
        neighbor_idx = neighbor_idx[finite]
        if len(neighbor_idx) == 0:
            continue
        local_weights = np.exp(-2.2 * distance[neighbor_idx]) * row_weights[neighbor_idx]
        if local_weights.sum() <= 0:
            continue
        memberships[i] = float(
            np.sum(local_weights * (y[neighbor_idx] == y[i])) / local_weights.sum()
        )
        ordinal_memberships[i] = float(
            np.sum(local_weights * (1.0 - np.abs(y[neighbor_idx] - y[i]) / 2.0))
            / local_weights.sum()
        )
    rough_membership = 0.72 * memberships + 0.28 * ordinal_memberships
    positive = rough_membership >= beta
    dependency = float(np.average(positive.astype(float), weights=row_weights))
    soft_quality = float(np.average(rough_membership, weights=row_weights))
    return 0.78 * dependency + 0.22 * soft_quality


class StabilityRoughReducer:
    def __init__(
        self,
        n_bootstrap: int = 10,
        min_features: int = 4,
        max_features: int = 8,
        random_state: int = 0,
    ):
        self.n_bootstrap = n_bootstrap
        self.min_features = min_features
        self.max_features = max_features
        self.random_state = random_state

    def _greedy(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        base_scores: np.ndarray,
    ) -> list[int]:
        selected: list[int] = []
        current = 0.0
        candidates = list(range(Z.shape[1]))
        while candidates and len(selected) < self.max_features:
            evaluations = []
            for feature in candidates:
                subset = selected + [feature]
                dep = neighborhood_dependency(Z[:, subset], y, weights)
                objective = dep + 0.035 * base_scores[feature] - 0.007 * len(subset)
                evaluations.append((objective, dep, feature))
            objective, dep, feature = max(evaluations)
            improvement = dep - current
            if len(selected) >= self.min_features and improvement < 0.012:
                break
            selected.append(feature)
            candidates.remove(feature)
            current = max(current, dep)
        if len(selected) < self.min_features:
            remaining = [j for j in np.argsort(base_scores)[::-1] if j not in selected]
            selected.extend(remaining[: self.min_features - len(selected)])
        return selected

    def fit(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        base_scores: np.ndarray,
        feature_names: list[str],
        eligible_mask: np.ndarray | None = None,
    ):
        rng = np.random.default_rng(self.random_state)
        groups = groups.astype(str)
        unique_groups = np.unique(groups)
        selection_counts = np.zeros(Z.shape[1], dtype=float)
        selections = []
        for bootstrap_id in range(self.n_bootstrap + 1):
            if bootstrap_id == 0:
                idx = np.arange(len(y))
            else:
                sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled_groups])
            weights = inverse_source_weights(groups[idx])
            selection = self._greedy(Z[idx], y[idx], weights, base_scores)
            selections.append(selection)
            selection_counts[selection] += 1
        frequency = selection_counts / len(selections)
        eligible = (
            np.ones(Z.shape[1], dtype=bool)
            if eligible_mask is None
            else np.asarray(eligible_mask, dtype=bool)
        )
        selection_counts[~eligible] = 0
        order = sorted(
            [j for j in range(Z.shape[1]) if eligible[j]],
            key=lambda j: (frequency[j], base_scores[j]),
            reverse=True,
        )
        selected = [j for j in order if frequency[j] >= 0.35][: self.max_features]
        if len(selected) < self.min_features:
            selected = order[: self.min_features]

        weights = inverse_source_weights(groups)
        individual = {
            j: neighborhood_dependency(Z[:, [j]], y, weights) for j in selected
        }
        interactions = []
        for left, right in combinations(selected, 2):
            pair = neighborhood_dependency(Z[:, [left, right]], y, weights)
            synergy = pair - max(individual[left], individual[right])
            interactions.append(
                {
                    "feature_1": feature_names[left],
                    "feature_2": feature_names[right],
                    "pair_dependency": pair,
                    "synergy": synergy,
                }
            )
        interactions.sort(key=lambda row: row["synergy"], reverse=True)
        self.selected_indices_ = selected
        self.selection_frequency_ = frequency
        self.interactions_ = interactions
        self.feature_names_ = feature_names
        return self


@dataclass
class Rule:
    attrs: tuple[int, ...]
    values: tuple[int, ...]
    decision: int
    confidence: float
    ordinal_purity: float
    support: int
    source_count: int
    source_stability: float
    score: float


class LocalMinimalRoughRules:
    def __init__(
        self,
        max_search_len: int = 6,
        min_support: int = 3,
        min_confidence: float = 0.56,
        min_quality: float = 0.66,
        max_rules: int = 360,
    ):
        self.max_search_len = max_search_len
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_quality = min_quality
        self.max_rules = max_rules

    @staticmethod
    def _is_redundant(candidate: Rule, kept: list[Rule]) -> bool:
        candidate_map = dict(zip(candidate.attrs, candidate.values))
        for short in kept:
            if short.decision != candidate.decision or len(short.attrs) >= len(candidate.attrs):
                continue
            if all(candidate_map.get(a) == v for a, v in zip(short.attrs, short.values)):
                if (
                    short.confidence >= candidate.confidence - 0.025
                    and short.source_stability >= candidate.source_stability - 0.10
                    and short.support >= candidate.support
                ):
                    return True
        return False

    def fit(
        self,
        Z_selected: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        selected_feature_names: list[str],
    ):
        self.feature_names_ = selected_feature_names
        self.prior_ = (np.bincount(y, minlength=3) + 1.0) / (len(y) + 3.0)
        groups = groups.astype(str)
        candidates: list[Rule] = []
        max_len = min(self.max_search_len, Z_selected.shape[1])
        for length in range(1, max_len + 1):
            for attrs in combinations(range(Z_selected.shape[1]), length):
                buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
                for row_id, values in enumerate(Z_selected[:, attrs]):
                    if np.any(values < 0):
                        continue
                    buckets[tuple(map(int, values))].append(row_id)
                for values, row_ids in buckets.items():
                    if len(row_ids) < self.min_support:
                        continue
                    local_y = y[row_ids]
                    counts = np.bincount(local_y, minlength=3)
                    decision = int(np.argmax(counts))
                    confidence = float((counts[decision] + 1.0) / (len(row_ids) + 3.0))
                    ordinal_purity = float(1.0 - np.mean(np.abs(local_y - decision)) / 2.0)
                    quality = 0.72 * confidence + 0.28 * ordinal_purity
                    if confidence < self.min_confidence or quality < self.min_quality:
                        continue
                    covered_groups = np.unique(groups[row_ids])
                    agreements = []
                    for group in covered_groups:
                        group_rows = [r for r in row_ids if groups[r] == group]
                        group_counts = np.bincount(y[group_rows], minlength=3)
                        agreements.append(int(np.argmax(group_counts) == decision))
                    source_stability = float(np.mean(agreements)) if agreements else 0.0
                    score = (
                        quality
                        * math.log1p(len(row_ids))
                        * (0.55 + 0.45 * source_stability)
                        * math.log1p(1 + len(covered_groups))
                        / (length ** 0.42)
                    )
                    candidates.append(
                        Rule(
                            attrs=tuple(attrs),
                            values=tuple(values),
                            decision=decision,
                            confidence=confidence,
                            ordinal_purity=ordinal_purity,
                            support=len(row_ids),
                            source_count=len(covered_groups),
                            source_stability=source_stability,
                            score=score,
                        )
                    )
        candidates.sort(key=lambda r: (len(r.attrs), -r.score))
        kept: list[Rule] = []
        for candidate in candidates:
            if not self._is_redundant(candidate, kept):
                kept.append(candidate)
        kept.sort(key=lambda r: r.score, reverse=True)
        class_counts = Counter()
        balanced = []
        per_class_limit = max(20, self.max_rules // 3)
        for rule in kept:
            if class_counts[rule.decision] < per_class_limit:
                balanced.append(rule)
                class_counts[rule.decision] += 1
        self.rules_ = balanced[: self.max_rules]
        return self

    def predict_proba(self, Z_selected: np.ndarray):
        probabilities = np.tile(self.prior_, (len(Z_selected), 1)) * 0.18
        certain = np.zeros(len(Z_selected), dtype=bool)
        evidence = []
        for i, row in enumerate(Z_selected):
            matched = []
            for rule in self.rules_:
                if all(row[a] == v for a, v in zip(rule.attrs, rule.values)):
                    matched.append(rule)
                    probabilities[i, rule.decision] += rule.score
            if matched:
                class_score = probabilities[i].copy()
                order = np.argsort(class_score)[::-1]
                top_rules = [r for r in matched if r.decision == int(order[0])]
                best = max(top_rules, key=lambda r: r.score) if top_rules else max(matched, key=lambda r: r.score)
                margin = float(
                    (class_score[order[0]] - class_score[order[1]])
                    / max(class_score[order[0]], 1e-9)
                )
                certain[i] = (
                    best.confidence >= 0.68
                    and best.source_count >= 2
                    and best.source_stability >= 0.67
                    and margin >= 0.18
                )
                evidence.append(
                    {
                        "matched_rules": len(matched),
                        "best_confidence": best.confidence,
                        "best_sources": best.source_count,
                        "best_length": len(best.attrs),
                        "margin": margin,
                    }
                )
            else:
                evidence.append(
                    {
                        "matched_rules": 0,
                        "best_confidence": 0.0,
                        "best_sources": 0,
                        "best_length": 0,
                        "margin": 0.0,
                    }
                )
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities, certain, evidence

    def table(self) -> pd.DataFrame:
        rows = []
        for rank, rule in enumerate(self.rules_, 1):
            conditions = []
            for attr, value in zip(rule.attrs, rule.values):
                conditions.append(f"{self.feature_names_[attr]}=bin{value}")
            rows.append(
                {
                    "rank": rank,
                    "conditions": " AND ".join(conditions),
                    "decision_class": rule.decision,
                    "rule_length": len(rule.attrs),
                    "confidence": rule.confidence,
                    "ordinal_purity": rule.ordinal_purity,
                    "support": rule.support,
                    "source_count": rule.source_count,
                    "source_stability": rule.source_stability,
                    "score": rule.score,
                }
            )
        return pd.DataFrame(rows)


class BoundaryExpertEnsemble:
    """Nested-selected boundary expert: tree, shrinkage-linear, or their blend."""

    CANDIDATE_WEIGHTS = {
        "tree_dominant": np.array([0.72, 0.10, 0.13, 0.05]),
        "linear_blend": np.array([0.08, 0.47, 0.40, 0.05]),
        "balanced_blend": np.array([0.36, 0.29, 0.30, 0.05]),
        "lda_dominant": np.array([0.08, 0.77, 0.10, 0.05]),
        "logit_dominant": np.array([0.08, 0.10, 0.77, 0.05]),
    }

    def __init__(self, random_state: int = 0, source_aware: bool = False):
        self.random_state = random_state
        self.source_aware = source_aware

    def _fit_models(self, Xi, Xs, y, sample_weight, seed):
        models = {}
        models["extra_trees"] = ExtraTreesClassifier(
            n_estimators=260,
            max_features=1.0,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ).fit(Xi, y)
        models["lda"] = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"
        ).fit(Xs, y)
        models["logit"] = LogisticRegression(
            C=0.20,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=seed,
        ).fit(Xs, y)
        ordinal_models = []
        for threshold in (0, 1):
            binary = (y > threshold).astype(int)
            if np.unique(binary).size < 2:
                ordinal_models.append(float(binary[0]))
            else:
                ordinal = LogisticRegression(
                    C=0.45,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=seed,
                )
                ordinal.fit(Xs, binary, sample_weight=sample_weight)
                ordinal_models.append(ordinal)
        models["ordinal"] = ordinal_models
        return models

    @staticmethod
    def _align(classes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        aligned = np.zeros((len(probabilities), 3), dtype=float)
        for j, cls in enumerate(classes):
            aligned[:, int(cls)] = probabilities[:, j]
        return aligned

    def _base_probabilities(self, models, Xi, Xs):
        extra = self._align(
            models["extra_trees"].classes_, models["extra_trees"].predict_proba(Xi)
        )
        lda = self._align(models["lda"].classes_, models["lda"].predict_proba(Xs))
        logit = self._align(
            models["logit"].classes_, models["logit"].predict_proba(Xs)
        )
        exceed = []
        for model in models["ordinal"]:
            if isinstance(model, float):
                exceed.append(np.full(len(Xs), model, dtype=float))
            else:
                exceed.append(model.predict_proba(Xs)[:, 1])
        gt0 = np.maximum(exceed[0], exceed[1])
        gt1 = np.minimum(exceed[0], exceed[1])
        ordinal = np.column_stack([1.0 - gt0, gt0 - gt1, gt1])
        return [extra, lda, logit, ordinal]

    def _choose_weights(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
        oof = [np.zeros((len(y), 3), dtype=float) for _ in range(4)]
        filled = np.zeros(len(y), dtype=bool)
        if self.source_aware and len(np.unique(groups)) >= 3:
            splitter = StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=self.random_state + 77
            )
            split_iterator = splitter.split(X, y, groups=groups)
        else:
            minimum_class = int(np.min(np.bincount(y, minlength=3)))
            n_splits = max(2, min(3, minimum_class))
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state + 77
            )
            split_iterator = splitter.split(X, y)
        for inner_fold, (train, validation) in enumerate(split_iterator):
            if np.unique(y[train]).size < 2:
                continue
            imputer = SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            )
            Xi_train = imputer.fit_transform(X.iloc[train])
            Xi_validation = imputer.transform(X.iloc[validation])
            scaler = RobustScaler().fit(Xi_train)
            Xs_train = scaler.transform(Xi_train)
            Xs_validation = scaler.transform(Xi_validation)
            weights = inverse_source_weights(groups[train])
            try:
                models = self._fit_models(
                    Xi_train, Xs_train, y[train], weights, self.random_state + inner_fold
                )
                probabilities = self._base_probabilities(
                    models, Xi_validation, Xs_validation
                )
            except ValueError:
                continue
            for j in range(4):
                oof[j][validation] = probabilities[j]
            filled[validation] = True
        if np.sum(filled) < max(12, len(y) // 2):
            return "balanced_blend", self.CANDIDATE_WEIGHTS["balanced_blend"]
        scores = []
        for name, weights in self.CANDIDATE_WEIGHTS.items():
            blended = sum(weights[j] * oof[j][filled] for j in range(4))
            prediction = np.argmax(blended, axis=1)
            macro = f1_score(y[filled], prediction, average="macro", zero_division=0)
            balanced = balanced_accuracy_score(y[filled], prediction)
            scores.append((0.58 * macro + 0.42 * balanced, name, weights))
        _, name, weights = max(scores, key=lambda item: item[0])
        return name, weights

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
        self.columns_ = list(X.columns)
        groups = np.asarray(groups).astype(str)
        self.selected_blend_, self.blend_weights_ = self._choose_weights(X, y, groups)
        self.imputer_ = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )
        Xi = self.imputer_.fit_transform(X)
        self.scaler_ = RobustScaler().fit(Xi)
        Xs = self.scaler_.transform(Xi)
        sample_weight = inverse_source_weights(groups)
        self.models_ = self._fit_models(
            Xi, Xs, y, sample_weight, self.random_state
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xi = self.imputer_.transform(X)
        Xs = self.scaler_.transform(Xi)
        base = self._base_probabilities(self.models_, Xi, Xs)
        probabilities = sum(
            self.blend_weights_[j] * base[j] for j in range(len(base))
        )
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


class SANTORF:
    def __init__(self, random_state: int = 0, source_aware: bool = False):
        self.random_state = random_state
        self.source_aware = source_aware

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
        self.columns_ = list(X.columns)
        self.binner_ = StableAdaptiveBinner(
            n_bootstrap=10, random_state=self.random_state
        ).fit(X, y, groups)
        Z = self.binner_.transform(X)
        self.reducer_ = StabilityRoughReducer(
            n_bootstrap=8, random_state=self.random_state
        ).fit(
            Z,
            y,
            groups,
            self.binner_.base_scores(),
            self.columns_,
            eligible_mask=np.array(
                [self.binner_.coverage_[name] >= 0.45 for name in self.columns_],
                dtype=bool,
            ),
        )
        selected = self.reducer_.selected_indices_
        selected_names = [self.columns_[j] for j in selected]
        self.rules_ = LocalMinimalRoughRules().fit(
            Z[:, selected], y, groups, selected_names
        )
        self.expert_ = BoundaryExpertEnsemble(
            random_state=self.random_state, source_aware=self.source_aware
        ).fit(
            X, y, groups
        )
        return self

    def components(self, X: pd.DataFrame):
        Z = self.binner_.transform(X)
        rule_p, certain, evidence = self.rules_.predict_proba(
            Z[:, self.reducer_.selected_indices_]
        )
        expert_p = self.expert_.predict_proba(X)
        return rule_p, expert_p, certain, evidence

    def predict_variants(self, X: pd.DataFrame):
        rule_p, expert_p, certain, evidence = self.components(X)
        hybrid_p = expert_p.copy()
        hybrid_p[certain] = 0.68 * rule_p[certain] + 0.32 * expert_p[certain]
        hybrid_p /= hybrid_p.sum(axis=1, keepdims=True)
        disagreement = np.argmax(rule_p, axis=1) != np.argmax(expert_p, axis=1)
        abstain = (np.max(hybrid_p, axis=1) < 0.47) | (
            disagreement & ~certain & (np.max(expert_p, axis=1) < 0.58)
        )
        return {
            "rules_only": (np.argmax(rule_p, axis=1), rule_p, np.zeros(len(X), bool)),
            "expert_only": (np.argmax(expert_p, axis=1), expert_p, np.max(expert_p, axis=1) < 0.47),
            "three_way_hybrid": (np.argmax(hybrid_p, axis=1), hybrid_p, abstain),
        }, certain, evidence

    def audit_tables(self):
        selected = set(self.reducer_.selected_indices_)
        feature_rows = []
        for j, name in enumerate(self.columns_):
            cuts = self.binner_.cuts_[name]
            feature_rows.append(
                {
                    "feature": name,
                    "feature_label": FEATURE_LABELS.get(name, name),
                    "selected": j in selected,
                    "selection_frequency": self.reducer_.selection_frequency_[j],
                    "supervised_gain": self.binner_.gain_[name],
                    "cut_stability": self.binner_.cut_stability_[name],
                    "observed_coverage": self.binner_.coverage_[name],
                    "cut_1": cuts[0],
                    "cut_2": cuts[1],
                }
            )
        return (
            pd.DataFrame(feature_rows),
            pd.DataFrame(self.reducer_.interactions_),
            self.rules_.table(),
        )


def metric_row(
    y_true: np.ndarray,
    prediction: np.ndarray,
    abstain: np.ndarray,
) -> dict[str, float]:
    accepted = ~abstain
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "quadratic_kappa": float(cohen_kappa_score(y_true, prediction, weights="quadratic")),
        "ordinal_mae": float(np.mean(np.abs(y_true - prediction))),
        "coverage": float(np.mean(accepted)),
        "selective_accuracy": float(accuracy_score(y_true[accepted], prediction[accepted]))
        if np.any(accepted)
        else np.nan,
    }


def make_splits(
    y: np.ndarray,
    groups: np.ndarray,
    split_kind: str,
    repeats: int = 3,
):
    if split_kind == "row_stratified":
        splitter = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=repeats, random_state=20260809
        )
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
            yield fold, train, test
    elif split_kind == "source_grouped":
        fold = 0
        for repeat in range(repeats):
            splitter = StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=20260809 + repeat
            )
            for train, test in splitter.split(np.zeros(len(y)), y, groups=groups):
                yield fold, train, test
                fold += 1
    else:
        raise ValueError(split_kind)


def run_cross_validation(
    data: pd.DataFrame,
    dataset_name: str,
    features: list[str],
    repeats: int = 3,
) -> pd.DataFrame:
    y = target_classes(data["yield_pct"].to_numpy(float))
    groups = data["source_id"].astype(str).to_numpy()
    X = data[features]
    rows = []
    for split_kind in ("row_stratified", "source_grouped"):
        for fold, train, test in make_splits(y, groups, split_kind, repeats=repeats):
            model = SANTORF(
                random_state=3100 + fold,
                source_aware=split_kind == "source_grouped",
            ).fit(
                X.iloc[train], y[train], groups[train]
            )
            variants, certain, evidence = model.predict_variants(X.iloc[test])
            selected_names = [
                features[j] for j in model.reducer_.selected_indices_
            ]
            for variant, (prediction, probabilities, abstain) in variants.items():
                row = {
                    "experiment": "dataset_cv",
                    "dataset": dataset_name,
                    "feature_set": "extended" if "particle_size_mid_um" in features else "core",
                    "split_kind": split_kind,
                    "fold": fold,
                    "model_variant": variant,
                    "train_n": len(train),
                    "test_n": len(test),
                    "train_sources": len(np.unique(groups[train])),
                    "test_sources": len(np.unique(groups[test])),
                    "certain_rule_coverage": float(np.mean(certain)),
                    "rule_count": len(model.rules_.rules_),
                    "median_rule_length": float(
                        np.median([len(r.attrs) for r in model.rules_.rules_])
                    )
                    if model.rules_.rules_
                    else np.nan,
                    "selected_features": "|".join(selected_names),
                    "boundary_expert_blend": model.expert_.selected_blend_,
                }
                row.update(metric_row(y[test], prediction, abstain))
                rows.append(row)
            print(
                f"{dataset_name:18s} {split_kind:15s} fold={fold:02d} "
                f"hybrid_acc={rows[-1]['accuracy']:.3f} hybrid_f1={rows[-1]['macro_f1']:.3f} "
                f"rules={len(model.rules_.rules_)} selected={','.join(selected_names)}",
                flush=True,
            )
    return pd.DataFrame(rows)


def paired_augmentation_test(
    original: pd.DataFrame,
    augmented: pd.DataFrame,
    repeats: int = 2,
) -> pd.DataFrame:
    y_original = target_classes(original["yield_pct"].to_numpy(float))
    groups_original = original["source_id"].astype(str).to_numpy()
    new_rows = augmented["dataset_origin"] == "Krinski_2025"
    external = augmented.loc[new_rows].copy()
    rows = []
    for fold, train, test in make_splits(
        y_original, groups_original, "source_grouped", repeats=repeats
    ):
        train_original = original.iloc[train].copy()
        test_original = original.iloc[test].copy()
        for training_condition in (
            "original_training",
            "plus_Krinski2025",
            "plus_two_new_sources",
        ):
            if training_condition == "original_training":
                training = train_original
            elif training_condition == "plus_Krinski2025":
                training = pd.concat([train_original, external], ignore_index=True)
            else:
                all_external = augmented[
                    augmented["dataset_origin"] != "original_99"
                ]
                training = pd.concat([train_original, all_external], ignore_index=True)
            y_train = target_classes(training["yield_pct"].to_numpy(float))
            train_groups = training["source_id"].astype(str).to_numpy()
            model = SANTORF(random_state=6100 + fold, source_aware=True).fit(
                training[CORE_FEATURES], y_train, train_groups
            )
            variants, certain, _ = model.predict_variants(test_original[CORE_FEATURES])
            prediction, probabilities, abstain = variants["three_way_hybrid"]
            row = {
                "experiment": "paired_augmentation",
                "dataset": training_condition,
                "feature_set": "core",
                "split_kind": "same_original_source_test",
                "fold": fold,
                "model_variant": "three_way_hybrid",
                "train_n": len(training),
                "test_n": len(test_original),
                "train_sources": len(np.unique(train_groups)),
                "test_sources": len(np.unique(groups_original[test])),
                "certain_rule_coverage": float(np.mean(certain)),
                "rule_count": len(model.rules_.rules_),
                "median_rule_length": float(
                    np.median([len(r.attrs) for r in model.rules_.rules_])
                )
                if model.rules_.rules_
                else np.nan,
                "selected_features": "|".join(
                    CORE_FEATURES[j] for j in model.reducer_.selected_indices_
                ),
                "boundary_expert_blend": model.expert_.selected_blend_,
            }
            row.update(metric_row(y_original[test], prediction, abstain))
            rows.append(row)
    return pd.DataFrame(rows)


def external_new_source_test(original: pd.DataFrame, augmented: pd.DataFrame):
    external = augmented[augmented["dataset_origin"] == "Krinski_2025"].copy()
    y_train = target_classes(original["yield_pct"].to_numpy(float))
    y_test = target_classes(external["yield_pct"].to_numpy(float))
    groups = original["source_id"].astype(str).to_numpy()
    model = SANTORF(random_state=8888, source_aware=True).fit(
        original[CORE_FEATURES], y_train, groups
    )
    variants, certain, _ = model.predict_variants(external[CORE_FEATURES])
    rows = []
    for variant, (prediction, probabilities, abstain) in variants.items():
        row = {
            "experiment": "external_new_source",
            "dataset": "train_original_test_Krinski2025",
            "feature_set": "core",
            "split_kind": "locked_external_source",
            "fold": 0,
            "model_variant": variant,
            "train_n": len(original),
            "test_n": len(external),
            "train_sources": original["source_id"].nunique(),
            "test_sources": 1,
            "certain_rule_coverage": float(np.mean(certain)),
            "rule_count": len(model.rules_.rules_),
            "median_rule_length": float(
                np.median([len(r.attrs) for r in model.rules_.rules_])
            )
            if model.rules_.rules_
            else np.nan,
            "selected_features": "|".join(
                CORE_FEATURES[j] for j in model.reducer_.selected_indices_
            ),
            "boundary_expert_blend": model.expert_.selected_blend_,
        }
        row.update(metric_row(y_test, prediction, abstain))
        rows.append(row)
    prediction, probabilities, abstain = variants["three_way_hybrid"]
    prediction_table = external[
        [
            "source_id",
            "doi",
            "yield_pct",
            "yield_sd_pct",
            "particle_size_mid_um",
            "density_g_ml",
            "bp_c",
            "ratio_g_ml",
            "time_h",
            "temperature_c",
        ]
    ].copy()
    prediction_table["true_class"] = y_test
    prediction_table["predicted_class"] = prediction
    prediction_table["prob_low"] = probabilities[:, 0]
    prediction_table["prob_medium"] = probabilities[:, 1]
    prediction_table["prob_high"] = probabilities[:, 2]
    prediction_table["abstain"] = abstain
    prediction_table.to_csv(OUT / "external_Krinski2025_predictions.csv", index=False)
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "experiment",
        "dataset",
        "feature_set",
        "split_kind",
        "model_variant",
    ]
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "quadratic_kappa",
        "ordinal_mae",
        "coverage",
        "selective_accuracy",
        "certain_rule_coverage",
        "rule_count",
        "median_rule_length",
    ]
    rows = []
    for keys, frame in detail.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["folds"] = len(frame)
        for metric in metric_columns:
            values = frame[metric].dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_ci95"] = (
                float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def data_quality_profile(original: pd.DataFrame, augmented: pd.DataFrame) -> dict:
    external = augmented[augmented["dataset_origin"] != "original_99"]
    return {
        "original_rows": int(len(original)),
        "augmented_rows": int(len(augmented)),
        "new_rows": int(len(external)),
        "original_sources": int(original["source_id"].nunique()),
        "augmented_sources": int(augmented["source_id"].nunique()),
        "yield_class_counts_original": Counter(
            map(int, target_classes(original["yield_pct"].to_numpy(float)))
        ),
        "yield_class_counts_augmented": Counter(
            map(int, target_classes(augmented["yield_pct"].to_numpy(float)))
        ),
        "core_missing_rate_original": {
            c: float(original[c].isna().mean()) for c in CORE_FEATURES
        },
        "core_missing_rate_augmented": {
            c: float(augmented[c].isna().mean()) for c in CORE_FEATURES
        },
        "particle_size_missing_rate_augmented": float(
            augmented["particle_size_mid_um"].isna().mean()
        ),
        "new_source_dois": sorted(map(str, external["doi"].dropna().unique())),
        "new_source_urls": sorted(map(str, external["source_url"].dropna().unique())),
        "new_source_ratio_derivation": "approximately 5 g dry SCG / 200 mL solvent = 0.025 g/mL",
        "new_source_temperature_rule": "Soxhlet extraction represented by each solvent boiling point",
        "Alhanif2026_duplicate_handling": "One exact duplicate condition in Table 2 consolidated and represented as n_replicates=2",
    }


def main():
    original, augmented = load_datasets()
    profile = data_quality_profile(original, augmented)
    with open(OUT / "data_quality_profile.json", "w", encoding="utf-8") as handle:
        json.dump(profile, handle, ensure_ascii=False, indent=2, default=dict)

    experiments = []
    def checkpoint():
        pd.concat(experiments, ignore_index=True).to_csv(
            OUT / "v4_experiment_detail_checkpoint.csv", index=False
        )

    experiments.append(
        run_cross_validation(original, "original_99", CORE_FEATURES, repeats=3)
    )
    checkpoint()
    experiments.append(
        run_cross_validation(augmented, "augmented_122", CORE_FEATURES, repeats=3)
    )
    checkpoint()
    experiments.append(
        run_cross_validation(augmented, "augmented_122", EXTENDED_FEATURES, repeats=3)
    )
    checkpoint()
    experiments.append(paired_augmentation_test(original, augmented, repeats=2))
    checkpoint()
    experiments.append(external_new_source_test(original, augmented))
    checkpoint()
    detail = pd.concat(experiments, ignore_index=True)
    summary = summarize(detail)
    detail.to_csv(OUT / "v4_experiment_detail.csv", index=False)
    summary.to_csv(OUT / "v4_experiment_summary.csv", index=False)

    y_augmented = target_classes(augmented["yield_pct"].to_numpy(float))
    final_model = SANTORF(random_state=9901, source_aware=True).fit(
        augmented[EXTENDED_FEATURES],
        y_augmented,
        augmented["source_id"].astype(str).to_numpy(),
    )
    features, interactions, rules = final_model.audit_tables()
    features.to_csv(OUT / "v4_feature_stability.csv", index=False)
    interactions.to_csv(OUT / "v4_interaction_stability.csv", index=False)
    rules.to_csv(OUT / "v4_representative_rules.csv", index=False)

    print("\n=== SUMMARY ===")
    print(
        summary[
            [
                "experiment",
                "dataset",
                "feature_set",
                "split_kind",
                "model_variant",
                "folds",
                "accuracy_mean",
                "balanced_accuracy_mean",
                "macro_f1_mean",
                "quadratic_kappa_mean",
                "ordinal_mae_mean",
                "coverage_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
