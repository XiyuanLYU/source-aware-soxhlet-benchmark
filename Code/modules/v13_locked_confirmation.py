from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import multifeedstock_v12 as v12
import multifeedstock_v12_calibration as v12cal
import v13_hierarchical_search as v13


OUT = Path(__file__).resolve().parent / "results_v13_hierarchical"
SEEDS = list(range(16101, 16116))
ELIGIBLE_MIN_N = 8
CLASSIFIER_NAME = "extra_feedstock_batch"
GUIDANCE_NAME = "delta_extra_material"
DELTA_NAME = "delta_gb_process"


def classifier_probability(
    data: pd.DataFrame,
    target_source: str,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> np.ndarray:
    y = data["absolute_class"].to_numpy(int)
    other_idx = data.index[~data["source_id"].astype(str).eq(target_source)].to_numpy()
    train_idx = np.concatenate([other_idx, cal_idx])
    candidate = next(item for item in v12.CANDIDATES if item.name == CLASSIFIER_NAME)
    model = v12.make_model(candidate, seed)
    model.fit(data.loc[train_idx], y[train_idx])
    probability = v12cal.aligned_probability(model, data.loc[test_idx])
    return v12cal.prior_shift(probability, y[train_idx], y[cal_idx], 1.0)


def fit_unlabeled_guidance_and_delta(
    data: pd.DataFrame,
    target_source: str,
    seed: int,
):
    source_mask = data["source_id"].astype(str).eq(target_source)
    train_frame = data.loc[~source_mask]
    weights = v13.source_balanced_weights(train_frame)
    target_delta = v13.delta_target(train_frame)

    guidance_spec = next(spec for spec in v13.REGRESSORS if spec.name == GUIDANCE_NAME)
    guidance = v13.make_regressor(guidance_spec, seed)
    guidance.fit(train_frame, target_delta, model__sample_weight=weights)

    delta_spec = next(spec for spec in v13.REGRESSORS if spec.name == DELTA_NAME)
    delta = v13.make_regressor(delta_spec, seed + 1)
    delta.fit(train_frame, target_delta, model__sample_weight=weights)
    return guidance, delta


def v13_probabilities(
    data: pd.DataFrame,
    source_frame: pd.DataFrame,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    pclass: np.ndarray,
    delta_model,
    gate_count: int,
) -> dict[str, np.ndarray]:
    y_class = data["absolute_class"].to_numpy(int)
    cal_y = data.loc[cal_idx, "yield_pct"].to_numpy(float)
    cal_classes = y_class[cal_idx]
    pknn = v13.calibration_knn_probability(source_frame, cal_idx, test_idx, cal_classes)
    cal_delta = delta_model.predict(data.loc[cal_idx])
    test_delta = delta_model.predict(data.loc[test_idx])
    baseline = float(np.mean(cal_y - cal_delta))
    cal_pred = baseline + cal_delta
    no_local_yhat = baseline + test_delta
    local = v13.local_residual_correction(
        source_frame, cal_idx, test_idx, cal_y - cal_pred
    )
    full_yhat = no_local_yhat + local
    preg_no_local = v13.regression_probability(no_local_yhat, 1.8)
    preg_full = v13.regression_probability(full_yhat, 1.8)

    no_local = 0.25 * pclass + 0.50 * preg_no_local + 0.25 * pknn
    no_gate = 0.25 * pclass + 0.50 * preg_full + 0.25 * pknn
    no_knn = 0.50 * pclass + 0.50 * preg_full
    counts = np.bincount(cal_classes, minlength=3)
    dominant = int(counts.max()) >= gate_count
    full = pclass if dominant else no_gate
    return {
        "v13_no_local": no_local,
        "v13_no_gate": no_gate,
        "v13_no_knn": no_knn,
        "v13_full": full,
    }


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = v12.load_all()
    y = data["absolute_class"].to_numpy(int)
    sizes = data.groupby("source_id").size()
    eligible = sorted(sizes[sizes >= ELIGIBLE_MIN_N].index.astype(str))
    prediction_rows: list[dict] = []

    for repeat, seed in enumerate(SEEDS):
        for source_index, source in enumerate(eligible):
            source_mask = data["source_id"].astype(str).eq(source)
            source_frame = data[source_mask]
            model_seed = seed * 10000 + source_index * 100
            guidance, delta_model = fit_unlabeled_guidance_and_delta(
                data, source, model_seed
            )
            guidance_score = guidance.predict(source_frame)

            configurations: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
            for k in (3, 5):
                cal_idx = v13.calibration_indices(
                    source_frame, k, seed + source_index * 101,
                    "delta_guided", guidance_score,
                )
                test_idx = source_frame.index.difference(cal_idx).to_numpy()
                pclass = classifier_probability(
                    data, source, cal_idx, test_idx, model_seed + k
                )
                configurations[f"guided_classifier_k{k}"] = (k, test_idx, pclass)
                variants = v13_probabilities(
                    data, source_frame, cal_idx, test_idx, pclass, delta_model,
                    gate_count=4 if k == 5 else 3,
                )
                for name, probability in variants.items():
                    configurations[f"{name}_k{k}"] = (k, test_idx, probability)

            legacy_cal = v12cal.calibration_indices(
                source_frame, 5, seed + source_index * 100
            )
            legacy_test = source_frame.index.difference(legacy_cal).to_numpy()
            legacy_probability = classifier_probability(
                data, source, legacy_cal, legacy_test, model_seed + 55
            )
            configurations["legacy_v12_random_k5"] = (5, legacy_test, legacy_probability)

            for config, (k, test_idx, probability) in configurations.items():
                prediction = np.argmax(probability, axis=1)
                for position, (idx, predicted) in enumerate(zip(test_idx, prediction)):
                    prediction_rows.append({
                        "phase": "locked_confirmation",
                        "repeat": repeat,
                        "seed": seed,
                        "config": config,
                        "k_calibration": k,
                        "source_id": source,
                        "feedstock_name": data.loc[idx, "feedstock_name"],
                        "is_external_feedstock": bool(
                            data.loc[idx, "feedstock_name"] != "spent_coffee_grounds"
                        ),
                        "row_id": data.loc[idx, "row_id"],
                        "actual_class": int(y[idx]),
                        "predicted_class": int(predicted),
                        "probability_low": float(probability[position, 0]),
                        "probability_mid": float(probability[position, 1]),
                        "probability_high": float(probability[position, 2]),
                    })

    predictions = pd.DataFrame(prediction_rows)
    metric_rows: list[dict] = []
    for (repeat, config, k), frame in predictions.groupby(
        ["repeat", "config", "k_calibration"]
    ):
        for scope, selected in (
            ("all_eligible_large_batches", frame),
            ("external_feedstocks_only", frame[frame["is_external_feedstock"]]),
        ):
            truth = selected["actual_class"].to_numpy(int)
            pred = selected["predicted_class"].to_numpy(int)
            sources = selected["source_id"].astype(str).to_numpy()
            feedstocks = selected["feedstock_name"].astype(str).to_numpy()
            source_accuracy = [
                float(np.mean(truth[sources == source] == pred[sources == source]))
                for source in np.unique(sources)
            ]
            metric_rows.append({
                "phase": "locked_confirmation",
                "repeat": int(repeat),
                "config": config,
                "k_calibration": int(k),
                "scope": scope,
                "n_test": len(selected),
                "worst_source_accuracy": min(source_accuracy),
                **v12.metrics(truth, pred, sources, feedstocks),
            })
    return pd.DataFrame(metric_rows), predictions


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = metrics.groupby(
        ["config", "k_calibration", "scope"], as_index=False
    ).agg(
        repeats=("repeat", "size"), n_test_per_repeat=("n_test", "first"),
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        publication_balanced_accuracy_mean=("publication_balanced_accuracy", "mean"),
        feedstock_balanced_accuracy_mean=("feedstock_balanced_accuracy", "mean"),
        worst_source_accuracy_mean=("worst_source_accuracy", "mean"),
        ordinal_mae_mean=("ordinal_mae", "mean"),
    )
    summary["accuracy_ci95_low"] = (
        summary["accuracy_mean"] - 1.96 * summary["accuracy_sd"] / np.sqrt(summary["repeats"])
    )
    summary["accuracy_ci95_high"] = (
        summary["accuracy_mean"] + 1.96 * summary["accuracy_sd"] / np.sqrt(summary["repeats"])
    )
    return summary.sort_values(["scope", "accuracy_mean"], ascending=[True, False])


def per_source(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.assign(
        correct=lambda item: item["actual_class"].eq(item["predicted_class"])
    )
    return frame.groupby(
        ["config", "k_calibration", "source_id", "feedstock_name", "is_external_feedstock"],
        as_index=False,
    ).agg(
        repeated_predictions=("row_id", "size"),
        unique_test_rows=("row_id", "nunique"),
        accuracy=("correct", "mean"),
    ).sort_values(["config", "accuracy", "source_id"])


def main() -> None:
    metrics, predictions = run()
    summary = summarize(metrics)
    sources = per_source(predictions)
    metrics.to_csv(OUT / "locked_confirmation_metrics.csv", index=False)
    predictions.to_csv(OUT / "locked_confirmation_predictions.csv", index=False)
    summary.to_csv(OUT / "locked_confirmation_summary.csv", index=False)
    sources.to_csv(OUT / "locked_confirmation_per_source.csv", index=False)

    overall = summary[summary["scope"].eq("all_eligible_large_batches")]
    full = overall[overall["config"].eq("v13_full_k5")].iloc[0]
    legacy = overall[overall["config"].eq("legacy_v12_random_k5")].iloc[0]
    decision = {
        "status": "confirmed" if full["accuracy_ci95_low"] > 0.70 else "not_confirmed",
        "confirmation_seeds": SEEDS,
        "locked_configuration": {
            "selection": "delta-guided low/median/high plus maximin fill; outcomes unused",
            "delta_model": "source-balanced GradientBoostingRegressor on within-source yield delta",
            "probability_blend": "0.25 classifier + 0.50 calibrated delta regression + 0.25 calibration-neighborhood",
            "gate": "use classifier when at least 4 of 5 calibration labels agree; otherwise use blend",
        },
        "v13_full": full.to_dict(),
        "legacy_v12_same_confirmation": legacy.to_dict(),
        "accuracy_improvement_points": float(
            100 * (full["accuracy_mean"] - legacy["accuracy_mean"])
        ),
        "macro_f1_improvement_points": float(
            100 * (full["macro_f1_mean"] - legacy["macro_f1_mean"])
        ),
        "claim_boundary": "Results apply after labeled calibration experiments from a new publication batch; they are not zero-shot results.",
    }
    (OUT / "locked_confirmation_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False), flush=True)
    print("\nDECISION", flush=True)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
