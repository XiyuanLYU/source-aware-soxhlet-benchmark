from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import multifeedstock_v12 as v12

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


OUT = Path(__file__).resolve().parent / "results_v12_multifeedstock"
ELIGIBLE_MIN_N = 8
REPEATS = [13101, 13102, 13103, 13104, 13105]


def calibration_indices(frame: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    columns = [
        "ratio_g_ml", "time_h", "temperature_c", "dry_temperature_c", "dry_time_h",
        "particle_size_mid_um", "density_g_ml", "bp_c", "dipole_moment",
    ]
    matrix = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(frame[columns])
    matrix = RobustScaler().fit_transform(matrix)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(frame)))
    selected = [first]
    while len(selected) < min(k, len(frame)):
        distance = np.sqrt(((matrix[:, None, :] - matrix[selected][None, :, :]) ** 2).mean(axis=2))
        nearest = distance.min(axis=1)
        nearest[selected] = -np.inf
        selected.append(int(np.argmax(nearest)))
    return frame.index.to_numpy()[np.asarray(selected, int)]


def aligned_probability(model, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(frame)
    classes = model.named_steps["model"].classes_.astype(int)
    probability = np.full((len(frame), 3), 1e-9)
    probability[:, classes] = raw
    return probability / probability.sum(axis=1, keepdims=True)


def prior_shift(probability: np.ndarray, y_train: np.ndarray, y_cal: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return probability
    global_counts = np.bincount(y_train, minlength=3).astype(float) + 1.0
    global_prior = global_counts / global_counts.sum()
    cal_counts = np.bincount(y_cal, minlength=3).astype(float) + 0.75
    cal_prior = cal_counts / cal_counts.sum()
    multiplier = np.power(cal_prior / global_prior, gamma)
    adjusted = probability * multiplier.reshape(1, -1)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def evaluate(data: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = data[target_column].to_numpy(int)
    sources = data["source_id"].astype(str).to_numpy()
    feedstocks = data["feedstock_name"].astype(str).to_numpy()
    source_sizes = data.groupby("source_id").size()
    eligible = sorted(source_sizes[source_sizes >= ELIGIBLE_MIN_N].index.astype(str))
    model_names = ["extra_material_batch", "extra_feedstock_batch"]
    candidates = [next(c for c in v12.CANDIDATES if c.name == name) for name in model_names]
    prediction_rows = []
    metric_rows = []

    for repeat, seed in enumerate(REPEATS):
        for k in (1, 3, 5):
            for candidate in candidates:
                for gamma in (0.0, 0.5, 1.0):
                    repeat_predictions = []
                    repeat_truth = []
                    repeat_sources = []
                    repeat_feedstocks = []
                    for source_index, source in enumerate(eligible):
                        source_frame = data[data["source_id"].astype(str).eq(source)]
                        cal = calibration_indices(source_frame, k, seed + source_index * 100)
                        test = source_frame.index.difference(cal).to_numpy()
                        other = data.index[~data["source_id"].astype(str).eq(source)].to_numpy()
                        train = np.concatenate([other, cal])
                        model = v12.make_model(candidate, seed + source_index)
                        model.fit(data.loc[train], y[train])
                        probability = aligned_probability(model, data.loc[test])
                        probability = prior_shift(probability, y[train], y[cal], gamma)
                        pred = np.argmax(probability, axis=1)
                        repeat_predictions.extend(pred.tolist())
                        repeat_truth.extend(y[test].tolist())
                        repeat_sources.extend(sources[test].tolist())
                        repeat_feedstocks.extend(feedstocks[test].tolist())
                        for idx, actual, predicted in zip(test, y[test], pred):
                            prediction_rows.append({
                                "target": target_column, "repeat": repeat, "k_calibration": k,
                                "candidate": candidate.name, "prior_gamma": gamma,
                                "row_id": data.loc[idx, "row_id"], "source_id": source,
                                "feedstock_name": feedstocks[idx], "actual_class": int(actual),
                                "predicted_class": int(predicted), "is_external_feedstock": bool(
                                    feedstocks[idx] != "spent_coffee_grounds"
                                ),
                            })
                    truth_arr = np.asarray(repeat_truth, int)
                    pred_arr = np.asarray(repeat_predictions, int)
                    metric_rows.append({
                        "target": target_column, "repeat": repeat, "k_calibration": k,
                        "candidate": candidate.name, "prior_gamma": gamma,
                        "n_test": len(truth_arr), "n_sources": len(set(repeat_sources)),
                        **v12.metrics(
                            truth_arr, pred_arr, np.asarray(repeat_sources), np.asarray(repeat_feedstocks)
                        ),
                    })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    summary = details.groupby(
        ["target", "k_calibration", "candidate", "prior_gamma"], as_index=False
    ).agg(
        repeats=("repeat", "size"), accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"), balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        publication_balanced_accuracy_mean=("publication_balanced_accuracy", "mean"),
        feedstock_balanced_accuracy_mean=("feedstock_balanced_accuracy", "mean"),
        ordinal_mae_mean=("ordinal_mae", "mean"),
    )
    summary["score"] = (
        0.35 * summary["accuracy_mean"] + 0.25 * summary["macro_f1_mean"]
        + 0.20 * summary["balanced_accuracy_mean"]
        + 0.10 * summary["publication_balanced_accuracy_mean"]
        + 0.10 * summary["feedstock_balanced_accuracy_mean"]
    )
    return summary.sort_values(["target", "score"], ascending=[True, False])


def external_only_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = predictions[predictions["is_external_feedstock"]].copy()
    for keys, frame in selected.groupby(["target", "repeat", "k_calibration", "candidate", "prior_gamma"]):
        target, repeat, k, candidate, gamma = keys
        truth = frame["actual_class"].to_numpy(int)
        pred = frame["predicted_class"].to_numpy(int)
        rows.append({
            "target": target, "repeat": repeat, "k_calibration": k,
            "candidate": candidate, "prior_gamma": gamma, "n_test": len(frame),
            **v12.metrics(
                truth, pred, frame["source_id"].astype(str).to_numpy(),
                frame["feedstock_name"].astype(str).to_numpy(),
            ),
        })
    return summarize(pd.DataFrame(rows))


def main() -> None:
    data = v12.load_all()
    target_frames = {
        "absolute_class": data,
        "relative_class": data[data["source_batch_n"] >= 3].reset_index(drop=True),
    }
    all_details, all_predictions = [], []
    for target, frame in target_frames.items():
        details, predictions = evaluate(frame, target)
        all_details.append(details)
        all_predictions.append(predictions)
    details = pd.concat(all_details, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize(details)
    external = external_only_summary(predictions)
    details.to_csv(OUT / "fewshot_calibration_details.csv", index=False)
    predictions.to_csv(OUT / "fewshot_calibration_predictions.csv", index=False)
    summary.to_csv(OUT / "fewshot_calibration_summary.csv", index=False)
    external.to_csv(OUT / "fewshot_external_only_summary.csv", index=False)
    decision = {
        "eligible_source_minimum_rows": ELIGIBLE_MIN_N,
        "calibration_selection": "maximin condition-space coverage; outcomes are not used to choose rows",
        "best_overall": summary.groupby("target").head(1).to_dict(orient="records"),
        "best_external_only": external.groupby("target").head(1).to_dict(orient="records"),
        "claim_boundary": "Few-shot results require k labeled calibration experiments from the new publication/feedstock.",
    }
    (OUT / "fewshot_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("OVERALL", flush=True)
    print(summary.groupby("target").head(12).to_string(index=False), flush=True)
    print("\nEXTERNAL ONLY", flush=True)
    print(external.groupby("target").head(12).to_string(index=False), flush=True)
    print("\nDECISION", flush=True)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
