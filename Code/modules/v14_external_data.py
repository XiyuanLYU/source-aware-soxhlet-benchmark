from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import multifeedstock_v12 as v12


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_v14_compatibility"
OUT.mkdir(parents=True, exist_ok=True)


SOLVENT_PROPERTIES = {
    "n_hexane": {"density_g_ml": 0.655, "bp_c": 69.0, "dipole_moment": 0.00},
    # The same physical descriptors are used for the 2-MeTHF records reported
    # by Mkhonto and Chetty (2021).  Boiling point is stated in the paper;
    # density and dipole moment are standard solvent descriptors and are
    # explicitly flagged as descriptor metadata rather than measured outcomes.
    "2_methyltetrahydrofuran": {
        "density_g_ml": 0.853,
        "bp_c": 80.0,
        "dipole_moment": 1.58,
    },
    "petroleum_ether": {"density_g_ml": 0.653, "bp_c": 60.0, "dipole_moment": 0.00},
    "ethanol": {"density_g_ml": 0.789, "bp_c": 78.4, "dipole_moment": 1.69},
    # The paper used a 50:50 methanol/chloroform blend. These three physical
    # descriptors are transparent arithmetic approximations and are flagged below.
    "methanol_chloroform_50_50": {
        "density_g_ml": (0.792 + 1.489) / 2,
        "bp_c": (64.7 + 61.2) / 2,
        "dipole_moment": (1.70 + 1.04) / 2,
    },
}


def make_row(
    *,
    source_id: str,
    run: int,
    feedstock_name: str,
    solvent: str,
    yield_pct: float,
    time_h: float,
    temperature_c: float,
    ratio_g_ml: float,
    particle_size_mid_um: float = np.nan,
    sample_weight_g: float = np.nan,
    solvent_volume_ml: float = np.nan,
    dry_temperature_c: float = np.nan,
    dry_time_h: float = np.nan,
    feedstock_family: str = "fruit_seed",
    plant_part: str = "seed",
    doi: str,
    source_url: str,
    title: str,
    quality_tier: str,
    recommended_use: str,
    provenance_note: str,
    method: str = "Soxhlet",
    method_domain: str = "primary_soxhlet",
    raw_ratio_text: str = "",
    raw_particle_text: str = "",
    quality_flags: str = "",
) -> dict:
    props = SOLVENT_PROPERTIES[solvent]
    return {
        "row_id": f"{source_id}_{run:02d}",
        **props,
        "pressure_bar": 1.0,
        "ratio_g_ml": ratio_g_ml,
        "time_h": time_h,
        "temperature_c": temperature_c,
        "dry_temperature_c": dry_temperature_c,
        "dry_time_h": dry_time_h,
        "particle_size_mid_um": particle_size_mid_um,
        "yield_pct": yield_pct,
        "source_id": source_id,
        "publication_cluster_id": source_id,
        "dataset_origin": "v14_literature_expansion",
        "doi": doi,
        "source_url": source_url,
        "source_title": title,
        "method": method,
        "method_domain": method_domain,
        "solvent": solvent,
        "feedstock_name": feedstock_name,
        "feedstock_family": feedstock_family,
        "plant_part": plant_part,
        "prior_processing": "native_seed",
        "quality_tier": quality_tier,
        "recommended_use": recommended_use,
        "sample_weight_g": sample_weight_g,
        "solvent_volume_ml": solvent_volume_ml,
        "raw_ratio_text": raw_ratio_text,
        "raw_particle_text": raw_particle_text,
        "quality_flags": quality_flags,
        "provenance_note": provenance_note,
    }


def hossain_2020() -> pd.DataFrame:
    """Table 2 experimental averages; replicate columns are not pseudo-replicated."""
    runs = [
        # time h, particle mm, replicate 1, replicate 2, reported average
        (6.5, 2.00, 17.10, 12.50, 14.80),
        (5.5, 1.40, 21.50, 13.50, 17.50),
        (7.0, 1.40, 15.00, 12.85, 13.93),
        (4.5, 2.00, 17.00, 13.70, 15.35),
        (4.5, 1.18, 21.00, 17.15, 19.08),
        (5.5, 1.40, 13.50, 15.00, 14.25),
        (5.5, 1.40, 17.00, 17.15, 17.08),
        (4.0, 1.40, 4.20, 5.00, 4.60),
        (5.5, 0.85, 27.00, 21.00, 24.00),
        (5.5, 3.75, 17.00, 21.00, 19.00),
        (6.5, 1.18, 27.50, 25.50, 26.50),
    ]
    rows = []
    for run, (time_h, particle_mm, rep1, rep2, average) in enumerate(runs, 1):
        rows.append(make_row(
            source_id="Hossain2020_Papaya",
            run=run,
            feedstock_name="carica_papaya_seed",
            solvent="n_hexane",
            yield_pct=average,
            time_h=time_h,
            temperature_c=80.0,
            ratio_g_ml=2.0 / 125.0,
            particle_size_mid_um=particle_mm * 1000.0,
            sample_weight_g=2.0,
            solvent_volume_ml=125.0,
            dry_temperature_c=50.0,
            dry_time_h=24.0,
            doi="10.1007/s13369-020-04551-1",
            source_url=(
                "https://www.researchgate.net/publication/341058192_Experimental_Study_"
                "and_Modeling_Approach_of_Response_Surface_Methodology_Coupled_with_Crow_"
                "Search_Algorithm_for_Optimizing_the_Extraction_Conditions_of_Papaya_Seed_Waste_Oil"
            ),
            title=(
                "Experimental Study and Modeling Approach of Response Surface Methodology "
                "Coupled with Crow Search Algorithm for Optimizing the Extraction Conditions "
                "of Papaya Seed Waste Oil"
            ),
            quality_tier="gold",
            recommended_use="primary_augmentation",
            provenance_note=(
                f"Table 2 reported average; replicate_1={rep1:.2f}, "
                f"replicate_2={rep2:.2f}."
            ),
            raw_ratio_text="2 g seed / 125 mL n-hexane",
            raw_particle_text=f"{particle_mm:g} mm",
        ))
    return pd.DataFrame(rows)


def andriana_2023() -> pd.DataFrame:
    """Full 2^3 factorial table; ratio conversion is explicit and auditable."""
    runs = [
        # time min, mesh, material:solvent g/g, yield
        (170, 10, "1:10", 49.51),
        (190, 10, "1:10", 35.69),
        (170, 30, "1:10", 54.14),
        (190, 30, "1:10", 55.03),
        (170, 10, "1:20", 46.11),
        (190, 10, "1:20", 42.57),
        (170, 30, "1:20", 53.00),
        (190, 30, "1:20", 59.43),
    ]
    mesh_to_um = {10: 2000.0, 30: 600.0}
    rows = []
    for run, (minutes, mesh, ratio_text, outcome) in enumerate(runs, 1):
        denominator = float(ratio_text.split(":")[1])
        ratio_g_g = 1.0 / denominator
        ratio_g_ml = ratio_g_g * SOLVENT_PROPERTIES["petroleum_ether"]["density_g_ml"]
        rows.append(make_row(
            source_id="Andriana2023_Papaya",
            run=run,
            feedstock_name="carica_papaya_seed",
            solvent="petroleum_ether",
            yield_pct=outcome,
            time_h=minutes / 60.0,
            temperature_c=60.0,
            ratio_g_ml=ratio_g_ml,
            particle_size_mid_um=mesh_to_um[mesh],
            dry_temperature_c=60.0,
            doi="10.14710/jvsar.v5i1.17338",
            source_url=(
                "https://www.researchgate.net/publication/371683106_Optimization_of_"
                "Soxhlet_Extraction_Papaya_Seed_Oil_Carica_papaya_L_with_Petroleum_Ether"
            ),
            title=(
                "Optimization of Soxhlet Extraction Papaya Seed Oil (Carica papaya L.) "
                "with Petroleum Ether"
            ),
            quality_tier="silver",
            recommended_use="sensitivity_augmentation",
            provenance_note=(
                "Factorial yield table. ratio_g_ml derived from nominal g/g ratio and "
                "petroleum-ether density; exact per-run mass/volume withheld."
            ),
            raw_ratio_text=f"{ratio_text} g/g",
            raw_particle_text=f"{mesh} mesh; numeric opening is a derived approximation",
            quality_flags="method_volume_ratio_conflict|derived_mesh_opening|derived_ratio_g_ml",
        ))
    return pd.DataFrame(rows)


def esonye_2021() -> pd.DataFrame:
    """Table 4: 17 BBD runs x 3 solvents, retained as sensitivity-only data."""
    runs = [
        (-1, -1, 0, 47.10, 30.15, 28.92),
        (1, -1, 0, 51.21, 35.25, 31.61),
        (-1, 1, 0, 45.50, 37.19, 30.04),
        (1, 1, 0, 43.00, 29.04, 27.44),
        (-1, 0, -1, 45.50, 37.19, 30.04),
        (1, 0, -1, 45.50, 37.19, 30.04),
        (-1, 0, 1, 48.10, 30.81, 28.50),
        (1, 0, 1, 50.10, 34.62, 30.94),
        (0, -1, -1, 44.24, 33.64, 29.74),
        (0, 1, -1, 48.10, 30.81, 28.50),
        (0, -1, 1, 56.30, 40.14, 33.78),
        (0, 1, 0, 43.30, 29.04, 27.44),
        (0, 0, 0, 47.21, 30.11, 28.00),
        (0, 0, 0, 48.50, 30.92, 28.97),
        (0, 0, 0, 50.21, 34.77, 31.56),
        (0, 0, 0, 49.61, 32.45, 28.84),
        (0, 0, 0, 45.50, 33.19, 30.04),
    ]
    particle_by_code = {-1: 300.0, 0: 400.0, 1: 500.0}
    time_by_code = {-1: 45.0 / 60.0, 0: 52.5 / 60.0, 1: 60.0 / 60.0}
    ratio_text_by_code = {-1: "1:4", 0: "1:5", 1: "1:6"}
    ratio_by_code = {-1: 1 / 4, 0: 1 / 5, 1: 1 / 6}
    solvent_specs = [
        ("n_hexane", 69.0, 3),
        ("ethanol", 78.0, 4),
        ("methanol_chloroform_50_50", 70.0, 5),
    ]
    rows = []
    row_number = 0
    for design_run, values in enumerate(runs, 1):
        particle_code, time_code, ratio_code = values[:3]
        for solvent, temperature, yield_position in solvent_specs:
            row_number += 1
            ratio_g_ml = ratio_by_code[ratio_code]
            sample_g = 20.0
            solvent_ml = sample_g / ratio_g_ml
            flags = ["repeated_reported_values"]
            if design_run == 12:
                flags.append("possible_bbd_design_point_anomaly")
            if solvent == "methanol_chloroform_50_50":
                flags.append("mixture_properties_approximated")
            rows.append(make_row(
                source_id="Esonye2021_Dacryodes",
                run=row_number,
                feedstock_name="dacryodes_edulis_seed",
                solvent=solvent,
                yield_pct=float(values[yield_position]),
                time_h=time_by_code[time_code],
                temperature_c=temperature,
                ratio_g_ml=ratio_g_ml,
                particle_size_mid_um=particle_by_code[particle_code],
                sample_weight_g=sample_g,
                solvent_volume_ml=solvent_ml,
                doi="10.1016/j.heliyon.2021.e06342",
                source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC7969337/",
                title=(
                    "Application of soft-computing techniques for statistical modeling "
                    "and optimization of Dacryodes edulis seed oil extraction using polar "
                    "and non-polar solvents"
                ),
                quality_tier="silver",
                recommended_use="sensitivity_only",
                provenance_note=f"Table 4 design run {design_run}; observed value only.",
                raw_ratio_text=ratio_text_by_code[ratio_code] + " g/mL",
                raw_particle_text=f"{particle_by_code[particle_code]:g} micrometres",
                quality_flags="|".join(flags),
            ))
    return pd.DataFrame(rows)


def senrayan_2018() -> pd.DataFrame:
    """Full central-composite table from a related, non-Soxhlet extraction domain."""
    runs = [
        # material:solvent, temperature C, time h, yield %
        ("1:15", 40.0, 3.0, 16.25),
        ("1:10", 40.0, 6.0, 14.93),
        ("1:5", 45.0, 6.0, 10.36),
        ("1:10", 45.0, 6.0, 18.90),
        ("1:10", 45.0, 9.0, 14.78),
        ("1:15", 45.0, 6.0, 20.56),
        ("1:10", 45.0, 6.0, 18.92),
        ("1:10", 45.0, 6.0, 18.19),
        ("1:10", 45.0, 6.0, 18.33),
        ("1:15", 40.0, 9.0, 11.58),
        ("1:5", 40.0, 9.0, 5.05),
        ("1:5", 50.0, 9.0, 15.88),
        ("1:5", 50.0, 3.0, 5.49),
        ("1:10", 45.0, 6.0, 18.84),
        ("1:5", 50.0, 9.0, 8.65),
        ("1:15", 50.0, 3.0, 15.36),
        ("1:10", 45.0, 6.0, 18.24),
        ("1:5", 40.0, 3.0, 4.84),
        ("1:10", 50.0, 6.0, 17.62),
        ("1:10", 45.0, 3.0, 16.37),
    ]
    rows = []
    for run, (ratio_text, temperature, time_h, outcome) in enumerate(runs, 1):
        denominator = float(ratio_text.split(":")[1])
        flags = ["related_non_soxhlet_method"]
        if run in (12, 15):
            flags.append("duplicate_reported_conditions_with_discordant_yield")
        rows.append(make_row(
            source_id="Senrayan2018_Papaya",
            run=run,
            feedstock_name="carica_papaya_seed",
            solvent="n_hexane",
            yield_pct=outcome,
            time_h=time_h,
            temperature_c=temperature,
            ratio_g_ml=1.0 / denominator,
            doi="10.1080/01496395.2018.1480632",
            source_url=(
                "https://www.researchgate.net/publication/325885398_Solvent-assisted_"
                "extraction_of_oil_from_papaya_Carica_papaya_L_seeds_evaluation_of_its_"
                "physiochemical_properties_and_fatty-acid_composition"
            ),
            title=(
                "Solvent-assisted extraction of oil from papaya (Carica papaya L.) seeds: "
                "evaluation of its physiochemical properties and fatty-acid composition"
            ),
            quality_tier="silver_related_method",
            recommended_use="method_transfer_sensitivity",
            provenance_note=(
                "Full central-composite experimental table; non-Soxhlet orbital-shaker "
                "method retained only for method-aware transfer experiments."
            ),
            method="Temperature-controlled orbital shaker",
            method_domain="related_batch_extraction",
            raw_ratio_text=ratio_text + " material:solvent",
            quality_flags="|".join(flags),
        ))
    return pd.DataFrame(rows)


def mkhonto_chetty_2021() -> pd.DataFrame:
    """Exact Figure 1b values for mixed spent-coffee-ground samples.

    The four legacy ``OriginalRef_10`` rows are not relabelled here: their
    values and 0.5 h duration are absent from the cited Thiex paper and do not
    match this publication.  Instead, this function reconstructs ten new,
    independently traceable observations from the numerical labels in Figure
    1b of Mkhonto and Chetty (2021).  The Results section states that the ratio
    experiment was run for 5.50 h; this value is used because it describes the
    plotted experiment itself (the earlier Methods paragraph says 6 h).
    """
    ratio_results = [
        # solids-to-solvent ratio (g/mL), hexane yield, 2-MeTHF yield
        (0.050, 18.11, 26.34),
        (0.056, 17.31, 25.89),
        (0.067, 18.88, 25.68),
        (0.100, 15.07, 17.88),
        (0.200, 10.87, 11.12),
    ]
    specs = [
        ("n_hexane", 69.0, 1),
        ("2_methyltetrahydrofuran", 80.0, 2),
    ]
    rows = []
    run = 0
    for ratio_g_ml, hexane_yield, methf_yield in ratio_results:
        for solvent, temperature, yield_position in specs:
            run += 1
            outcome = (hexane_yield, methf_yield)[yield_position - 1]
            rows.append(make_row(
                source_id="Mkhonto2021_CET89",
                run=run,
                feedstock_name="spent_coffee_grounds",
                feedstock_family="coffee_processing_residue",
                plant_part="spent_ground",
                solvent=solvent,
                yield_pct=outcome,
                time_h=5.5,
                temperature_c=temperature,
                ratio_g_ml=ratio_g_ml,
                particle_size_mid_um=481.22,
                sample_weight_g=35.0,
                solvent_volume_ml=35.0 / ratio_g_ml,
                dry_temperature_c=110.0,
                dry_time_h=5.5,
                doi="10.3303/CET2189069",
                source_url="https://www.cetjournal.it/cet/21/89/069.pdf",
                title=(
                    "2-Methyltetrahydrofuran as a Potential Green Solvent for "
                    "Lipids Extraction from Spent Coffee Grounds for Fuel Grade "
                    "Hydrocarbons Production"
                ),
                quality_tier="gold",
                recommended_use="primary_augmentation",
                provenance_note=(
                    "Figure 1b exact bar label; mixed dried SCG; ratio experiment "
                    "described in Results as fixed at 5.50 h. Methods states 6 h, "
                    "so the within-paper duration discrepancy is retained in the audit note."
                ),
                raw_ratio_text=f"{1 / ratio_g_ml:g}:1 solvent-to-SCG (% w/v); paper also reports {ratio_g_ml:g}",
                raw_particle_text="reported mean particle size 481.22 micrometres",
                quality_flags=(
                    "within_paper_time_5.5h_results_vs_6h_methods|"
                    "solvent_descriptors_from_standard_metadata"
                ),
            ))
    frame = pd.DataFrame(rows)
    frame["prior_processing"] = "brewed_oven_dried_and_mixed"
    frame["manuscript_reference_number"] = 10
    frame["manuscript_reference_citation"] = "[10]"
    frame["reference_link_status"] = "linked"
    frame["manuscript_reference_title"] = (
        "2-Methyltetrahydrofuran as a Potential Green Solvent for Lipids "
        "Extraction from Spent Coffee Grounds for Fuel Grade Hydrocarbons Production"
    )
    frame["manuscript_reference_doi"] = "10.3303/CET2189069"
    frame["manuscript_reference_full"] = (
        "Mkhonto B, Chetty M. 2-Methyltetrahydrofuran as a Potential Green "
        "Solvent for Lipids Extraction from Spent Coffee Grounds for Fuel Grade "
        "Hydrocarbons Production. Chemical Engineering Transactions. "
        "2021;89:409-414. doi:10.3303/CET2189069."
    )
    return frame


def build_candidates() -> pd.DataFrame:
    return pd.concat(
        [
            mkhonto_chetty_2021(), hossain_2020(), andriana_2023(),
            esonye_2021(), senrayan_2018(),
        ],
        ignore_index=True,
        sort=False,
    )


def prepare_augmented(inclusion: str) -> pd.DataFrame:
    candidates = build_candidates()
    allowed = {
        "v13": [],
        "hossain": ["Hossain2020_Papaya"],
        "hossain_andriana": ["Hossain2020_Papaya", "Andriana2023_Papaya"],
        "hossain_senrayan": ["Hossain2020_Papaya", "Senrayan2018_Papaya"],
        "hossain_andriana_senrayan": [
            "Hossain2020_Papaya", "Andriana2023_Papaya", "Senrayan2018_Papaya"
        ],
        "all_sensitivity": [
            "Hossain2020_Papaya", "Andriana2023_Papaya", "Esonye2021_Dacryodes",
            "Senrayan2018_Papaya",
        ],
    }
    if inclusion not in allowed:
        raise ValueError(f"Unknown inclusion set: {inclusion}")
    base = v12.load_all().copy()
    # The verified 2021 SCG/Soxhlet source replaces the four unsupported
    # OriginalRef_10 legacy rows in every current analysis corpus.
    verified_primary = candidates[
        candidates["source_id"].eq("Mkhonto2021_CET89")
    ].copy()
    base = pd.concat([base, verified_primary], ignore_index=True, sort=False)
    if not allowed[inclusion]:
        for column in v12.v4.CORE_FEATURES + ["particle_size_mid_um", "yield_pct"]:
            base[column] = pd.to_numeric(base[column], errors="coerce")
        return v12.add_targets_and_batch_features(base).reset_index(drop=True)
    add = candidates[candidates["source_id"].isin(allowed[inclusion])].copy()
    combined = pd.concat([base, add], ignore_index=True, sort=False)
    for column in v12.v4.CORE_FEATURES + ["particle_size_mid_um", "yield_pct"]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    # Re-engineer source-relative ranks after the augmentation.
    return v12.add_targets_and_batch_features(combined).reset_index(drop=True)


def audit_candidates(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    feature_columns = [
        "ratio_g_ml", "time_h", "temperature_c", "particle_size_mid_um",
        "sample_weight_g", "solvent_volume_ml", "yield_pct",
    ]
    source_rows = []
    for source, frame in candidates.groupby("source_id", sort=True):
        condition_cols = [
            "feedstock_name", "solvent", "ratio_g_ml", "time_h",
            "temperature_c", "particle_size_mid_um",
        ]
        exact_condition_duplicates = int(frame.duplicated(condition_cols, keep=False).sum())
        source_rows.append({
            "source_id": source,
            "rows": len(frame),
            "feedstock": frame["feedstock_name"].iloc[0],
            "solvents": "|".join(sorted(frame["solvent"].unique())),
            "quality_tier": frame["quality_tier"].iloc[0],
            "recommended_use": frame["recommended_use"].iloc[0],
            "low_n": int((frame["yield_pct"] <= v12.ABSOLUTE_CUTS[0]).sum()),
            "mid_n": int(((frame["yield_pct"] > v12.ABSOLUTE_CUTS[0]) &
                           (frame["yield_pct"] <= v12.ABSOLUTE_CUTS[1])).sum()),
            "high_n": int((frame["yield_pct"] > v12.ABSOLUTE_CUTS[1]).sum()),
            "yield_min": float(frame["yield_pct"].min()),
            "yield_max": float(frame["yield_pct"].max()),
            "condition_duplicate_rows": exact_condition_duplicates,
            "flagged_rows": int(frame["quality_flags"].ne("").sum()),
        })
    source_audit = pd.DataFrame(source_rows)
    missing = candidates.groupby("source_id")[feature_columns].agg(
        lambda column: float(column.isna().mean())
    ).reset_index()
    summary = {
        "candidate_rows": int(len(candidates)),
        "candidate_sources": int(candidates["source_id"].nunique()),
        "primary_augmentation_rows": int(
            candidates["recommended_use"].eq("primary_augmentation").sum()
        ),
        "sensitivity_rows": int(
            candidates["recommended_use"].ne("primary_augmentation").sum()
        ),
        "all_methods_soxhlet": bool(candidates["method_domain"].eq("primary_soxhlet").all()),
        "primary_soxhlet_rows": int(candidates["method_domain"].eq("primary_soxhlet").sum()),
        "related_method_rows": int(candidates["method_domain"].ne("primary_soxhlet").sum()),
        "locked_absolute_cuts": v12.ABSOLUTE_CUTS.tolist(),
        "decision_rule": (
            "Primary Soxhlet augmentation uses Hossain. Senrayan is related-method "
            "auxiliary data only. Andriana and Esonye can enter only pre-declared "
            "sensitivity analyses because of documented table/method ambiguity."
        ),
    }
    return source_audit, missing, summary


def main() -> None:
    candidates = build_candidates()
    candidates = v12.add_targets_and_batch_features(candidates)
    source_audit, missing, summary = audit_candidates(candidates)
    candidates.to_csv(OUT / "v14_literature_candidates.csv", index=False)
    source_audit.to_csv(OUT / "v14_source_quality_audit.csv", index=False)
    missing.to_csv(OUT / "v14_missingness_by_source.csv", index=False)
    for inclusion in (
        "v13", "hossain", "hossain_andriana", "hossain_senrayan",
        "hossain_andriana_senrayan", "all_sensitivity",
    ):
        prepare_augmented(inclusion).to_csv(
            OUT / f"training_{inclusion}.csv", index=False
        )
    (OUT / "v14_data_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(source_audit.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
