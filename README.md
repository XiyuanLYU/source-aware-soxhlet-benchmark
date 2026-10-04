# Source-aware Soxhlet extraction benchmark: data and code

Data and computational materials accompanying **Source-aware classification of reported biomass extraction yields: a Soxhlet development benchmark**.

This snapshot contains the **197-record, 22-publication, five-feedstock** literature input used in the current manuscript. It is separate from earlier corpus revisions and must not be substituted with those revisions. The records are compiled from published studies; they are not 197 newly performed laboratory experiments. Bibliographic attribution is retained in `Data/source_inventory.csv` and the row-level input file.

## Reproduction scope

The executable workflow reproduces the eleven-method comparison in **Table 6 and Figure 2**. Saved predictions and partition assignments are supplied for inspection. This snapshot does not claim to reproduce every analysis or figure in the manuscript. The five separate SCG laboratory measurements and the 27-condition experimental design are not included in this snapshot.

No manuscript files, author contact lists, third-party article PDFs, authentication material, local machine paths, or Python environment folders are part of the intended public release.

## Attribution and reuse

Please acknowledge this benchmark and the original source publications when referring to the compiled records. Original publications retain their respective rights. No additional open-source or data-reuse license has been selected for this preparation snapshot; see `RIGHTS.md`.

## Model comparison reproduction

This package refits the eleven methods reported in Table 6 from the 197 experimental records in `Data/model_input_197.csv`. The records represent 22 publications and five feedstocks. The reference numbers in both data files match the manuscript and Supporting Information.

## Run

Use Python 3.12 and the versions listed in `requirements.txt`. From this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe Code\reproduce_table6.py --workers 4 --output Results_New
.\.venv\Scripts\python.exe Code\export_tables_and_figure.py --results Results_New --output Exports_New
```

The first command refits the models; the second exports the comparison table and Figure 2. Choose new empty output folders for each run.

## Evaluation

Eleven publications with at least eight records are each held out in turn. Five measurements from a held-out publication are used for calibration; its remaining records are used for testing. The combined test set contains 118 records per seed. The complete procedure uses 15 seeds, giving 165 publication/seed partitions. All methods receive the same calibration and test records.

`Results_New/verification.json` reports the run checks. With the supplied input and default settings, `status` should be `PASS`. Reference predictions are read after fitting and scoring, only to check reproduction. They are not used to train the models or calculate the new predictions.

## Files

- `Data`: experimental records and the publication inventory.
- `Code/reproduce_table6.py`: the fitting and evaluation entry point.
- `Code/additional_controls.py`: the adjusted-classifier, neighbourhood-vote, five-label-majority and five-yield-mean comparisons.
- `Code/modules`: model components imported by the entry point.
- `Code/export_tables_and_figure.py`: exports Table 6 and Figure 2 from a selected run. The eleven-method output is the version used in the manuscript; the seven-method output is an optional subset view.
- `Config/manuscript_defaults.json`: the fixed manuscript configuration.
- `Config/tuning_example.json`: an optional configuration example. Use `--experiment --config Config/tuning_example.json` to run it separately.
- `Reference_Results`: reference predictions and full-precision comparison metrics.
- `Audit/module_provenance.json`: checksums used to verify the imported model modules.
- `Results/table6_full_precision.csv`: the reproduced eleven-method summary supplied with this package.

The package entry point reproduces Table 6 and Figure 2. Other analyses and illustrations are documented in the manuscript and Supporting Information.
