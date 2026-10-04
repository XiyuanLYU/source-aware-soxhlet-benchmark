from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVC

import multifeedstock_v12 as v12
import v14_external_data as v14data

HERE = Path(__file__).resolve().parent
SUBMISSION = HERE.parent
DATA_DIR = SUBMISSION / "05_Data_Sources"
CORE = DATA_DIR / "Core_Comparison"
LOCKED = DATA_DIR / "Locked_Results"
OUT = DATA_DIR / "Conventional_Baselines"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(19101, 19116))
CLASSES = np.array([0, 1, 2], dtype=int)
NUMERIC = list(v12.BATCH_NUMERIC)
CATEGORICAL = list(v12.MATERIAL_FULL)


def aligned_probability(model, frame):
    raw = model.predict_proba(frame)
    out = np.zeros((len(frame), 3), dtype=float)
    for j, cls in enumerate(model.named_steps["model"].classes_):
        out[:, int(cls)] = raw[:, j]
    out = np.clip(out, 1e-12, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def make_model(name: str, seed: int, n_train: int):
    scale = name in {"logistic_regression", "svm_rbf", "knn"}
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", RobustScaler() if scale else "passthrough"),
        ]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=True)
    if name == "logistic_regression":
        model = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced", random_state=seed)
    elif name == "random_forest":
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features=0.8,
                                       class_weight="balanced_subsample", random_state=seed, n_jobs=1)
    elif name == "extra_trees":
        model = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, max_features=0.8,
                                     class_weight="balanced", random_state=seed, n_jobs=1)
    elif name == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=120, learning_rate=0.045, max_depth=2,
                                           min_samples_leaf=3, random_state=seed)
    elif name == "svm_rbf":
        model = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
                    probability=True, random_state=seed)
    elif name == "knn":
        model = KNeighborsClassifier(n_neighbors=min(7, max(3, int(math.sqrt(n_train)))),
                                     weights="distance", metric="minkowski", p=2)
    else:
        raise ValueError(name)
    return Pipeline([("prep", prep), ("model", model)])


def metric_values(frame):
    y=frame.actual_class.to_numpy(int); pred=frame.predicted_class.to_numpy(int)
    source_acc=frame.assign(ok=y==pred).groupby("source_id").ok.mean()
    return {
        "n_test": len(frame), "sources": frame.source_id.nunique(),
        "accuracy": accuracy_score(y,pred),
        "balanced_accuracy": balanced_accuracy_score(y,pred),
        "macro_f1": f1_score(y,pred,labels=CLASSES,average="macro",zero_division=0),
        "publication_balanced_accuracy": source_acc.mean(),
        "worst_source_accuracy": source_acc.min(),
        "ordinal_mae": np.mean(np.abs(y-pred)),
    }


def probabilistic_scores(frame, bins=10):
    y=frame.actual_class.to_numpy(int)
    p=frame[["probability_low","probability_mid","probability_high"]].to_numpy(float)
    one=np.eye(3)[y]
    brier=np.mean(np.sum((p-one)**2,axis=1))
    conf=p.max(axis=1); correct=(p.argmax(axis=1)==y).astype(float)
    edges=np.linspace(0,1,bins+1); ece=0.; rows=[]
    for i in range(bins):
        mask=(conf>=edges[i]) & ((conf<edges[i+1]) if i<bins-1 else (conf<=edges[i+1]))
        if mask.any():
            ece += mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
            rows.append((i,edges[i],edges[i+1],int(mask.sum()),conf[mask].mean(),correct[mask].mean()))
    return brier,ece,rows


def run_baselines():
    # The historical modules resolve their input table beside the code file.
    # Point that constant to the packaged, unchanged input table.
    v12.PRIMARY = DATA_DIR / "Model_Input_Data" / "training_primary_v10_1.csv"
    data=v14data.prepare_augmented("hossain_senrayan").copy()
    by_row={str(v):i for i,v in data.row_id.items()}
    core=pd.read_csv(CORE/"core_comparison_common_test_predictions.csv")
    guided=core[core.config.eq("guided5_evidence_gated")]
    models=["logistic_regression","random_forest","extra_trees","gradient_boosting","svm_rbf","knn"]
    rows=[]
    for (seed,source),g in guided.groupby(["seed","source_id"],sort=True):
        cal_ids=str(g.calibration_rows.iloc[0]).split(";")
        cal_idx=np.array([by_row[x] for x in cal_ids],dtype=int)
        test_idx=np.array([by_row[x] for x in g.row_id.astype(str)],dtype=int)
        train_other=data.index[~data.source_id.astype(str).eq(str(source))].to_numpy(int)
        train_idx=np.concatenate([train_other,cal_idx])
        y=data.absolute_class.to_numpy(int)
        for mi,name in enumerate(models):
            model=make_model(name,int(seed)*10000+mi*100+len(rows)%97,len(train_idx))
            model.fit(data.loc[train_idx],y[train_idx])
            prob=aligned_probability(model,data.loc[test_idx]); pred=prob.argmax(axis=1)
            for pos,idx in enumerate(test_idx):
                rows.append({"seed":seed,"model":name,"source_id":source,
                             "feedstock_name":data.loc[idx,"feedstock_name"],"row_id":data.loc[idx,"row_id"],
                             "actual_class":int(y[idx]),"predicted_class":int(pred[pos]),
                             "probability_low":prob[pos,0],"probability_mid":prob[pos,1],"probability_high":prob[pos,2],
                             "calibration_rows":";".join(cal_ids)})
    pred=pd.DataFrame(rows); pred.to_csv(OUT/"conventional_baseline_predictions.csv",index=False)
    metrics=[]
    for (seed,name),g in pred.groupby(["seed","model"]):
        m=metric_values(g); b,e,_=probabilistic_scores(g); metrics.append({"seed":seed,"model":name,**m,"multiclass_brier":b,"ece_10bin":e})
    metrics=pd.DataFrame(metrics); metrics.to_csv(OUT/"conventional_baseline_metrics.csv",index=False)
    summary=metrics.groupby("model",as_index=False).agg(seeds=("seed","nunique"),n_test_per_seed=("n_test","first"),
        accuracy_mean=("accuracy","mean"),accuracy_sd=("accuracy","std"),balanced_accuracy_mean=("balanced_accuracy","mean"),
        macro_f1_mean=("macro_f1","mean"),publication_balanced_accuracy_mean=("publication_balanced_accuracy","mean"),
        worst_source_accuracy_mean=("worst_source_accuracy","mean"),ordinal_mae_mean=("ordinal_mae","mean"),
        multiclass_brier_mean=("multiclass_brier","mean"),ece_10bin_mean=("ece_10bin","mean"))
    summary.to_csv(OUT/"conventional_baseline_summary.csv",index=False)
    return pred,metrics,summary


def locked_calibration():
    p=pd.read_csv(LOCKED/"locked_v14_predictions.csv")
    p=p[p.config.eq("v14_router_full")].copy()
    metrics=[]; bins=[]
    for seed,g in p.groupby("seed"):
        b,e,br=probabilistic_scores(g)
        metrics.append({"seed":seed,"n_test":len(g),"multiclass_brier":b,"ece_10bin":e})
        for row in br: bins.append({"seed":seed,"bin":row[0],"lower":row[1],"upper":row[2],"n":row[3],"mean_confidence":row[4],"observed_accuracy":row[5]})
    pd.DataFrame(metrics).to_csv(OUT/"locked_probability_calibration_metrics.csv",index=False)
    pd.DataFrame(bins).to_csv(OUT/"locked_probability_reliability_bins_by_seed.csv",index=False)
    # Pooled reliability is descriptive; seed-level metrics remain inferential unit.
    b,e,br=probabilistic_scores(p)
    pooled=pd.DataFrame(br,columns=["bin","lower","upper","n","mean_confidence","observed_accuracy"])
    pooled.to_csv(OUT/"locked_probability_reliability_pooled.csv",index=False)
    summary={"seeds":int(p.seed.nunique()),"predictions":int(len(p)),"rows_per_seed":int(len(p)/p.seed.nunique()),
             "multiclass_brier_mean":float(pd.DataFrame(metrics).multiclass_brier.mean()),
             "multiclass_brier_sd":float(pd.DataFrame(metrics).multiclass_brier.std(ddof=1)),
             "ece_10bin_mean":float(pd.DataFrame(metrics).ece_10bin.mean()),
             "ece_10bin_sd":float(pd.DataFrame(metrics).ece_10bin.std(ddof=1)),
             "pooled_multiclass_brier":b,"pooled_ece_10bin":e,
             "definition":"Multiclass Brier = mean sum of squared probability error across three classes; ECE uses 10 fixed confidence bins."}
    (OUT/"locked_probability_calibration_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary,pooled


if __name__=="__main__":
    _,_,s=run_baselines(); cal,_=locked_calibration()
    print(s.to_string(index=False)); print(json.dumps(cal,indent=2))
