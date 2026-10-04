"""Refit all eleven Table 6 methods from the 197-row input, with no result cache.

Reference_Results is read ONLY AFTER all predictions and scores are computed.
Original acquisition, model parameters and fit-seed conventions are preserved.
"""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
import argparse,hashlib,importlib.metadata,json,platform,sys,time,warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent/'modules'))
import conventional_baselines_and_calibration as conventional
import v13_locked_confirmation as locked
import v13_hierarchical_search as v13
import additional_controls
MODELS=['logistic_regression','random_forest','extra_trees','gradient_boosting','svm_rbf','knn']
PROBS=['probability_low','probability_mid','probability_high']
KEYS=['seed','source_id','model','row_id']
SEEDS=list(range(19101,19116))

def fold(job):
    data,seed,source,si,baseline_rows_before,config=job
    warnings.filterwarnings('ignore',category=FutureWarning)
    warnings.filterwarnings('ignore',message='y_pred contains classes not in y_true')
    frame=data.loc[data.source_id.eq(source)]
    model_seed=seed*10000+si*100
    guidance,delta=locked.fit_unlabeled_guidance_and_delta(data,source,model_seed)
    cal=v13.calibration_indices(frame,5,seed+si*101,'delta_guided',guidance.predict(frame))
    test=frame.index.difference(cal).to_numpy()
    pclass=locked.classifier_probability(data,source,cal,test,model_seed+5)
    probs={'calibrated_ensemble':locked.v13_probabilities(data,frame,cal,test,pclass,delta,gate_count=config['gate_count'])['v13_full']}
    probs.update(additional_controls.probabilities(data,frame,cal,test,pclass))
    train=np.concatenate([data.index[~data.source_id.eq(source)].to_numpy(),cal])
    y=data.absolute_class.to_numpy(int)
    fit_seeds={}
    for mi,name in enumerate(MODELS):
        # Exact legacy run_baselines convention: len(rows) counts all previous
        # six-model prediction rows, including previous sources and seeds.
        fit_seed=seed*10000+mi*100+(baseline_rows_before+mi*len(test))%97
        model=conventional.make_model(name,fit_seed,len(train))
        overrides=config['baseline_parameters'].get(name,{})
        if overrides:model.set_params(**{'model__'+k:v for k,v in overrides.items()})
        model.fit(data.loc[train],y[train])
        probs[name]=conventional.aligned_probability(model,data.loc[test])
        fit_seeds[name]=fit_seed
    cal_ids='|'.join(data.loc[cal,'row_id'].astype(str))
    rows=[]
    for name,p in probs.items():
        assert p.shape==(len(test),3) and np.isfinite(p).all() and np.allclose(p.sum(axis=1),1)
        for j,idx in enumerate(test):
            rows.append({'seed':seed,'source_id':source,'model':name,'row_id':data.loc[idx,'row_id'],
                'actual_class':int(y[idx]),'predicted_class':int(p[j].argmax()),'calibration_rows':cal_ids,
                **dict(zip(PROBS,map(float,p[j])))})
    manifest={'seed':seed,'source_id':source,'source_index':si,'calibration_rows':cal_ids,
        'test_rows':'|'.join(data.loc[test,'row_id'].astype(str)),'n_calibration':5,'n_test':len(test),
        'guidance_seed':model_seed,'delta_seed':model_seed+1,'class_branch_seed':model_seed+5,
        **{k+'_fit_seed':v for k,v in fit_seeds.items()}}
    return rows,manifest

def score(g):
    y=g.actual_class.to_numpy(int);p=g.predicted_class.to_numpy(int)
    return {'n_test':len(g),'sources':g.source_id.nunique(),'accuracy':accuracy_score(y,p),
        'balanced_accuracy':balanced_accuracy_score(y,p),
        'macro_f1':f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0),
        'ordinal_mae':float(np.abs(y-p).mean())}

def verify(pred,summary):
    ref=ROOT/'Reference_Results'
    e=pd.read_csv(ref/'original_ensemble_predictions.csv').assign(model='calibrated_ensemble')
    b=pd.read_csv(ref/'original_six_baseline_predictions.csv')
    a=pd.read_csv(ref/'review_added_four_predictions.csv').rename(columns={'config':'model'})
    expect=pd.concat([e,b,a],ignore_index=True)
    assert len(expect)==len(pred)==19470 and not expect.duplicated(KEYS).any()
    pair=pred.merge(expect[KEYS+['actual_class','predicted_class','calibration_rows']+PROBS],on=KEYS,suffixes=('_new','_reference'),validate='one_to_one')
    rows=[]
    for name,g in pair.groupby('model'):
        rows.append({'model':name,'predictions':len(g),
            'class_mismatches':int((g.predicted_class_new!=g.predicted_class_reference).sum()),
            'truth_mismatches':int((g.actual_class_new!=g.actual_class_reference).sum()),
            'calibration_list_mismatches':int((g.calibration_rows_new!=g.calibration_rows_reference).sum()),
            'max_probability_difference':max(float(np.abs(g[c+'_new']-g[c+'_reference']).max()) for c in PROBS)})
    audit=pd.DataFrame(rows)
    es=pd.read_csv(ref/'expected_11_method_summary.csv').set_index('model');ss=summary.set_index('model')
    diffs=[]
    for name in ss.index:
        for m in ['accuracy','balanced_accuracy','macro_f1','ordinal_mae']:
            diffs.append({'model':name,'metric':m,'computed':ss.loc[name,m+'_mean'],'reference':es.loc[name,m+'_mean'],'absolute_difference':abs(ss.loc[name,m+'_mean']-es.loc[name,m+'_mean'])})
    diff=pd.DataFrame(diffs)
    passed=bool(audit.class_mismatches.eq(0).all() and audit.truth_mismatches.eq(0).all() and audit.calibration_list_mismatches.eq(0).all() and audit.max_probability_difference.max()<1e-10 and diff.absolute_difference.max()<1e-12)
    return audit,diff,passed

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--output',type=Path,default=ROOT/'Results_New')
    ap.add_argument('--experiment',action='store_true',help='Record a new exploratory run; do not certify a match to the manuscript.')
    ap.add_argument('--config',type=Path,help='JSON overrides; requires --experiment. See Config/tuning_example.json.')
    args=ap.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    if args.config and not args.experiment:ap.error('--config requires --experiment')
    config={'gate_count':4,'baseline_parameters':{}}
    if args.config:
        supplied=json.loads(args.config.read_text(encoding='utf-8-sig'))
        unknown=set(supplied)-set(config)
        if unknown:raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        config.update(supplied)
    if type(config['gate_count']) is not int or not 1<=config['gate_count']<=6:raise ValueError('gate_count must be an integer from 1 to 6; 6 disables the gate for five labels.')
    if not isinstance(config['baseline_parameters'],dict):raise ValueError('baseline_parameters must be a JSON object')
    for name,params in config['baseline_parameters'].items():
        if name not in MODELS or not isinstance(params,dict):raise ValueError(f'Invalid baseline specification: {name}')
        base=conventional.make_model(name,0,197).named_steps['model'].get_params()
        if set(params)-set(base):raise ValueError(f'Unknown parameters for {name}: {set(params)-set(base)}')
        if any('__' in k for k in params):raise ValueError('Only estimator parameters are supported by this configuration file.')
    provenance=json.loads((ROOT/'Audit/module_provenance.json').read_text(encoding='utf-8'))
    unchanged=all(hashlib.sha256((ROOT/'Code/modules'/r['module']).read_bytes()).hexdigest()==r['delivered_sha256'] for r in provenance)
    if not unchanged and not args.experiment:raise ValueError('Core modules were edited. Use --experiment for exploratory results, or restore the original package for manuscript verification.')
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new empty output directory; cached results are never used.')
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    path=ROOT/'Data/model_input_197.csv';data=pd.read_csv(path).reset_index(drop=True)
    assert len(data)==data.row_id.nunique()==197 and data.source_id.nunique()==22
    assert data.method_domain.eq('primary_soxhlet').all()
    input_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    if not args.experiment and input_hash!='1d377f6eb17c8c50b8f857b96ac51f8486014ed067fcae8210ef1f189d2119c8':raise ValueError('Input file differs from the manuscript input. Use --experiment for a changed input.')
    sources=sorted(data.groupby('source_id').size().loc[lambda v:v>=8].index.astype(str));assert len(sources)==11
    jobs=[];offset=0
    for seed in SEEDS:
        for si,source in enumerate(sources):
            jobs.append((data,seed,source,si,offset,config));offset+=6*(int(data.source_id.eq(source).sum())-5)
    print(f'Refitting 11 methods on {len(jobs)} source/seed partitions; no cached predictions.',flush=True)
    rows=[];manifests=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(part,manifest) in enumerate(pool.map(fold,jobs),1):
            rows.extend(part);manifests.append(manifest)
            if i%11==0:print(f'{i}/165 folds completed in {time.monotonic()-start:.1f} s',flush=True)
    pred=pd.DataFrame(rows).sort_values(KEYS)
    assert len(pred)==19470 and pred.groupby(['seed','model']).size().eq(118).all()
    manifest=pd.DataFrame(manifests);assert len(manifest)==165
    for (seed,source),g in pred.groupby(['seed','source_id']):
        assert g.calibration_rows.nunique()==1
        cal=set(g.calibration_rows.iloc[0].split('|'));assert len(cal)==5
        expected=set(data.loc[data.source_id.eq(source),'row_id'])-cal
        assert all(set(h.row_id)==expected for _,h in g.groupby('model'))
        assert not cal & set(g.row_id)
    per=pd.DataFrame([{'seed':s,'model':m,**score(g)} for (s,m),g in pred.groupby(['seed','model'])])
    sums=[]
    for model,g in per.groupby('model'):
        r={'model':model,'seeds':15,'sources':11,'n_test_per_seed':118}
        for m in ['accuracy','balanced_accuracy','macro_f1','ordinal_mae']:r[m+'_mean']=g[m].mean();r[m+'_sd']=g[m].std(ddof=1)
        sums.append(r)
    summary=pd.DataFrame(sums)
    pred.to_csv(out/'all_11_method_predictions.csv',index=False);manifest.to_csv(out/'partitions_and_fit_seeds.csv',index=False)
    per.to_csv(out/'metrics_by_seed.csv',index=False);summary.to_csv(out/'table6_full_precision.csv',index=False)
    passed=None
    comparison={}
    if not args.experiment:
        audit,diff,passed=verify(pred,summary)
        audit.to_csv(out/'prediction_comparison.csv',index=False);diff.to_csv(out/'metric_comparison.csv',index=False)
        comparison={'all_predicted_classes_match_reference':bool(audit.class_mismatches.eq(0).all()),
            'max_probability_difference':float(audit.max_probability_difference.max()),'max_metric_difference':float(diff.absolute_difference.max())}
    report={'status':'EXPERIMENT_COMPLETED' if args.experiment else ('PASS' if passed else 'FAIL'),'input_rows':197,'source_publications':22,'eligible_sources':11,'seeds':SEEDS,
        'source_seed_partitions':165,'methods':11,'test_rows_per_seed_per_method':118,'predictions':len(pred),
        'used_saved_predictions_for_calculation':False,'regenerated_calibration_selection':True,
        'all_partition_checks_passed':True,'reference_comparison_performed':not args.experiment,**comparison,
        'effective_configuration':config,'core_modules_unchanged':unchanged,
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'Code').rglob('*.py'))},
        'input_sha256':input_hash,'python':platform.python_version(),
        'dependencies':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','matplotlib']},'elapsed_seconds':time.monotonic()-start}
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(summary[['model','accuracy_mean','balanced_accuracy_mean','macro_f1_mean']].to_string(index=False),flush=True)
    print(json.dumps(report,indent=2),flush=True)
    if passed is False:raise SystemExit('FAIL: inspect prediction_comparison.csv and metric_comparison.csv')
if __name__=='__main__':main()
