"""Export Table 6 and a matched-method figure from one explicitly selected run.

Quantitative grid: overall accuracy, class-balanced recall and macro F1 answer
different classification questions. Points are 15-seed means; bars are sample
SD (ddof=1), not independent-source confidence intervals. No significance claim.
Width 183 mm, editable vector text, 600 dpi PNG, minimum font size 7 pt.
Both scopes are exported: original seven and all eleven methods. The four
excluded from the seven-method view are the added branch/rule comparators.
"""
from pathlib import Path
import argparse,html,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORIGINAL=['calibrated_ensemble','extra_trees','random_forest','gradient_boosting','knn','logistic_regression','svm_rbf']
ADDED=['p_class_only','target_neighbour_only','five_label_majority','five_yield_mean_class']
LABELS={
    'calibrated_ensemble':'Calibrated ensemble',
    'extra_trees':'Extra-Trees','random_forest':'Random forest',
    'gradient_boosting':'Gradient boosting','knn':'K-nearest neighbours',
    'logistic_regression':'Logistic regression','svm_rbf':'Radial-basis SVM',
    'p_class_only':'Adjusted classifier only',
    'target_neighbour_only':'Neighbourhood vote only',
    'five_label_majority':'Five-label majority',
    'five_yield_mean_class':'Class of five-yield mean',
}
METRICS=[('accuracy','Accuracy (%)'),('balanced_accuracy','Balanced accuracy (%)'),('macro_f1','Macro F1 (%)')]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=json.loads((args.results/'verification.json').read_text(encoding='utf-8'))
    if report['status'] not in ['PASS','EXPERIMENT_COMPLETED']:raise ValueError('Do not export a failed reproduction run.')
    summary=pd.read_csv(args.results/'table6_full_precision.csv').set_index('model')
    seeds=pd.read_csv(args.results/'metrics_by_seed.csv')
    assert set(summary.index)==set(ORIGINAL+ADDED) and len(seeds)==165
    assert seeds.groupby('model').seed.nunique().eq(15).all()
    for metric,_ in METRICS:
        check=seeds.groupby('model')[metric].agg(['mean','std']).loc[summary.index]
        assert np.allclose(check['mean'],summary[metric+'_mean'])
        assert np.allclose(check['std'],summary[metric+'_sd'])
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new empty export directory.')
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':0.6,'savefig.facecolor':'white'})
    exported={}
    for scope,order in [('11_methods',ORIGINAL+ADDED),('7_methods',ORIGINAL)]:
        data=summary.loc[order].copy()
        data.to_csv(out/f'figure_source_{scope}.csv')
        table=pd.DataFrame({'Method':[LABELS[x] for x in order],
            'Group':['Original comparison' if x in ORIGINAL else 'Added comparator' for x in order]})
        for metric,label in METRICS:table[label]=[f'{v*100:.1f}' for v in data[metric+'_mean']]
        table['Ordinal MAE']=[f'{v:.3f}' for v in data.ordinal_mae_mean]
        table.to_csv(out/f'Table6_{scope}.csv',index=False,encoding='utf-8-sig')
        note='Means over 15 seeds; 118 test rows per seed from 11 publications. Class labels: 0, 1, 2.'
        content='<html><meta charset="utf-8"><style>body{font:16px Arial;margin:40px;max-width:1100px}td,th{padding:9px;border-bottom:1px solid #bbb;text-align:left}th{background:#edf1f4}table{border-collapse:collapse}p{line-height:1.6}</style>'
        content+=f'<h1>Table 6 — {scope.replace("_"," ")}</h1><p>{html.escape(note)}</p><p>Run status: {report["status"]}</p>'+table.to_html(index=False,escape=True)
        (out/f'Table6_{scope}.html').write_text(content,encoding='utf-8')
        height=118 if len(order)==11 else 90
        fig,axes=plt.subplots(1,3,figsize=(7.204724409448819,height/25.4),sharey=True)
        fig.subplots_adjust(left=0.295,right=0.985,bottom=0.15,top=0.86,wspace=0.22)
        y=np.arange(len(order))
        for panel,(ax,(metric,label)) in enumerate(zip(axes,METRICS)):
            for i,model in enumerate(order):
                color='#176684' if model=='calibrated_ensemble' else ('#956339' if model in ADDED else '#59636a')
                marker='s' if model in ADDED else 'o'
                ax.errorbar(100*data.loc[model,metric+'_mean'],i,xerr=100*data.loc[model,metric+'_sd'],
                    fmt=marker,color=color,markersize=3.4,capsize=2,elinewidth=0.75,capthick=0.75)
            ax.set_xlim(0,100);ax.set_xticks([0,50,100]);ax.set_ylim(len(order)-0.45,-0.65)
            ax.set_title(label,pad=17)
            ax.text(-0.03,1.04,chr(97+panel),transform=ax.transAxes,fontsize=8,fontweight='bold')
            ax.set_yticks(y,labels=[LABELS[m] for m in order]);ax.tick_params(axis='y',length=0,pad=6)
            ax.grid(axis='x',color='#dde2e5',linewidth=0.5);ax.set_axisbelow(True)
            if len(order)==11:ax.axhline(6.5,color='#bfc5c8',linewidth=0.6)
        fig.text(0.295,0.04,'Points: mean; bars: ±1 seed SD. 15 seeds; 118 test records per seed.',fontsize=7)
        fig.savefig(out/f'Figure2_{scope}.pdf')
        fig.savefig(out/f'Figure2_{scope}.svg')
        fig.savefig(out/f'Figure2_{scope}.png',dpi=600)
        plt.close(fig)
        exported[scope]={'included_methods':order,'excluded_methods':[m for m in ORIGINAL+ADDED if m not in order]}
    (out/'export_manifest.json').write_text(json.dumps({'source_results':str(args.results.resolve()),
        'run_status':report['status'],'summary':'mean across 15 seeds','spread':'sample SD, ddof=1',
        'independent_source_confidence_interval':False,'variants':exported},indent=2),encoding='utf-8')
    print(f'Tables, source data and both figure scopes exported to {out}')

if __name__=='__main__':main()
