"""Four comparators added during the 2026-09-30 review, not four new experiments.

All use the full ensemble's five selected rows. p_class_only is the fitted
and prior-adjusted ensemble branch, not the conventional Extra-Trees baseline.
"""
import numpy as np
import multifeedstock_v12 as v12
import v13_hierarchical_search as v13

def probabilities(data, source_frame, cal, test, adjusted_classifier_probability):
    labels=data.loc[cal,'absolute_class'].to_numpy(int)
    yields=data.loc[cal,'yield_pct'].to_numpy(float)
    majority=int(np.bincount(labels,minlength=3).argmax())
    mean_class=int(np.digitize(yields.mean(),v12.ABSOLUTE_CUTS,right=True))
    return {
        'p_class_only': adjusted_classifier_probability,
        'target_neighbour_only': v13.calibration_knn_probability(source_frame,cal,test,labels),
        'five_label_majority': np.tile(np.eye(3)[majority],(len(test),1)),
        'five_yield_mean_class': np.tile(np.eye(3)[mean_class],(len(test),1)),
    }
