import numpy as np

def uncertainty_from_pseudo_label(pseudo_label, thres_low=0.1, thres_high=0.9):
    pseudo1 = pseudo_label.copy()
    pseudo1[pseudo1 <= thres_low] = 0
    pseudo1[pseudo1 >= thres_high] = 1
    pseudo1[(pseudo1 > thres_low) & (pseudo1 < thres_high)] = 0.5
    uncertainty = np.abs(pseudo1 - 0.5) * 2
    return uncertainty
