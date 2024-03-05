
import numpy as np
import pandas as pd
import sys

def haversine(pred, gt):
    """
    Haversine distance between two points on the Earth surface.

    Parameters
    -----
    pred: numpy array of shape (N, 2)
        Contains predicted (LATITUDE, LONGITUDE).
    gt: numpy array of shape (N, 2)
        Contains ground-truth (LATITUDE, LONGITUDE).

    Returns
    ------
    numpy array of shape (N,)
        Contains haversine distance between predictions
        and ground truth.
    """
    pred_lat = np.radians(pred[:, 0])
    pred_long = np.radians(pred[:, 1])
    gt_lat = np.radians(gt[:, 0])
    gt_long = np.radians(gt[:, 1])

    dlat = gt_lat - pred_lat
    dlon = gt_long - pred_long

    a = np.sin(dlat/2)**2 + np.cos(pred_lat) * np.cos(gt_lat) * np.sin(dlon/2)**2

    d = 2 * 6371 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return d
     
if __name__ == '__main__':

    ground_truth = pd.read_csv('solutions.csv', index_col = 'TRIP_ID')
    submission = pd.read_csv(sys.argv[1], index_col = 'TRIP_ID')

    submission = submission.loc[ground_truth.index]
        
    gt_public = ground_truth[ground_truth['PUBLIC'] == True]
    gt_private = ground_truth[ground_truth['PUBLIC'] == False]

    sub_public = submission.loc[gt_public.index]
    sub_private = submission.loc[gt_private.index]

    hvpublic = haversine(gt_public[['LATITUDE','LONGITUDE']].values, sub_public[['LATITUDE','LONGITUDE']].values).mean()
    hvprivate = haversine(gt_private[['LATITUDE','LONGITUDE']].values, sub_private[['LATITUDE','LONGITUDE']].values).mean()

    print("Haversine score:\n    Public: %.8f\n    Private: %.8f" % (hvpublic, hvprivate))
