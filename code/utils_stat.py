import numpy as np
import pandas as pd


def haversine(pred, gt):
    """
    Havarsine distance between two points on the Earth surface.

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

    a = np.sin(dlat/2)**2 + np.cos(pred_lat) * \
        np.cos(gt_lat) * np.sin(dlon/2)**2

    d = 2 * 6371 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return d


def load_train_data(csv_path, nb_rows=None):
    """
    Reads a CSV file (train or test) and returns the data contained.

    Parameters
    ----------
    csv_path : String
        Path to the CSV file to be read.
        e.g., "train.csv"

    Returns
    -------
    data : Pandas DataFrame 
        Data read from CSV file.
    n_samples : Integer
        Number of rows (samples) in the dataset.
    """
    data = pd.read_csv(csv_path, index_col="TRIP_ID")
    # Sample random rows
    df = data.sample(n=nb_rows)

    return df, len(df)


def load_test_data(csv_path, nb_rows=None):
    """
    Reads a CSV file (train or test) and returns the data contained.

    Parameters
    ----------
    csv_path : String
        Path to the CSV file to be read.
        e.g., "train.csv"

    Returns
    -------
    data : Pandas DataFrame 
        Data read from CSV file.
    n_samples : Integer
        Number of rows (samples) in the dataset.
    """
    data = pd.read_csv(csv_path, index_col="TRIP_ID", nrows=nb_rows)

    return data, len(data)


def write_submission(trip_ids, destinations, file_name="submission"):
    """
    This function writes a submission csv file given the trip ids, 
    and the predicted destinations.

    Parameters
    ----------
    trip_id : List of Strings
        List of trip ids (e.g., "T1").
    destinations : NumPy Array of Shape (n_samples, 2) with float values
        Array of destinations (latitude and longitude) for each trip.
    file_name : String
        Name of the submission file to be saved.
        Default: "submission".
    """
    n_samples = len(trip_ids)
    assert destinations.shape == (n_samples, 2)

    submission = pd.DataFrame(
        data={
            'LATITUDE': destinations[:, 0],
            'LONGITUDE': destinations[:, 1],
        },
        columns=["LATITUDE", "LONGITUDE"],
        index=trip_ids,
    )

    # Write file
    submission.to_csv(file_name + ".csv", index_label="TRIP_ID")
