# %% import libraries

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold
import rampwf as rw
 
problem_title = "Classification de restaurant au guide Michelin"

# Correspondence between categories and int8 categories
# Mapping int to categories
int_to_cat = {
   1 : 'ONE_STAR',
   2 : 'TWO_STARS',
   3 : 'THREE_STARS',
}

_event_label_int = list(int_to_cat)

Predictions = rw.prediction_types.make_multiclass(label_names=_event_label_int)
workflow = rw.workflows.Classifier()

# Mapping categories to int
cat_to_int = {v: k for k, v in int_to_cat.items()}

score_types = [
    rw.score_types.BalancedAccuracy(
        name="bal_acc", precision=3, adjusted=False
    ),
    rw.score_types.Accuracy(name="acc", precision=3),
]

def _get_data(path=".", split="train", cat_to_int = cat_to_int):
    # Load data from csv files into pd.DataFrame

    data_df = pd.read_csv(os.path.join(path, "data", split + ".csv"))

    data_df["cuisine1"] = data_df["cuisine1"].astype("category")
    data_df["cuisine2"] = data_df["cuisine2"].astype("category")

    # usefull columns
    subset = [
        'name',
        'blurb',
        'michelin_award',
        'city',
        'country',
        'lat',
        'lon',
        # 'image',
        'chef',
        'cuisine1',
        'cuisine2',
        # 'url',
        # 'url2'
    ]

    X = data_df[subset]

    # labels
    y = np.array(data_df["michelin_award"].map(cat_to_int).fillna(-1).astype("int8"))

    return X, y

groups = None

def get_train_data(path="."):
    data = pd.read_csv(os.path.join(path, "data", "train.csv"))
    data["name"] = data["name"].astype("category")
    Name = np.array(data["name"].cat.codes)
    global groups
    groups = Name
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")

def get_cv(X, y):
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=2)
    return cv.split(X, y, groups)
