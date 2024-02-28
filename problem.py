# %% import libraries

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer


# workflow libraires
from sklearn.base import is_classifier
from sklearn.utils import _safe_indexing
# from ..utils.importing import import_module_from_source

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
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")

def get_cv(X, y):
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=2)
    return cv.split(X, y, groups)

# %% define classification model

X_train_df, y_train = get_train_data()
X_test_df, y_test = get_test_data()

# The model will take numpy arrays as input.
X_train = X_train_df.to_numpy()

X_test = X_test_df.to_numpy()


numeric_transformer = Pipeline(
    steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),  
        ("vectorizer", TfidfVectorizer()),  # Convert text to numbers
        ("scaler", StandardScaler())  
])

clf_lr = Pipeline(
    steps=[("transformer", numeric_transformer), ("classifier", LogisticRegression(max_iter=500))]
)

clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)


print("balanced accuracy score linear regressor (score to beat) = ", balanced_accuracy_score(y_test, y_pred_lr))
