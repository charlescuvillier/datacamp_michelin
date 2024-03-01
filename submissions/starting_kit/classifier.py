from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression


class Classifier(BaseEstimator):
    def __init__(self):
        text_column=['blurb']
        cat_column=['name','city','country','chef','cuisine1','cuisine2']
        num_column=['lat','lon']
        #Preprocessing for numeric data
        numeric_transformer = Pipeline(
        steps = [
        ("imputer", SimpleImputer(strategy="median")),  
        ("scaler", StandardScaler())  
        ])
        #Preprocessing for categorical data
        cat_transformer=Pipeline(
            steps=[
                ("imputer2", SimpleImputer(strategy="most_frequent")),
                ("encoder",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))
                ]
        )


        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_column),
                ("cat", cat_transformer, cat_column),
            ],remainder='drop' # Here not use the description of the restaurant
        )

        #The first model to beat use a logistic linear regressor
        self.model = LogisticRegression(max_iter=500)
        self.pipe = make_pipeline(self.preprocessor, self.model)

    def fit(self, X, y):
        self.pipe.fit(X, y)

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)