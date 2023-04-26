import numpy as np
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import hdbscan


default_clf = joblib.load('default_risk_lgbm.pkl')
clusterer = joblib.load('hdbscan_clusterer.pkl')
application_clf = joblib.load('application_clf_lgbm.pkl')


def data_preprocessing(df: pd.DataFrame) -> None:
    df = df.astype(
      {col:'category' for col in df.select_dtypes('object').columns})
    
    df = df.replace(['XNA', 365243.0], np.nan)

    return df


app = Flask(__name__)


@app.route("/default_risk_prediction", methods=["POST"])
def default_risk_prediction():
    df = pd.read_json(request.json, orient='split')

    df = data_preprocessing(df)

    pred = default_clf.predict_proba(df)[:, 1]

    data = {"default_risk": pred.tolist()}

    return jsonify(data)


@app.route("/clustering", methods=["POST"])
def clustering():
    df = pd.read_json(request.json, orient='split')

    df = data_preprocessing(df)

    cluster_labels = hdbscan.approximate_predict(
        clusterer['clusterer'],
        clusterer['preprocessor'].transform(df)
    )

    data = {"cluster_labels": cluster_labels[0].tolist()}

    return jsonify(data)


@app.route("/application_outcome_prediction", methods=["POST"])
def application_outcome_prediction():
    df = pd.read_json(request.json, orient='split')

    df = data_preprocessing(df)

    pred = application_clf.predict_proba(df)[:, 1]

    data = {"application_outcome": pred.tolist()}

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)