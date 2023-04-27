# Home Credit Dataset

### Introduction

The aim of this project is to use the Homecredit data to see what kind of models can be built for prediction, most importantly default risk prediction.
Final models are then deployed to Google Cloud Platform as APIs. API functionality is demonstrated using a Streamlit app.

### Technologies

Typical data science tools were used like `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, but also `duckdb` for querying large pandas dataframes in memory. 
Also gradient boosting frameworks were used like `XGBoost` and `LightGBM`. For API creation and deployment `Flask`, `Streamlit` and `Google Cloud Platform` were used.
