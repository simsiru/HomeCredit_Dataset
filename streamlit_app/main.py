import streamlit as st
import requests
import pandas as pd


df = pd.read_csv('test_default.csv')
sample_default_df = df.drop(columns='TARGET').sample(10)

df = pd.read_csv('test_application.csv')
sample_application_df = df.drop(
    columns=['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).sample(10)


st.title('ML capstone project')


tab1, tab2, tab3 = st.tabs(['Default risk prediction', 'Clustering', 
                            'Application outcome prediction'])


tab1.header('Default risk prediction')

tab1.subheader('Sample input data')

if tab1.button('Show', key=0):
    tab1.dataframe(sample_default_df)

tab1.subheader('Generate output')

if tab1.button('Predict', key=1):
    json_data = sample_default_df.to_json(orient='split')
    resp = requests.post(
        'https://projectapi-t3oyhqidxq-uc.a.run.app/default_risk_prediction', 
        json=json_data)
    tab1.write(resp.json())


tab2.header('Clustering')

tab2.subheader('Sample input data')

if tab2.button('Show', key=2):
    tab2.dataframe(sample_default_df)

tab2.subheader('Generate output')

if tab2.button('Assign cluster labels', key=3):
    json_data = sample_default_df.to_json(orient='split')
    resp = requests.post(
        'https://projectapi-t3oyhqidxq-uc.a.run.app/clustering', 
        json=json_data)
    tab2.write(resp.json())


tab3.header('Application outcome prediction')

tab3.subheader('Sample input data')

if tab3.button('Show', key=4):
    tab3.dataframe(sample_application_df)

tab3.subheader('Generate output')

if tab3.button('Predict', key=5):
    json_data = sample_application_df.to_json(orient='split')
    resp = requests.post(
        'https://projectapi-t3oyhqidxq-uc.a.run.app/application_outcome_prediction', 
        json=json_data)
    tab3.write(resp.json())