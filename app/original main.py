import sys
sys.path.append('C:/Users/solace.dark/Documents/Jupyter Notebook/Projects/churn_prediction')
from explore import wrangle

import streamlit as st
import math
import numpy as np
import joblib

def add_sidebar():
    st.sidebar.header('Customer Features')
    data = wrangle('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
    slider_labels = [('tenure', 'tenure'),
                     ('monthly charges', 'MonthlyCharges'),
                     ('total charges', 'TotalCharges')]
    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(math.ceil(data[key].max()))
        )
    return input_dict

def add_predictions(input_data):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    model = joblib.load('catboost_model.pkl')
    prediction = model.predict(input_array)

    if prediction[]

def main():
    st.set_page_config(
        page_title='Telco Customer Churn Prediction',
        layout='wide',
        initial_sidebar_state= 'expanded'
    )

    input_data = add_sidebar()

    with st.container():
        st.title('Telco Customer Churn Prediction')
        st.write('This app predicts whether a customer will leave the telco company.')

    col1, col2 = st.columns([4, 1])

    with col1:
    #    radar_chart = get_radar_chart(input_data)
    #    st.plotly_chart(radar_chart)
        st.write('This is column 1')
    with col2:
        add_predictions(input_data)


if __name__ == '__main__':
    main()