from explore import wrangle

import streamlit as st
import math
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

def add_sidebar():
    st.sidebar.header('Customer Features')
    data = wrangle('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
    input_dict = {}

    numeric_features =  [col for col in data.columns if col in data.select_dtypes('number').columns and col != 'Churn']
    num_slider_labels = []
    for feature in numeric_features:
        feature_tuple = (feature, feature)
        num_slider_labels.append(feature_tuple)
    
    for label, key in num_slider_labels:
        if isinstance(data[key].max(), float):
            input_dict[key] = st.sidebar.slider(
                label=label,
                min_value=float(0),
                max_value=float(math.ceil(data[key].max())))
        else:
            input_dict[key] = st.sidebar.slider(
                label=label,
                min_value=0,
                max_value=data[key].max())

    category_features = [col for col in data.columns if col in data.select_dtypes('object').columns]
    cat_slider_labels = []
    for feature in category_features:
        feature_tuple = (feature, feature)
        cat_slider_labels.append(feature_tuple)
    
    for i, (label, key) in enumerate(cat_slider_labels):
        input_dict[key] = st.sidebar.radio(label, data[key].unique(), key=i)

    if st.sidebar.button("Serve"):
        return input_dict
    
    return None

model = joblib.load('./catboost_model.pkl')

def get_input(model, input_data):
    input_array = np.array([input_data[feature] for feature in model.feature_names_]).reshape(1, -1)
    prediction = model.predict(input_array)

    return input_array, prediction

def add_predictions(input, prediction):
    with st.container(border=True):
        st.subheader('Prediction On Customer')
        if prediction[0] == 1:
            st.write('<p style="color:red"><b>Customer churns</p>', unsafe_allow_html=True)
            st.write(f'Probability customer churns:')
            st.write(f'<p style="color:white;font-size:80px;"><b>{round(model.predict_proba(input)[0][1] * 1e2)}%</b></p>', unsafe_allow_html=True)
        else:
            st.write('<p style="color:green"><b>Customer stays</p>', unsafe_allow_html=True)
            st.write(f'Probability customer stays:')
            st.write(f'<p style="color:white;font-size:80px;"><b>{round(model.predict_proba(input)[0][0] * 1e2)}%</b></p>', unsafe_allow_html=True)
    


def plot_decision(model, input):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input)
    expected_value = explainer.expected_value

    fig, ax = plt.subplots(figsize=(5, 4))
    shap.plots.decision(expected_value, shap_vals[0], feature_names=model.feature_names_)
    st.pyplot(fig)
    plt.close()

def main():
    st.set_page_config(
        page_title='Telco Customer Churn Prediction',
        layout='wide',
        initial_sidebar_state= 'collapsed'
    )

    with st.container():
        st.title('Telco Customer Churn Prediction')
        st.write('''
                This app is designed to predict whether a customer will leave the telco company and marks substantial advancement in customer retention analytics.
                Based on predictions, decision makers can take proactive measures to retain valuable customers.
                ''')

    input_data = add_sidebar()
    if input_data is not None:
        input, prediction = get_input(model, input_data)

        with st.container():
            col1, col2 = st.columns([3.5, 1.5])

            with col2:
                add_predictions(input, prediction)

            with col1:
                st.subheader('Decision Plot')

                plot_decision(model, input)
                st.write('''
                        The decision plot displays which features, at what magnitude contributes most to whether a customer leaves the company.
                        A branch that points right(red) means the feature tends to push the customer to churn. And to the left(blue) pushes the customer to stay.
                        Decision makers can use this plot to understand which factors matter the most to retaining the customers.
                        ''')
    else:
        st.write('<p style="color:yellow;font-size:50px;"><b>Choose customer features on the sidebar to serve predictions</b></p>', unsafe_allow_html=True)
        

if __name__ == '__main__':
    main()
