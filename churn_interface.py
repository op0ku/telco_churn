#import required libraries
from explore import wrangle  #module created to wrangle the dataset

import shap
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier


MODEL_PATH = "C:/Users/solace.dark/Documents/Jupyter Notebook/catboost_model.skops"
DATA_PATH = "C:/Users/solace.dark/Documents/Jupyter Notebook/Practical Projects/Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"

st.set_page_config(page_title='Telco Customer Churn Prediction',  layout='wide')

@st.cache_resource




#return shap values
def compute_shap(model, X_train, X_test):
    explainer = shap.TreeExplainer(model)
    train_shap_vals = explainer.shap_values(X_train)
    test_shap_vals = explainer.shap_values(X_test)
    return explainer, train_shap_vals, test_shap_vals

#visualize shap values for specific customer
def viz_shap_values(model, explainer, train_shap_vals, test_shap_vals, customer_id, X_train, X_test):
    customer_index = X_test

st.title('Telco Customer Churn Prediction')