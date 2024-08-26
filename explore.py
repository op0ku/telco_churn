# Module contains functions to wrangle and perform EDA on the churn dataset

import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


def wrangle(filepath):
    """
    Import and clean the dataset.

    Parameters:
    - filepath (string): path to the location of the csv file to clean.

    Returns:
    cleaned csv file as DataFrame.
    """
    #import csv file
    df = pd.read_csv(filepath)

    #remove unnecessary column
    df.drop(['customerID'], axis=1, inplace=True)

    # Changing categorical variables to numeric:
    df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

    #change type for required features
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('str')

    #fill in missing values
    df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'])

    #replace values in features
    df['SeniorCitizen'] = df['SeniorCitizen'].replace({'0': 'No', '1':'Yes'})
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
    columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'] 

    for col in columns_to_replace:
        df[col] = df[col].replace('No internet service', 'No')
    
    return df

#initialize empty list to store plots from perform_eda()
all_plots = []

def perform_eda(df, columns=None, label='Churn'):
    """
    Perform exploratory data analysis (EDA) on the cleaned DataFrame.

    Parameters:
    - df (pd.DataFrame): cleaned DataFrame to perform EDA on.
    - columns (list): list of column names to perform EDA on. if None, all columns in the DataFrame
    - label: the response feature to be predicted

    Returns:
    EDA results and plots.
    """
    if columns == None:
        columns = df.columns.to_list()
    elif isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        #visualize percentage of churned customers
        if col == label:
            plt.pie(df[label].value_counts(),
                    labels=[i for i in df[label].value_counts().index],
                    autopct='%1.2f%%')
            plt.title(f'{label} Distribution')
            plt.show()
            all_plots.append(plt) #add output plot to the list

            #return percentage of churned customers
            churn_pct = round((df[label].value_counts()[1] / len(df[label])) * 100, 2)
            no_churn_pct = round((df[label].value_counts()[0] / len(df[label])) * 100, 2)
            print(f'In the last month, {no_churn_pct}% of customers continue to use the telco services. {churn_pct}% of customers churned.')

        #analyze distribution of numeric features and their relationships with the label(Churn) using histogram
        elif is_numeric_dtype(df[col]):
            sns.histplot(data=df, x=col, hue=label, multiple='stack', kde=True)
            plt.title(f'Distribution of {col} vs {label}')
            plt.xlabel(f'{col} values')
            plt.ylabel(f'Density')
            plt.show()
            all_plots.append(plt)
         
        #analyze distribution of category features using Pie chart
        else:
            plt.pie(df[col].value_counts(),
                    labels=[i for i in df[col].value_counts().index],
                    autopct='%1.2f%%')
            plt.title(f'{col} Distribution')
            plt.show()
            all_plots.append(plt)
            
            #analyze category features and their relationships with the label(Churn) using bar chart
            sns.countplot(data=df, x=col, hue=label)
            plt.title(f'Bar Chart: {col} vs {label}')
            plt.xlabel(f'{col}')
            plt.ylabel(f'count')
            plt.show()
            all_plots.append(plt)

            #return churn probabilities for the category features
            churn_probabilities = df.groupby(col)[label].value_counts(normalize=True) * 100
            print(f'Churn Probabilities of {col}:')
            for category_value in df[col].unique():
                churn_rate = churn_probabilities[category_value][1] if 1 in churn_probabilities[category_value].index else 0
                print(f'A {category_value} customer has a churn probability of {round(churn_rate, 2)}%')
            print('\n')
    return all_plots