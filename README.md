TELCO CUSTOMER CHURN PREDICTION

The aim of this  project is to find a solution for predicting customer churn in a telecommunications company. A few binary classification models are trained and the best performing model is selected. The business metric is to predict churn rate; the proportion of customers that left the telco company in the last month. And the business impact is to reduce churn rate because it is much less expensive to retain existing customers than to acquire new customers.

SHAP(SHapley Additive exPlanations) is used to explain to what extent each feature used in the model affects the model outcome.
Then Streamlit is used to present the output of the model through a user-friendly interface.

Users can interact with the churn app interface [here](https://telcocustomerchurn-ozbbmwnmbawcx4eakmdatp.streamlit.app/).

The dataset used is a sample dataset from IBM that can be accessed on Kaggle(https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It is made up of 7083 customers with 21 features. The features include customer account information, demographic information, and registered services. The label (Churn) contains information on whether a customer has churned

Models used in predicting customer churn
1. Logistic regression is widely used for binary classification tasks, such as predicting churn (yes/no).
It is a glassbox model i.e. it is simple and interpretable. It performs well when the relationship between features and the target is  linear.
Logistic Regression is trained as the baseline model.


2. Random Forests are an ensemble of decision trees. They combine multiple trees to improve prediction accuracy and reduce overfitting.
This model is robust against noise and outliers and handles high-dimensional data well.
Random forests are effective for improving model performance and handling complex relationships.


3. CatBoost model is designed for categorical data. This model performs well with datasets that have categorical features like “yes” or “no,” or different categories. It handles features well without requiring extensive preprocessing.

4. Explainable Boosting Machine(EBM) is a glassbox model that takes into account high accuracy while maintaining interpretability. This classifier employs cyclic gradient boosting and can perform automatic interaction detection.
