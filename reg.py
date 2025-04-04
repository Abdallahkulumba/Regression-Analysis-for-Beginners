import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import os
import webbrowser


# Title of the app
st.title(" Regression Analysis ")

# Sidebar: About the Author Section
st.sidebar.write("## About the Author")
st.sidebar.write("""
**Author:** Sserujja Abdallah Kulumba  
**Affiliation:** Islamic University of Technology  
**Email:** abdallahkulumba@iut-dhaka.edu  
**GitHub:** [github.com/Abdallahkulumba](https://github.com/Abdallahkulumba)  
**LinkedIn:** [linkedin.com/in/Abdallahkulumba](https://www.linkedin.com/in/abdallah-kulumba-sserujja/)  
**Facebook:** [facebook.com/Abdallahkulumba](https://www.facebook.com/abdallah.ed.ak)  
""")

# Sidebar for selecting regression type
st.sidebar.header("Regression Options")
option = st.sidebar.selectbox("Select Regression Type", 
                               ("Simple Linear Regression", 
                                "Multiple Linear Regression", 
                                "Polynomial Regression"))

# Sidebar for dataset upload or selection
st.sidebar.header("Dataset Options")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
available_datasets = ["economic_index.csv", "height-weight.csv"]
selected_dataset = None

if not uploaded_file:
    selected_dataset = st.sidebar.selectbox("Or select an available dataset", available_datasets)

# Load the dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.write(df.head())
elif selected_dataset:
    dataset_path = os.path.join(os.getcwd(), selected_dataset)
    df = pd.read_csv(dataset_path)
    st.write(f"### Selected Dataset: {selected_dataset}")
    st.write(df.head())
else:
    st.write("### Please upload a dataset or select one from the available options to proceed.")

if 'df' in locals():
    if option == "Simple Linear Regression":
        st.header("Simple Linear Regression")
        st.write("""
        Simple Linear Regression is a statistical method used to model the relationship between two variables: 
        one independent variable (X) and one dependent variable (y). It assumes a linear relationship between 
        the variables and fits a straight line (y = mx + c) to the data points.

        ### Key Features:
        - **Visualization**: Scatter plots are used to visualize the relationship between X and y.
        - **Assumptions**: Assumes a linear relationship, no multicollinearity, and homoscedasticity.
        - **Use CSSSSase**: Predicting a dependent variable based on one independent variable, e.g., predicting house prices based on size.
        """)

        # Select columns for X and y
        X_col = st.selectbox("Select the independent variable (X)", df.columns)
        y_col = st.selectbox("Select the dependent variable (y)", df.columns)
        
        X = df[[X_col]]
        y = df[y_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")
        
        # Plotting
        plt.scatter(X_test, y_test, color='blue', label='Actual')
        plt.plot(X_test, y_pred, color='red', label='Predicted')
        plt.xlabel(X_col)
        plt.ylabel(y_col)
        plt.title("Simple Linear Regression")
        plt.legend()
        st.pyplot(plt)

    elif option == "Multiple Linear Regression":
        st.header("Multiple Linear Regression")
        st.write("""
        Multiple Linear Regression is an extension of simple linear regression that models the relationship 
        between one dependent variable (y) and multiple independent variables (X). It fits a linear equation 
        (y = b0 + b1X1 + b2X2 + ... + bnXn) to the data.

        ### Key Features:
        - **Data Preprocessing**: Requires handling missing values, scaling, and encoding categorical variables.
        - **Assumptions**: Assumes linearity, no multicollinearity, and homoscedasticity.
        - **Use Case**: Predicting a dependent variable based on multiple factors, e.g., predicting salary based on education, experience, and age.
        """)

        # Select columns for X and y
        X_cols = st.multiselect("Select the independent variables (X)", df.columns)
        y_col = st.selectbox("Select the dependent variable (y)", df.columns)
        
        X = df[X_cols]
        y = df[y_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")

    elif option == "Polynomial Regression":
        st.header("Polynomial Regression")
        st.write("""
        Polynomial Regression is a type of regression analysis used to model non-linear relationships 
        between the independent variable (X) and the dependent variable (y). It transforms the features 
        into polynomial features and fits a linear model to the transformed features.

        ### Key Features:
        - **Feature Transformation**: Converts the independent variable into polynomial features (e.g., X, X^2, X^3).
        - **Flexibility**: Can model complex, non-linear relationships.
        - **Use Case**: Predicting outcomes where the relationship between variables is non-linear, e.g., predicting growth rates or stock prices.

        ### Note:
        Be cautious about overfitting when using higher-degree polynomials.
        """)

        # Select columns for X and y
        X_col = st.selectbox("Select the independent variable (X)", df.columns)
        y_col = st.selectbox("Select the dependent variable (y)", df.columns)
        
        X = df[[X_col]]
        y = df[y_col]
        
        # Slider for polynomial degree
        degree = st.slider("Select the degree of the polynomial", min_value=2, max_value=5, value=2)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Polynomial transformation
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        
        # Model training
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Predictions
        y_pred = model.predict(poly.transform(X_test))
        
        # Performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")
        
        # Plotting
        plt.scatter(X, y, color='green', label='Data')
        plt.scatter(X_test, y_pred, color='red', label='Predicted')
        plt.xlabel(X_col)
        plt.ylabel(y_col)
        plt.title(f"Polynomial Regression (Degree {degree})")
        plt.legend()
        st.pyplot(plt)

