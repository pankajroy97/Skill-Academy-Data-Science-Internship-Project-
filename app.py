import pandas as pd 
import numpy as np
import pickle
from sklearn import *
import streamlit as st 
import warnings
warnings.filterwarnings('ignore')

# Load the data and model
df = pickle.load(open('dataframe.pkl', 'rb'))
model=pickle.load(open('random_forest.pkl','rb'))

def predict_sales( Country, Region, Segment, Category, SubCategory, Discount_Percentage, Actual_Discount, Quantity, Order_month):
    # Create a new dataframe with the user inputs
    inputs = pd.DataFrame([[Country, Region, Segment, Category, SubCategory, Discount_Percentage, Actual_Discount, Quantity, Order_month]],
                          columns=['Country', 'Region', 'Segment', 'Category', 'Sub-Category','Discount_Percentage', 'Actual_Discount', 'Quantity', 'Order_month'])
    
    
    # Check that the input data has the expected number of features
    if inputs.shape[1] != 9:
        raise ValueError(f'Expected 9 features but got {inputs.shape[1]}')

    # Make the prediction
    sales = model.predict(inputs)

    return sales[0]


# Create the user interface
st.title('Sales prediction specific category')
st.markdown('Use this app to predict sales for a specific category based on various input parameters.')
Country = st.selectbox('Country', df['Country'].unique())
Region = st.selectbox('Region', df['Region'].unique())
Segment = st.selectbox('Segment', df['Segment'].unique())
Category = st.selectbox('Category', df['Category'].unique())
subcategories = df.loc[df['Category'] == Category, 'Sub-Category'].unique()
SubCategory = st.selectbox('Sub-Category', subcategories)
Discount_Percentage = float(st.selectbox('Discount Percentage', df['Discount_Percentage'].unique()))
Actual_Discount = st.number_input('Actual Discount', min_value=0)
Quantity = st.number_input('Quantity', min_value=1,max_value=9)
Order_month = st.selectbox('Order month', df['Order_month'].unique())


# Add a button to trigger the prediction
if st.button('Predict Price'):
    Sales = predict_sales(Country, Region, Segment, Category, SubCategory, Discount_Percentage, Actual_Discount, Quantity, Order_month)
    st.success(f'The predicted sales for {Category} in {Country} is {Sales:,.0f} Indian Rupees.')
