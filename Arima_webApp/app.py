import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
import datetime
import plotly.express as px
import plotly.graph_objs as go


def make_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'].astype(dtype=str), 
                        y=df['food_sales'],
                        marker_color='red', text="sales" ,name='Food Sales'))
    fig.add_trace(go.Scatter(x=df['date'].astype(dtype=str), 
                        y=df['hobbies_sales'],
                        marker_color='green', text="hobbies", name='Hobbies Sales'))
    fig.add_trace(go.Scatter(x=df['date'].astype(dtype=str), 
                        y=df['household_sales'],
                        marker_color='blue', text="household", name='Household Sales'))                    
    fig.update_layout({"title": 'Sales Prediction - n steps ahead forecast with ARIMA',
                   "xaxis": {"title":"date"},
                   "yaxis": {"title":"sales"},
                   "showlegend": True})
    return fig

'''
# Retail Sales Analysis 
To make accurate predictions for product sales for next 10 days
in advance (the data set includes daily unit sales per product).
Download the data set - [Retail Dataset](https://github.com/bhanusumanth/ai-sales-predict-dataset)

'''
'''
Sales Prediction for All Categories (Food, Hobbies and Household) using ARIMA (AutoRegressive Integrated Moving Average)
'''
s_date = '2011-01-29'
e_date = '2016-06-19'
# create date day number map
date_day_data = {'day_num': range(1,1970), 'date':pd.date_range(s_date,e_date, tz=None)}
date_df = pd.DataFrame(data=date_day_data)
date_df['date_str']=date_df['date'].astype(str)
date_day_map = dict(zip(list(date_df['date_str']), list(date_df['day_num'])))
day_date_map = dict(zip( list(date_df['day_num']), list(date_df['date_str'])))


start_date = st.date_input("Pick a date for prediction", value=datetime.date(2011, 7, 6),min_value=datetime.date(2011, 1,29), max_value=datetime.date(2016, 4,1))
forecast_number = st.number_input('Number of days to forecast (n-steps ahead)', value=10, max_value = 31)
food_model = joblib.load('food_arima_compressed.pkl')
hobbies_model = joblib.load('hobbies_arima_compressed.pkl')
household_model = joblib.load('household_arima_compressed.pkl')
# convert to proper date string
start_date = start_date.strftime("%Y-%m-%d")
start_day = date_day_map[start_date]
if st.button('Predict'):
    food_predictions = food_model.predict(start = start_day, end = start_day + forecast_number)
    hobbies_predictions = hobbies_model.predict(start = start_day, end = start_day + forecast_number)
    household_predictions = household_model.predict(start = start_day, end = start_day + forecast_number)
    display_dates = pd.date_range(s_date,e_date)
    food_results = pd.DataFrame({'date': pd.date_range(day_date_map[start_day],day_date_map[start_day+forecast_number], tz=None), 'food_sales': food_predictions,
    'hobbies_sales': hobbies_predictions, 'household_sales':household_predictions})
    food_results['date']=food_results['date'].astype(str)
    st.write(food_results)
    st.plotly_chart(make_plot(food_results), use_container_width=True)






