# MTA Ridership Analysis
## Introduction
For this project, we used the MTA daily ridership from 2020-2025. Using ARIMA or Prophet model, we were be able to predict the ridership on specific days, and also analyze the recovery rate of each transportation mode after COVID-19 lockdown. Then, we predicted the maximum recovery rate over the next 10 years.

## The Dataset
Source: https://catalog.data.gov/dataset/mta-daily-ridership-data-beginning-2020

Timeframe: 2020 - 2025

Columns Used:
- Date
- Subways: Total Estimated Ridership
- Buses: Total Estimated Ridership
- LIRR: Total Estimated Ridership
- Metro-North: Total Estimated Ridership
- Staten Island Railway: Total Estimated Ridership

## MTA Daily Ridership Data: 2020 - 2025
![MTA Ridership Overview](visualizations/all_data.png)

As we can see in the graph, there is a huge drop in ridership around the beginning of 2020. If we follow the trend of each transportation mode, we notice that the ridership gradually increases. The growth are noticable for all mode except Staten Island Railway due to higher volume of riderships of the other tranportations. Nevertheless, we included all modes in our analysis.


## MTA Subway Ridership
![MTA Subway Ridership](visualizations/subway_ridership.png)

The MTA Subway has the largest ridership among all the transporation mode in NYC. Given the large data, we applied both ARIMA and Prophet models. ARIMA served as our initial learning model to forecasting while Prophet offered tools for visualizing trends and seasonality.

### ARIMA
We started with ARIMA because it's a classic baseline model for time series forecasting. This model is design to work for single variable time series data and capture trend and seasonality with differencing.

ARIMA has 3 variables:
1) AR (Auto-Regressive): uses past values as inputs in regression
2) I (Integrated): the data is stationary
3) MA (Moving Average): uses past errors as inputs in regression

Before fitting the model, we prepared subway's data by splitting into training and testing date sets. And then,
we run an ADF test to determine if the data is stationary or not.
```python
from statsmodels.tsa.stattools import adfuller

Run ADF test
result = adfuller(train['Subway'])

print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

The result was:
- ADF Statistic: -2.014709719230118
- p-value: 0.28016087630726183

The p-value is larger than 0.05. This meant the data was not currently stationary. Therefore, we applied differencing again, and we got a result of p-value less than 0.05. Since we used differencing twice, our I (Integrated) is equal to 2. This value was used to make the model.
```python
#Make the Model
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train, order=(1, 2, 1))  # (p, d, q)
model_fit = model.fit()

print(model_fit.summary())
```
We used the parameters ARIMA(1,2,1) as a baseline model to understand how well a simple ARIMA setup could forecast ridership before tuning or switching models. And, then we forecasted future dates of the training data and compared the prediction to actual rideship in the test data.

![MTA Subway ARIMA 121 Prediction](visualizations/arima121.png)

We see that the prediction line is linear despite the up and down peaks cause by weekend dips in ridership in the actual data. This suggest that ARIMA(1,2,1) model struggles to fully capture the weekly seasonality. We calculated the MAPE (Mean Absolute Percentage Error) and it resulted to 20.80%. As a general rule of thumb, the lower the MAPE is, the more accurate the model. Values under 10% considered highly accurate, 10–20% reasonable, and anything above 20% should be improve.

We decided to try another parameter, ARIMA(7,2,7), to see if it could better capture weekly trends and seasonality, especially the consistent ridership dips during weekends. By increasing the lag terms, we aimed to incorporate the repeating 7-day cycle into the model.

![MTA Subway ARIMA 727 Prediction](visualizations/arima727.png)

We see that this model capture the peaks and the dips in ridership unlike the first ARIMA model. The calculated MAPE is 16.33%. This is an improvement compare to the first ARIMA model. However, as we look closer to the graph, we observe that the predicted peaks are remaining relatively flat while the dips are growing deeper each weekend. This suggest that while the model picks up on weekly seasonality, it struggles adapt how much the ridership rises and falls.