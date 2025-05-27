# MTA Ridership Analysis
## Introduction
For this project, we used the MTA daily ridership from 2020-2025 (https://catalog.data.gov/dataset/mta-daily-ridership-data-beginning-2020). Using ARIMA or Prophet model, we were be able to predict the ridership on specific days, and also analyze the recovery rate of each transportation mode after COVID-19 lockdown. Then, we predicted the maximum recovery rate over the next 10 years.

## MTA Daily Ridership Data: 2020 - 2025
![MTA Ridership Overview](visualizations/all_data.png)

As we can see in the graph, there is a huge drop in ridership around the beginning of 2020. If we follow the trend of each transportation mode, we can notice that the ridership gradually increases. The growth are noticable for all mode except Staten Island Railway due to higher volume of riderships of the other tranportations. Nevertheless, we included all modes in our analysis.


## MTA Subway Ridership
![MTA Subway Ridership](visualizations/subway_ridership.png)

The MTA Subway has the largest ridership among all the transporation mode in NYC. Given the large data, we applied both ARIMA and Prophet models. ARIMA served as our initial learning model to forecasting while Prophet offered tools for visualizing trends and seasonality.

## Modeling Approach

### ARIMA
We used ARIMA because ...

split data/ train and tes
