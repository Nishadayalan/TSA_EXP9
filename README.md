# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 28-10-25

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
data = pd.read_csv("/content/plane_crash.csv")
print(data.columns)


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Drop rows with missing or invalid dates
data = data.dropna(subset=['Date'])

# Sort by Date
data = data.sort_values(by='Date')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Define ARIMA model function
def arima_model(data, target_variable, order):
    # Split into train and test sets (80/20)
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Fit ARIMA model
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    # Forecast future values
    forecast = fitted_model.forecast(steps=len(test_data))

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data', color='red')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

# Run ARIMA model
arima_model(data, 'Fatalities', order=(5,1,0))

```

### OUTPUT:



<img width="1026" height="620" alt="image" src="https://github.com/user-attachments/assets/3b39d019-6b26-4fa1-8013-b040d4ef82c2" />




### RESULT:
Thus the program run successfully based on the ARIMA model using python.
