import pandas as pd
from prophet import Prophet
import yaml
import mlflow
import mlflow.prophet
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_model():

    # 1. Load configuration from YAML file
    with open("config/model_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # 2. Load the historical BTC-USD data
    data_path = "data/raw/btc_usd_historical_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please run the data download script first.")
    
    # 3. Prepare the data for Prophet
    df = pd.read_csv(data_path, header=[0, 1], index_col=0)

    # 4. Reset index and rename columns
    df = df.reset_index()
    df.columns =['ds', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    df['y'] = df['Close']
    df = df[['ds', 'y']]

    # 5. Mlflow setup
    mlflow.set_experiment("BTC_Price_Forecasting")
    with mlflow.start_run():
        print("Training the Prophet model...")
        # 5.1. Initialize and train the Prophet model
        cps = config['model_params']['changepoint_prior_scale']
        sm = config['model_params']['seasonality_mode']

        # 5.2. Log model parameters
        mlflow.log_param("changepoint_prior_scale", cps)
        mlflow.log_param("seasonality_mode", sm)

        # 5.3. Create and fit the model
        model = Prophet(changepoint_prior_scale=cps, seasonality_mode=sm, daily_seasonality=True)
        model.fit(df)

        # 6. Predict future values
        horizon =config.get('forecast_horizon_days', 1460)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

        historical_forecast = forecast.iloc[:len(df)]
        mae = mean_absolute_error(df['y'], historical_forecast['yhat'])
        rmse = np.sqrt(mean_squared_error(df['y'], historical_forecast['yhat']))
        
        print(f"Metryki modelu: MAE={mae:.2f}, RMSE={rmse:.2f}")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # B. Generate and save forecast plot
        fig = model.plot(forecast)
        fig.suptitle(f"BTC Price Forecast - {horizon} days")
        plot_path = "models/btc_forecast_plot.png"
        os.makedirs("models", exist_ok=True)
        plt.savefig(plot_path)
        
        # C. Log the plot as an artifact
        mlflow.log_artifact(plot_path)

        # C. Save forecast to CSV and log as artifact
        forecast_csv = "data/processed/btc_forecast.csv"
        os.makedirs("data/processed", exist_ok=True)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_csv, index=False)
        mlflow.log_artifact(forecast_csv)

        # 7. Save model to MLflow
        mlflow.prophet.log_model(model, artifact_path="model")
        print("Model training completed and logged to MLflow.")
if __name__ == "__main__":
    train_model()