import yfinance as yf
import pandas as pd
import os
import yaml
def download_btc_data():

    # 1. Load configuration from YAML file
    with open("config/model_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    ticker = config.get("ticker", "BTC-USD")
    start_date = config.get("start_date", "2010-01-01")


    # 2. Prepare the dictory to save data
    os.makedirs("data/raw", exist_ok=True)

    # 3. Download historical BTC-USD data
    print(f"Downloading data for {ticker} starting from {start_date}...")
    df = yf.download(ticker, start=start_date)
    
    # 4. Save the data to a CSV file
    output_path = "data/raw/btc_usd_historical_data.csv"
    df.to_csv(output_path)

    print(f"Data saved to {output_path}")
    return df
if __name__ == "__main__":
    download_btc_data()