import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import yaml
import os

# Set professional page config
st.set_page_config(
    page_title="BTC Cycle Forecaster",
    page_icon="📈",
    layout="wide"
)

def load_config():
    """Load configuration from YAML file."""
    with open("config/model_config.yaml", "r") as f:
        return yaml.safe_load(f)

@st.cache_data
def load_data(file_path):
    """Load and cache historical data."""
    df = pd.read_csv(file_path, header=[0, 1], index_col=0)
    df = df.reset_index()
    # Handle multi-index columns from yfinance
    df.columns = ['ds', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    df['y'] = df['Close']
    return df

def run_forecast(df, horizon, cps):
    """Train Prophet model and generate forecast."""
    model = Prophet(
        changepoint_prior_scale=cps,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    return model, forecast

def main():
    # --- UI Header ---
    st.title("Bitcoin Halving Cycle Forecast Dashboard")
    st.markdown("""
    This dashboard provides an AI-driven forecast for Bitcoin prices based on historical 
    halving cycles and trend analysis using the **Meta Prophet** model.
    """)

    # --- Sidebar Configuration ---
    st.sidebar.header("Model Configuration")
    config = load_config()
    
    ticker = st.sidebar.text_input("Ticker Symbol", value=config['ticker'])
    horizon = st.sidebar.slider("Forecast Horizon (Days)", 365, 1460, config['forecast_horizon_days'])
    cps = st.sidebar.select_slider(
        "Model Flexibility (Changepoint Prior Scale)", 
        options=[0.001, 0.01, 0.05, 0.1, 0.5], 
        value=config['model_params']['changepoint_prior_scale']
    )

    # --- Data Loading ---
    data_path = "data/raw/btc_usd_historical_data.csv"
    if os.path.exists(data_path):
        df = load_data(data_path)
        
        # --- Model Execution ---
        with st.spinner('Calculating AI Forecast...'):
            model, forecast = run_forecast(df, horizon, cps)

        # --- Metrics Display ---
        col1, col2, col3 = st.columns(3)
        latest_price = df['y'].iloc[-1]
        predicted_price = forecast['yhat'].iloc[-1]
        change = ((predicted_price - latest_price) / latest_price) * 100
        
        col1.metric("Current Price", f"${latest_price:,.2f}")
        col2.metric("Forecasted Price (End of Horizon)", f"${predicted_price:,.2f}", f"{change:.2f}%")
        col3.metric("Model Engine", "Prophet ML")

        # --- Interactive Visualization ---
        st.subheader("Interactive Forecast Chart")
        fig = go.Figure()

        # Actual data
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="Historical Price", line=dict(color="#f2a900")))
        
        # Forecast data
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted Trend", line=dict(color="#00ff00")))
        
        # Uncertainty interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="Uncertainty Range"
        ))

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Year",
            yaxis_title="Price (USD)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Data Summary ---
        with st.expander("View Raw Forecast Data"):
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

    else:
        st.error(f"Data file not found at {data_path}. Please run download_data.py first.")

if __name__ == "__main__":
    main()