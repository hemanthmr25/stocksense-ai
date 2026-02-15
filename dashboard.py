import streamlit as st
import matplotlib.pyplot as plt
import math

from data.fetch_data import get_stock_data
from technical_engine.features import add_features
from technical_engine.label import create_labels
from technical_engine.model import train_model, predict_latest
from news_engine.sentiment import analyze_news
from fusion_engine.decision import final_decision


# ---------------- CONFIG ----------------
st.set_page_config(page_title="StockSense AI", layout="centered")

st.title("📈 StockSense AI")
st.markdown(
    "ML-powered Buy / Sell / Hold prediction using **Technical Indicators + Live News Sentiment**"
)

# ---------------- HELPERS ----------------
def scalar_value(x):
    try:
        if hasattr(x, "values"):
            val = float(x.values[0])
        else:
            val = float(x)
        if math.isnan(val):
            return None
        return val
    except Exception:
        return None

# ---------------- TIME RANGE ----------------
time_range = st.selectbox(
    "Select Time Range",
    ["1 Day", "1 Week", "1 Month", "1 Year", "5 Years"],
    index=3
)

period_map = {
    "1 Day": "1d",
    "1 Week": "5d",
    "1 Month": "1mo",
    "1 Year": "1y",
    "5 Years": "5y"
}

selected_period = period_map[time_range]

# ---------------- NIFTY 50 ----------------
NIFTY_50 = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS",
    "BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS","BHARTIARTL.NS","BPCL.NS",
    "BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS",
    "EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","INDUSINDBK.NS",
    "INFY.NS","ITC.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS",
    "M&M.NS","MARUTI.NS","NESTLEIND.NS","NTPC.NS","ONGC.NS",
    "POWERGRID.NS","RELIANCE.NS","SBIN.NS","SUNPHARMA.NS","TATACONSUM.NS",
    "TATAMOTORS.NS","TATASTEEL.NS","TECHM.NS","TITAN.NS","ULTRACEMCO.NS",
    "UPL.NS","WIPRO.NS","LTIM.NS","ADANIGREEN.NS"
]

stock = st.selectbox("Select Stock", sorted(NIFTY_50))

# ---------------- MAIN ACTION ----------------
if st.button("🔍 Analyze Stock"):

    with st.spinner("Analyzing stock... Please wait"):

        try:
            # --------- DATA FETCH ----------
            # ML needs stable history
            df_ml = get_stock_data(stock, period="1y")
            df_ml = add_features(df_ml)
            df_ml = create_labels(df_ml)

            if df_ml.empty or len(df_ml) < 60:
                st.warning("Not enough historical data for ML analysis.")
                st.stop()

            # Graph can use selected period
            df_plot = get_stock_data(stock, period=selected_period)
            

            # --------- ML MODEL ----------
            model = train_model(df_ml)
            tech_pred, tech_prob = predict_latest(df_ml, model)

            # --------- NEWS ----------
            news_signal, news_score, headlines, api_status = analyze_news(stock)

            # --------- FINAL DECISION ----------
            decision = final_decision(tech_pred, tech_prob, news_signal)
            signal_map = {1: "BUY", 0: "HOLD", -1: "SELL"}

            st.success(f"📌 Final Decision: **{decision}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Technical Signal", signal_map.get(tech_pred, "HOLD"))
            col2.metric("Confidence", round(float(tech_prob), 2))
            col3.metric("News Sentiment", news_signal)

            # --------- LIVE SNAPSHOT ----------
            st.markdown("### 📊 Live Market Snapshot")

            latest = df_plot[["Open","High","Low","Close","Volume"]].dropna().iloc[-1]

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric("Open", round(scalar_value(latest["Open"]), 2))
            c2.metric("High", round(scalar_value(latest["High"]), 2))
            c3.metric("Low", round(scalar_value(latest["Low"]), 2))
            c4.metric("Close", round(scalar_value(latest["Close"]), 2))
            c5.metric("Volume", f"{int(scalar_value(latest['Volume'])):,}")

            st.caption(f"Last updated: {latest.name}")

            # --------- PRICE GRAPH ----------
            st.markdown(f"### 📈 Stock Price ({time_range})")

            fig, ax = plt.subplots()
            ax.plot(df_plot.index, df_plot["Close"], label="Close Price")

            if "sma_20" in df_ml.columns:
                ax.plot(df_ml.index, df_ml["sma_20"], label="SMA 20")

            if "sma_50" in df_ml.columns:
                ax.plot(df_ml.index, df_ml["sma_50"], label="SMA 50")

            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()

            st.pyplot(fig)

            # --------- INTERPRETATION ----------
            st.markdown("### 🧠 Interpretation")

            latest_ml = df_ml.iloc[-1]

            rsi_val = scalar_value(latest_ml.get("rsi", 0))
            close_val = scalar_value(latest_ml.get("Close", 0))
            sma50_val = scalar_value(latest_ml.get("sma_50", 0))
            hammer_val = int(scalar_value(latest_ml.get("hammer", 0)) or 0)

            reasons = []

            if rsi_val is not None:
                if rsi_val < 30:
                    reasons.append("RSI indicates oversold conditions")
                elif rsi_val > 70:
                    reasons.append("RSI indicates overbought conditions")

            if close_val > sma50_val:
                reasons.append("Price above 50-day moving average (bullish)")
            else:
                reasons.append("Price below 50-day moving average (bearish)")

            if hammer_val == 1:
                reasons.append("Bullish hammer candlestick pattern detected")

            if news_signal == "POSITIVE":
                reasons.append("Positive recent news sentiment")
            elif news_signal == "NEGATIVE":
                reasons.append("Negative recent news sentiment")
            else:
                reasons.append("Neutral news sentiment")

            for r in reasons:
                st.write("•", r)

            # --------- NEWS HEADLINES ----------
            st.markdown("### 📰 Latest News Headlines")

            if api_status != "OK":
                st.warning(f"News unavailable due to: {api_status}")

            if headlines:
                for h in headlines[:5]:
                    st.write("-", h)
            else:
                st.info("No recent company-specific news found.")
                            # ==========================================
            # 💼 PORTFOLIO TRACKER (FIXED)
            # ==========================================
            st.markdown("---")
            st.header("💼 Portfolio Tracker")

            col1, col2 = st.columns(2)

            with col1:
                buy_price = st.number_input("Buy Price ₹", min_value=0.0)

            with col2:
                qty = st.number_input("Quantity", min_value=1, step=1)

            if st.button("Calculate Profit/Loss"):
                live_price = float(df["Close"].iloc[-1])

                invested = buy_price * qty
                current_val = live_price * qty
                profit = current_val - invested
                percent = (profit / invested) * 100 if invested > 0 else 0

                st.subheader("📊 Portfolio Result")

                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"₹{round(live_price,2)}")
                c2.metric("Invested", f"₹{round(invested,2)}")
                c3.metric("Current Value", f"₹{round(current_val,2)}")

                if profit >= 0:
                    st.success(f"Profit: ₹{round(profit,2)} ({round(percent,2)}%)")
                else:
                    st.error(f"Loss: ₹{round(profit,2)} ({round(percent,2)}%)")

           
      
        except Exception as e:
                st.error("⚠️ Something went wrong while analyzing the stock.")
                st.write("Debug info:", e)

