from data.fetch_data import get_stock_data
from technical_engine.features import add_features
from technical_engine.label import create_labels
from technical_engine.model import train_model, predict_latest
from news_engine.sentiment import analyze_news
from fusion_engine.decision import final_decision


def format_score(s):
    return f"{s:+.2f}"

def safe_scalar(x):
    try:
        if hasattr(x, "iloc"):
            return float(x.iloc[0])
        elif hasattr(x, "__len__") and not isinstance(x, str):
            return float(x[0])
        else:
            return float(x)
    except Exception:
        return 0.0


print("\n🚀 StockSense AI — Interactive Stock Analyzer\n")

symbol = input("Enter Stock Symbol (e.g., TCS.NS, INFY.NS, RELIANCE.NS): ").strip().upper()

try:
    df = get_stock_data(symbol)
except Exception as e:
    print("❌ Error fetching data for", symbol)
    print("Reason:", e)
    raise SystemExit(1)

df = add_features(df)
df = create_labels(df)

model = train_model(df)

tech_pred, tech_prob = predict_latest(df, model)
signal_map = {1: "BUY", 0: "HOLD", -1: "SELL"}

# 🔥 Yahoo Finance News (NO API KEY)
news_label, news_avg_score, headlines, api_status = analyze_news(symbol)

decision = final_decision(tech_pred, tech_prob, news_label)

print("\n==============================")
print("STOCK:", symbol)
print("Technical Signal:", signal_map[tech_pred])
print("Technical Confidence:", round(float(tech_prob), 2))
print("News Signal:", news_label, f"(avg score {format_score(news_avg_score)})")
print("News Status:", api_status)
print("FINAL DECISION:", decision)
print("==============================\n")

# 🔥 EXPLAINABLE REASONS

reasons = []

latest = df.iloc[-1]

rsi_val = safe_scalar(latest["rsi"])
close_val = safe_scalar(latest["Close"])
sma50_val = safe_scalar(latest["sma_50"])
hammer_val = int(safe_scalar(latest["hammer"]))

if rsi_val < 30:
    reasons.append("RSI indicates oversold conditions")
elif rsi_val > 70:
    reasons.append("RSI indicates overbought conditions")

if close_val > sma50_val:
    reasons.append("Price is above 50-day moving average (bullish)")
else:
    reasons.append("Price is below 50-day moving average (bearish)")

if hammer_val == 1:
    reasons.append("Bullish hammer candlestick pattern detected")

if headlines:
    for h in headlines[:3]:
        reasons.append(f"News: {h}")
else:
    reasons.append("No recent company-specific news found (news treated as neutral)")

print("🧠 REASONS FOR DECISION:")
for r in reasons:
    print("-", r)

if headlines:
    print("\n📰 Top News Headlines:")
    for h in headlines[:6]:
        print("-", h)
else:
    print("\n📰 No headlines available.")

print("\n✅ Analysis Complete\n")
