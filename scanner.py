from data.fetch_data import get_stock_data
from technical_engine.features import add_features
from technical_engine.label import create_labels
from technical_engine.model import train_model, predict_latest
from news_engine.sentiment import analyze_news
from fusion_engine.decision import final_decision

STOCKS = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS",
    "AXISBANK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "BPCL.NS","BHARTIARTL.NS","BRITANNIA.NS","CIPLA.NS","COALINDIA.NS",
    "DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS","HCLTECH.NS",
    "HDFCBANK.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS",
    "HINDUNILVR.NS","ICICIBANK.NS","ITC.NS","INDUSINDBK.NS","INFY.NS",
    "JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS",
    "NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS","RELIANCE.NS",
    "SBIN.NS","SUNPHARMA.NS","TCS.NS","TATAMOTORS.NS","TATASTEEL.NS",
    "TECHM.NS","TITAN.NS","ULTRACEMCO.NS","UPL.NS","WIPRO.NS"
]

print("\n🚀 StockSense AI — NIFTY 50 Scanner\n")

results = []

for symbol in STOCKS:
    try:
        df = get_stock_data(symbol)
        df = add_features(df)
        df = create_labels(df)

        model = train_model(df)
        tech_pred, tech_prob = predict_latest(df, model)

        company_name = symbol.replace(".NS", "")
        try:
            news_label, _, _ = analyze_news(company_name)
        except Exception:
            news_label = "NEUTRAL"

        decision = final_decision(tech_pred, tech_prob, news_label)

        results.append({
            "stock": symbol,
            "final": decision,
            "confidence": round(float(tech_prob), 2)
        })

        print(f"{symbol:15}  {decision:5}  Confidence: {round(float(tech_prob),2)}")

    except Exception as e:
        print(f"❌ Skipping {symbol}")

# 🔥 RANKING (FIXED)
buys = sorted(
    [r for r in results if r["final"] == "BUY"],
    key=lambda x: x["confidence"],
    reverse=True
)

sells = sorted(
    [r for r in results if r["final"] == "SELL"],
    key=lambda x: x["confidence"],
    reverse=True
)

print("\n========= 🔥 TOP BUY STOCKS =========\n")
top_buys = buys[:5] if len(buys) >= 5 else buys
for i, r in enumerate(top_buys, start=1):
    print(f"{i}. {r['stock']:15}  BUY   Confidence: {r['confidence']}")

if not top_buys:
    print("No strong BUY signals today.")

print("\n========= 🔴 TOP SELL STOCKS =========\n")
top_sells = sells[:5] if len(sells) >= 5 else sells
for i, r in enumerate(top_sells, start=1):
    print(f"{i}. {r['stock']:15}  SELL  Confidence: {r['confidence']}")

if not top_sells:
    print("No strong SELL signals today.")

print("\n✅ Scan Complete\n")
