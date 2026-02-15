import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def fetch_yahoo_news(symbol, max_articles=6):
    try:
        ticker = yf.Ticker(symbol)
        news_items = ticker.news

        if not news_items and "." in symbol:
            base = symbol.split(".")[0]
            ticker = yf.Ticker(base)
            news_items = ticker.news

        if not news_items:
            return []

        headlines = []
        for item in news_items[:max_articles]:
            title = item.get("title")
            if title:
                headlines.append(title)

        return list(set(headlines))
    except Exception:
        return []


def fetch_google_news(company_name, max_articles=6):
    try:
        query = company_name.replace(".NS", "")
        rss_url = f"https://news.google.com/rss/search?q={query}+stock+India"

        feed = feedparser.parse(rss_url)
        entries = feed.entries[:max_articles]

        headlines = []
        for entry in entries:
            title = entry.title
            if title:
                headlines.append(title)

        return list(set(headlines))
    except Exception:
        return []


def analyze_news(symbol):
    try:
        # 1️⃣ Try Yahoo first
        headlines = fetch_yahoo_news(symbol)

        # 2️⃣ Fallback to Google RSS
        if not headlines:
            headlines = fetch_google_news(symbol)

        if not headlines:
            return "NEUTRAL", 0.0, [], "NO_NEWS"

        scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        avg_score = sum(scores) / len(scores)

        if avg_score > 0.15:
            sentiment = "POSITIVE"
        elif avg_score < -0.15:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

        return sentiment, avg_score, headlines, "OK"

    except Exception:
        return "NEUTRAL", 0.0, [], "UNKNOWN_ERROR"
