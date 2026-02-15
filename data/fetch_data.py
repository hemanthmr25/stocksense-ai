import yfinance as yf

def get_stock_data(symbol, period="5y"):
    """
    Fetch stock data dynamically from Yahoo Finance
    """
    df = yf.download(symbol, period=period, progress=False)
    df.dropna(inplace=True)
    return df
