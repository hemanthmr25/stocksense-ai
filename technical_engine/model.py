from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

FEATURES = ['sma_20','sma_50','rsi','macd','atr','hammer']

def train_model(df):
    X = df[FEATURES]
    y = df['signal']

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        n_jobs=1,          # 🔥 IMPORTANT FIX
        random_state=42
    )

    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, _ in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

    return model

def predict_latest(df, model):
    latest = df[FEATURES].iloc[-1].values.reshape(1, -1)
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest).max()
    return pred, prob
