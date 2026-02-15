def final_decision(tech_pred, tech_prob, news_signal):
    if tech_pred == 1 and news_signal == "POSITIVE":
        return "BUY"
    elif tech_pred == -1 and news_signal == "NEGATIVE":
        return "SELL"
    else:
        return "HOLD"
