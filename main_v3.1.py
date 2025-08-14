import streamlit as st
import yfinance as yf
import time
import datetime
import csv
import logging
from statistics import stdev
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json
import os

# File paths
CSV_FILE = 'portfolio_export.csv'
ERROR_LOG = 'portfolio_errors.log'
PORTFOLIO_JSON = 'portfolio.json'
WATCHLIST_JSON = 'watchlist.json'
SETTINGS_JSON = 'settings.json'
RECOMMENDATIONS_JSON = 'recommendations.json'

# Finnhub API key
FINNHUB_API_KEY = 'd25rhs9r01qhge4e2bjgd25rhs9r01qhge4e2bk0'

# Set up error logging
logging.basicConfig(filename=ERROR_LOG, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Starting script execution")

# Default settings
DEFAULT_SETTINGS = {
    'refresh_interval': 60,
    'alert_threshold': 5.0,
    'portfolio_visible_columns': ['Ticker', 'Shares', 'Avg Buy Price', 'Current Price', 'Value', 'P/L'],
    'watchlist_visible_columns': ['Ticker', 'Current Price', 'Daily Change'],
    'auto_refresh': True,
}

def init_json_files():
    """Initialize JSON files if they don't exist."""
    try:
        if not os.path.exists(PORTFOLIO_JSON):
            with open(PORTFOLIO_JSON, 'w') as f:
                json.dump({}, f)
        if not os.path.exists(WATCHLIST_JSON):
            with open(WATCHLIST_JSON, 'w') as f:
                json.dump([], f)
        if not os.path.exists(SETTINGS_JSON):
            with open(SETTINGS_JSON, 'w') as f:
                json.dump(DEFAULT_SETTINGS, f)
        if not os.path.exists(RECOMMENDATIONS_JSON):
            with open(RECOMMENDATIONS_JSON, 'w') as f:
                json.dump([], f)
    except Exception as e:
        logging.error(f"Error initializing JSON files: {e}")

def load_settings():
    """Load settings from JSON."""
    try:
        init_json_files()
        with open(SETTINGS_JSON, 'r') as f:
            settings = json.load(f)
        return {**DEFAULT_SETTINGS, **settings}
    except Exception as e:
        logging.error(f"Error loading settings: {e}")
        return DEFAULT_SETTINGS

def save_settings(settings):
    """Save settings to JSON."""
    try:
        init_json_files()
        with open(SETTINGS_JSON, 'w') as f:
            json.dump(settings, f)
    except Exception as e:
        logging.error(f"Error saving settings: {e}")

def load_portfolio():
    """Load the portfolio from JSON."""
    try:
        init_json_files()
        with open(PORTFOLIO_JSON, 'r') as f:
            portfolio = json.load(f)
        # Ensure buy_date exists for all purchases
        for ticker, purchases in portfolio.items():
            for p in purchases:
                p['buy_date'] = p.get('buy_date', None)
        return portfolio
    except Exception as e:
        logging.error(f"Error loading portfolio: {e}")
        return {}

def save_portfolio(portfolio):
    """Save the portfolio to JSON."""
    try:
        init_json_files()
        with open(PORTFOLIO_JSON, 'w') as f:
            json.dump(portfolio, f)
    except Exception as e:
        logging.error(f"Error saving portfolio: {e}")

def load_watchlist():
    """Load the watchlist from JSON."""
    try:
        init_json_files()
        with open(WATCHLIST_JSON, 'r') as f:
            watchlist = json.load(f)
        return watchlist
    except Exception as e:
        logging.error(f"Error loading watchlist: {e}")
        return []

def save_watchlist(watchlist):
    """Save the watchlist to JSON."""
    try:
        init_json_files()
        with open(WATCHLIST_JSON, 'w') as f:
            json.dump(watchlist, f)
    except Exception as e:
        logging.error(f"Error saving watchlist: {e}")

def save_recommendations(recommendations):
    """Save recommendations to JSON."""
    try:
        init_json_files()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rec_list = []
        for ticker, recs in recommendations.items():
            for rec in recs:
                full_rec = f"{ticker}: {rec}"
                rec_list.append({"ticker": ticker, "recommendation": full_rec, "timestamp": timestamp})
        with open(RECOMMENDATIONS_JSON, 'w') as f:
            json.dump(rec_list, f)
    except Exception as e:
        logging.error(f"Error saving recommendations: {e}")

def fetch_finnhub_news(ticker, api_key, retries=3, backoff=3):
    """Fetch news from Finnhub API with VADER sentiment analysis."""
    try:
        logging.debug(f"Fetching Finnhub news for {ticker}")
        analyzer = SentimentIntensityAnalyzer()
        news_items = []
        for attempt in range(retries):
            try:
                from_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
                to_date = datetime.datetime.now().strftime('%Y-%m-%d')
                url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={api_key}"
                response = requests.get(url)
                response.raise_for_status()
                news = response.json()[:5]
                for item in news:
                    title = item.get('headline', '')
                    if not title:
                        continue
                    publisher = item.get('source', 'Unknown')
                    pub_date = item.get('datetime', 'Unknown date')
                    if pub_date != 'Unknown date':
                        try:
                            pub_date = datetime.datetime.fromtimestamp(int(pub_date)).strftime('%Y-%m-%d %H:%M')
                        except:
                            pub_date = 'Unknown date'
                    link = item.get('url', '')
                    sentiment_scores = analyzer.polarity_scores(title)
                    sentiment = "Positive" if sentiment_scores['compound'] > 0.05 else \
                                "Negative" if sentiment_scores['compound'] < -0.05 else "Neutral"
                    news_items.append({
                        'ticker': ticker,
                        'title': title,
                        'publisher': publisher,
                        'date': pub_date,
                        'sentiment': sentiment,
                        'compound': sentiment_scores['compound'],
                        'link': link
                    })
                return news_items if news_items else [{'ticker': ticker, 'title': f"No recent news for {ticker}", 'publisher': '', 'date': '', 'sentiment': '', 'compound': 0, 'link': ''}]
            except Exception as e:
                if attempt == retries - 1:
                    logging.error(f"Failed to fetch Finnhub news for {ticker} after {retries} attempts: {e}")
                    return [{'ticker': ticker, 'title': f"ERROR: Failed to fetch Finnhub news for {ticker}: {e}", 'publisher': '', 'date': '', 'sentiment': '', 'compound': 0, 'link': ''}]
                time.sleep(backoff * (2 ** attempt))
        return [{'ticker': ticker, 'title': f"No news available for {ticker}", 'publisher': '', 'date': '', 'sentiment': '', 'compound': 0, 'link': ''}]
    except Exception as e:
        logging.error(f"Error in Finnhub news fetch for {ticker}: {e}")
        return [{'ticker': ticker, 'title': f"ERROR: Unable to fetch Finnhub news for {ticker}: {e}", 'publisher': '', 'date': '', 'sentiment': '', 'compound': 0, 'link': ''}]

def fetch_yfinance_news(ticker, retries=3, backoff=3):
    """Fallback to yfinance news with VADER sentiment analysis."""
    try:
        logging.debug(f"Fetching yfinance news for {ticker}")
        stock = yf.Ticker(ticker)
        info = stock.info
        quote_type = info.get('quoteType', '').upper()
        if quote_type in ['MUTUALFUND', 'FUTURE']:
            logging.debug(f"Skipping news for {ticker} (type: {quote_type})")
            return [{'ticker': ticker, 'title': f"No news available for {ticker} ({quote_type.lower()})", 'publisher': '', 'date': '', 'sentiment': '', 'compound': 0, 'link': ''}]
        
        analyzer = SentimentIntensityAnalyzer()
        news_items = []
        for attempt in range(retries):
            try:
                news = stock.news[:5]
                for item in news:
                    title = item.get('title', '')
                    if not title or title.lower() == 'no title':
                        continue
                    publisher = item.get('publisher', 'Unknown')
                    pub_time = item.get('providerPublishTime', 0)
                    pub_date = (datetime.datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
                                if pub_time > 0 else 'Unknown date')
                    link = item.get('link', '')
                    sentiment_scores = analyzer.polarity_scores(title)
                    sentiment = "Positive" if sentiment_scores['compound'] > 0.05 else \
                                "Negative" if sentiment_scores['compound'] < -0.05 else "Neutral"
                    news_items.append({
                        'ticker': ticker,
                        'title': title,
                        'publisher': publisher,
                        'date': pub_date,
                        'sentiment': sentiment,
                        'compound': sentiment_scores['compound'],
                        'link': link
                    })
                return news_items if news_items else [{'ticker': ticker, 'title': f"No recent news for {ticker}", 'publisher': '', 'date': '', 'sentiment': '', 'compound': 0, 'link': ''}]
            except Exception as e:
                if attempt == retries - 1:
                    logging.error(f"Failed to fetch yfinance news for {ticker} after {retries} attempts: {e}")
                    return [{'ticker': ticker, 'title': f"ERROR: Failed to fetch yfinance news for {ticker}: {e}", 'publisher': '', 'date': '', 'sentiment': '', 'compound': 0, 'link': ''}]
                time.sleep(backoff * (2 ** attempt))
        return [{'ticker': ticker, 'title': f"No news available for {ticker}", 'publisher': '', 'date': '', 'sentiment': '', 'compound': 0, 'link': ''}]
    except Exception as e:
        logging.error(f"Error checking quote type for {ticker}: {e}")
        return [{'ticker': ticker, 'title': f"ERROR: Unable to fetch yfinance news for {ticker}: {e}", 'publisher': '', 'date': '', 'sentiment': '', 'compound': 0, 'link': ''}]

def fetch_news_with_retries(ticker, retries=3, backoff=3):
    """Fetch news, preferring Finnhub, falling back to yfinance."""
    if 'news_cache' not in st.session_state:
        st.session_state.news_cache = {}
    if ticker in st.session_state.news_cache and time.time() - st.session_state.news_cache[ticker]['timestamp'] < 3600:
        return st.session_state.news_cache[ticker]['news']
    news = fetch_finnhub_news(ticker, FINNHUB_API_KEY, retries, backoff)
    if news and not any("ERROR" in item['title'] or "No news available" in item['title'] for item in news):
        st.session_state.news_cache[ticker] = {'news': news, 'timestamp': time.time()}
        return news
    news = fetch_yfinance_news(ticker, retries, backoff)
    st.session_state.news_cache[ticker] = {'news': news, 'timestamp': time.time()}
    return news

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_recommendations(portfolio, watchlist):
    """Generate enhanced AI-driven recommendations."""
    try:
        logging.debug("Generating recommendations")
        recommendations = {}
        total_value = 0
        ticker_values = {}

        for ticker, purchases in portfolio.items():
            recommendations[ticker] = []
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                history = stock.history(period='1mo')
                if len(history) < 10:
                    recommendations[ticker].append("Insufficient data for analysis.")
                    continue
                
                current_price = history['Close'].iloc[-1]
                total_shares = sum(p['shares'] for p in purchases)
                ticker_value = total_shares * current_price
                total_value += ticker_value
                ticker_values[ticker] = ticker_value
                
                # Technical indicators
                prices = history['Close'].tolist()
                ma_10 = sum(prices[-10:]) / 10 if len(prices) >= 10 else None
                ma_20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else None
                if ma_10 and ma_20:
                    if ma_10 > ma_20 and current_price > ma_10:
                        recommendations[ticker].append(f"Bullish trend (10-day MA: ${ma_10:.2f} > 20-day MA: ${ma_20:.2f}), consider holding.")
                    elif ma_10 < ma_20 and current_price < ma_10:
                        recommendations[ticker].append(f"Bearish trend (10-day MA: ${ma_10:.2f} < 20-day MA: ${ma_20:.2f}), consider reviewing.")
                
                # RSI
                rsi = calculate_rsi(prices)
                if rsi and rsi > 70:
                    recommendations[ticker].append(f"Overbought (RSI: {rsi:.1f}), consider selling.")
                elif rsi and rsi < 30:
                    recommendations[ticker].append(f"Oversold (RSI: {rsi:.1f}), potential buying opportunity.")
                
                # Volatility
                if len(prices) >= 6:  # Need 5 returns for 5 days
                    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
                    volatility = stdev(returns[-5:]) * 100 if len(returns) >= 5 else 0
                    if volatility > 10:
                        recommendations[ticker].append(f"High volatility ({volatility:.1f}%), exercise caution.")
                    elif volatility < 5:
                        recommendations[ticker].append(f"Low volatility ({volatility:.1f}%), stable investment.")
                
                # Fundamental analysis
                pe_ratio = info.get('trailingPE', None)
                dividend_yield = info.get('dividendYield', None)
                if pe_ratio and pe_ratio > 30:
                    recommendations[ticker].append(f"High P/E ratio ({pe_ratio:.1f}), may be overvalued.")
                if dividend_yield and dividend_yield > 0.05:
                    recommendations[ticker].append(f"High dividend yield ({dividend_yield*100:.1f}%), attractive for income.")
                
                # Sector performance
                sector = info.get('sector', 'Unknown')
                if sector != 'Unknown':
                    recommendations[ticker].append(f"Sector ({sector}) performance should be monitored.")
                
                # MACD
                if len(prices) >= 26:
                    df_prices = pd.Series(prices)
                    ema12 = df_prices.ewm(span=12, adjust=False).mean()
                    ema26 = df_prices.ewm(span=26, adjust=False).mean()
                    macd_line = ema12 - ema26
                    signal_line = macd_line.ewm(span=9, adjust=False).mean()
                    macd = macd_line.iloc[-1]
                    signal = signal_line.iloc[-1]
                    if macd > signal:
                        recommendations[ticker].append(f"MACD bullish (MACD: {macd:.2f} > Signal: ${signal:.2f}), potential buy signal.")
                    elif macd < signal:
                        recommendations[ticker].append(f"MACD bearish (MACD: {macd:.2f} < Signal: ${signal:.2f}), potential sell signal.")
                
                # Bollinger Bands
                if len(prices) >= 20:
                    ma20 = sum(prices[-20:]) / 20
                    std = stdev(prices[-20:])
                    upper = ma20 + 2 * std
                    lower = ma20 - 2 * std
                    if current_price > upper:
                        recommendations[ticker].append(f"Price above upper Bollinger Band (${upper:.2f}), possible overbought.")
                    elif current_price < lower:
                        recommendations[ticker].append(f"Price below lower Bollinger Band (${lower:.2f}), possible oversold.")
                
                # Momentum (5-day price change)
                if len(prices) >= 5:
                    price_change = ((current_price - prices[-5]) / prices[-5] * 100) if prices[-5] != 0 else 0
                    if price_change > 5:
                        recommendations[ticker].append(f"Strong momentum ({price_change:.1f}% over 5 days), potential continuation.")
                    elif price_change < -5:
                        recommendations[ticker].append(f"Weak momentum ({price_change:.1f}% over 5 days), potential reversal.")
                
                # Volume Trend
                if len(history) >= 10 and 'Volume' in history.columns:
                    volumes = history['Volume'].tail(10).tolist()
                    avg_volume = sum(volumes) / len(volumes)
                    recent_volume = volumes[-1]
                    if recent_volume > avg_volume * 1.5:
                        recommendations[ticker].append(f"High volume spike ({recent_volume:.0f} vs avg {avg_volume:.0f}), increased activity.")
                    elif recent_volume < avg_volume * 0.5:
                        recommendations[ticker].append(f"Low volume ({recent_volume:.0f} vs avg {avg_volume:.0f}), reduced activity.")
                
                # Stochastic Oscillator (%K)
                if len(prices) >= 14:
                    low14 = min(prices[-14:])
                    high14 = max(prices[-14:])
                    k = 100 * (current_price - low14) / (high14 - low14) if high14 != low14 else 50
                    if k > 80:
                        recommendations[ticker].append(f"Stochastic overbought (K: {k:.1f}), consider selling.")
                    elif k < 20:
                        recommendations[ticker].append(f"Stochastic oversold (K: {k:.1f}), potential buying opportunity.")
                
                # CCI
                if len(history) >= 20 and 'High' in history.columns and 'Low' in history.columns:
                    typical_price = (history['High'] + history['Low'] + history['Close']) / 3
                    tp_last20 = typical_price[-20:]
                    ma20 = tp_last20.mean()
                    mean_dev = (tp_last20 - ma20).abs().mean()
                    cci = (typical_price.iloc[-1] - ma20) / (0.015 * mean_dev) if mean_dev != 0 else 0
                    if cci > 100:
                        recommendations[ticker].append(f"CCI overbought ({cci:.1f}), possible sell.")
                    elif cci < -100:
                        recommendations[ticker].append(f"CCI oversold ({cci:.1f}), possible buy.")
            except Exception as e:
                logging.error(f"Error analyzing {ticker} for recommendations: {e}")
                recommendations[ticker].append(f"Error in analysis ({e}).")

        if total_value > 0 and ticker_values:
            for ticker, value in ticker_values.items():
                allocation = value / total_value * 100
                if allocation > 50:
                    recommendations[ticker].append(f"High allocation ({allocation:.1f}%), consider diversifying.")

        for ticker in watchlist:
            if ticker not in recommendations:
                recommendations[ticker] = []
            try:
                stock = yf.Ticker(ticker)
                history = stock.history(period='1mo')
                if len(history) < 10:
                    continue
                prices = history['Close'].tolist()
                ma_10 = sum(prices[-10:]) / 10 if len(prices) >= 10 else None
                ma_20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else None
                if ma_10 and ma_20 and ma_10 > ma_20:
                    recommendations[ticker].append(f"Bullish trend (10-day MA: ${ma_10:.2f} > 20-day MA: ${ma_20:.2f}), consider adding.")
                rsi = calculate_rsi(prices)
                if rsi and rsi < 30:
                    recommendations[ticker].append(f"Oversold (RSI: {rsi:.1f}), potential buying opportunity.")
            except Exception as e:
                logging.error(f"Error analyzing watchlist {ticker} for recommendations: {e}")

        # Save recommendations
        save_recommendations(recommendations)
        return recommendations
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}")
        return {}

def fetch_historical_portfolio_value(portfolio, period='1y'):
    """Calculate historical portfolio value over a given period."""
    try:
        logging.debug("Fetching historical portfolio value")
        if not portfolio:
            logging.info("Empty portfolio, returning empty DataFrame")
            return pd.DataFrame()

        all_dates = pd.date_range(end=datetime.date.today(), periods=365, freq='D')  # Assume 1 year
        df = pd.DataFrame(index=all_dates)
        df.index.name = 'Date'
        df = df.sort_index()  # Ensure ascending
        for ticker, purchases in portfolio.items():
            try:
                stock = yf.Ticker(ticker)
                history = stock.history(period=period)['Close']
                if history.empty:
                    continue
                history = history.tz_localize(None)  # Remove timezone
                history = history.reindex(df.index, method='ffill')
                total_shares = sum(p['shares'] for p in purchases)
                df[ticker] = history * total_shares
            except Exception as e:
                logging.error(f"Error fetching history for {ticker}: {e}")
                continue

        if df.empty or df.columns.empty:
            logging.info("No valid historical data for any ticker")
            return pd.DataFrame()

        df['Total_Value'] = df.sum(axis=1, skipna=True)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        logging.debug(f"Portfolio value DataFrame shape: {df.shape}")
        return df[['Date', 'Total_Value']]
    except Exception as e:
        logging.error(f"Critical error in fetch_historical_portfolio_value: {e}")
        return pd.DataFrame()

def fetch_data(portfolio, watchlist, alert_threshold):
    """Fetch data for portfolio and watchlist with enhanced error handling."""
    try:
        all_tickers = list(portfolio.keys()) + watchlist
        if not all_tickers:
            logging.info("No tickers in portfolio or watchlist.")
            return {}, {}, [], []

        portfolio_data = {}
        watchlist_data = {}
        news_items = []
        alerts = []
        data_cache = {}  # Local cache for this fetch

        for ticker in all_tickers:
            try:
                if ticker in data_cache and time.time() - data_cache[ticker]['timestamp'] < 300:
                    info = data_cache[ticker]['info']
                else:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    if not info or 'symbol' not in info:
                        logging.error(f"No data returned for ticker {ticker}")
                        raise ValueError(f"No data for ticker {ticker}")
                    data_cache[ticker] = {'info': info, 'timestamp': time.time()}
                
                quote_type = info.get('quoteType', '').upper()
                
                if quote_type == 'MUTUALFUND':
                    current_price = info.get('navPrice', info.get('regularMarketPrice', info.get('previousClose', 0)))
                    prev_close = info.get('previousClose', current_price)
                else:
                    history = stock.history(period='5d', interval='1d')
                    if len(history) < 2:
                        current_price = info.get('regularMarketPrice', info.get('previousClose', 0))
                        prev_close = info.get('previousClose', current_price)
                    else:
                        current_price = history['Close'].iloc[-1]
                        prev_close = history['Close'].iloc[-2]
                
                if pd.isna(current_price) or current_price == 0:
                    current_price = 'N/A'
                    logging.warning(f"Invalid current price for {ticker}")
                if pd.isna(prev_close) or prev_close == 0:
                    prev_close = current_price if current_price != 'N/A' else 'N/A'
                
                if current_price == 'N/A':
                    daily_change = 'N/A'
                    daily_change_pct = 'N/A'
                else:
                    daily_change = current_price - prev_close if prev_close != 'N/A' else 'N/A'
                    daily_change_pct = (daily_change / prev_close * 100) if prev_close != 'N/A' and prev_close > 0 else 'N/A'
                    if daily_change_pct != 'N/A' and daily_change_pct < -alert_threshold:
                        alerts.append(f"{ticker}: {daily_change_pct:.2f}% drop")
                
                if ticker in portfolio or ticker in watchlist:
                    news_items.extend(fetch_news_with_retries(ticker))
                
                if ticker in portfolio:
                    purchases = portfolio[ticker]
                    total_shares = sum(p['shares'] for p in purchases)
                    has_buy_prices = all(p['buy_price'] is not None for p in purchases)
                    
                    total_value = total_shares * current_price if current_price != 'N/A' else 'N/A'
                    yesterday_value = total_shares * prev_close if prev_close != 'N/A' else 'N/A'
                    
                    if has_buy_prices:
                        total_invested = sum(p['shares'] * p['buy_price'] for p in purchases)
                        avg_buy_price = total_invested / total_shares if total_shares > 0 else 0
                        total_pl = total_value - total_invested if total_value != 'N/A' else 'N/A'
                    else:
                        avg_buy_price = 0
                        total_invested = 0
                        total_pl = daily_change * total_shares if daily_change != 'N/A' else 'N/A'
                    
                    portfolio_data[ticker] = {
                        'total_shares': total_shares,
                        'avg_buy_price': avg_buy_price,
                        'current_price': current_price,
                        'total_value': total_value,
                        'total_pl': total_pl,
                        'yesterday_value': yesterday_value,
                        'daily_change_pct': daily_change_pct,
                        'total_invested': total_invested
                    }
                else:
                    watchlist_data[ticker] = {
                        'current_price': current_price,
                        'daily_change': daily_change,
                        'daily_change_pct': daily_change_pct
                    }
            except Exception as e:
                logging.error(f"Error processing {ticker}: {e}")
                if ticker in portfolio:
                    purchases = portfolio[ticker]
                    total_shares = sum(p['shares'] for p in purchases)
                    has_buy_prices = all(p['buy_price'] is not None for p in purchases)
                    if has_buy_prices:
                        total_invested = sum(p['shares'] * p['buy_price'] for p in purchases)
                        avg_buy_price = total_invested / total_shares if total_shares > 0 else 0
                        total_pl = 'N/A'
                    else:
                        avg_buy_price = 0
                        total_invested = 0
                        total_pl = 'N/A'
                    portfolio_data[ticker] = {
                        'total_shares': total_shares,
                        'avg_buy_price': avg_buy_price,
                        'current_price': 'N/A',
                        'total_value': 'N/A',
                        'total_pl': total_pl,
                        'yesterday_value': 'N/A',
                        'daily_change_pct': 'N/A',
                        'total_invested': total_invested
                    }
                else:
                    watchlist_data[ticker] = {
                        'current_price': 'N/A',
                        'daily_change': 'N/A',
                        'daily_change_pct': 'N/A'
                    }
        
        return portfolio_data, watchlist_data, news_items, alerts
    except Exception as e:
        logging.error(f"Critical error in fetch_data: {e}")
        return {}, {}, [], []

def main():
    st.set_page_config(page_title="Portfolio Manager", layout="wide")
    st.title("📈 Portfolio Manager")

    # Initialize JSON files
    init_json_files()

    # Load settings and state
    settings = load_settings()
    portfolio = load_portfolio()
    watchlist = load_watchlist()

    # Initialize session state
    if 'next_refresh' not in st.session_state:
        st.session_state.next_refresh = time.time() + settings['refresh_interval']
    if 'alert_threshold' not in st.session_state:
        st.session_state.alert_threshold = settings.get('alert_threshold', 5.0)
    if 'portfolio_visible_columns' not in st.session_state:
        st.session_state.portfolio_visible_columns = settings['portfolio_visible_columns']
    if 'watchlist_visible_columns' not in st.session_state:
        st.session_state.watchlist_visible_columns = settings['watchlist_visible_columns']
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = settings['auto_refresh']
    if 'refresh_key' not in st.session_state:
        st.session_state.refresh_key = 0
    if 'news_cache' not in st.session_state:
        st.session_state.news_cache = {}
    if 'recommendations_cache' not in st.session_state:
        st.session_state.recommendations_cache = {'recommendations': {}, 'timestamp': 0}
    if 'pie_colors' not in st.session_state:
        st.session_state.pie_colors = {}

    # Auto-refresh
    if st.session_state.auto_refresh:
        refresh_interval = settings['refresh_interval'] * 1000  # Convert to milliseconds
        st_autorefresh(interval=refresh_interval, key="auto_refresh")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        st.checkbox("Enable Auto-Refresh", value=st.session_state.auto_refresh, key="auto_refresh_toggle",
                    on_change=lambda: st.session_state.update(auto_refresh=not st.session_state.auto_refresh))
        new_refresh_interval = st.number_input("Refresh Interval (seconds)", min_value=10, value=settings['refresh_interval'], step=10, key="refresh_interval")
        if st.button("Update Refresh", key="update_refresh"):
            settings['refresh_interval'] = new_refresh_interval
            save_settings(settings)
            st.session_state.next_refresh = time.time() + new_refresh_interval
            st.success(f"Refresh interval set to {new_refresh_interval} seconds.")
            st.rerun()

        new_alert_threshold = st.number_input("Alert Drop Threshold (%)", min_value=0.0, value=settings['alert_threshold'], step=0.1, key="alert_threshold")
        if st.button("Update Alert", key="update_alert"):
            settings['alert_threshold'] = new_alert_threshold
            save_settings(settings)
            st.session_state.alert_threshold = new_alert_threshold
            st.success(f"Alert threshold set to {new_alert_threshold}%.")

        st.subheader("Portfolio Columns")
        portfolio_cols = ['Ticker', 'Shares', 'Avg Buy Price', 'Current Price', 'Value', 'P/L']
        selected_portfolio_cols = [col for col in portfolio_cols if st.checkbox(col, value=col in st.session_state.portfolio_visible_columns, key=f"pf_{col}")]
        if st.button("Apply Portfolio Columns", key="apply_portfolio_cols"):
            st.session_state.portfolio_visible_columns = selected_portfolio_cols
            settings['portfolio_visible_columns'] = selected_portfolio_cols
            save_settings(settings)
            st.rerun()

        st.subheader("Watchlist Columns")
        watchlist_cols = ['Ticker', 'Current Price', 'Daily Change']
        selected_watchlist_cols = [col for col in watchlist_cols if st.checkbox(col, value=col in st.session_state.watchlist_visible_columns, key=f"wf_{col}")]
        if st.button("Apply Watchlist Columns", key="apply_watchlist_cols"):
            st.session_state.watchlist_visible_columns = selected_watchlist_cols
            settings['watchlist_visible_columns'] = selected_watchlist_cols
            save_settings(settings)
            st.rerun()

        st.subheader("Pie Chart Colors")
        tickers = sorted(portfolio.keys())
        color_palette = px.colors.qualitative.Plotly
        for i, ticker in enumerate(tickers):
            default_color = color_palette[i % len(color_palette)]
            st.session_state.pie_colors[ticker] = st.color_picker(f"Color for {ticker}", st.session_state.pie_colors.get(ticker, default_color))

    # Input section
    st.header("➕ Manage Portfolio and Watchlist")
    with st.expander("Add/Remove Stocks", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Add Purchase")
            ticker = st.text_input("Ticker", key="add_ticker")
            shares = st.number_input("Shares", min_value=0.0, step=0.01, key="add_shares")
            buy_price = st.number_input("Buy Price (optional)", min_value=0.0, step=0.01, value=0.0, key="add_buy_price")
            buy_date = st.date_input("Buy Date (optional)", value=datetime.date.today(), key="add_buy_date")
            if st.button("Add Purchase", key="add_purchase"):
                try:
                    if not ticker:
                        st.error("Ticker cannot be empty.")
                    else:
                        ticker = ticker.upper()
                        if shares <= 0:
                            raise ValueError("Shares must be positive.")
                        buy_price = buy_price if buy_price > 0 else None
                        buy_date_str = buy_date.strftime('%Y-%m-%d') if buy_date else None
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        if not info or 'symbol' not in info:
                            st.error(f"Ticker '{ticker}' not found or no data available.")
                        else:
                            if ticker not in portfolio:
                                portfolio[ticker] = []
                            portfolio[ticker].append({'shares': shares, 'buy_price': buy_price, 'buy_date': buy_date_str})
                            save_portfolio(portfolio)
                            st.success(f"Added purchase for {ticker} to portfolio.")
                            st.rerun()
                except ValueError as e:
                    st.error(f"Invalid input: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    logging.error(f"Unexpected error adding stock {ticker}: {e}")

        with col2:
            st.subheader("Add to Watchlist")
            watchlist_ticker = st.text_input("Ticker", key="watchlist_ticker")
            if st.button("Add to Watchlist", key="add_watchlist"):
                try:
                    if not watchlist_ticker:
                        st.error("Ticker cannot be empty.")
                    elif watchlist_ticker.upper() in watchlist:
                        st.error(f"{watchlist_ticker.upper()} is already in the watchlist.")
                    else:
                        ticker = watchlist_ticker.upper()
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        if not info or 'symbol' not in info:
                            st.error(f"Ticker '{ticker}' not found or no data available.")
                        else:
                            watchlist.append(ticker)
                            save_watchlist(watchlist)
                            st.success(f"Added {ticker} to watchlist.")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error adding to watchlist: {e}")
                    logging.error(f"Error adding {ticker} to watchlist: {e}")

        with col3:
            st.subheader("Remove Purchase")
            ticker_options = list(portfolio.keys()) if portfolio else ["No tickers"]
            selected_ticker = st.selectbox("Select Ticker", ticker_options, key="remove_ticker")
            if selected_ticker and selected_ticker != "No tickers":
                purchases = portfolio.get(selected_ticker, [])
                purchase_options = [f"Shares: {p['shares']:.2f}, Buy Price: {'N/A' if p['buy_price'] is None else f'${p['buy_price']:.2f}'}, Buy Date: {p.get('buy_date', 'N/A')}" for p in purchases]
                selected_purchase = st.selectbox("Select Purchase", purchase_options, key="remove_purchase")
                if st.button("Remove Purchase", key="remove_purchase_btn"):
                    try:
                        if not selected_ticker or selected_ticker == "No tickers":
                            st.error("Please select a ticker.")
                        elif not selected_purchase:
                            st.error("Please select a purchase to remove.")
                        else:
                            index = purchase_options.index(selected_purchase)
                            removed = purchases.pop(index)
                            if not purchases:
                                del portfolio[selected_ticker]
                            save_portfolio(portfolio)
                            buy_price_str = f"${removed['buy_price']:.2f}" if removed['buy_price'] is not None else "N/A"
                            buy_date_str = removed.get('buy_date', 'N/A')
                            st.success(f"Removed purchase: Shares: {removed['shares']:.2f}, Buy Price: {buy_price_str}, Buy Date: {buy_date_str} for {selected_ticker}.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error removing purchase: {e}")
                        logging.error(f"Error removing purchase for {selected_ticker}: {e}")

        with col4:
            st.subheader("Remove from Watchlist")
            ticker_options = watchlist if watchlist else ["No tickers"]
            watchlist_ticker = st.selectbox("Select Ticker", ticker_options, key="remove_watchlist_ticker")
            if st.button("Remove from Watchlist", key="remove_watchlist"):
                try:
                    if not watchlist_ticker or watchlist_ticker == "No tickers":
                        st.error("Please select a ticker.")
                    elif watchlist_ticker in watchlist:
                        watchlist.remove(watchlist_ticker)
                        save_watchlist(watchlist)
                        st.success(f"Removed {watchlist_ticker} from watchlist.")
                        st.rerun()
                    else:
                        st.error(f"{watchlist_ticker} not in watchlist.")
                except Exception as e:
                    st.error(f"Error removing from watchlist: {e}")
                    logging.error(f"Error removing {watchlist_ticker} from watchlist: {e}")

    # Export/Import CSV
    st.header("📁 Export/Import Portfolio (CSV)")
    col_export, col_import = st.columns(2)
    with col_export:
        if st.button("Export to CSV", key="export_csv"):
            try:
                with open(CSV_FILE, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Ticker', 'Shares', 'Buy Price', 'Buy Date'])
                    for ticker, purchases in portfolio.items():
                        for p in purchases:
                            buy_price = p['buy_price'] if p['buy_price'] is not None else ''
                            buy_date = p.get('buy_date', '')
                            writer.writerow([ticker, p['shares'], buy_price, buy_date])
                st.success(f"Portfolio exported to {CSV_FILE}.")
            except Exception as e:
                st.error(f"Error exporting to CSV: {e}")
                logging.error(f"Error exporting to CSV: {e}")

    with col_import:
        uploaded_file = st.file_uploader("Import from CSV", type=['csv'], key="csv_uploader")
        if uploaded_file and st.button("Import CSV", key="import_csv"):
            try:
                df = pd.read_csv(uploaded_file)
                expected_header = ['Ticker', 'Shares', 'Buy Price', 'Buy Date']
                if list(df.columns)[:3] != expected_header[:3]:
                    st.error(f"Invalid CSV format. Expected header at least: {','.join(expected_header[:3])}")
                else:
                    for _, row in df.iterrows():
                        ticker = row['Ticker'].upper()
                        try:
                            shares = float(row['Shares'])
                            if shares <= 0:
                                st.error(f"WARNING: Skipping {ticker}: Shares must be positive.")
                                continue
                            buy_price = float(row['Buy Price']) if pd.notna(row['Buy Price']) else None
                            if buy_price is not None and buy_price <= 0:
                                st.error(f"WARNING: Skipping {ticker}: Buy price must be positive.")
                                continue
                            buy_date = row.get('Buy Date', None)
                            if buy_date and not pd.isna(buy_date):
                                try:
                                    datetime.datetime.strptime(str(buy_date), '%Y-%m-%d')
                                except ValueError:
                                    st.error(f"WARNING: Invalid buy date format for {ticker}. Expected YYYY-MM-DD.")
                                    buy_date = None
                            else:
                                buy_date = None
                            
                            stock = yf.Ticker(ticker)
                            info = stock.info
                            if not info or 'symbol' not in info:
                                st.error(f"WARNING: Ticker '{ticker}' not found or no data available.")
                                continue
                            
                            if ticker not in portfolio:
                                portfolio[ticker] = []
                            portfolio[ticker].append({'shares': shares, 'buy_price': buy_price, 'buy_date': buy_date})
                        except ValueError:
                            st.error(f"WARNING: Skipping {ticker}: Invalid shares or buy price in CSV.")
                            continue
                        except Exception as e:
                            st.error(f"WARNING: Skipping {ticker}: {e}")
                            logging.error(f"Error importing {ticker} from CSV: {e}")
                    
                    save_portfolio(portfolio)
                    st.success(f"Portfolio imported from CSV.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing from CSV: {e}")
                logging.error(f"Error importing from CSV: {e}")

    # Export/Import All as JSON
    st.header("📁 Export/Import All Data (JSON)")
    col_export_json, col_import_json = st.columns(2)
    with col_export_json:
        if st.button("Export All to JSON", key="export_json"):
            try:
                all_data = {
                    "portfolio": portfolio,
                    "watchlist": watchlist,
                    "settings": settings,
                }
                json_str = json.dumps(all_data)
                st.download_button("Download JSON", json_str, file_name="portfolio_all.json", mime="application/json")
                st.success("Exported all data to JSON.")
            except Exception as e:
                st.error(f"Error exporting JSON: {e}")
                logging.error(f"Error exporting JSON: {e}")

    with col_import_json:
        uploaded_json = st.file_uploader("Import All from JSON", type=['json'], key="json_uploader")
        if uploaded_json and st.button("Import JSON", key="import_json"):
            try:
                all_data = json.load(uploaded_json)
                portfolio = all_data.get("portfolio", {})
                watchlist = all_data.get("watchlist", [])
                settings = all_data.get("settings", DEFAULT_SETTINGS)
                # Ensure buy_date exists in imported portfolio
                for ticker, purchases in portfolio.items():
                    for p in purchases:
                        p['buy_date'] = p.get('buy_date', None)
                save_portfolio(portfolio)
                save_watchlist(watchlist)
                save_settings(settings)
                st.success("Imported all data from JSON.")
                st.rerun()
            except Exception as e:
                st.error(f"Error importing JSON: {e}")
                logging.error(f"Error importing JSON: {e}")

    # Portfolio and Watchlist Display
    st.header("📊 Portfolio and Watchlist")
    portfolio_data, watchlist_data, news_items, alerts = fetch_data(portfolio, watchlist, st.session_state.alert_threshold)

    # Style positive and negative values
    def style_values(val):
        try:
            if isinstance(val, str) and val != 'N/A':
                # Handle strings with dollar signs or percentages
                num_str = val.replace('$', '').replace('%', '')
                if '(' in num_str:
                    num_str = num_str.split('(')[0].strip()  # Extract number before percentage
                num = float(num_str)
                return 'color: green' if num > 0 else 'color: red' if num < 0 else ''
            return ''
        except:
            return ''

    col_portfolio, col_watchlist = st.columns(2)

    with col_portfolio:
        st.subheader("Portfolio")
        if portfolio_data:
            df_data = []
            for ticker, data in portfolio_data.items():
                df_data.append({
                    'Ticker': ticker,
                    'Shares': f"{data['total_shares']:.2f}",
                    'Avg Buy Price': f"${data['avg_buy_price']:.2f}",
                    'Current Price': f"${data['current_price']:.2f}" if data['current_price'] != 'N/A' else 'N/A',
                    'Value': f"${data['total_value']:.2f}" if data['total_value'] != 'N/A' else 'N/A',
                    'P/L': f"${data['total_pl']:.2f}" if data['total_pl'] != 'N/A' else 'N/A'
                })
            df = pd.DataFrame(df_data)
            columns = [col for col in st.session_state.portfolio_visible_columns if col in df.columns]
            # Apply styling only to P/L
            styled_df = df[columns].style.applymap(style_values, subset=['P/L'])
            st.dataframe(styled_df, use_container_width=True, height=300)
        else:
            st.info("No portfolio data available.")

    with col_watchlist:
        st.subheader("Watchlist")
        if watchlist_data:
            df_data = []
            for ticker, data in watchlist_data.items():
                df_data.append({
                    'Ticker': ticker,
                    'Current Price': f"${data['current_price']:.2f}" if data['current_price'] != 'N/A' else 'N/A',
                    'Daily Change': f"${data['daily_change']:.2f} ({data['daily_change_pct']:.2f}%)" if data['current_price'] != 'N/A' else 'N/A'
                })
            df = pd.DataFrame(df_data)
            columns = [col for col in st.session_state.watchlist_visible_columns if col in df.columns]
            # Apply styling only to Daily Change
            styled_df = df[columns].style.applymap(style_values, subset=['Daily Change'])
            st.dataframe(styled_df, use_container_width=True, height=300)
        else:
            st.info("No watchlist data available.")

    # Portfolio Summary with red/green styling
    st.header("💹 Portfolio Summary")
    total_value_all = sum(data['total_value'] for data in portfolio_data.values() if data['total_value'] != 'N/A')
    total_invested_all = sum(data['total_invested'] for data in portfolio_data.values())
    total_pl_all = sum(data['total_pl'] for data in portfolio_data.values() if data['total_pl'] != 'N/A')
    total_yesterday_value = sum(data['yesterday_value'] for data in portfolio_data.values() if data['yesterday_value'] != 'N/A')
    
    daily_change = total_value_all - total_yesterday_value
    daily_change_pct = (daily_change / total_yesterday_value * 100) if total_yesterday_value > 0 else 0

    col_metrics1, col_metrics2 = st.columns(2)
    with col_metrics1:
        st.markdown(f"Total Portfolio Value: ${total_value_all:.2f}")
        daily_change_style = 'color: green' if daily_change > 0 else 'color: red' if daily_change < 0 else ''
        st.markdown(f"Daily Change: <span style='{daily_change_style}'>${daily_change:.2f}</span>", unsafe_allow_html=True)
        st.markdown(f"Total Tracked Invested: ${total_invested_all:.2f}")
    with col_metrics2:
        pl_style = 'color: green' if total_pl_all > 0 else 'color: red' if total_pl_all < 0 else ''
        st.markdown(f"Total Profit/Loss: <span style='{pl_style}'>${total_pl_all:.2f}</span>", unsafe_allow_html=True)
        daily_change_pct_style = 'color: green' if daily_change_pct > 0 else 'color: red' if daily_change_pct < 0 else ''
        st.markdown(f"Daily Change (%): <span style='{daily_change_pct_style}'>{daily_change_pct:.2f}%</span>", unsafe_allow_html=True)

    # Market open/close countdown
    st.header("🕒 Market Status")
    market_status = st.empty()
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4)))  # EDT
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if now < market_open:
        time_left = market_open - now
        hours, remainder = divmod(time_left.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        market_status.metric(f"🕒 Market closed", f"Opens in {hours:02d}:{minutes:02d}:{seconds:02d}")
    elif market_open <= now < market_close:
        time_left = market_close - now
        hours, remainder = divmod(time_left.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        market_status.metric(f"🕒 Market open", f"Closes in {hours:02d}:{minutes:02d}:{seconds:02d}")
    else:
        next_open = (now + datetime.timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
        time_left = next_open - now
        hours, remainder = divmod(time_left.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        market_status.metric(f"🕒 Market closed", f"Opens in {hours:02d}:{minutes:02d}:{seconds:02d}")

    # Portfolio Visualization
    st.header("📈 Portfolio Visualization")
    st.subheader("Allocation")
    if total_value_all > 0:
        labels = [t for t in portfolio_data.keys() if portfolio_data[t]['total_value'] != 'N/A']
        sizes = [portfolio_data[t]['total_value'] for t in labels]
        df_pie = pd.DataFrame({'Ticker': labels, 'Value': sizes})
        fig = px.pie(df_pie, values='Value', names='Ticker', title='Portfolio Allocation',
                     template='plotly_dark', hole=0.3)
        colors_list = [st.session_state.pie_colors.get(label, px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]) for i, label in enumerate(labels)]
        fig.update_traces(marker=dict(colors=colors_list), textposition='inside', textinfo='percent+label')
        fig.update_layout(
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for portfolio allocation.")

    # Historical Portfolio Value Chart
    st.subheader("Historical Value")
    historical_values = fetch_historical_portfolio_value(portfolio, period='1y')
    if not historical_values.empty:
        fig_hist = px.line(historical_values, x='Date', y='Total_Value',
                           title='Portfolio Value Over Time', template='plotly_dark')
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No historical data available for portfolio value. Please ensure portfolio tickers have recent data or try a shorter time period.")

    # News and Recommendations
    st.header("📰 News & Recommendations")
    if time.time() - st.session_state.recommendations_cache['timestamp'] > 3600:
        recommendations = generate_recommendations(portfolio, watchlist)
        st.session_state.recommendations_cache = {'recommendations': recommendations, 'timestamp': time.time()}
    else:
        recommendations = st.session_state.recommendations_cache['recommendations']
    
    all_tickers = set(portfolio.keys()) | set(watchlist)
    for ticker in sorted(all_tickers):
        news_for_ticker = [item for item in news_items if item['ticker'] == ticker]
        recs_for_ticker = recommendations.get(ticker, [])
        if news_for_ticker or recs_for_ticker:
            with st.expander(f"**{ticker}** ({'Portfolio' if ticker in portfolio else 'Watchlist'})"):
                if news_for_ticker:
                    st.markdown("#### News")
                    for item in news_for_ticker:
                        sentiment_color = 'green' if item['sentiment'] == 'Positive' else 'red' if item['sentiment'] == 'Negative' else 'grey'
                        st.markdown(
                            f"""
                            <div style='padding: 10px; border-bottom: 1px solid #333;'>
                                <p style='margin: 0;'><a href='{item['link']}' target='_blank'>{item['title']}</a></p>
                                <p style='margin: 0; color: #aaa; font-size: 0.9em;'>{item['publisher']} | {item['date']}</p>
                                <p style='margin: 0; color: {sentiment_color}; font-size: 0.9em;'>Sentiment: <span style='color: {sentiment_color};'>{item['sentiment']}</span> ({item['compound']:.2f})</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    # Overall sentiment
                    avg_compound = sum(item['compound'] for item in news_for_ticker if 'compound' in item) / len(news_for_ticker) if news_for_ticker else 0
                    overall_sentiment = "Positive" if avg_compound > 0.05 else "Negative" if avg_compound < -0.05 else "Neutral"
                    overall_color = 'green' if overall_sentiment == 'Positive' else 'red' if overall_sentiment == 'Negative' else 'grey'
                    st.markdown(f"**Overall News Sentiment**: <span style='color: {overall_color};'>{overall_sentiment}</span> (average score: {avg_compound:.2f})", unsafe_allow_html=True)
                    if overall_sentiment == "Positive":
                        st.markdown("The recent news suggests <span style='color: green;'>positive</span> developments for the stock.", unsafe_allow_html=True)
                    elif overall_sentiment == "Negative":
                        st.markdown("The recent news indicates <span style='color: red;'>potential challenges</span> for the stock.", unsafe_allow_html=True)
                    else:
                        st.markdown("The news sentiment is neutral, with no strong positive or negative indicators.")
                if recs_for_ticker:
                    st.markdown("#### Recommendations")
                    for rec in recs_for_ticker:
                        rec_lower = rec.lower()
                        if "bullish" in rec_lower or "buy" in rec_lower or "oversold" in rec_lower:
                            color = "green"
                        elif "bearish" in rec_lower or "sell" in rec_lower or "overbought" in rec_lower:
                            color = "red"
                        else:
                            color = "white"
                        st.markdown(f"<span style='color: {color};'>- {rec}</span>", unsafe_allow_html=True)

    # Alerts with red styling
    if alerts:
        alert_text = "<br>".join([f'<span style="color: red">{alert}</span>' for alert in alerts])
        st.markdown(f"🚨 **Portfolio Alerts**:<br>{alert_text}", unsafe_allow_html=True)

    # Display refresh countdown
    refresh_placeholder = st.empty()
    seconds_left = max(0, int(st.session_state.next_refresh - time.time()))
    refresh_placeholder.metric("🔄 Next refresh in", f"{seconds_left}s")
    if seconds_left <= 0:
        st.session_state.next_refresh = time.time() + settings['refresh_interval']
        st.session_state.refresh_key += 1
        st.rerun()

if __name__ == "__main__":
    main()