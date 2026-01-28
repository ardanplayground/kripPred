import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from statistics import mean, stdev
import math
import json

# Konfigurasi halaman
st.set_page_config(
    page_title="AI Crypto Forecast Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom - Enhanced
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 6px solid #ffc107;
        padding: 20px;
        margin: 20px 0;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #a8e6cf 100%);
        border-left: 6px solid #28a745;
        padding: 20px;
        margin: 20px 0;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 1px solid #dee2e6;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
    }
    .signal-bullish {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .signal-bearish {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 AI Crypto Forecast Pro</h1>', unsafe_allow_html=True)

# Enhanced Disclaimer
st.markdown("""
<div class="warning-box">
    <strong>⚠️ ADVANCED DISCLAIMER:</strong><br>
    • Model AI ini menggunakan ensemble methods dengan target akurasi 60-65%<br>
    • Kombinasi: LSTM-inspired predictions + Technical Analysis + Sentiment Scoring<br>
    • Crypto markets sangat volatil - TIDAK ADA model yang 100% akurat<br>
    • Ini BUKAN financial advice - gunakan untuk riset & edukasi<br>
    • Selalu gunakan risk management & jangan invest lebih dari yang mampu Anda tanggung
</div>
""", unsafe_allow_html=True)

# ============= ADVANCED FUNCTIONS =============

@st.cache_data(ttl=300)
def search_coins(query):
    """Mencari coin berdasarkan query"""
    try:
        url = "https://api.coingecko.com/api/v3/search"
        params = {"query": query}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get('coins', [])
        return []
    except Exception as e:
        st.error(f"Error searching coins: {e}")
        return []

@st.cache_data(ttl=300)
def get_trending_coins():
    """Ambil trending coins"""
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get('coins', [])
        return []
    except Exception as e:
        return []

@st.cache_data(ttl=300)
def get_coin_data(coin_id, days=30, currency="usd"):
    """Ambil data historical coin"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": currency,
            "days": days,
            "interval": "daily" if days > 1 else "hourly"
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            market_caps = data.get('market_caps', [])
            
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            if volumes:
                df_vol = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                df_vol['timestamp'] = pd.to_datetime(df_vol['timestamp'], unit='ms')
                df = df.merge(df_vol, on='timestamp', how='left')
            
            if market_caps:
                df_mc = pd.DataFrame(market_caps, columns=['timestamp', 'market_cap'])
                df_mc['timestamp'] = pd.to_datetime(df_mc['timestamp'], unit='ms')
                df = df.merge(df_mc, on='timestamp', how='left')
            
            return df
        return None
    except Exception as e:
        st.error(f"Error getting coin data: {e}")
        return None

@st.cache_data(ttl=300)
def get_coin_info(coin_id):
    """Ambil informasi detail coin"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "community_data": "true",
            "developer_data": "true"
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def get_global_crypto_data():
    """Ambil data global crypto market"""
    try:
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get('data', {})
        return {}
    except Exception:
        return {}

# ============= TECHNICAL INDICATORS - ADVANCED =============

def calculate_sma(data, period):
    """Simple Moving Average"""
    if len(data) < period:
        return [None] * len(data)
    sma = []
    for i in range(len(data)):
        if i < period - 1:
            sma.append(None)
        else:
            sma.append(mean(data[i-period+1:i+1]))
    return sma

def calculate_ema(data, period):
    """Exponential Moving Average"""
    if len(data) < period:
        return [None] * len(data)
    
    ema = [None] * (period - 1)
    ema.append(mean(data[:period]))
    multiplier = 2 / (period + 1)
    
    for i in range(period, len(data)):
        ema.append((data[i] - ema[-1]) * multiplier + ema[-1])
    
    return ema

def calculate_rsi(data, period=14):
    """Relative Strength Index"""
    if len(data) < period + 1:
        return [None] * len(data)
    
    changes = [data[i] - data[i-1] for i in range(1, len(data))]
    gains = [max(0, change) for change in changes]
    losses = [abs(min(0, change)) for change in changes]
    
    rsi = [None] * period
    avg_gain = mean(gains[:period])
    avg_loss = mean(losses[:period])
    
    for i in range(period, len(changes)):
        if avg_loss == 0:
            rsi.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    return [None] + rsi

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Bollinger Bands"""
    sma = calculate_sma(data, period)
    
    upper = []
    lower = []
    
    for i in range(len(data)):
        if i < period - 1:
            upper.append(None)
            lower.append(None)
        else:
            window = data[i-period+1:i+1]
            std = stdev(window) if len(window) > 1 else 0
            upper.append(sma[i] + (std_dev * std))
            lower.append(sma[i] - (std_dev * std))
    
    return upper, lower

def calculate_macd(data, fast=12, slow=26, signal=9):
    """MACD Indicator"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    
    macd_line = []
    for i in range(len(data)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)
    
    signal_line = calculate_ema([x for x in macd_line if x is not None], signal)
    signal_line = [None] * (len(macd_line) - len(signal_line)) + signal_line
    
    histogram = []
    for i in range(len(macd_line)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram.append(macd_line[i] - signal_line[i])
        else:
            histogram.append(None)
    
    return macd_line, signal_line, histogram

def calculate_stochastic(highs, lows, closes, period=14):
    """Stochastic Oscillator"""
    k_values = []
    
    for i in range(len(closes)):
        if i < period - 1:
            k_values.append(None)
        else:
            high_max = max(highs[i-period+1:i+1])
            low_min = min(lows[i-period+1:i+1])
            
            if high_max - low_min == 0:
                k_values.append(50)
            else:
                k = ((closes[i] - low_min) / (high_max - low_min)) * 100
                k_values.append(k)
    
    d_values = calculate_sma([x for x in k_values if x is not None], 3)
    d_values = [None] * (len(k_values) - len(d_values)) + d_values
    
    return k_values, d_values

def calculate_atr(highs, lows, closes, period=14):
    """Average True Range - Volatility Indicator"""
    tr_values = []
    
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])
        tr = max(high_low, high_close, low_close)
        tr_values.append(tr)
    
    atr = [None]
    if len(tr_values) >= period:
        atr.append(mean(tr_values[:period]))
        
        for i in range(period, len(tr_values)):
            atr.append((atr[-1] * (period - 1) + tr_values[i]) / period)
    
    while len(atr) < len(closes):
        atr.append(None)
    
    return atr

def calculate_obv(closes, volumes):
    """On-Balance Volume"""
    obv = [0]
    
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    
    return obv

# ============= ADVANCED FORECASTING =============

def lstm_inspired_forecast(prices, days_ahead=7):
    """
    LSTM-inspired forecast using pattern recognition
    Simulates LSTM behavior dengan weighted moving patterns
    """
    if len(prices) < 20:
        return None
    
    # Extract patterns dari historical data
    window_size = 10
    patterns = []
    
    for i in range(len(prices) - window_size):
        window = prices[i:i+window_size]
        # Normalize pattern
        mean_val = mean(window)
        normalized = [(x - mean_val) / mean_val if mean_val != 0 else 0 for x in window]
        patterns.append(normalized)
    
    # Cari pattern yang mirip dengan current state
    current_window = prices[-window_size:]
    current_mean = mean(current_window)
    current_normalized = [(x - current_mean) / current_mean if current_mean != 0 else 0 for x in current_window]
    
    # Calculate similarity scores
    similarities = []
    for pattern in patterns:
        # Euclidean distance
        dist = math.sqrt(sum((current_normalized[i] - pattern[i])**2 for i in range(window_size)))
        similarities.append(1 / (1 + dist))  # Convert to similarity score
    
    # Weighted prediction based on similar patterns
    top_k = min(5, len(similarities))
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    
    forecasts = []
    last_price = prices[-1]
    
    for day in range(1, days_ahead + 1):
        weighted_changes = []
        
        for idx in top_indices:
            if idx + window_size + day < len(prices):
                future_price = prices[idx + window_size + day]
                current_price = prices[idx + window_size]
                change = (future_price - current_price) / current_price if current_price != 0 else 0
                weight = similarities[idx]
                weighted_changes.append(change * weight)
        
        if weighted_changes:
            avg_change = sum(weighted_changes) / sum(similarities[i] for i in top_indices)
            forecast_price = last_price * (1 + avg_change)
            last_price = forecast_price
        else:
            forecast_price = last_price
        
        forecasts.append(forecast_price)
    
    return forecasts

def ensemble_forecast(df, days_ahead=7):
    """
    Ensemble forecasting - combine multiple methods
    """
    prices = df['price'].tolist()
    
    # Method 1: Trend-based
    recent = prices[-14:]
    x = list(range(len(recent)))
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(recent)
    sum_xy = sum(x[i] * recent[i] for i in range(n))
    sum_x2 = sum(i**2 for i in x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2) if (n * sum_x2 - sum_x**2) != 0 else 0
    intercept = (sum_y - slope * sum_x) / n
    
    trend_forecasts = [intercept + slope * (len(recent) + i) for i in range(1, days_ahead + 1)]
    
    # Method 2: LSTM-inspired
    lstm_forecasts = lstm_inspired_forecast(prices, days_ahead)
    if lstm_forecasts is None:
        lstm_forecasts = trend_forecasts
    
    # Method 3: EMA-based
    ema_21 = calculate_ema(prices, 21)
    last_ema = ema_21[-1] if ema_21[-1] is not None else prices[-1]
    momentum = (prices[-1] - last_ema) / last_ema if last_ema != 0 else 0
    ema_forecasts = [prices[-1] * (1 + momentum * (i * 0.5)) for i in range(1, days_ahead + 1)]
    
    # Combine with weights
    ensemble = []
    volatility = stdev(prices[-30:]) if len(prices) >= 30 else stdev(prices)
    
    for i in range(days_ahead):
        # Dynamic weighting based on market conditions
        if volatility / mean(prices[-30:]) > 0.1:  # High volatility
            combined = (lstm_forecasts[i] * 0.5 + trend_forecasts[i] * 0.3 + ema_forecasts[i] * 0.2)
        else:  # Low volatility
            combined = (trend_forecasts[i] * 0.4 + lstm_forecasts[i] * 0.4 + ema_forecasts[i] * 0.2)
        
        upper = combined + (volatility * 2)
        lower = max(0, combined - (volatility * 2))
        
        ensemble.append({
            'price': combined,
            'upper': upper,
            'lower': lower,
            'confidence': calculate_confidence(prices, i+1)
        })
    
    return ensemble

def calculate_confidence(prices, days_ahead):
    """Calculate forecast confidence score"""
    # Factors: volatility, trend strength, data availability
    
    volatility = stdev(prices[-30:]) / mean(prices[-30:]) if len(prices) >= 30 else 1
    
    # Trend consistency
    recent_14 = prices[-14:]
    uptrend = sum(1 for i in range(1, len(recent_14)) if recent_14[i] > recent_14[i-1])
    trend_strength = abs(uptrend - 7) / 7  # 0 to 1, where 1 is strongest trend
    
    # Time decay - confidence decreases with forecast horizon
    time_decay = 1 / (1 + (days_ahead * 0.1))
    
    # Combine factors
    confidence = (
        (1 - min(volatility, 1)) * 0.4 +  # Lower volatility = higher confidence
        trend_strength * 0.3 +              # Stronger trend = higher confidence
        time_decay * 0.3                    # Nearer forecast = higher confidence
    )
    
    return max(0.3, min(0.85, confidence))  # Clamp between 30-85%

# ============= ADVANCED ANALYSIS =============

def calculate_advanced_indicators(df):
    """Calculate all technical indicators"""
    prices = df['price'].tolist()
    volumes = df['volume'].tolist() if 'volume' in df.columns else [0] * len(prices)
    
    # Create high/low approximations (since we only have closing prices)
    highs = [p * 1.01 for p in prices]  # Approximate
    lows = [p * 0.99 for p in prices]   # Approximate
    
    indicators = {}
    
    # Moving Averages
    indicators['sma_7'] = calculate_sma(prices, 7)
    indicators['sma_25'] = calculate_sma(prices, 25)
    indicators['sma_50'] = calculate_sma(prices, 50)
    indicators['ema_12'] = calculate_ema(prices, 12)
    indicators['ema_26'] = calculate_ema(prices, 26)
    indicators['ema_50'] = calculate_ema(prices, 50)
    
    # RSI
    indicators['rsi'] = calculate_rsi(prices, 14)
    
    # Bollinger Bands
    upper, lower = calculate_bollinger_bands(prices, 20, 2)
    indicators['bb_upper'] = upper
    indicators['bb_lower'] = lower
    indicators['bb_middle'] = calculate_sma(prices, 20)
    
    # MACD
    macd, signal, histogram = calculate_macd(prices, 12, 26, 9)
    indicators['macd'] = macd
    indicators['macd_signal'] = signal
    indicators['macd_histogram'] = histogram
    
    # Stochastic
    k, d = calculate_stochastic(highs, lows, prices, 14)
    indicators['stoch_k'] = k
    indicators['stoch_d'] = d
    
    # ATR
    indicators['atr'] = calculate_atr(highs, lows, prices, 14)
    
    # OBV
    if len(volumes) == len(prices):
        indicators['obv'] = calculate_obv(prices, volumes)
    
    return indicators

def generate_advanced_signals(df, indicators, coin_info):
    """Generate advanced trading signals with scoring"""
    signals = []
    scores = {'bullish': 0, 'bearish': 0, 'neutral': 0}
    
    current_price = df['price'].iloc[-1]
    prices = df['price'].tolist()
    
    # RSI Analysis
    if indicators['rsi'][-1]:
        rsi = indicators['rsi'][-1]
        if rsi < 30:
            signals.append({
                'type': 'bullish',
                'indicator': 'RSI',
                'message': f"OVERSOLD (RSI: {rsi:.1f}) - Strong Buy Signal",
                'strength': 'high'
            })
            scores['bullish'] += 3
        elif rsi > 70:
            signals.append({
                'type': 'bearish',
                'indicator': 'RSI',
                'message': f"OVERBOUGHT (RSI: {rsi:.1f}) - Strong Sell Signal",
                'strength': 'high'
            })
            scores['bearish'] += 3
        elif 40 <= rsi <= 60:
            signals.append({
                'type': 'neutral',
                'indicator': 'RSI',
                'message': f"NEUTRAL (RSI: {rsi:.1f}) - No clear signal",
                'strength': 'low'
            })
            scores['neutral'] += 1
    
    # Moving Average Crossovers
    if indicators['sma_7'][-1] and indicators['sma_25'][-1]:
        sma7_curr = indicators['sma_7'][-1]
        sma25_curr = indicators['sma_25'][-1]
        
        if len(indicators['sma_7']) > 1 and indicators['sma_7'][-2] and indicators['sma_25'][-2]:
            sma7_prev = indicators['sma_7'][-2]
            sma25_prev = indicators['sma_25'][-2]
            
            # Golden Cross
            if sma7_prev <= sma25_prev and sma7_curr > sma25_curr:
                signals.append({
                    'type': 'bullish',
                    'indicator': 'MA Cross',
                    'message': "GOLDEN CROSS - SMA7 crossed above SMA25 (Strong Buy)",
                    'strength': 'high'
                })
                scores['bullish'] += 4
            # Death Cross
            elif sma7_prev >= sma25_prev and sma7_curr < sma25_curr:
                signals.append({
                    'type': 'bearish',
                    'indicator': 'MA Cross',
                    'message': "DEATH CROSS - SMA7 crossed below SMA25 (Strong Sell)",
                    'strength': 'high'
                })
                scores['bearish'] += 4
        
        if sma7_curr > sma25_curr:
            scores['bullish'] += 1
        else:
            scores['bearish'] += 1
    
    # MACD Analysis
    if indicators['macd'][-1] and indicators['macd_signal'][-1]:
        macd_curr = indicators['macd'][-1]
        signal_curr = indicators['macd_signal'][-1]
        
        if len(indicators['macd']) > 1 and indicators['macd'][-2] and indicators['macd_signal'][-2]:
            macd_prev = indicators['macd'][-2]
            signal_prev = indicators['macd_signal'][-2]
            
            # MACD Crossover
            if macd_prev <= signal_prev and macd_curr > signal_curr:
                signals.append({
                    'type': 'bullish',
                    'indicator': 'MACD',
                    'message': "MACD Bullish Crossover - Buy Signal",
                    'strength': 'medium'
                })
                scores['bullish'] += 2
            elif macd_prev >= signal_prev and macd_curr < signal_curr:
                signals.append({
                    'type': 'bearish',
                    'indicator': 'MACD',
                    'message': "MACD Bearish Crossover - Sell Signal",
                    'strength': 'medium'
                })
                scores['bearish'] += 2
    
    # Bollinger Bands
            if indicators['bb_upper'][-1] and indicators['bb_lower'][-1]:
        bb_upper = indicators['bb_upper'][-1]
        bb_lower = indicators['bb_lower'][-1]
        
        if current_price >= bb_upper:
            signals.append({
                'type': 'bearish',
                'indicator': 'Bollinger',
                'message': f"Price at Upper Band ({currency_symbol}{bb_upper:.2f}) - Overbought",
                'strength': 'medium'
            })
            scores['bearish'] += 2
        elif current_price <= bb_lower:
            signals.append({
                'type': 'bullish',
                'indicator': 'Bollinger',
                'message': f"Price at Lower Band ({currency_symbol}{bb_lower:.2f}) - Oversold",
                'strength': 'medium'
            })
            scores['bullish'] += 2
    
    # Stochastic
    if indicators['stoch_k'][-1] and indicators['stoch_d'][-1]:
        k = indicators['stoch_k'][-1]
        d = indicators['stoch_d'][-1]
        
        if k < 20 and d < 20:
            signals.append({
                'type': 'bullish',
                'indicator': 'Stochastic',
                'message': f"Stochastic Oversold (K:{k:.1f}, D:{d:.1f})",
                'strength': 'medium'
            })
            scores['bullish'] += 2
        elif k > 80 and d > 80:
            signals.append({
                'type': 'bearish',
                'indicator': 'Stochastic',
                'message': f"Stochastic Overbought (K:{k:.1f}, D:{d:.1f})",
                'strength': 'medium'
            })
            scores['bearish'] += 2
    
    # Volume Analysis
    if 'volume' in df.columns:
        avg_volume = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        
        if current_volume > avg_volume * 1.5:
            price_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
            if price_change > 0:
                signals.append({
                    'type': 'bullish',
                    'indicator': 'Volume',
                    'message': f"High Volume Breakout (+{(current_volume/avg_volume-1)*100:.1f}% avg)",
                    'strength': 'high'
                })
                scores['bullish'] += 2
            else:
                signals.append({
                    'type': 'bearish',
                    'indicator': 'Volume',
                    'message': f"High Volume Selloff (+{(current_volume/avg_volume-1)*100:.1f}% avg)",
                    'strength': 'high'
                })
                scores['bearish'] += 2
    
    # Calculate overall sentiment
    total_score = sum(scores.values())
    if total_score > 0:
        bull_pct = (scores['bullish'] / total_score) * 100
        bear_pct = (scores['bearish'] / total_score) * 100
        neutral_pct = (scores['neutral'] / total_score) * 100
    else:
        bull_pct = bear_pct = neutral_pct = 33.33
    
    overall_sentiment = {
        'bullish': bull_pct,
        'bearish': bear_pct,
        'neutral': neutral_pct,
        'recommendation': 'BUY' if bull_pct > 50 else 'SELL' if bear_pct > 50 else 'HOLD'
    }
    
    return signals, overall_sentiment

def calculate_risk_metrics(df, forecast_data):
    """Calculate advanced risk metrics"""
    prices = df['price'].tolist()
    
    # Volatility
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    volatility = stdev(returns) * math.sqrt(252) if len(returns) > 1 else 0  # Annualized
    
    # VaR (Value at Risk) - 95% confidence
    sorted_returns = sorted(returns)
    var_95 = sorted_returns[int(len(sorted_returns) * 0.05)] if len(sorted_returns) > 0 else 0
    
    # Sharpe Ratio (assuming 2% risk-free rate)
    avg_return = mean(returns) if returns else 0
    sharpe = ((avg_return * 252) - 0.02) / volatility if volatility != 0 else 0
    
    # Max Drawdown
    peak = prices[0]
    max_dd = 0
    for price in prices:
        if price > peak:
            peak = price
        dd = (peak - price) / peak
        if dd > max_dd:
            max_dd = dd
    
    # Forecast Risk
    forecast_prices = [f['price'] for f in forecast_data]
    forecast_volatility = stdev(forecast_prices) / mean(forecast_prices) if forecast_prices else 0
    
    return {
        'volatility': volatility * 100,
        'var_95': var_95 * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd * 100,
        'forecast_volatility': forecast_volatility * 100
    }

# ============= STREAMLIT UI =============

# Sidebar
st.sidebar.title("🔍 AI Crypto Analyzer")

# Trending Coins
with st.sidebar.expander("🔥 Trending Now", expanded=True):
    trending = get_trending_coins()
    if trending:
        for coin in trending[:5]:
            item = coin.get('item', {})
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{item.get('name', 'N/A')}** ({item.get('symbol', 'N/A').upper()})")
            with col2:
                rank = item.get('market_cap_rank', 'N/A')
                st.write(f"#{rank}")

# Search Box
search_query = st.sidebar.text_input("🔎 Cari coin:", placeholder="Bitcoin, ETH, BNB...")

selected_coin = None
coin_id = None

if search_query:
    with st.spinner("🔍 Searching..."):
        results = search_coins(search_query)
        
        if results:
            st.sidebar.subheader(f"📊 Hasil ({len(results)} coins)")
            
            for coin in results[:10]:
                col1, col2, col3 = st.sidebar.columns([1, 3, 2])
                
                with col1:
                    if coin.get('thumb'):
                        st.image(coin['thumb'], width=30)
                
                with col2:
                    st.write(f"**{coin.get('name', 'N/A')}**")
                    st.caption(coin.get('symbol', 'N/A').upper())
                
                with col3:
                    if st.button("Pilih", key=coin['id']):
                        selected_coin = coin
                        coin_id = coin['id']
                
                st.sidebar.divider()
        else:
            st.sidebar.warning("Tidak ada hasil.")

# Settings
st.sidebar.subheader("⚙️ Settings")

# Currency Selection
currency = st.sidebar.selectbox(
    "💱 Currency:",
    options=["usd", "idr"],
    format_func=lambda x: "USD ($)" if x == "usd" else "IDR (Rp)",
    index=0
)

# Currency symbol and format
currency_symbol = "$" if currency == "usd" else "Rp"
currency_rate = 1 if currency == "usd" else 15800  # Approximate USD to IDR rate

timeframe = st.sidebar.selectbox(
    "Data Period:",
    options=[7, 14, 30, 60, 90, 180, 365],
    format_func=lambda x: f"{x} hari",
    index=2
)

forecast_days = st.sidebar.slider("Forecast Days:", 1, 30, 7)

show_advanced = st.sidebar.checkbox("🧠 Show Advanced Indicators", value=True)

# Main Content
if coin_id or selected_coin:
    if not coin_id and selected_coin:
        coin_id = selected_coin['id']
    
    st.markdown(f"""
    <div class="success-box">
        <strong>✅ Analyzing:</strong> {selected_coin.get('name', coin_id)} ({selected_coin.get('symbol', '').upper()})
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch data
    with st.spinner("🤖 AI Processing... Analyzing patterns, calculating predictions..."):
        df = get_coin_data(coin_id, timeframe, currency)
        coin_info = get_coin_info(coin_id)
        global_data = get_global_crypto_data()
    
    if df is not None and not df.empty and coin_info:
        # Current Info
        st.subheader("💎 Market Overview")
        
        market_data = coin_info.get('market_data', {})
        current_price = market_data.get('current_price', {}).get(currency, 0)
        price_change_24h = market_data.get('price_change_percentage_24h', 0)
        price_change_7d = market_data.get('price_change_percentage_7d', 0)
        market_cap = market_data.get('market_cap', {}).get(currency, 0)
        market_cap_rank = coin_info.get('market_cap_rank', 'N/A')
        volume_24h = market_data.get('total_volume', {}).get(currency, 0)
        circulating_supply = market_data.get('circulating_supply', 0)
        ath = market_data.get('ath', {}).get(currency, 0)
        ath_change = market_data.get('ath_change_percentage', {}).get(currency, 0)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("💰 Price", f"{currency_symbol}{current_price:,.2f}")
        
        with col2:
            st.metric("📈 24h", f"{price_change_24h:.2f}%", 
                     delta=f"{price_change_24h:.2f}%")
        
        with col3:
            st.metric("📊 7d", f"{price_change_7d:.2f}%",
                     delta=f"{price_change_7d:.2f}%")
        
        with col4:
            st.metric("🏆 Rank", f"#{market_cap_rank}")
        
        with col5:
            if currency == "usd":
                st.metric("💎 Market Cap", 
                         f"${market_cap/1e9:.2f}B" if market_cap > 1e9 else f"${market_cap/1e6:.2f}M")
            else:
                st.metric("💎 Market Cap", 
                         f"Rp{market_cap/1e12:.2f}T" if market_cap > 1e12 else f"Rp{market_cap/1e9:.2f}B")
        
        with col6:
            if currency == "usd":
                st.metric("📦 Volume 24h", 
                         f"${volume_24h/1e9:.2f}B" if volume_24h > 1e9 else f"${volume_24h/1e6:.2f}M")
            else:
                st.metric("📦 Volume 24h", 
                         f"Rp{volume_24h/1e12:.2f}T" if volume_24h > 1e12 else f"Rp{volume_24h/1e9:.2f}B")
        
        # Calculate indicators
        indicators = calculate_advanced_indicators(df)
        
        # Generate signals
        signals, sentiment = generate_advanced_signals(df, indicators, coin_info, currency_symbol)
        
        # AI FORECAST SECTION
        st.markdown("---")
        st.subheader("🤖 AI-Powered Ensemble Forecast")
        
        with st.spinner("🧠 Running AI models... (LSTM + Trend + EMA)"):
            forecast_data = ensemble_forecast(df, forecast_days)
        
        if forecast_data:
            # Forecast Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            last_forecast = forecast_data[-1]
            first_forecast = forecast_data[0]
            avg_confidence = mean([f['confidence'] for f in forecast_data])
            
            with col1:
                change_pct = ((last_forecast['price'] - current_price) / current_price) * 100
                st.metric(
                    f"🎯 Day {forecast_days} Prediction",
                    f"{currency_symbol}{last_forecast['price']:,.2f}",
                    f"{change_pct:+.2f}%"
                )
            
            with col2:
                confidence_class = "confidence-high" if avg_confidence > 0.7 else "confidence-medium" if avg_confidence > 0.5 else "confidence-low"
                st.markdown(f"**📊 Avg Confidence**")
                st.markdown(f"<h2 class='{confidence_class}'>{avg_confidence*100:.1f}%</h2>", unsafe_allow_html=True)
            
            with col3:
                price_range = last_forecast['upper'] - last_forecast['lower']
                st.metric("📏 Price Range", f"{currency_symbol}{price_range:,.2f}")
            
            with col4:
                direction = "🟢 BULLISH" if change_pct > 0 else "🔴 BEARISH"
                st.markdown(f"**🎲 Direction**")
                st.markdown(f"<h2>{direction}</h2>", unsafe_allow_html=True)
            
            # Forecast Chart
            st.subheader("📈 Price Forecast Visualization")
            
            forecast_fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('AI Price Forecast', 'Confidence Score'),
                row_heights=[0.7, 0.3]
            )
            
            # Historical prices
            forecast_fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['price'],
                    name='Historical',
                    line=dict(color='#667eea', width=2),
                    hovertemplate=f'<b>Historical</b><br>Date: %{{x}}<br>Price: {currency_symbol}%{{y:,.2f}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Forecast
            last_date = df['timestamp'].iloc[-1]
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_prices = [f['price'] for f in forecast_data]
            forecast_upper = [f['upper'] for f in forecast_data]
            forecast_lower = [f['lower'] for f in forecast_data]
            
            # Confidence interval
            forecast_fig.add_trace(
                go.Scatter(
                    x=forecast_dates + forecast_dates[::-1],
                    y=forecast_upper + forecast_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 165, 0, 0.2)',
                    line=dict(width=0),
                    name='Confidence Interval',
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # Forecast line
            forecast_fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast_prices,
                    name='AI Forecast',
                    line=dict(color='orange', width=3, dash='dash'),
                    hovertemplate=f'<b>Forecast</b><br>Date: %{{x}}<br>Price: {currency_symbol}%{{y:,.2f}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Confidence scores
            confidence_scores = [f['confidence'] * 100 for f in forecast_data]
            forecast_fig.add_trace(
                go.Bar(
                    x=forecast_dates,
                    y=confidence_scores,
                    name='Confidence',
                    marker_color='lightblue',
                    hovertemplate='<b>Confidence</b><br>Date: %{x}<br>Score: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            forecast_fig.update_layout(
                height=700,
                hovermode='x unified',
                template='plotly_white',
                showlegend=True
            )
            
            forecast_fig.update_yaxes(title_text=f"Price ({currency.upper()})", row=1, col=1)
            forecast_fig.update_yaxes(title_text="Confidence %", row=2, col=1)
            
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            # Forecast Table
            st.subheader("📋 Detailed Forecast Table")
            
            forecast_table = []
            for i, (date, f) in enumerate(zip(forecast_dates, forecast_data)):
                change = ((f['price'] - current_price) / current_price) * 100
                forecast_table.append({
                    'Day': i + 1,
                    'Date': date.strftime('%Y-%m-%d'),
                    'Predicted Price': f'{currency_symbol}{f["price"]:,.2f}',
                    'Lower Bound': f'{currency_symbol}{f["lower"]:,.2f}',
                    'Upper Bound': f'{currency_symbol}{f["upper"]:,.2f}',
                    'Change %': f'{change:+.2f}%',
                    'Confidence': f'{f["confidence"]*100:.1f}%'
                })
            
            st.dataframe(pd.DataFrame(forecast_table), use_container_width=True)
            
            # Risk Analysis
            st.subheader("⚠️ Risk Assessment & Metrics")
            
            risk_metrics = calculate_risk_metrics(df, forecast_data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                vol_color = "🟢" if risk_metrics['volatility'] < 50 else "🟡" if risk_metrics['volatility'] < 100 else "🔴"
                st.metric(f"{vol_color} Annualized Volatility", f"{risk_metrics['volatility']:.2f}%")
            
            with col2:
                st.metric("📉 Value at Risk (95%)", f"{risk_metrics['var_95']:.2f}%")
            
            with col3:
                sharpe_color = "🟢" if risk_metrics['sharpe_ratio'] > 1 else "🟡" if risk_metrics['sharpe_ratio'] > 0 else "🔴"
                st.metric(f"{sharpe_color} Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
            
            with col4:
                st.metric("📊 Max Drawdown", f"{risk_metrics['max_drawdown']:.2f}%")
            
            # Risk interpretation
            risk_level = "LOW" if risk_metrics['volatility'] < 50 else "MEDIUM" if risk_metrics['volatility'] < 100 else "HIGH"
            risk_color = "green" if risk_level == "LOW" else "orange" if risk_level == "MEDIUM" else "red"
            
            st.markdown(f"""
            **Overall Risk Level:** <span style='color:{risk_color}; font-size:1.5em; font-weight:bold;'>{risk_level}</span>
            """, unsafe_allow_html=True)
        
        # Trading Signals
        st.markdown("---")
        st.subheader("🎯 AI Trading Signals & Analysis")
        
        # Overall Sentiment
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Overall Market Sentiment")
            
            fig_sentiment = go.Figure()
            
            fig_sentiment.add_trace(go.Bar(
                x=[sentiment['bullish'], sentiment['bearish'], sentiment['neutral']],
                y=['Bullish', 'Bearish', 'Neutral'],
                orientation='h',
                marker=dict(
                    color=['green', 'red', 'gray'],
                    line=dict(color='white', width=2)
                ),
                text=[f"{sentiment['bullish']:.1f}%", f"{sentiment['bearish']:.1f}%", f"{sentiment['neutral']:.1f}%"],
                textposition='auto',
            ))
            
            fig_sentiment.update_layout(
                title="Sentiment Distribution",
                xaxis_title="Percentage",
                height=250,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            st.markdown("### 🎲 Recommendation")
            recommendation = sentiment['recommendation']
            rec_color = "green" if recommendation == "BUY" else "red" if recommendation == "SELL" else "orange"
            
            st.markdown(f"""
            <div style='text-align:center; padding:30px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius:15px;'>
                <h1 style='color:{rec_color}; font-size:3em; margin:0;'>{recommendation}</h1>
                <p style='margin-top:10px;'>Based on {len(signals)} indicators</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Individual Signals
        st.markdown("### 📡 Individual Signals")
        
        bullish_signals = [s for s in signals if s['type'] == 'bullish']
        bearish_signals = [s for s in signals if s['type'] == 'bearish']
        neutral_signals = [s for s in signals if s['type'] == 'neutral']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🟢 Bullish Signals")
            for sig in bullish_signals:
                strength_badge = "🔥" if sig['strength'] == 'high' else "⚡" if sig['strength'] == 'medium' else "💫"
                st.markdown(f"""
                <div class="signal-bullish">
                    <strong>{strength_badge} {sig['indicator']}</strong><br>
                    {sig['message']}
                </div>
                """, unsafe_allow_html=True)
            
            if not bullish_signals:
                st.info("No bullish signals")
        
        with col2:
            st.markdown("#### 🔴 Bearish Signals")
            for sig in bearish_signals:
                strength_badge = "🔥" if sig['strength'] == 'high' else "⚡" if sig['strength'] == 'medium' else "💫"
                st.markdown(f"""
                <div class="signal-bearish">
                    <strong>{strength_badge} {sig['indicator']}</strong><br>
                    {sig['message']}
                </div>
                """, unsafe_allow_html=True)
            
            if not bearish_signals:
                st.info("No bearish signals")
        
        with col3:
            st.markdown("#### 🟡 Neutral Signals")
            for sig in neutral_signals:
                st.markdown(f"""
                <div class="signal-neutral">
                    <strong>💫 {sig['indicator']}</strong><br>
                    {sig['message']}
                </div>
                """, unsafe_allow_html=True)
            
            if not neutral_signals:
                st.info("No neutral signals")
        
        # Technical Charts
        if show_advanced:
            st.markdown("---")
            st.subheader("📊 Advanced Technical Analysis")
            
            # Create comprehensive chart
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Bollinger Bands', 'MACD', 'RSI & Stochastic', 'Volume & OBV'),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Price & Bollinger Bands
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['price'], name='Price',
                          line=dict(color='#667eea', width=2)),
                row=1, col=1
            )
            
            if indicators['bb_upper'][-1]:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=indicators['bb_upper'],
                              name='BB Upper', line=dict(color='red', dash='dot')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=indicators['bb_lower'],
                              name='BB Lower', line=dict(color='green', dash='dot'),
                              fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
                    row=1, col=1
                )
            
            if indicators['sma_50'][-1]:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=indicators['sma_50'],
                              name='SMA 50', line=dict(color='orange', dash='dash')),
                    row=1, col=1
                )
            
            # MACD
            if indicators['macd'][-1]:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=indicators['macd'],
                              name='MACD', line=dict(color='blue')),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=indicators['macd_signal'],
                              name='Signal', line=dict(color='red', dash='dash')),
                    row=2, col=1
                )
                
                if indicators['macd_histogram'][-1]:
                    colors = ['green' if h and h > 0 else 'red' for h in indicators['macd_histogram']]
                    fig.add_trace(
                        go.Bar(x=df['timestamp'], y=indicators['macd_histogram'],
                              name='Histogram', marker_color=colors),
                        row=2, col=1
                    )
            
            # RSI & Stochastic
            if indicators['rsi'][-1]:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=indicators['rsi'],
                              name='RSI', line=dict(color='purple')),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
            
            if indicators['stoch_k'][-1]:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=indicators['stoch_k'],
                              name='Stoch %K', line=dict(color='blue', width=1)),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=indicators['stoch_d'],
                              name='Stoch %D', line=dict(color='red', dash='dash', width=1)),
                    row=3, col=1
                )
            
            # Volume & OBV
            if 'volume' in df.columns:
                colors = ['green' if i > 0 and df['price'].iloc[i] > df['price'].iloc[i-1] else 'red'
                         for i in range(len(df))]
                
                fig.add_trace(
                    go.Bar(x=df['timestamp'], y=df['volume'],
                          name='Volume', marker_color=colors),
                    row=4, col=1
                )
            
            if 'obv' in indicators and indicators['obv'][-1]:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=indicators['obv'],
                              name='OBV', line=dict(color='orange'), yaxis='y2'),
                    row=4, col=1
                )
            
            fig.update_layout(
                height=1000,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            fig.update_yaxes(title_text=f"Price ({currency.upper()})", row=1, col=1)
            fig.update_yaxes(title_text="MACD", row=2, col=1)
            fig.update_yaxes(title_text="RSI/Stoch", row=3, col=1)
            fig.update_yaxes(title_text="Volume", row=4, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional Info
        with st.expander("ℹ️ Coin Information & Links"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Name:** {coin_info.get('name', 'N/A')}")
                st.write(f"**Symbol:** {coin_info.get('symbol', 'N/A').upper()}")
                st.write(f"**Market Cap Rank:** #{market_cap_rank}")
                st.write(f"**All-Time High:** {currency_symbol}{ath:,.2f} ({ath_change:+.2f}%)")
                
                if circulating_supply:
                    st.write(f"**Circulating Supply:** {circulating_supply:,.0f}")
                
                max_supply = market_data.get('max_supply')
                if max_supply:
                    st.write(f"**Max Supply:** {max_supply:,.0f}")
            
            with col2:
                links = coin_info.get('links', {})
                
                if links.get('homepage') and links['homepage'][0]:
                    st.write(f"**Website:** {links['homepage'][0]}")
                
                if links.get('blockchain_site'):
                    explorers = [site for site in links['blockchain_site'] if site]
                    if explorers:
                        st.write(f"**Explorer:** {explorers[0]}")
                
                if links.get('subreddit_url'):
                    st.write(f"**Reddit:** {links['subreddit_url']}")
                
                community = coin_info.get('community_data', {})
                if community.get('twitter_followers'):
                    st.write(f"**Twitter Followers:** {community['twitter_followers']:,}")
            
            if coin_info.get('description', {}).get('en'):
                st.markdown("**Description:**")
                description = coin_info['description']['en'][:800] + "..."
                st.write(description)
    
    else:
        st.error("❌ Failed to fetch data. Please try again or select another coin.")

else:
    # Welcome Screen
    st.markdown("""
    <div class="success-box">
        <h2>👋 Welcome to AI Crypto Forecast Pro!</h2>
        <p>Advanced cryptocurrency analysis powered by AI & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🚀 Features:
    
    #### 🤖 **AI-Powered Forecasting:**
    - **Ensemble Method**: Combines LSTM-inspired predictions, trend analysis, and EMA
    - **Confidence Scoring**: Each prediction comes with accuracy confidence
    - **Multiple Timeframes**: Forecast 1-30 days ahead
    - **Risk Assessment**: Comprehensive risk metrics (VaR, Sharpe, Volatility)
    
    #### 📊 **Advanced Technical Analysis:**
    - **10+ Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR, OBV
    - **Trading Signals**: Automated buy/sell/hold recommendations
    - **Sentiment Analysis**: Bullish/Bearish scoring system
    - **Volume Analysis**: Track money flow and market interest
    
    #### 📈 **Visual Analytics:**
    - Interactive charts with zoom & pan
    - Price forecast with confidence intervals
    - Multi-indicator overlay charts
    - Real-time data from CoinGecko
    
    ### 📖 How to Use:
    1. **Search** for any cryptocurrency in the sidebar
    2. **Select** your coin from search results
    3. **Configure** timeframe and forecast period
    4. **Analyze** AI predictions, signals, and risk metrics
    5. **Make informed decisions** based on comprehensive data
    
    ### ⚠️ Important Notes:
    - Target accuracy: **60-65%** (realistic for crypto markets)
    - Use **multiple confirmations** before trading
    - Always apply **risk management**
    - This is **NOT financial advice**
    - Past performance ≠ future results
    
    ### 🎯 Best Practices:
    - Compare multiple coins before deciding
    - Check confidence scores - higher is better
    - Look for signal alignment (multiple bullish/bearish)
    - Consider overall market sentiment
    - Use stop-losses and position sizing
    """)
    
    # Global Crypto Stats
    global_data = get_global_crypto_data()
    
    if global_data:
        st.subheader("🌍 Global Crypto Market")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_market_cap = global_data.get('total_market_cap', {}).get('usd', 0)
        total_volume = global_data.get('total_volume', {}).get('usd', 0)
        btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
        eth_dominance = global_data.get('market_cap_percentage', {}).get('eth', 0)
        
        with col1:
            st.metric("Total Market Cap", f"${total_market_cap/1e12:.2f}T")
        
        with col2:
            st.metric("24h Volume", f"${total_volume/1e9:.2f}B")
        
        with col3:
            st.metric("BTC Dominance", f"{btc_dominance:.2f}%")
        
        with col4:
            st.metric("ETH Dominance", f"{eth_dominance:.2f}%")
    
    # Show trending as examples
    st.subheader("🔥 Currently Trending")
    trending = get_trending_coins()
    
    if trending:
        cols = st.columns(3)
        for idx, coin in enumerate(trending[:6]):
            item = coin.get('item', {})
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{item.get('name', 'N/A')}</h4>
                    <p><strong>Symbol:</strong> {item.get('symbol', 'N/A').upper()}</p>
                    <p><strong>Rank:</strong> #{item.get('market_cap_rank', 'N/A')}</p>
                    <p><strong>Score:</strong> {item.get('score', 0)}</p>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
    <h3>🤖 AI Crypto Forecast Pro</h3>
    <p>📊 Powered by CoinGecko API | 🧠 Advanced ML Algorithms | ⚠️ Educational Purposes Only</p>
    <p><strong>Target Accuracy: 60-65%</strong> | Always use risk management | Not financial advice</p>
</div>
""", unsafe_allow_html=True)
