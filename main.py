import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from statistics import mean, stdev
import math

# Konfigurasi halaman
st.set_page_config(
    page_title="Crypto Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# CSS Custom
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚀 Crypto Forecast Dashboard</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <strong>⚠️ DISCLAIMER PENTING:</strong><br>
    • Forecast cryptocurrency memiliki akurasi rendah (40-60% untuk jangka pendek)<br>
    • Ini BUKAN saran investasi - gunakan untuk referensi saja<br>
    • Pasar crypto sangat volatil dan unpredictable<br>
    • Selalu lakukan riset sendiri (DYOR) sebelum trading
</div>
""", unsafe_allow_html=True)

# Fungsi API CoinGecko
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
def get_coin_list():
    """Ambil list semua coin"""
    try:
        url = "https://api.coingecko.com/api/v3/coins/list"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error getting coin list: {e}")
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
        st.error(f"Error getting trending coins: {e}")
        return []

@st.cache_data(ttl=300)
def get_coin_data(coin_id, days=30):
    """Ambil data historical coin"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily" if days > 1 else "hourly"
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            if volumes:
                df_vol = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                df_vol['timestamp'] = pd.to_datetime(df_vol['timestamp'], unit='ms')
                df = df.merge(df_vol, on='timestamp', how='left')
            
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
            "developer_data": "false"
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error getting coin info: {e}")
        return None

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

def simple_forecast(df, days_ahead=7):
    """Forecast sederhana menggunakan moving average dan trend"""
    prices = df['price'].tolist()
    
    # Hitung trend
    recent_prices = prices[-14:]
    if len(recent_prices) < 2:
        return None
    
    # Linear trend
    x = list(range(len(recent_prices)))
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(recent_prices)
    sum_xy = sum(x[i] * recent_prices[i] for i in range(n))
    sum_x2 = sum(i**2 for i in x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    intercept = (sum_y - slope * sum_x) / n
    
    # Forecast
    last_price = prices[-1]
    forecasts = []
    
    for i in range(1, days_ahead + 1):
        forecast_price = intercept + slope * (len(recent_prices) + i)
        
        # Tambahkan volatilitas
        volatility = stdev(recent_prices) if len(recent_prices) > 1 else 0
        
        forecasts.append({
            'day': i,
            'price': forecast_price,
            'upper': forecast_price + (volatility * 1.5),
            'lower': max(0, forecast_price - (volatility * 1.5)),
            'volatility': volatility
        })
    
    return forecasts

def calculate_technical_indicators(df):
    """Hitung indikator teknikal"""
    prices = df['price'].tolist()
    
    indicators = {}
    
    # Moving Averages
    indicators['sma_7'] = calculate_sma(prices, 7)
    indicators['sma_25'] = calculate_sma(prices, 25)
    indicators['ema_12'] = calculate_ema(prices, 12)
    indicators['ema_26'] = calculate_ema(prices, 26)
    
    # RSI
    indicators['rsi'] = calculate_rsi(prices, 14)
    
    # MACD
    if indicators['ema_12'][-1] and indicators['ema_26'][-1]:
        macd = [
            (indicators['ema_12'][i] - indicators['ema_26'][i]) 
            if indicators['ema_12'][i] and indicators['ema_26'][i] else None
            for i in range(len(prices))
        ]
        indicators['macd'] = macd
    
    return indicators

def analyze_signals(df, indicators, coin_info):
    """Analisis sinyal trading"""
    signals = []
    current_price = df['price'].iloc[-1]
    
    # RSI Analysis
    if indicators['rsi'][-1]:
        rsi = indicators['rsi'][-1]
        if rsi < 30:
            signals.append(("🟢 OVERSOLD", f"RSI {rsi:.1f} - Potensi Buy Signal"))
        elif rsi > 70:
            signals.append(("🔴 OVERBOUGHT", f"RSI {rsi:.1f} - Potensi Sell Signal"))
        else:
            signals.append(("🟡 NEUTRAL", f"RSI {rsi:.1f} - Tidak ada sinyal kuat"))
    
    # Moving Average Crossover
    if indicators['sma_7'][-1] and indicators['sma_25'][-1]:
        sma7 = indicators['sma_7'][-1]
        sma25 = indicators['sma_25'][-1]
        
        if sma7 > sma25:
            signals.append(("🟢 BULLISH", "SMA 7 di atas SMA 25 - Trend Naik"))
        else:
            signals.append(("🔴 BEARISH", "SMA 7 di bawah SMA 25 - Trend Turun"))
    
    # Price vs MA
    if indicators['sma_25'][-1]:
        if current_price > indicators['sma_25'][-1]:
            signals.append(("📈 ABOVE MA", "Harga di atas MA 25 - Momentum Positif"))
        else:
            signals.append(("📉 BELOW MA", "Harga di bawah MA 25 - Momentum Negatif"))
    
    # Volatility
    prices = df['price'].tail(14).tolist()
    if len(prices) > 1:
        volatility = (stdev(prices) / mean(prices)) * 100
        if volatility > 10:
            signals.append(("⚠️ HIGH VOLATILITY", f"{volatility:.1f}% - Risk Tinggi"))
        elif volatility < 3:
            signals.append(("😴 LOW VOLATILITY", f"{volatility:.1f}% - Pergerakan Lambat"))
    
    return signals

# Sidebar - Search & Select
st.sidebar.title("🔍 Cari Cryptocurrency")

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
search_query = st.sidebar.text_input("Cari coin (nama/symbol):", placeholder="Bitcoin, ETH, BNB...")

selected_coin = None
coin_id = None

if search_query:
    with st.spinner("Mencari..."):
        results = search_coins(search_query)
        
        if results:
            st.sidebar.subheader(f"📊 Hasil ({len(results)} coins)")
            
            # Display search results
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
            st.sidebar.warning("Tidak ada hasil. Coba keyword lain.")

# Timeframe selection
st.sidebar.subheader("⏱️ Timeframe")
timeframe = st.sidebar.selectbox(
    "Pilih periode data:",
    options=[7, 14, 30, 90, 180, 365],
    format_func=lambda x: f"{x} hari",
    index=2
)

forecast_days = st.sidebar.slider("Forecast berapa hari ke depan?", 1, 30, 7)

# Main Content
if coin_id or selected_coin:
    if not coin_id and selected_coin:
        coin_id = selected_coin['id']
    
    st.success(f"✅ Menganalisis: **{selected_coin.get('name', coin_id)}** ({selected_coin.get('symbol', '').upper()})")
    
    # Fetch data
    with st.spinner("Mengambil data dari CoinGecko..."):
        df = get_coin_data(coin_id, timeframe)
        coin_info = get_coin_info(coin_id)
    
    if df is not None and not df.empty and coin_info:
        # Current Info
        st.subheader("💰 Informasi Terkini")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        market_data = coin_info.get('market_data', {})
        current_price = market_data.get('current_price', {}).get('usd', 0)
        price_change_24h = market_data.get('price_change_percentage_24h', 0)
        market_cap = market_data.get('market_cap', {}).get('usd', 0)
        volume_24h = market_data.get('total_volume', {}).get('usd', 0)
        circulating_supply = market_data.get('circulating_supply', 0)
        
        with col1:
            st.metric("Harga (USD)", f"${current_price:,.2f}")
        
        with col2:
            st.metric("24h Change", f"{price_change_24h:.2f}%", 
                     delta=f"{price_change_24h:.2f}%")
        
        with col3:
            st.metric("Market Cap", f"${market_cap/1e9:.2f}B" if market_cap > 1e9 else f"${market_cap/1e6:.2f}M")
        
        with col4:
            st.metric("Volume 24h", f"${volume_24h/1e9:.2f}B" if volume_24h > 1e9 else f"${volume_24h/1e6:.2f}M")
        
        with col5:
            st.metric("Circulating Supply", f"{circulating_supply/1e6:.2f}M" if circulating_supply > 1e6 else f"{circulating_supply:,.0f}")
        
        # Calculate indicators
        indicators = calculate_technical_indicators(df)
        
        # Charts
        st.subheader("📈 Historical Chart & Technical Analysis")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Moving Averages', 'Volume', 'RSI'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price & MA
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['price'], name='Price', 
                      line=dict(color='#667eea', width=2)),
            row=1, col=1
        )
        
        if indicators['sma_7'][-1]:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=indicators['sma_7'], 
                          name='SMA 7', line=dict(color='orange', dash='dash')),
                row=1, col=1
            )
        
        if indicators['sma_25'][-1]:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=indicators['sma_25'], 
                          name='SMA 25', line=dict(color='red', dash='dash')),
                row=1, col=1
            )
        
        # Volume
        if 'volume' in df.columns:
            colors = ['red' if df['price'].iloc[i] < df['price'].iloc[i-1] else 'green' 
                     for i in range(1, len(df))]
            colors.insert(0, 'gray')
            
            fig.add_trace(
                go.Bar(x=df['timestamp'], y=df['volume'], name='Volume',
                      marker_color=colors),
                row=2, col=1
            )
        
        # RSI
        if indicators['rsi'][-1]:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=indicators['rsi'], 
                          name='RSI', line=dict(color='purple')),
                row=3, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast
        st.subheader(f"🔮 Forecast {forecast_days} Hari Ke Depan")
        
        forecasts = simple_forecast(df, forecast_days)
        
        if forecasts:
            # Forecast Chart
            forecast_fig = go.Figure()
            
            # Historical
            forecast_fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['price'], 
                          name='Historical', line=dict(color='#667eea', width=2))
            )
            
            # Forecast
            last_date = df['timestamp'].iloc[-1]
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_prices = [f['price'] for f in forecasts]
            forecast_upper = [f['upper'] for f in forecasts]
            forecast_lower = [f['lower'] for f in forecasts]
            
            forecast_fig.add_trace(
                go.Scatter(x=forecast_dates, y=forecast_prices, 
                          name='Forecast', line=dict(color='orange', width=2, dash='dash'))
            )
            
            # Confidence interval
            forecast_fig.add_trace(
                go.Scatter(x=forecast_dates, y=forecast_upper, 
                          name='Upper Bound', line=dict(width=0), 
                          showlegend=False, hoverinfo='skip')
            )
            
            forecast_fig.add_trace(
                go.Scatter(x=forecast_dates, y=forecast_lower, 
                          name='Confidence Interval', 
                          fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)',
                          line=dict(width=0), mode='lines')
            )
            
            forecast_fig.update_layout(
                title=f"Price Forecast - {forecast_days} Days",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            # Forecast Table
            st.subheader("📊 Detail Forecast")
            
            forecast_df = pd.DataFrame(forecasts)
            forecast_df['date'] = forecast_dates
            forecast_df['change_pct'] = ((forecast_df['price'] - current_price) / current_price * 100)
            
            forecast_display = forecast_df[['date', 'price', 'lower', 'upper', 'change_pct', 'volatility']].copy()
            forecast_display.columns = ['Tanggal', 'Prediksi Harga', 'Lower Bound', 'Upper Bound', 'Change %', 'Volatilitas']
            
            st.dataframe(
                forecast_display.style.format({
                    'Prediksi Harga': '${:.2f}',
                    'Lower Bound': '${:.2f}',
                    'Upper Bound': '${:.2f}',
                    'Change %': '{:.2f}%',
                    'Volatilitas': '${:.2f}'
                }),
                use_container_width=True
            )
            
            # Summary
            avg_forecast = mean(forecast_prices)
            last_forecast = forecast_prices[-1]
            change_pct = ((last_forecast - current_price) / current_price) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Prediksi Hari ke-{forecast_days}", f"${last_forecast:,.2f}", 
                         f"{change_pct:+.2f}%")
            with col2:
                st.metric("Rata-rata Forecast", f"${avg_forecast:,.2f}")
            with col3:
                avg_volatility = mean([f['volatility'] for f in forecasts])
                st.metric("Avg Volatilitas", f"${avg_volatility:.2f}")
        
        # Trading Signals
        st.subheader("🎯 Analisis & Sinyal Trading")
        
        signals = analyze_signals(df, indicators, coin_info)
        
        for signal_type, description in signals:
            st.info(f"**{signal_type}**: {description}")
        
        # Risk Assessment
        st.subheader("⚠️ Risk Assessment")
        
        prices_14d = df['price'].tail(14).tolist()
        if len(prices_14d) > 1:
            volatility = (stdev(prices_14d) / mean(prices_14d)) * 100
            
            risk_level = "🟢 LOW" if volatility < 3 else "🟡 MEDIUM" if volatility < 10 else "🔴 HIGH"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("14-Day Volatility", f"{volatility:.2f}%")
            with col2:
                st.metric("Risk Level", risk_level)
        
        # Additional Info
        with st.expander("ℹ️ Informasi Tambahan"):
            st.write(f"**Nama:** {coin_info.get('name', 'N/A')}")
            st.write(f"**Symbol:** {coin_info.get('symbol', 'N/A').upper()}")
            st.write(f"**Rank:** #{coin_info.get('market_cap_rank', 'N/A')}")
            
            if coin_info.get('description', {}).get('en'):
                description = coin_info['description']['en'][:500] + "..."
                st.write(f"**Deskripsi:** {description}")
            
            links = coin_info.get('links', {})
            if links.get('homepage'):
                st.write(f"**Website:** {links['homepage'][0]}")
    
    else:
        st.error("❌ Gagal mengambil data. Coba lagi atau pilih coin lain.")

else:
    # Welcome Screen
    st.info("👈 Gunakan sidebar untuk mencari cryptocurrency dan mulai analisis!")
    
    st.markdown("""
    ### 📖 Cara Menggunakan:
    1. **Cari coin** di sidebar menggunakan nama atau symbol (contoh: Bitcoin, ETH, BNB)
    2. **Pilih coin** dari hasil pencarian
    3. **Atur timeframe** dan periode forecast
    4. **Analisis** chart, indikator teknikal, dan forecast
    
    ### 🎯 Fitur:
    - ✅ Data real-time dari CoinGecko API
    - ✅ Technical indicators (SMA, EMA, RSI, MACD)
    - ✅ Price forecast dengan confidence interval
    - ✅ Trading signals & risk assessment
    - ✅ Trending coins
    - ✅ Volume analysis
    
    ### 📊 Indikator yang Tersedia:
    - **SMA (Simple Moving Average)**: Trend jangka pendek & menengah
    - **RSI (Relative Strength Index)**: Overbought/oversold conditions
    - **Volume Analysis**: Konfirmasi pergerakan harga
    - **Volatility**: Ukuran risk & movement
    
    ### ⚠️ Catatan:
    - Forecast menggunakan trend analysis & moving averages
    - Akurasi terbatas untuk pasar crypto yang sangat volatil
    - Gunakan sebagai referensi, bukan keputusan final
    - Always DYOR (Do Your Own Research)!
    """)
    
    # Show trending as examples
    st.subheader("🔥 Trending Coins Saat Ini")
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
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("📊 Data dari CoinGecko API | ⚠️ Bukan saran keuangan | 🔬 Untuk edukasi & riset saja")
st.caption("Made with ❤️ using Streamlit")
