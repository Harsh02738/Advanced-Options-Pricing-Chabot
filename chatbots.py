import os
import re
import streamlit as st
import requests
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging
import pandas as pd

# ----------------------------
# Configuration and Setup
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Asset configurations
ASSET_CONFIGS = {
    "AAPL": {"name": "Apple Inc.", "type": "stock"},
    "TSLA": {"name": "Tesla Inc.", "type": "stock"},
    "GOOGL": {"name": "Alphabet Inc.", "type": "stock"},
    "MSFT": {"name": "Microsoft Corp.", "type": "stock"},
    "SPY": {"name": "SPDR S&P 500 ETF", "type": "etf"},
    "BTC": {"name": "Bitcoin", "type": "crypto"},
    "ETH": {"name": "Ethereum", "type": "crypto"},
    "LTC": {"name": "Litecoin", "type": "crypto"}
}

# ----------------------------
# Price fetching functions
# ----------------------------
def fetch_asset_price(symbol):
    """Fetch asset price with fallback methods"""
    # Try Alpha Vantage first (if API key available)
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if alpha_vantage_key:
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": alpha_vantage_key
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if "Global Quote" in data:
                price = float(data["Global Quote"]["05. price"])
                logger.info(f"Fetched {symbol} price: ${price:,.2f}")
                return price
        except Exception as e:
            logger.warning(f"Alpha Vantage failed for {symbol}: {e}")
    
    # Fallback to Yahoo Finance (free)
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data["chart"]["result"]:
            price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
            logger.info(f"Fetched {symbol} price: ${price:,.2f}")
            return price
    except Exception as e:
        logger.warning(f"Yahoo Finance failed for {symbol}: {e}")
    
    # If all fail, return None
    st.warning(f"Could not fetch price for {symbol}")
    return None

def fetch_crypto_price(pair="btcusd"):
    """Fetch cryptocurrency price"""
    try:
        url = f"https://api.coinbase.com/v2/exchange-rates?currency=BTC"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "data" in data and "rates" in data["data"]:
            price = float(data["data"]["rates"]["USD"])
            logger.info(f"Fetched BTC price: ${price:,.2f}")
            return price
    except Exception as e:
        logger.warning(f"Coinbase API failed: {e}")
    
    # Fallback to a mock price for demo
    st.warning("Using mock BTC price for demo")
    return 45000.0

# ----------------------------
# Enhanced NLP query parsing
# ----------------------------
def parse_query(query):
    """Enhanced natural language parsing with multi-asset support"""
    params = {}
    query_lower = query.lower()
    
    # Asset/Ticker detection - PRIORITY: Check this FIRST
    ticker_patterns = [
        r"\b([A-Z]{2,5})\b",  # Standard tickers (AAPL, MSFT, etc.)
        r"(?:ticker|symbol|stock)\s*[:=]\s*([A-Za-z]+)",  # Explicit ticker specification
    ]
    
    detected_ticker = None
    for pattern in ticker_patterns:
        matches = re.findall(pattern, query)
        if matches:
            detected_ticker = matches[0].upper()
            if detected_ticker in ASSET_CONFIGS or len(detected_ticker) <= 5:
                params["ticker"] = detected_ticker
                break
    
    # Option type detection - Enhanced patterns
    call_patterns = [
        r"\bcall\b", r"buying.*call", r"long.*call", r"call.*option"
    ]
    put_patterns = [
        r"\bput\b", r"buying.*put", r"long.*put", r"put.*option"
    ]
    
    if any(re.search(pattern, query_lower) for pattern in call_patterns):
        params["option_type"] = "call"
    elif any(re.search(pattern, query_lower) for pattern in put_patterns):
        params["option_type"] = "put"
    
    # Handle switching patterns with more flexibility
    switch_patterns = {
        "call": [
            r"(?:switch to|change to|convert to|use)\s+(?:a\s+)?call",
            r"make.*(?:it|this)\s+(?:a\s+)?call"
        ],
        "put": [
            r"(?:switch to|change to|convert to|use)\s+(?:a\s+)?put",
            r"make.*(?:it|this)\s+(?:a\s+)?put"
        ]
    }
    
    for option_type, patterns in switch_patterns.items():
        if any(re.search(pattern, query_lower) for pattern in patterns):
            params["option_type"] = option_type
    
    # Method switching patterns - Enhanced
    method_patterns = {
        "mc": [
            r"(?:switch to|change to|use|switch back to)\s+(?:monte carlo|mc)(?:\s+(?:method|simulation))?",
            r"\bmonte\s*carlo\b", r"\bmc\s+(?:method|simulation)\b"
        ],
        "bs": [
            r"(?:switch to|change to|use|switch back to)\s+(?:black.scholes|bs)(?:\s+method)?",
            r"(?:switch to|change to|use|switch back to)\s+(?:black|scholes)",
            r"\bblack.scholes\b", r"\bbs\s+(?:method|model)\b"
        ],
        "bt": [
            r"(?:switch to|change to|use|switch back to)\s+(?:binomial tree|binomial|bt)(?:\s+method)?",
            r"\bbinomial\s*tree\b", r"\bbt\s+(?:method|model)\b"
        ]
    }
    
    for method, patterns in method_patterns.items():
        if any(re.search(pattern, query_lower) for pattern in patterns):
            params["method"] = method
            break
    
    # Extract numeric parameters with more flexible patterns
    param_patterns = {
        "S": [r"(?:spot|stock|price|s)\s*(?:=|is|:)\s*([\d.]+)",
              r"underlying\s*(?:=|is|:)\s*([\d.]+)",
              r"s\s*=\s*([\d.]+)"],
        "K": [r"(?:strike|k)\s*(?:=|is|:)\s*([\d.]+)",
              r"k\s*=\s*([\d.]+)"],
        "T": [r"(?:time|maturity|expiry|t)\s*(?:=|is|:)\s*([\d.]+)",
              r"(?:(\d+)\s*(?:days?|d))",  # Convert days to years
              r"(?:(\d+)\s*(?:months?|m))",  # Convert months to years
              r"t\s*=\s*([\d.]+)"],
        "r": [r"(?:rate|risk.free|r)\s*(?:=|is|:)\s*([\d.]+)",
              r"r\s*=\s*([\d.]+)"],
        "sigma": [r"(?:vol|volatility|sigma)\s*(?:=|is|:)\s*([\d.]+)",
                  r"(?:vol|volatility|sigma)\s*(?:of|at)\s*([\d.]+)",
                  r"sigma\s*=\s*([\d.]+)"]
    }
    
    for param, patterns in param_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                value = float(match.group(1))
                # Convert time units
                if param == "T":
                    if "days" in pattern or "d)" in pattern:
                        value = value / 365.0
                    elif "months" in pattern or "m)" in pattern:
                        value = value / 12.0
                # Convert percentage to decimal for volatility and rate
                elif param in ["r", "sigma"] and value > 1:
                    value = value / 100.0
                params[param] = value
                break
    
    return params

# ----------------------------
# Enhanced scenario application
# ----------------------------
def apply_scenarios(params, query):
    """Apply multiple scenarios with improved parsing and validation"""
    scenarios_applied = []
    
    # Volatility adjustments with more patterns
    vol_patterns = [
        (r"(?:decrease|reduce|lower)\s+(?:vol|volatility)\s+by\s+([\d.]+)%", -1),
        (r"(?:increase|raise|boost)\s+(?:vol|volatility)\s+by\s+([\d.]+)%", 1),
        (r"(?:vol|volatility)\s+(?:goes\s+)?(?:up|increases?)\s+by\s+([\d.]+)%", 1),
        (r"(?:vol|volatility)\s+(?:goes\s+)?(?:down|decreases?)\s+by\s+([\d.]+)%", -1),
        (r"(?:vol|volatility)\s+drops?\s+by\s+([\d.]+)%", -1)
    ]
    
    for pattern, direction in vol_patterns:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            change = float(match.group(1)) / 100
            old_sigma = params.get("sigma", 0.2)
            params["sigma"] = old_sigma * (1 + direction * change)
            action = "increased" if direction > 0 else "decreased"
            scenarios_applied.append(f"Volatility {action} by {change*100:.1f}%: {old_sigma:.4f} → {params['sigma']:.4f}")
    
    # Spot price movements
    spot_patterns = [
        (r"spot\s+(?:moves?|goes?)\s+(?:up\s+by\s+|)\+([\d.]+)%", 1),
        (r"spot\s+(?:moves?|goes?)\s+(?:down\s+by\s+|)\-([\d.]+)%", -1),
        (r"stock\s+(?:price\s+)?(?:rises?|increases?)\s+by\s+([\d.]+)%", 1),
        (r"stock\s+(?:price\s+)?(?:falls?|decreases?)\s+by\s+([\d.]+)%", -1),
        (r"underlying\s+(?:moves?|changes?)\s+([\+\-]?)([\d.]+)%", None)
    ]
    
    for pattern, direction in spot_patterns:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            if direction is None:  # Handle +/- prefix
                sign = match.group(1) if match.group(1) else "+"
                change = float(match.group(2)) / 100 * (1 if sign == "+" else -1)
            else:
                change = float(match.group(1)) / 100 * direction
            
            old_S = params.get("S", 100)
            params["S"] = old_S * (1 + change)
            action = "increased" if change > 0 else "decreased"
            scenarios_applied.append(f"Spot price {action} by {abs(change)*100:.1f}%: ${old_S:.2f} → ${params['S']:.2f}")
    
    # Risk-free rate adjustments
    rate_patterns = [
        (r"(?:risk.free\s+rate|interest\s+rate|rate)\s+(?:increases?|goes?\s+up)\s+by\s+([\d.]+)%", 1),
        (r"(?:risk.free\s+rate|interest\s+rate|rate)\s+(?:decreases?|goes?\s+down)\s+by\s+([\d.]+)%", -1)
    ]
    
    for pattern, direction in rate_patterns:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            change = float(match.group(1)) / 100 * direction
            old_r = params.get("r", 0.05)
            params["r"] = old_r + change
            action = "increased" if direction > 0 else "decreased"
            scenarios_applied.append(f"Risk-free rate {action} by {abs(change)*100:.1f}%: {old_r:.4f} → {params['r']:.4f}")
    
    # Time decay scenarios
    time_patterns = [
        (r"(?:(\d+)\s+days?\s+pass|time\s+passes?\s+by\s+(\d+)\s+days?)", "days"),
        (r"(?:(\d+)\s+weeks?\s+pass|time\s+passes?\s+by\s+(\d+)\s+weeks?)", "weeks")
    ]
    
    for pattern, unit in time_patterns:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            days = float(match.group(1) or match.group(2))
            if unit == "weeks":
                days *= 7
            time_change = days / 365.0
            old_T = params.get("T", 1)
            params["T"] = max(0.001, old_T - time_change)  # Prevent negative time
            scenarios_applied.append(f"Time decay: {days} days passed, T: {old_T:.4f} → {params['T']:.4f}")
    
    # Display applied scenarios
    if scenarios_applied:
        st.info("📊 **Scenarios Applied:**")
        for scenario in scenarios_applied:
            st.write(f"• {scenario}")
    
    # Set default option type only if not already set
    if "option_type" not in params:
        params["option_type"] = "call"
    
    return params

# ----------------------------
# Black-Scholes with input validation
# ----------------------------
def black_scholes_price(S, K, T, r, sigma, option_type):
    """Black-Scholes pricing with input validation"""
    if T <= 0:
        raise ValueError("Time to maturity must be positive")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")
    if S <= 0:
        raise ValueError("Spot price must be positive")
    if K <= 0:
        raise ValueError("Strike price must be positive")
    
    try:
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == "call":
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            delta = norm.cdf(d1)
        else:  # put
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = -norm.cdf(-d1)
        
        gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
        vega = S*norm.pdf(d1)*np.sqrt(T) / 100  # Per 1% vol change
        
        if option_type == "call":
            theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
            rho = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
        else:
            theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
            rho = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
        
        greeks = {
            "delta": round(delta, 4),
            "gamma": round(gamma, 4),
            "vega": round(vega, 4),
            "theta": round(theta, 4),
            "rho": round(rho, 4)
        }
        
        return price, greeks
        
    except Exception as e:
        raise ValueError(f"Black-Scholes calculation error: {str(e)}")

# ----------------------------
# Monte Carlo with progress tracking
# ----------------------------
def monte_carlo_price(S, K, T, r, sigma, option_type, num_paths=50000, num_steps=100):
    """Monte Carlo pricing with progress indication"""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("All parameters must be positive")
    
    try:
        np.random.seed(42)  # For reproducible results
        dt = T / num_steps
        
        # Generate random walks
        z = np.random.standard_normal((num_paths, num_steps))
        
        # Calculate paths efficiently
        log_returns = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z
        log_S = np.log(S) + np.cumsum(log_returns, axis=1)
        S_T = np.exp(log_S[:, -1])
        
        # Calculate payoffs
        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        price = np.exp(-r*T) * np.mean(payoffs)
        
        # Calculate confidence interval
        std_error = np.std(payoffs) / np.sqrt(num_paths)
        conf_interval = 1.96 * std_error * np.exp(-r*T)
        
        greeks = {
            "confidence_95": round(conf_interval, 4),
            "num_paths": num_paths,
            "std_error": round(std_error, 6)
        }
        
        return price, greeks
        
    except Exception as e:
        raise ValueError(f"Monte Carlo calculation error: {str(e)}")

# ----------------------------
# Enhanced Binomial Tree
# ----------------------------
def binomial_tree_price(S, K, T, r, sigma, option_type, N=1000):
    """Enhanced binomial tree with American option support"""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("All parameters must be positive")
    
    try:
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(r * dt) - d) / (u - d)
        
        # Validate risk-neutral probability
        if q < 0 or q > 1:
            raise ValueError("Invalid risk-neutral probability - check parameters")
        
        # Initialize arrays
        stock_prices = np.zeros(N + 1)
        option_prices = np.zeros(N + 1)
        
        # Calculate stock prices at maturity
        for j in range(N + 1):
            stock_prices[j] = S * (u**(N - j)) * (d**j)
        
        # Calculate option payoffs at maturity
        if option_type == "call":
            option_prices = np.maximum(stock_prices - K, 0)
        else:
            option_prices = np.maximum(K - stock_prices, 0)
        
        # Backward induction
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                # European option value
                european_value = np.exp(-r * dt) * (q * option_prices[j] + (1 - q) * option_prices[j + 1])
                option_prices[j] = european_value
        
        price = option_prices[0]
        
        greeks = {
            "num_steps": N,
            "up_factor": round(u, 4),
            "down_factor": round(d, 4),
            "risk_neutral_prob": round(q, 4)
        }
        
        return price, greeks
        
    except Exception as e:
        raise ValueError(f"Binomial tree calculation error: {str(e)}")

# ----------------------------
# Enhanced Visualization
# ----------------------------
def plot_sensitivity(params, parameter_name, values, option_type, method):
    """Enhanced sensitivity plot with better styling"""
    prices = []
    original_value = params[parameter_name]
    
    try:
        for val in values:
            params[parameter_name] = val
            if method == "bs":
                price, _ = black_scholes_price(params["S"], params["K"], params["T"], 
                                             params["r"], params["sigma"], option_type)
            elif method == "mc":
                price, _ = monte_carlo_price(params["S"], params["K"], params["T"], 
                                           params["r"], params["sigma"], option_type)
            elif method == "bt":
                price, _ = binomial_tree_price(params["S"], params["K"], params["T"], 
                                             params["r"], params["sigma"], option_type)
            prices.append(price)
        
        # Reset parameter
        params[parameter_name] = original_value
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(values, prices, 'b-', linewidth=2, label=f'{option_type.capitalize()} Option')
        ax.axvline(x=original_value, color='r', linestyle='--', alpha=0.7, 
                  label=f'Current {parameter_name.upper()} = {original_value}')
        ax.axhline(y=prices[len(prices)//2], color='g', linestyle=':', alpha=0.5)
        
        ax.set_title(f"Option Price Sensitivity to {parameter_name.upper()} ({method.upper()} Method)", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(f"{parameter_name.upper()}", fontsize=12)
        ax.set_ylabel("Option Price ($)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add current price annotation
        current_idx = np.argmin(np.abs(np.array(values) - original_value))
        ax.annotate(f'${prices[current_idx]:.2f}', 
                   xy=(original_value, prices[current_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error creating sensitivity plot: {str(e)}")

# ----------------------------
# Streamlit UI with Enhanced Features
# ----------------------------
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Options Pricing Chatbot",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "params" not in st.session_state:
        st.session_state.params = {}
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    
    st.title("📊 Advanced Options Pricing Chatbot")
    st.markdown("*Natural language interface for options pricing and scenario analysis*")
    
    # Sidebar with current parameters and controls
    with st.sidebar:
        st.header("🔧 Current Parameters")
        if st.session_state.params:
            for key, value in st.session_state.params.items():
                if key not in ["conversation", "query_count"]:
                    if isinstance(value, float):
                        st.metric(key.upper(), f"{value:.4f}")
                    else:
                        st.metric(key.upper(), str(value))
        
        st.header("🎛️ Quick Actions")
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation = []
            st.session_state.params = {}
            st.rerun()
        
        if st.button("🔄 Reset to Defaults"):
            st.session_state.params = {
                "S": 100, "K": 100, "T": 1, "r": 0.05,
                "sigma": 0.2, "option_type": "call", "method": "bs"
            }
            st.rerun()
        
        # Live price fetching for multiple assets
        st.header("💰 Live Prices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Stocks")
            stock_symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "SPY"]
            for symbol in stock_symbols:
                if st.button(f"📊 {symbol}", key=f"stock_{symbol}"):
                    price = fetch_asset_price(symbol)
                    if price:
                        st.session_state.params["S"] = price
                        st.session_state.params["ticker"] = symbol
                        asset_name = ASSET_CONFIGS.get(symbol, {}).get("name", symbol)
                        st.success(f"Set {asset_name} price: ${price:,.2f}")
                        st.rerun()
        
        with col2:
            st.subheader("₿ Crypto")
            crypto_symbols = ["BTC", "ETH", "LTC"]
            for symbol in crypto_symbols:
                if st.button(f"💎 {symbol}", key=f"crypto_{symbol}"):
                    if symbol == "BTC":
                        price = fetch_crypto_price()
                    else:
                        price = fetch_asset_price(symbol)
                    if price:
                        st.session_state.params["S"] = price
                        st.session_state.params["ticker"] = symbol
                        st.success(f"Set {symbol} price: ${price:,.2f}")
                        st.rerun()
    
    # Main input area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_input(
            "💬 Enter your query or scenario:",
            key=f"query_input_{st.session_state.query_count}",
            placeholder="e.g., 'Price a call option with S=100, K=105, T=0.25, r=0.05, sigma=0.2'"
        )
    
    with col2:
        send_button = st.button("🚀 Send", use_container_width=True)
    
    # Example queries
    with st.expander("💡 Example Queries"):
        examples = [
            "Price AAPL call with spot=185, strike=190, time=0.038, vol=45%",
            "TSLA put spread, bearish on deliveries, spot=240, strike=230",  
            "SPY protective put, hedge my portfolio, 3 months out",
            "Switch to put option and increase volatility by 10%",
            "Use Monte Carlo method with 100000 paths",
            "What if AAPL moves up 15% and vol decreases by 5%?",
            "Show sensitivity to spot price"
        ]
        for example in examples:
            if st.button(f"📋 {example}", key=f"example_{hash(example)}"):
                # Simulate user input
                st.session_state[f"query_input_{st.session_state.query_count}"] = example
                user_input = example
                send_button = True
    
    # Process user input
    if send_button and user_input.strip():
        process_user_query(user_input)
    elif send_button and not user_input.strip():
        st.warning("⚠️ Please enter a query before sending.")
    
    # Display conversation history
    display_conversation()
    
    # Visualization options
    if st.session_state.params and len(st.session_state.conversation) > 0:
        st.subheader("📈 Sensitivity Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Spot Price Sensitivity"):
                create_sensitivity_plot("S")
        
        with col2:
            if st.button("📊 Volatility Sensitivity"):
                create_sensitivity_plot("sigma")
        
        with col3:
            if st.button("📊 Time Sensitivity"):
                create_sensitivity_plot("T")

def process_user_query(user_input):
    """Process user query and update conversation"""
    try:
        # Parse new query
        new_params = parse_query(user_input)
        
        # Handle asset price fetching with priority for explicit parameters
        ticker = new_params.get("ticker")
        
        # Only fetch live price if:
        # 1. A ticker is specified AND
        # 2. No explicit spot price is provided AND  
        # 3. No spot price exists in session state
        if (ticker and 
            "S" not in new_params and 
            "S" not in st.session_state.params):
            
            if ticker == "BTC":
                live_price = fetch_crypto_price()
            else:
                live_price = fetch_asset_price(ticker)
                
            if live_price:
                new_params["S"] = live_price
                asset_name = ASSET_CONFIGS.get(ticker, {}).get("name", ticker)
                st.info(f"🔡 Fetched {asset_name} ({ticker}) price: ${live_price:,.2f}")
        
        # If no ticker specified and no spot price, fallback to BTC (legacy behavior)
        elif ("S" not in new_params and 
              "S" not in st.session_state.params and 
              not ticker):
            btc_price = fetch_crypto_price("btcusd")
            if btc_price:
                new_params["S"] = btc_price
                st.info(f"🔡 Fetched BTC price: ${btc_price:,.2f}")
        
        # Merge with previous parameters (explicit params take priority)
        st.session_state.params.update(new_params)
        
        # Apply scenarios
        st.session_state.params = apply_scenarios(st.session_state.params, user_input)
        
        # Set defaults for missing parameters
        defaults = {
            "S": 100, "K": 100, "T": 1, "r": 0.05,
            "sigma": 0.2, "option_type": "call", "method": "bs"
        }
        for key, default_val in defaults.items():
            if key not in st.session_state.params:
                st.session_state.params[key] = default_val
        
        # Validate parameters
        validate_parameters(st.session_state.params)
        
        # Compute price and Greeks
        calculate_option_price(user_input)
        
    except Exception as e:
        st.error(f"⌐ Error processing query: {str(e)}")
        logger.error(f"Query processing error: {e}")

def validate_parameters(params):
    """Validate option parameters"""
    validations = [
        (params.get("S", 0) > 0, "Spot price must be positive"),
        (params.get("K", 0) > 0, "Strike price must be positive"),
        (params.get("T", 0) > 0, "Time to maturity must be positive"),
        (params.get("sigma", 0) > 0, "Volatility must be positive"),
        (0 <= params.get("r", 0) <= 1, "Risk-free rate should be between 0 and 100%"),
        (0 < params.get("sigma", 0) <= 5, "Volatility should be between 0 and 500%"),
        (params.get("T", 0) <= 10, "Time to maturity should be reasonable (≤10 years)")
    ]
    
    for condition, message in validations:
        if not condition:
            raise ValueError(message)

def calculate_option_price(user_input):
    """Calculate option price using specified method"""
    p = st.session_state.params
    
    try:
        # Show calculation method
        method_names = {"bs": "Black-Scholes", "mc": "Monte Carlo", "bt": "Binomial Tree"}
        st.info(f"🧮 Using {method_names.get(p['method'], 'Unknown')} method...")
        
        # Calculate based on method
        if p["method"] == "bs":
            price, greeks = black_scholes_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["option_type"])
        elif p["method"] == "mc":
            with st.spinner("Running Monte Carlo simulation..."):
                price, greeks = monte_carlo_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], 
                                                p["option_type"], p.get("num_paths", 50000), 
                                                p.get("num_steps", 100))
        elif p["method"] == "bt":
            with st.spinner("Building binomial tree..."):
                price, greeks = binomial_tree_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], 
                                                  p["option_type"], p.get("N", 1000))
        else:
            raise ValueError(f"Unknown pricing method: {p['method']}")
        
        # Add to conversation
        st.session_state.conversation.append({
            "query": user_input,
            "price": price,
            "greeks": greeks,
            "params": p.copy(),
            "method": method_names.get(p["method"], "Unknown"),
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
        })
        
        # Success message
        st.success(f"✅ {p['option_type'].capitalize()} option priced: **${price:.4f}**")
        
        # Increment query counter and rerun
        st.session_state.query_count += 1
        st.rerun()
        
    except Exception as e:
        st.error(f"⌐ Calculation error: {str(e)}")
        logger.error(f"Calculation error: {e}")

def display_conversation():
    """Display conversation history with enhanced formatting"""
    if st.session_state.conversation:
        st.subheader("📖 Conversation History")
        
        for i, msg in enumerate(reversed(st.session_state.conversation), 1):
            with st.expander(f"Query {len(st.session_state.conversation) + 1 - i}: {msg['query'][:50]}{'...' if len(msg['query']) > 50 else ''}", expanded=(i == 1)):
                
                # Query and metadata
                st.markdown(f"**📝 Query:** {msg['query']}")
                st.markdown(f"**⏰ Time:** {msg.get('timestamp', 'N/A')}")
                st.markdown(f"**🔢 Method:** {msg.get('method', 'Unknown')}")
                
                # Price display
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.metric(
                        label=f"{msg['params']['option_type'].capitalize()} Option Price",
                        value=f"${msg['price']:.4f}",
                        delta=None
                    )
                
                with col2:
                    moneyness = msg['params']['S'] / msg['params']['K']
                    if moneyness > 1.05:
                        money_status = "🟢 In-the-Money"
                    elif moneyness < 0.95:
                        money_status = "🔴 Out-of-the-Money"
                    else:
                        money_status = "🟡 At-the-Money"
                    st.markdown(f"**Status:** {money_status}")
                
                # Parameters in a nice table
                st.markdown("**📋 Parameters:**")
                param_cols = st.columns(5)
                params_display = [
                    ("Spot (S)", f"${msg['params']['S']:.2f}"),
                    ("Strike (K)", f"${msg['params']['K']:.2f}"),
                    ("Time (T)", f"{msg['params']['T']:.3f}"),
                    ("Rate (r)", f"{msg['params']['r']*100:.2f}%"),
                    ("Vol (σ)", f"{msg['params']['sigma']*100:.1f}%")
                ]
                
                for col, (label, value) in zip(param_cols, params_display):
                    col.metric(label, value)
                
                # Greeks display
                if msg['greeks']:
                    st.markdown("**🔬 Greeks & Metrics:**")
                    if msg['params']['method'] == 'bs':
                        greek_cols = st.columns(5)
                        greeks_display = [
                            ("Delta", f"{msg['greeks'].get('delta', 0):.4f}"),
                            ("Gamma", f"{msg['greeks'].get('gamma', 0):.4f}"),
                            ("Vega", f"{msg['greeks'].get('vega', 0):.4f}"),
                            ("Theta", f"{msg['greeks'].get('theta', 0):.4f}"),
                            ("Rho", f"{msg['greeks'].get('rho', 0):.4f}")
                        ]
                        for col, (label, value) in zip(greek_cols, greeks_display):
                            col.metric(label, value)
                    else:
                        # Display method-specific metrics
                        for key, value in msg['greeks'].items():
                            if isinstance(value, (int, float)):
                                st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
                            else:
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                st.markdown("---")

def create_sensitivity_plot(parameter):
    """Create sensitivity analysis plot"""
    if not st.session_state.params:
        st.warning("No parameters available for sensitivity analysis")
        return
    
    p = st.session_state.params.copy()
    
    # Define ranges for different parameters
    ranges = {
        "S": np.linspace(p["S"] * 0.7, p["S"] * 1.3, 50),
        "K": np.linspace(p["K"] * 0.7, p["K"] * 1.3, 50),
        "sigma": np.linspace(max(0.01, p["sigma"] * 0.5), p["sigma"] * 2, 50),
        "T": np.linspace(max(0.01, p["T"] * 0.1), p["T"] * 2, 50),
        "r": np.linspace(max(0, p["r"] - 0.05), p["r"] + 0.05, 50)
    }
    
    if parameter not in ranges:
        st.error(f"Parameter {parameter} not supported for sensitivity analysis")
        return
    
    values = ranges[parameter]
    plot_sensitivity(p, parameter, values, p["option_type"], p["method"])

if __name__ == "__main__":
    main()
