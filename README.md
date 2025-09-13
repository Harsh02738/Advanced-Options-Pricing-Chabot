# 📈 Options Pricing Chatbot

An interactive **Streamlit-powered chatbot** for **options pricing and analysis**.  
It combines **financial models, natural language query parsing, and scenario simulations** to help users explore option strategies with ease.

---

## 🚀 Features

- **Live Market Data**
  - Fetches real-time stock prices (via Alpha Vantage / Yahoo Finance).
  - Fetches crypto prices (via Coinbase API).

- **Supported Pricing Models**
  - **Black–Scholes (BSM)** with Greeks (Δ, Γ, Vega, Theta, Rho).
  - **Monte Carlo Simulation** with confidence intervals.
  - **Binomial Tree** pricing with customizable steps.

- **Natural Language Understanding**
  - Parse queries like:
    ```
    Price a call option on AAPL with strike 180, expiring in 6 months, volatility 25%, risk-free rate 3%.
    ```
  - Auto-detects:
    - Option type (call/put)
    - Underlying asset (AAPL, TSLA, BTC, etc.)
    - Pricing method (Black-Scholes, Monte Carlo, Binomial)
    - Parameters (S, K, T, r, σ)

- **Scenario Analysis**
  - Apply market shocks:
    - Volatility changes
    - Spot price adjustments
    - Interest rate shifts
    - Time decay effects

- **Visualizations**
  - Sensitivity plots for:
    - Spot price vs Option value
    - Volatility vs Option value
    - Time to maturity vs Option value

- **Streamlit Chat Interface**
  - Chat-like interaction
  - Conversation history
  - Example queries for onboarding

---

## 📂 Project Structure
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .env # API keys (not committed)


---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/options-pricing-chatbot.git
   cd options-pricing-chatbot


Set up environment

python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows


Install dependencies

pip install -r requirements.txt


Set environment variables in .env:

ALPHA_VANTAGE_API_KEY=your_api_key_here
GEMINI_API_KEY=your_gemini_key_here   # optional (for crypto)

▶️ Usage

Run the app with:

streamlit run app.py


Then open the URL (default: http://localhost:8501) in your browser.

💡 Example Queries

Price a European call option on AAPL with strike 150, maturity 1 year, volatility 20%, risk-free 5%.

Use Monte Carlo to price a put option on BTC at strike 25,000, expiring in 3 months.

Show binomial tree pricing for TSLA call with strike 280, 6 months maturity.

Price a call on NFLX, but increase volatility by 10% and decay time by 2 months.

🛠️ Tech Stack

Python

Streamlit (UI framework)

NumPy, Pandas (computations)

Matplotlib (visualizations)

yfinance, Alpha Vantage, Coinbase (market data)
