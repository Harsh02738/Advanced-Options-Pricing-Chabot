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

