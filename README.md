
📈 Options Pricing Chatbot
An interactive Streamlit-powered chatbot designed for intuitive options pricing and analysis. This tool combines robust financial models (Black-Scholes, Monte Carlo, Binomial Trees), natural language query parsing, and interactive scenario simulations to help users explore complex option strategies with ease.

📋 Table of Contents
🚀 Features

🛠️ Tech Stack

⚙️ Installation and Setup

▶️ Usage

💡 Example Queries

📂 Project Structure

🤝 Contributing

📜 License

🚀 Features
This chatbot provides a comprehensive suite of tools for both novice and experienced traders:

Live Market Data Integration

Fetches real-time equity prices using Alpha Vantage and Yahoo Finance.

Retrieves current cryptocurrency prices (BTC, ETH, etc.) via the Coinbase API.

Advanced Pricing Models

Black-Scholes (BSM): Calculates theoretical European option prices and all associated Greeks (Delta, Gamma, Vega, Theta, Rho).

Monte Carlo Simulation: Prices options by simulating thousands of potential asset paths, complete with configurable simulations and confidence intervals.

Binomial Tree: Provides step-by-step CRR (Cox-Ross-Rubinstein) binomial pricing with customizable tree depths (steps).

Natural Language Understanding (NLU)

Simply type your request in plain English. The bot parses queries to identify key parameters automatically.

Auto-detects:

Option Type (Call/Put)

Underlying Asset (e.g., AAPL, TSLA, BTC-USD)

Pricing Model (Black-Scholes, Monte Carlo, Binomial)

Financial Parameters (Spot S, Strike K, Time-to-Maturity T, Risk-Free Rate r, Volatility σ)

Interactive Scenario Analysis

Instantly model "what-if" scenarios by applying market shocks.

Test the impact of:

Volatility spikes or crashes.

Spot price adjustments.

Shifts in the risk-free interest rate.

Automatic time decay (Theta) effects.

Dynamic Visualizations

Generates interactive sensitivity plots to visualize option behavior.

Plots include:

Spot Price vs. Option Value (Payoff diagrams)

Volatility Smile/Skew effects vs. Value

Time to Maturity vs. Value (Theta decay curve)

Streamlit Chat Interface

Modern, responsive chat-based UI.

Maintains a persistent conversation history for your session.

Includes helpful example queries to onboard new users instantly.

🛠️ Tech Stack
Core: Python 3.9+

UI Framework: Streamlit

Computations & Data: NumPy, Pandas

Financial Modeling: SciPy (for BSM calculations)

Visualizations: Matplotlib, Plotly (recommended enhancement)

Market Data APIs: yfinance, alpha_vantage, coinbase

NLU/AI (Optional): Google Gemini or other LLM for parsing

⚙️ Installation and Setup
Follow these steps to get the chatbot running locally.

1. Clone the Repository
Bash

git clone https://github.com/your-username/options-pricing-chatbot.git
cd options-pricing-chatbot
2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

Mac/Linux:

Bash

python3 -m venv venv
source venv/bin/activate
Windows (Command Prompt):

Bash

python -m venv venv
venv\Scripts\activate
3. Install Dependencies
Install all required Python packages using the requirements.txt file.

Bash

pip install -r requirements.txt
4. Configure Environment Variables
This application requires API keys for market data (and optionally for NLU).

Create a file named .env in the root of the project directory.

Add your API keys to this file. (Note: The .env file is listed in .gitignore and should never be committed to source control.)

Ini, TOML

# .env file content

# Required for fetching real-time stock data
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Required for NLU parsing or (as defined) crypto data
GEMINI_API_KEY=your_gemini_key_here

# Note: You may also need to add Coinbase API keys if using the full API
# COINBASE_API_KEY=...
# COINBASE_API_SECRET=...
▶️ Usage
Once your installation is complete and your .env file is configured, run the Streamlit application:

Bash

streamlit run app.py
Streamlit will automatically open the application in your default web browser (usually at http://localhost:8501).

💡 Example Queries
Interact with the chatbot using natural language. Try queries like these:

"Price a European call option on AAPL with strike 150, maturity 1 year, volatility 20%, and risk-free 5%."

"Use Monte Carlo with 10000 simulations to price a put option on BTC-USD at strike 25,000, expiring in 3 months."

"Show binomial tree pricing using 50 steps for a TSLA call with strike 280, 6 months maturity."

"What are the Greeks for an NVDA 900-strike call expiring in 45 days?"

"Price a call on NFLX, but show me a scenario analysis if volatility increases by 10% and time decays by 2 months."

📂 Project Structure
Plaintext

options-pricing-chatbot/
├── app.py              # Main Streamlit application file
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .env                # Local environment variables (API keys - Not committed)
└── .gitignore          # Files to ignore for Git tracking
🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request
