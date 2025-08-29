from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from engine.blacksholes import BlackScholes
from engine.montecarlo import MonteCarloOption

app = FastAPI(title="AI Quant Bot API")

# Request model
class OptionRequest(BaseModel):
    S: float        # Spot price
    K: float        # Strike price
    T: float        # Time to maturity (years)
    r: float        # Risk-free rate
    sigma: float    # Volatility
    option_type: str = "call"  # "call" or "put"
    method: str = "bs"         # "bs" or "mc"
    num_paths: int = 50000     # MC paths (if method="mc")
    num_steps: int = 100       # MC steps per path

# Endpoint to price option and return Greeks
@app.post("/price_option")
def price_option(req: OptionRequest):
    try:
        if req.method.lower() == "bs":
            bs = BlackScholes(req.S, req.K, req.T, req.r, req.sigma)
            price = bs.call_price() if req.option_type=="call" else bs.put_price()
            greeks = {
                "delta": bs.delta(req.option_type),
                "gamma": bs.gamma(),
                "vega": bs.vega(),
                "theta": bs.theta(req.option_type),
                "rho": bs.rho(req.option_type)
            }
        elif req.method.lower() == "mc":
            mc = MonteCarloOption(req.S, req.K, req.T, req.r, req.sigma,
                                  num_paths=req.num_paths, num_steps=req.num_steps)
            price = mc.european_call_price() if req.option_type=="call" else mc.european_put_price()
            greeks = "Not implemented for Monte Carlo yet"
        else:
            raise HTTPException(status_code=400, detail="Method must be 'bs' or 'mc'")

        return {
            "price": price,
            "greeks": greeks,
            "method": req.method.lower()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
