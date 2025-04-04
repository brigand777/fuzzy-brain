# fastapi_backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from core.optimizer import run_optimizers
from core.utils import dynamic_backtest_portfolio

app = FastAPI()

# Optional: allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    assets: Dict[str, float]  # {"BTC": 0.5, "ETH": 0.3, ...}
    price_data: Dict[str, List[float]]  # date-indexed historical prices
    lookback_days: int
    nonnegative: bool = True

@app.post("/optimize")
def optimize_portfolio(req: OptimizeRequest):
    df = pd.DataFrame(req.price_data)
    df.index = pd.to_datetime(df.index)
    lookback = df.tail(req.lookback_days)
    allocations = run_optimizers(lookback, nonnegative_mvo=req.nonnegative)
    return {method: alloc.to_dict() for method, alloc in allocations.items()}

# You can add more endpoints like /simulate, /portfolio, etc
