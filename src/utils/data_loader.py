# src/train/data_loader.py
import pandas as pd
from typing import Dict

def load_market_data() -> Dict[str, pd.DataFrame]:
    """
    Loads market data from CSV files. Adjust file paths as needed.
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with asset keys.
    """
    base_path = "./data"
    data_files = {
        "btc": f"{base_path}/btc_usdt/btc_usdt_1h.csv",
        "sol": f"{base_path}/sol_usdt/sol_usdt_1h.csv",
        "xrp": f"{base_path}/xrp_usdt/xrp_usdt_1h.csv",
        "shib": f"{base_path}/shib_usdt/shib_usdt_1h.csv",
        "doge": f"{base_path}/doge_usdt/doge_usdt_1h.csv",
        "bnb": f"{base_path}/bnb_usdt/bnb_usdt_1h.csv",
        "eth": f"{base_path}/eth_usdt/eth_usdt_1h.csv"
    }
    return {key: pd.read_csv(path) for key, path in data_files.items()}
