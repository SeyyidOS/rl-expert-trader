from typing import List, Dict, Any

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TradeLogger:
    """
    A dedicated class for logging trade events.

    This class collects trade events and can save the log to a CSV file for later analysis.
    """

    def __init__(self) -> None:
        self.trades: List[Dict[str, Any]] = []

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """Appends a trade dictionary to the log."""
        self.trades.append(trade)

    def save(self, path: str) -> None:
        """Saves the trade log to a CSV file at the given path."""
        if not self.trades:
            logging.warning("No trades to save.")
            return
        df = pd.DataFrame(self.trades)
        df.to_csv(path, index=False)
        logging.info("Trade log saved to %s", path)

    def clear(self) -> None:
        """Clears the logged trades."""
        self.trades.clear()
