import os
import time
import ccxt
import pandas as pd
from typing import List, Any


class DataGathering:
    TIMEFRAME_TO_MS = {
        '1m': 60_000,
        '5m': 300_000,
        '15m': 900_000,
        '30m': 1_800_000,
        '1h': 3_600_000,
        '4h': 14_400_000,
        '1d': 86_400_000,
    }

    def __init__(
            self,
            exchange_name: str = 'binance',
            symbol: str = 'ETH/USDT',
            timeframe: str = '1h',
            limit: int = 1000,
            start_date: str = '2017-08-17T00:00:00Z',
            data_dir: str = 'data'
    ):
        self.exchange = self._initialize_exchange(exchange_name)
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.duration = self._get_duration_from_timeframe(timeframe)
        self.since = self.exchange.parse8601(start_date)
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _initialize_exchange(self, name: str):
        try:
            exchange_class = getattr(ccxt, name)
            return exchange_class()
        except AttributeError:
            raise ValueError(f"Exchange '{name}' is not supported by ccxt.")

    def _get_duration_from_timeframe(self, timeframe: str) -> int:
        if timeframe not in self.TIMEFRAME_TO_MS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return self.TIMEFRAME_TO_MS[timeframe]

    def _fetch_ohlcv_batch(self, since: int) -> List[List[Any]]:
        return self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, since=since, limit=self.limit)

    def _deduplicate_and_sort(self, data: List[List[Any]]) -> List[List[Any]]:
        unique = {row[0]: row for row in data}
        sorted_data = sorted(unique.values(), key=lambda x: x[0])
        return sorted_data

    def gather_data(self) -> pd.DataFrame:
        all_ohlcv = []
        current_since = self.since

        while True:
            try:
                readable_time = pd.to_datetime(current_since, unit='ms')
                print(f"[INFO] Fetching from: {readable_time}")
                ohlcv = self._fetch_ohlcv_batch(current_since)

                if not ohlcv:
                    print("[INFO] No more data to fetch.")
                    break

                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + self.duration
                time.sleep(0.6)  # Respect API rate limits
            except Exception as e:
                print(f"[ERROR] Fetch failed: {e}")
                time.sleep(5)

        all_ohlcv = self._deduplicate_and_sort(all_ohlcv)

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def save_to_csv(self, df: pd.DataFrame):
        filename = f"{self.symbol.lower().replace('/', '_')}_{self.timeframe}.csv"
        self.data_dir = os.path.join(self.data_dir, f"{self.symbol.lower().replace('/', '_')}")
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"[INFO] Saved {len(df)} rows to {filepath}")


if __name__ == "__main__":
    dg = DataGathering(
        exchange_name='binance',
        symbol='XRP/USDT',
        timeframe='1h',
        limit=1000,
        start_date='2013-08-17T00:00:00Z',
        data_dir='../../../data'
    )
    df = dg.gather_data()
    dg.save_to_csv(df)
