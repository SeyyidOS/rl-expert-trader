import pandas as pd
import numpy as np


class PriceActionFeaturesGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_df(self):
        return self.df

    def level1_features(self):
        self.df['upper_wick'] = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        self.df['lower_wick'] = self.df[['open', 'close']].min(axis=1) - self.df['low']
        self.df['body'] = abs(self.df['close'] - self.df['open'])

        # Determine candle type
        self.df['candle_type'] = np.where(
            self.df['close'] > self.df['open'], 'bullish',
            np.where(self.df['close'] < self.df['open'], 'bearish', 'doji')
        )

    def level2_features(self):
        self._engulfing_patterns()
        self._swing_high_low()
        self._fvg()
        self._OB()

    def _engulfing_patterns(self):
        # We use shifted columns to access the previous candle
        self.df['prev_open'] = self.df['open'].shift(1)
        self.df['prev_close'] = self.df['close'].shift(1)

        # Bullish Engulfing: Previous candle bearish, current candle bullish,
        # and current body completely engulfs previous body
        self.df['bullish_engulfing'] = (
                (self.df['prev_close'] < self.df['prev_open']) &  # Previous candle bearish
                (self.df['close'] > self.df['open']) &  # Current candle bullish
                (self.df['open'] < self.df['prev_close']) &  # Current open below previous close
                (self.df['close'] > self.df['prev_open'])  # Current close above previous open
        )

        # Bearish Engulfing: Previous candle bullish, current candle bearish,
        # and current body completely engulfs previous body
        self.df['bearish_engulfing'] = (
                (self.df['prev_close'] > self.df['prev_open']) &  # Previous candle bullish
                (self.df['close'] < self.df['open']) &  # Current candle bearish
                (self.df['open'] > self.df['prev_close']) &  # Current open above previous close
                (self.df['close'] < self.df['prev_open'])  # Current close below previous open
        )

    def _swing_high_low(self, n: int = 3):
        # Define a window size: how many candles before and after to consider
        n = 3  # you can tweak this for sensitivity

        # Create boolean columns for swing highs and swing lows
        self.df['swing_high'] = self.df['high'].rolling(window=2 * n + 1, center=True).apply(
            lambda x: int(x[n] == max(x)), raw=True)

        self.df['swing_low'] = self.df['low'].rolling(window=2 * n + 1, center=True).apply(
            lambda x: int(x[n] == min(x)), raw=True)

        # Fill NaNs with False for clarity
        self.df['swing_high'].fillna(0, inplace=True)
        self.df['swing_low'].fillna(0, inplace=True)

        # Convert to boolean
        self.df['swing_high'] = self.df['swing_high'].astype(bool)
        self.df['swing_low'] = self.df['swing_low'].astype(bool)

    def _fvg(self):
        # Shifted values
        self.df['high_n_2'] = self.df['high'].shift(2)
        self.df['low_n_2'] = self.df['low'].shift(2)

        # Conditions
        self.df['bullish_fvg'] = self.df['low'] > self.df['high_n_2']
        self.df['bearish_fvg'] = self.df['high'] < self.df['low_n_2']

    def _OB(self):
        # Calculate candle direction
        self.df['candle_direction'] = self.df.apply(
            lambda row: 1 if row['close'] > row['open'] else (-1 if row['close'] < row['open'] else 0),
            axis=1
        )

        # Initialize OB flags
        self.df['bullish_ob'] = False
        self.df['bearish_ob'] = False

        # Set N for confirmation candles
        N = 3

        # Loop through and apply OB detection logic
        for i in range(len(self.df) - N):
            current_dir = self.df.loc[i, 'candle_direction']
            next_dirs = self.df.loc[i + 1:i + N, 'candle_direction'].tolist()

            if current_dir == -1 and all(d == 1 for d in next_dirs):  # Bullish OB
                self.df.at[i, 'bullish_ob'] = True
            elif current_dir == 1 and all(d == -1 for d in next_dirs):  # Bearish OB
                self.df.at[i, 'bearish_ob'] = True
