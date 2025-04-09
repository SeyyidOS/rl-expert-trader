from preprocessing.price_action_features import PriceActionFeaturesGenerator

import matplotlib.pyplot as plt
import pandas as pd


def visualize(df: pd.DataFrame):
    # Create a simplified candlestick-style plot using matplotlib (no mplfinance)

    # Choose a sample range for plotting
    sample = df.iloc[100:150].copy()
    sample['timestamp'] = pd.to_datetime(sample['timestamp'])

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 7))

    for idx, row in sample.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        # Draw candle body
        ax.plot([row['timestamp'], row['timestamp']], [row['low'], row['high']], color='black')  # Wick
        ax.add_patch(plt.Rectangle(
            (row['timestamp'], min(row['open'], row['close'])),
            width=pd.Timedelta(hours=0.8),
            height=abs(row['close'] - row['open']),
            color=color
        ))

    # Add swing highs
    swing_highs = sample[sample['swing_high']]
    ax.plot(swing_highs['timestamp'], swing_highs['high'], '^', markersize=10, color='red', label='Swing High')

    # Add swing lows
    swing_lows = sample[sample['swing_low']]
    ax.plot(swing_lows['timestamp'], swing_lows['low'], 'v', markersize=10, color='green', label='Swing Low')

    # Add engulfing patterns
    engulfing = sample[sample['bullish_engulfing'] | sample['bearish_engulfing']]
    ax.plot(engulfing['timestamp'], engulfing['close'], '*', markersize=12, color='blue', label='Engulfing')

    # Add FVG regions
    bullish_fvg = sample[sample['bullish_fvg']]
    bearish_fvg = sample[sample['bearish_fvg']]

    # Bullish FVG markers (green band from high_n_2 to low)
    for _, row in bullish_fvg.iterrows():
        ax.axhspan(row['high_n_2'], row['low'], xmin=0, xmax=1, facecolor='lime', alpha=0.3, label='Bullish FVG')

    # Bearish FVG markers (red band from low_n_2 to high)
    for _, row in bearish_fvg.iterrows():
        ax.axhspan(row['high'], row['low_n_2'], xmin=0, xmax=1, facecolor='red', alpha=0.3, label='Bearish FVG')

    # Final touches
    plt.title("Candlestick Chart with Price Action Features")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv('../../data/btc_usdt/btc_usdt_1h.csv')
    pa = PriceActionFeaturesGenerator(df)

    pa.level1_features()
    pa.level2_features()
    df = pa.get_df()

    visualize(df)

if __name__ == '__main__':
    main()
