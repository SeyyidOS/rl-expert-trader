import os
import re
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt


def plot_data(log_dir):
    """
    Reads CSVs named like:
      <process_id>_trade_log_episode_<episode_id>.csv
    For each file, computes the mean net PnL from the 'net_pnl' column and,
    if present, extracts the pair from the 'pair' column.
    """
    # Regex to capture process and episode (e.g., "0_trade_log_episode_3.csv")
    pattern = re.compile(r'^(\d+)_trade_log_episode_(\d+)\.csv$')
    data_records = []
    file_pattern = os.path.join(log_dir, '*_trade_log_episode_*.csv')
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        print(f"No CSV files found in: {log_dir}")
        return None

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match:
            process_id_str, episode_id_str = match.groups()
            process_id = int(process_id_str)
            episode_id = int(episode_id_str)

            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            if 'net_pnl' not in df.columns:
                print(f"Warning: 'net_pnl' column not found in {filename}. Skipping.")
                continue

            mean_pnl = df['net_pnl'].mean()

            # Extract pair info if the 'pair' column exists (assume one value per file)
            pair_val = None
            if 'pair' in df.columns:
                unique_pairs = df['pair'].unique()
                pair_val = unique_pairs[0] if len(unique_pairs) > 0 else None

            data_records.append((process_id, episode_id, mean_pnl, pair_val))

    # Create DataFrame of aggregated results.
    df_agg = pd.DataFrame(data_records, columns=['process_id', 'episode_id', 'mean_net_pnl', 'pair'])
    if df_agg.empty:
        print("No valid data to plot.")
        return None

    # Sort by process then episode for nicer plotting.
    df_agg.sort_values(['process_id', 'episode_id'], inplace=True)
    return df_agg


def plot_process_subplots(df_agg):
    """
    Creates a subplot grid (two columns) where each subplot corresponds to
    a process and shows mean net PnL over episodes.
    Annotates each data point with its pair information if available.
    """
    unique_processes = df_agg['process_id'].unique()
    num_procs = len(unique_processes)
    ncols = 2
    nrows = math.ceil(num_procs / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, proc_id in enumerate(unique_processes):
        ax = axes_flat[i]
        proc_data = df_agg[df_agg['process_id'] == proc_id]
        ax.plot(proc_data['episode_id'], proc_data['mean_net_pnl'],
                marker='o', linestyle='-', label=f'Process {proc_id}')
        # # Annotate each data point with pair if available.
        # for _, row in proc_data.iterrows():
        #     if pd.notna(row['pair']):
        #         ax.annotate(str(row['pair']),
        #                     (row['episode_id'], row['mean_net_pnl']),
        #                     textcoords="offset points", xytext=(5, 5),
        #                     ha='center', fontsize=8)
        ax.set_title(f'Process {proc_id} - Mean Net PnL')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Net PnL')
        ax.legend()

    # Turn off unused axes if the grid is larger than needed.
    for j in range(len(unique_processes), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.show()


def plot_overall_mean(df_agg):
    """
    Plots the overall mean net PnL over episodes (aggregated across all processes).
    Also scatters individual data points and annotates them with their pair information.
    """
    # Compute overall mean net PnL per episode (averaged over processes).
    overall = df_agg.groupby('episode_id')['mean_net_pnl'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(overall['episode_id'], overall['mean_net_pnl'], marker='o',
            linestyle='-', color='blue', label='Overall Mean Net PnL')

    # Scatter all individual data points and annotate with pair info.
    # ax.scatter(df_agg['episode_id'], df_agg['mean_net_pnl'], color='red')
    # for _, row in df_agg.iterrows():
    #     if pd.notna(row['pair']):
    #         ax.annotate(str(row['pair']),
    #                     (row['episode_id'], row['mean_net_pnl']),
    #                     textcoords="offset points", xytext=(5, 5),
    #                     ha='center', fontsize=8)

    ax.set_title("Overall Mean Net PnL over Episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Net PnL")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Adjust this path to where your CSV logs are stored.
    log_directory = r"D:\Projects\rl-expert-trader\logs\tensorboard\SpotTrading-CustomLSTM-PNL\PPO_run_20250415_223147\trade_logs"

    df_data = plot_data(log_directory)
    if df_data is not None:
        # Plot each process in subplots (two columns).
        plot_process_subplots(df_data)
        # Plot overall mean net PnL over episodes.
        plot_overall_mean(df_data)
