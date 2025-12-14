import re
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set matplotlib backend to Agg to avoid display issues
plt.switch_backend('Agg')

ARTIFACT_DIR = 'artifacts'
CSV_FILE = os.path.join(ARTIFACT_DIR, 'snn_lr_alpha_grid_parsed.csv')
LOG_FILE = os.path.join(ARTIFACT_DIR, 'snn_search_result.log')

def parse_log(file_path):
    records = []
    # Regex to match lines like:
    # SuperSpike | lr=1e-04, alpha=0.05 | best_acc=27.72% | baseline_epoch=None
    # SuperSpike | lr=2e-03, alpha=7.07 | best_acc=72.45% | baseline_epoch=5
    pattern = re.compile(r"^(\w+)\s+\|\s+lr=([0-9e.-]+),\s+alpha=([0-9e.+]+)\s+\|\s+best_acc=([0-9.]+)%\s+\|\s+baseline_epoch=(None|\d+)")

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            match = pattern.match(line)
            if match:
                surrogate = match.group(1)
                lr = float(match.group(2))
                alpha = float(match.group(3))
                best_acc = float(match.group(4))
                baseline_epoch_str = match.group(5)
                
                baseline_epoch = int(baseline_epoch_str) if baseline_epoch_str != 'None' else None
                
                records.append({
                    "surrogate": surrogate,
                    "lr": lr,
                    "alpha": alpha,
                    "best_acc": best_acc,
                    "baseline_epoch": baseline_epoch
                })
    return records

def plot_heatmap(df, surrogate_name, outfile):
    # Filter data for the specific surrogate
    subset = df[df['surrogate'] == surrogate_name]
    
    if subset.empty:
        print(f"No data found for surrogate: {surrogate_name}")
        return

    # Pivot the table
    # We want alpha on Y-axis and lr on X-axis
    pivot = subset.pivot_table(index="alpha", columns="lr", values="best_acc")
    
    # Sort index and columns to ensure correct order
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(pivot.values, origin="lower", cmap="viridis", aspect='auto')
    
    plt.title(f"{surrogate_name} Accuracy Heatmap")
    plt.xlabel("Learning Rate")
    plt.ylabel("Alpha")
    
    # Set ticks
    plt.xticks(ticks=range(len(pivot.columns)), labels=[f"{lr:.0e}" for lr in pivot.columns], rotation=45)
    plt.yticks(ticks=range(len(pivot.index)), labels=[f"{a:.3g}" for a in pivot.index])
    
    plt.colorbar(im, label="Best Test Accuracy (%)")
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    plt.savefig(outfile)
    plt.close()
    print(f"Saved heatmap to {outfile}")

def plot_combined_figure(df, baseline_infos, outfile):
    surrogates = ['SuperSpike', 'Sigmoid', 'Esser']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, surrogate in enumerate(surrogates):
        ax = axes[i]
        subset = df[df['surrogate'] == surrogate]
        if subset.empty:
            ax.text(0.5, 0.5, f"No data for {surrogate}", ha='center', va='center')
            continue
            
        pivot = subset.pivot_table(index="alpha", columns="lr", values="best_acc")
        pivot = pivot.sort_index(ascending=True)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        
        im = ax.imshow(pivot.values, origin="lower", cmap="viridis", aspect='auto')
        ax.set_title(f"{surrogate} Accuracy Heatmap")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Alpha")
        
        # Set ticks
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{lr:.0e}" for lr in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{a:.3g}" for a in pivot.index])
        
        fig.colorbar(im, ax=ax, label="Best Test Accuracy (%)")

    # 4th subplot for text summary
    ax_text = axes[3]
    ax_text.axis('off')
    
    summary_text = "Baseline Accuracy Target: 70% (from run_cifar10.py)\n\n"
    summary_text += "Fastest Baseline Results:\n"
    summary_text += "-" * 60 + "\n"
    
    for surrogate in surrogates:
        info = baseline_infos.get(surrogate)
        if info is not None:
            summary_text += f"Model: {surrogate} | Fastest Baseline lr={info['lr']:.0e}, alpha={info['alpha']:.3g}, epoch={int(info['baseline_epoch'])} | best_acc={info['best_acc']:.2f}%\n"
        else:
            summary_text += f"Model: {surrogate} | Did not reach baseline.\n"

    ax_text.text(0.05, 0.9, summary_text, fontsize=10, va='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"Saved combined figure to {outfile}")

def main():
    print(f"Parsing {LOG_FILE}...")
    records = parse_log(LOG_FILE)
    
    if not records:
        print("No matching records found in the log file.")
        return

    df = pd.DataFrame(records)
    
    # Save to CSV
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"Saved parsed data to {CSV_FILE}")
    
    # Get unique surrogates
    surrogates = df['surrogate'].unique()
    print(f"Found surrogates: {surrogates}")
    
    baseline_infos = {}

    for surrogate in surrogates:
        heatmap_path = os.path.join(ARTIFACT_DIR, f"heatmap_{surrogate}_parsed.png")
        plot_heatmap(df, surrogate, heatmap_path)
        
        # Find best baseline
        subset = df[df['surrogate'] == surrogate]
        reached_baseline = subset[subset['baseline_epoch'].notna()]
        if not reached_baseline.empty:
            best_baseline = reached_baseline.loc[reached_baseline['baseline_epoch'].idxmin()]
            print(f"[{surrogate}] Fastest Baseline: Epoch {int(best_baseline['baseline_epoch'])} "
                  f"(lr={best_baseline['lr']:.0e}, alpha={best_baseline['alpha']:.3g}, acc={best_baseline['best_acc']:.2f}%)")
            baseline_infos[surrogate] = best_baseline
        else:
            print(f"[{surrogate}] No configuration reached baseline.")
    
    # Plot combined figure
    combined_outfile = os.path.join(ARTIFACT_DIR, "combined_heatmap_summary.png")
    plot_combined_figure(df, baseline_infos, combined_outfile)

if __name__ == "__main__":
    main()
