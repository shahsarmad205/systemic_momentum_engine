import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def run_bootstrap(
    equity_csv: str = "output/backtests/daily_equity.csv",
    n_simulations: int = 1000,
    block_size: int = 21,  # monthly blocks preserve autocorrelation
    output_dir: str = "output/research",
):
    # Load daily returns
    e = pd.read_csv(equity_csv)
    e["ret"] = pd.to_numeric(e["equity"], errors="coerce").pct_change()
    returns = e["ret"].dropna().values
    n = len(returns)

    print(f"Bootstrap analysis: {n} daily returns")
    print(f"Simulations: {n_simulations}")
    print(f"Block size: {block_size} days")
    print()

    # Block bootstrap — preserves autocorrelation structure
    simulated_sharpes = []
    simulated_cagrs = []
    simulated_maxdds = []

    for _ in range(n_simulations):
        # Sample blocks with replacement
        n_blocks = int(np.ceil(n / block_size))
        blocks = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n - block_size)
            blocks.extend(returns[start : start + block_size])
        sim_returns = np.array(blocks[:n])

        # Compute metrics for this simulation
        sharpe = sim_returns.mean() / sim_returns.std() * np.sqrt(252)
        cagr = (1 + sim_returns).prod() ** (252 / n) - 1

        # Max drawdown
        cumulative = (1 + sim_returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_dd = drawdowns.min()

        simulated_sharpes.append(sharpe)
        simulated_cagrs.append(cagr)
        simulated_maxdds.append(max_dd)

    sharpes = np.array(simulated_sharpes)
    cagrs = np.array(simulated_cagrs)
    maxdds = np.array(simulated_maxdds)

    # Compute actual metrics
    actual_sharpe = returns.mean() / returns.std() * np.sqrt(252)
    actual_cagr = (1 + returns).prod() ** (252 / n) - 1

    # Print results
    print("=" * 55)
    print("BOOTSTRAP VALIDATION RESULTS")
    print("=" * 55)
    print()
    print(
        f"{'Metric':<20} {'Actual':>10} {'p10':>8} "
        f"{'p25':>8} {'p50':>8} {'p75':>8} {'p90':>8}"
    )
    print("-" * 70)

    for metric, actual, sims in [
        ("Sharpe", actual_sharpe, sharpes),
        ("CAGR", actual_cagr, cagrs),
        ("Max Drawdown", None, maxdds),
    ]:
        p10 = np.percentile(sims, 10)
        p25 = np.percentile(sims, 25)
        p50 = np.percentile(sims, 50)
        p75 = np.percentile(sims, 75)
        p90 = np.percentile(sims, 90)

        if actual is not None:
            print(
                f"{metric:<20} {actual:>10.3f} {p10:>8.3f} "
                f"{p25:>8.3f} {p50:>8.3f} {p75:>8.3f} {p90:>8.3f}"
            )
        else:
            print(
                f"{metric:<20} {'N/A':>10} {p10:>8.3f} "
                f"{p25:>8.3f} {p50:>8.3f} {p75:>8.3f} {p90:>8.3f}"
            )

    print()

    # Key questions
    prob_sharpe_above_half = (sharpes > 0.5).mean()
    prob_sharpe_above_one = (sharpes > 1.0).mean()
    prob_positive_cagr = (cagrs > 0).mean()

    print("=" * 55)
    print("KEY QUESTIONS")
    print("=" * 55)
    print(f"P(Sharpe > 0.5):  {prob_sharpe_above_half:.1%}")
    print(f"P(Sharpe > 1.0):  {prob_sharpe_above_one:.1%}")
    print(f"P(CAGR > 0):      {prob_positive_cagr:.1%}")
    print()

    # Verdict
    print("=" * 55)
    print("VERDICT")
    print("=" * 55)
    if prob_sharpe_above_half > 0.80 and prob_positive_cagr > 0.90:
        print("✅ STRONG: Edge is robust across simulations")
        print("   Suitable for paper trading")
    elif prob_sharpe_above_half > 0.60 and prob_positive_cagr > 0.75:
        print("⚠️  MODERATE: Edge exists but uncertain")
        print("   More data or stronger signal needed")
    else:
        print("❌ WEAK: Edge may not be real")
        print("   Do not trade live")

    # Plot distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Bootstrap Validation — 1000 Simulations", fontsize=14, fontweight="bold")

    for ax, data, actual, label, threshold in [
        (axes[0], sharpes, actual_sharpe, "Sharpe Ratio", 1.0),
        (axes[1], cagrs * 100, actual_cagr * 100, "CAGR (%)", 5.0),
        (axes[2], maxdds * 100, None, "Max Drawdown (%)", -20.0),
    ]:
        ax.hist(data, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(np.percentile(data, 10), color="red", linestyle="--", label="10th pct", linewidth=2)
        ax.axvline(np.percentile(data, 90), color="green", linestyle="--", label="90th pct", linewidth=2)
        if actual is not None:
            ax.axvline(actual, color="orange", linestyle="-", label="Actual", linewidth=2)
        ax.axvline(
            threshold,
            color="black",
            linestyle=":",
            label=f"Target {threshold}",
            linewidth=1.5,
        )
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/bootstrap_validation.png", dpi=150, bbox_inches="tight")
    print()
    print(f"Plot saved: {output_dir}/bootstrap_validation.png")

    # Save raw results
    results_df = pd.DataFrame(
        {
            "sharpe": sharpes,
            "cagr": cagrs,
            "max_drawdown": maxdds,
        }
    )
    results_df.to_csv(f"{output_dir}/bootstrap_results.csv", index=False)

    return {
        "actual_sharpe": actual_sharpe,
        "p10_sharpe": np.percentile(sharpes, 10),
        "p50_sharpe": np.percentile(sharpes, 50),
        "p90_sharpe": np.percentile(sharpes, 90),
        "prob_sharpe_above_half": prob_sharpe_above_half,
        "prob_sharpe_above_one": prob_sharpe_above_one,
        "prob_positive_cagr": prob_positive_cagr,
    }


if __name__ == "__main__":
    run_bootstrap()
