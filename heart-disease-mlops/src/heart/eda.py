"""Exploratory data analysis: writes figures to reports/figures and a summary file.

    python -m src.heart.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ROOT, TARGET, load_clean  # noqa: E402

FIG = ROOT / "reports" / "figures"
sns.set_theme(style="whitegrid", palette="deep")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_clean()

    # 1. class balance
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["No disease (0)", "Disease (1)"], counts.values, color=["#4c72b0", "#c44e52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, f"{v} ({v / len(df):.0%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("patients")
    fig.tight_layout()
    fig.savefig(FIG / "class_balance.png", dpi=150)
    plt.close(fig)

    # 2. numeric histograms split by target
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, bins=20, ax=ax, stat="density",
                     common_norm=False)
        ax.set_title(f"Distribution of {col}")
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "histograms.png", dpi=150)
    plt.close(fig)

    # 3. correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature correlation heatmap")
    fig.tight_layout()
    fig.savefig(FIG / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 4. categorical features vs target
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=TARGET, ax=ax)
        ax.set_title(f"{col} vs target")
    fig.tight_layout()
    fig.savefig(FIG / "categorical_vs_target.png", dpi=150)
    plt.close(fig)

    # 5. boxplots for outlier inspection
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "boxplots.png", dpi=150)
    plt.close(fig)

    summary = [
        f"rows={len(df)} cols={df.shape[1]}",
        "missing values:", df.isna().sum()[df.isna().sum() > 0].to_string() or "none",
        "target balance:", df[TARGET].value_counts(normalize=True).round(3).to_string(),
        "describe:", df[NUMERIC_FEATURES].describe().round(2).to_string(),
    ]
    (ROOT / "reports" / "eda_summary.txt").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
