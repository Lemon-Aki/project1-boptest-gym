from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# Settings
# =========================================================

INPUT_XLSX = Path("./test_result/denver_test_raw_new_300k_75k_85k_seed1-3.xlsx")
SHEET = "Raw_Data"

OUTDIR = Path("denver_pareto_outputs")
OUTDIR.mkdir(exist_ok=True)

COOLING_TESTS = ["Peak Cool", "Typical Cool"]
HEATING_TESTS = ["Peak Heat", "Typical Heat"]
ALL_FOUR_TESTS = ["Peak Cool", "Typical Cool", "Peak Heat", "Typical Heat"]

# ---------------------------------------------------------
# CONSISTENT STYLE FOR ALL FIGURES (01~09)
# shape = model / season
# color = reward weight type
# ---------------------------------------------------------

MODEL_MARKERS = {
    "All": "o",
    "Summer": "^",
    "Winter": "s",
}

WEIGHT_COLORS = {
    "8": "tab:blue",
    "16": "tab:orange",
    "26": "tab:green",
    "PI": "tab:red",
}

# display order
WEIGHT_ORDER_BY_MODEL = {
    "All": [8, 16, 26, 29],
    "Summer": [8, 13, 16, 26],
    "Winter": [8, 16, 26, 76],
}

# ---------------------------------------------------------
# Manual label offsets for the "direct-label" figures
# (05~09 and also 02~04 if needed)
# Adjust if labels overlap too much.
# ---------------------------------------------------------
LABEL_OFFSETS = {
    # Peak Cool
    ("Peak Cool", "All", 8): (8, 8),
    ("Peak Cool", "All", 16): (8, -14),
    ("Peak Cool", "All", 26): (-48, 8),
    ("Peak Cool", "All", 29): (-62, 6),

    ("Peak Cool", "Summer", 8): (8, 8),
    ("Peak Cool", "Summer", 16): (-55, -16),
    ("Peak Cool", "Summer", 26): (-55, 8),
    ("Peak Cool", "Summer", 13): (8, -16),

    # Typical Cool
    ("Typical Cool", "All", 8): (8, 8),
    ("Typical Cool", "All", 16): (8, -14),
    ("Typical Cool", "All", 26): (-48, 8),
    ("Typical Cool", "All", 29): (-60, 6),

    ("Typical Cool", "Summer", 8): (8, 8),
    ("Typical Cool", "Summer", 16): (-55, -16),
    ("Typical Cool", "Summer", 26): (-55, 8),
    ("Typical Cool", "Summer", 13): (8, -16),

    # Peak Heat
    ("Peak Heat", "All", 8): (8, 8),
    ("Peak Heat", "All", 16): (8, 8),
    ("Peak Heat", "All", 26): (-55, 8),
    ("Peak Heat", "All", 29): (-58, 8),

    ("Peak Heat", "Winter", 8): (8, 12),
    ("Peak Heat", "Winter", 16): (-55, -14),
    ("Peak Heat", "Winter", 26): (8, -16),
    ("Peak Heat", "Winter", 76): (-62, 8),

    # Typical Heat
    ("Typical Heat", "All", 8): (8, 8),
    ("Typical Heat", "All", 16): (8, 8),
    ("Typical Heat", "All", 26): (-55, 8),
    ("Typical Heat", "All", 29): (-58, 8),

    ("Typical Heat", "Winter", 8): (8, 12),
    ("Typical Heat", "Winter", 16): (-55, -14),
    ("Typical Heat", "Winter", 26): (8, -16),
    ("Typical Heat", "Winter", 76): (-62, 8),
}


# =========================================================
# Data helpers
# =========================================================

def load_data():
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET)

    # reward weight group for color consistency
    def weight_group(row):
        if str(row["weight_type"]).lower().startswith("pi"):
            return "PI"
        return str(int(row["weight"]))

    # full label written next to points
    def full_label(row):
        if str(row["weight_type"]).lower().startswith("pi"):
            return f'{row["model"]} PI {int(row["weight"])}:1'
        return f'{row["model"]} {int(row["weight"])}:1'

    df["weight_group"] = df.apply(weight_group, axis=1)
    df["plot_label"] = df.apply(full_label, axis=1)
    return df


def aggregate_seed_mean(df):
    return (
        df.groupby(
            ["model", "weight", "weight_type", "weight_group", "plot_label", "test"],
            as_index=False
        )
        .agg(
            n=("seed_no", "count"),
            cost_mean=("cost_tot", "mean"),
            cost_std=("cost_tot", "std"),
            tdis_mean=("tdis_tot", "mean"),
            tdis_std=("tdis_tot", "std"),
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
        )
    )


def pareto_mask(cost, tdis):
    """
    Nondominated points for minimizing BOTH cost and tdis.
    """
    cost = np.asarray(cost, dtype=float)
    tdis = np.asarray(tdis, dtype=float)
    keep = np.ones(len(cost), dtype=bool)

    for i in range(len(cost)):
        dominated = (
            (cost <= cost[i]) &
            (tdis <= tdis[i]) &
            ((cost < cost[i]) | (tdis < tdis[i]))
        )
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep


def marker_for_model(model):
    return MODEL_MARKERS[model]


def color_for_weight_group(weight_group):
    return WEIGHT_COLORS[weight_group]


def label_offset(test, model, weight):
    return LABEL_OFFSETS.get((test, model, int(weight)), (6, 5))


# =========================================================
# Legend helpers
# =========================================================

def add_shape_and_color_legend(fig, include_models=("All", "Summer", "Winter"),
                               include_weight_groups=("8", "16", "26", "PI"),
                               add_pareto=True,
                               bbox=(0.5, 0.01)):
    """
    Figure-level legends:
      - shape = model
      - color = weight group
    """
    shape_handles = [
        Line2D(
            [0], [0],
            marker=MODEL_MARKERS[m],
            linestyle="none",
            markersize=8,
            markerfacecolor="lightgray",
            markeredgecolor="black",
            label=m
        )
        for m in include_models
    ]

    color_label_map = {
        "8": "8:1",
        "16": "16:1",
        "26": "26:1",
        "PI": "PI",
    }

    color_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor=WEIGHT_COLORS[g],
            markeredgecolor="black",
            label=color_label_map[g]
        )
        for g in include_weight_groups
    ]

    handles = shape_handles + color_handles
    labels = [h.get_label() for h in handles]

    if add_pareto:
        pareto_handle = Line2D(
            [0], [0],
            linestyle="--",
            color="tab:blue",
            linewidth=1.5,
            label="Empirical Pareto front"
        )
        handles.append(pareto_handle)
        labels.append("Pareto front")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=bbox,
        ncol=min(len(handles), 6),
        frameon=True,
        fontsize=9,
    )


# =========================================================
# Common plotting helpers
# =========================================================

def set_axis(ax, title):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("cost_tot")
    ax.set_ylabel("tdis_tot")
    ax.grid(True, alpha=0.25)


def plot_direct_labeled_points(ax, g, test, use_mean=True):
    """
    Plot points with consistent shape/color and direct text labels.
    If use_mean=True, expects columns cost_mean / tdis_mean.
    Else expects cost_tot / tdis_tot.
    """
    xcol = "cost_mean" if use_mean else "cost_tot"
    ycol = "tdis_mean" if use_mean else "tdis_tot"

    for _, r in g.iterrows():
        ax.scatter(
            r[xcol],
            r[ycol],
            marker=marker_for_model(r["model"]),
            s=75,
            color=color_for_weight_group(r["weight_group"]),
            edgecolors="none",
            zorder=3,
        )

        dx, dy = label_offset(test, r["model"], r["weight"])
        ax.annotate(
            r["plot_label"],
            (r[xcol], r[ycol]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            ha="left",
            va="center",
        )


def draw_pareto(ax, x, y):
    pm = pareto_mask(x, y)
    front = pd.DataFrame({"x": x, "y": y})
    front = front.loc[pm].sort_values("x")

    if len(front) >= 2:
        ax.plot(
            front["x"],
            front["y"],
            linestyle="--",
            linewidth=1.5,
            color="tab:blue",
            zorder=2,
        )


# =========================================================
# 01. All seed distribution
# =========================================================

def plot_all_seed_distribution(df):
    """
    4 panels:
      Peak Cool / Typical Cool / Peak Heat / Typical Heat

    Cooling panels -> All vs Summer
    Heating panels -> All vs Winter

    Seed points:
      same style rule as all other figures
      shape=model, color=weight group

    Mean points:
      same shape/color, but larger and black-edged
    """
    specs = [
        ("Peak Cool", ["All", "Summer"], "Cooling — Peak Cool"),
        ("Typical Cool", ["All", "Summer"], "Cooling — Typical Cool"),
        ("Peak Heat", ["All", "Winter"], "Heating — Peak Heat"),
        ("Typical Heat", ["All", "Winter"], "Heating — Typical Heat"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()

    for ax, (test, models, title) in zip(axes, specs):
        sub = df[(df["test"] == test) & (df["model"].isin(models))].copy()

        for (model, weight, weight_type, weight_group, label), g in sub.groupby(
            ["model", "weight", "weight_type", "weight_group", "plot_label"]
        ):
            xmean = g["cost_tot"].mean()
            ymean = g["tdis_tot"].mean()

            # seed points
            for _, r in g.iterrows():
                ax.scatter(
                    r["cost_tot"],
                    r["tdis_tot"],
                    marker=marker_for_model(model),
                    s=55,
                    color=color_for_weight_group(weight_group),
                    alpha=0.80,
                    edgecolors="none",
                    zorder=3,
                )
                ax.annotate(
                    f'S{int(r["seed_no"])}',
                    (r["cost_tot"], r["tdis_tot"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

                # connector to mean
                ax.plot(
                    [r["cost_tot"], xmean],
                    [r["tdis_tot"], ymean],
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.35,
                    color=color_for_weight_group(weight_group),
                    zorder=1,
                )

            # mean point (same style, larger)
            ax.scatter(
                xmean,
                ymean,
                marker=marker_for_model(model),
                s=110,
                color=color_for_weight_group(weight_group),
                edgecolors="black",
                linewidths=0.7,
                zorder=4,
            )

        set_axis(ax, title)

    add_shape_and_color_legend(
        fig,
        include_models=("All", "Summer", "Winter"),
        include_weight_groups=("8", "16", "26", "PI"),
        add_pareto=False,
        bbox=(0.5, 0.01)
    )

    fig.suptitle(
        "01. Seed distribution by reward weight — cost–comfort plane",
        fontsize=15,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])

    fig.savefig(OUTDIR / "01_all_seed_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 02~04. Model-wise reward-weight Pareto (seed mean)
# =========================================================

def plot_model_weight_pareto(agg, model, tests, filename, fig_title):
    """
    02 All model
    03 Summer model
    04 Winter model
    """
    sub = agg[(agg["model"] == model) & (agg["test"].isin(tests))].copy()

    fig, axes = plt.subplots(1, len(tests), figsize=(6.5 * len(tests), 5.2), squeeze=False)
    axes = axes.ravel()

    for ax, test in zip(axes, tests):
        g = sub[sub["test"] == test].copy()

        # weight order for that model
        order = WEIGHT_ORDER_BY_MODEL[model]
        g["order"] = g["weight"].apply(lambda x: order.index(int(x)) if int(x) in order else 999)
        g = g.sort_values("order")

        plot_direct_labeled_points(ax, g, test, use_mean=True)
        draw_pareto(ax, g["cost_mean"].to_numpy(), g["tdis_mean"].to_numpy())
        set_axis(ax, f"{model} — {test}")

    # Only need color legend + shape legend
    add_shape_and_color_legend(
        fig,
        include_models=(model,),
        include_weight_groups=("8", "16", "26", "PI"),
        add_pareto=True,
        bbox=(0.5, 0.01)
    )

    fig.suptitle(fig_title, fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    fig.savefig(OUTDIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 05~07. Seed-wise Pareto
# =========================================================

def plot_seed_pareto(df, seed_no):
    """
    For each seed:
      Cooling -> All vs Summer
      Heating -> All vs Winter
    """
    seed_df = df[df["seed_no"] == seed_no].copy()

    specs = [
        ("Peak Cool", ["All", "Summer"], "Peak Cool"),
        ("Typical Cool", ["All", "Summer"], "Typical Cool"),
        ("Peak Heat", ["All", "Winter"], "Peak Heat"),
        ("Typical Heat", ["All", "Winter"], "Typical Heat"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()

    for ax, (test, models, title) in zip(axes, specs):
        g = seed_df[(seed_df["test"] == test) & (seed_df["model"].isin(models))].copy()

        plot_direct_labeled_points(ax, g, test, use_mean=False)
        draw_pareto(ax, g["cost_tot"].to_numpy(), g["tdis_tot"].to_numpy())
        set_axis(ax, title)

    add_shape_and_color_legend(
        fig,
        include_models=("All", "Summer", "Winter"),
        include_weight_groups=("8", "16", "26", "PI"),
        add_pareto=True,
        bbox=(0.5, 0.01)
    )

    fig.suptitle(
        f"{4 + seed_no:02d}. Seed {seed_no}: All vs seasonal models — cost–comfort Pareto",
        fontsize=15,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])

    fig.savefig(OUTDIR / f"{4 + seed_no:02d}_seed{seed_no}_pareto.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 08~09. Seed-mean All vs Seasonal comparison
# =========================================================

def plot_mean_all_vs_seasonal(agg, season_model, tests, filename, fig_title):
    """
    08 mean All vs Summer
    09 mean All vs Winter
    """
    sub = agg[(agg["model"].isin(["All", season_model])) & (agg["test"].isin(tests))].copy()

    fig, axes = plt.subplots(1, len(tests), figsize=(6.8 * len(tests), 5.2), squeeze=False)
    axes = axes.ravel()

    for ax, test in zip(axes, tests):
        g = sub[sub["test"] == test].copy()

        plot_direct_labeled_points(ax, g, test, use_mean=True)
        draw_pareto(ax, g["cost_mean"].to_numpy(), g["tdis_mean"].to_numpy())
        set_axis(ax, f"{fig_title.split(':')[0]} — {test}")

    include_models = ("All", season_model)

    add_shape_and_color_legend(
        fig,
        include_models=include_models,
        include_weight_groups=("8", "16", "26", "PI"),
        add_pareto=True,
        bbox=(0.5, 0.01)
    )

    fig.suptitle(fig_title, fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    fig.savefig(OUTDIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# CSV summaries
# =========================================================

def save_summary_tables(df, agg):
    agg.to_csv(OUTDIR / "reward_weight_seed_mean_summary.csv", index=False)

    cols = [
        "model", "weight", "weight_type", "weight_group",
        "plot_label", "seed_no", "test", "cost_tot", "tdis_tot", "reward"
    ]
    df[cols].sort_values(
        ["seed_no", "model", "test", "weight"]
    ).to_csv(OUTDIR / "seed_level_cost_tdis.csv", index=False)


# =========================================================
# Main
# =========================================================

def main():
    df = load_data()
    agg = aggregate_seed_mean(df)

    # 01
    plot_all_seed_distribution(df)

    # 02~04
    plot_model_weight_pareto(
        agg,
        model="All",
        tests=ALL_FOUR_TESTS,
        filename="02_model_weight_pareto_all.png",
        fig_title="02. All model: reward-weight Pareto (seed mean)",
    )

    plot_model_weight_pareto(
        agg,
        model="Summer",
        tests=COOLING_TESTS,
        filename="03_model_weight_pareto_summer.png",
        fig_title="03. Summer model: reward-weight Pareto (seed mean)",
    )

    plot_model_weight_pareto(
        agg,
        model="Winter",
        tests=HEATING_TESTS,
        filename="04_model_weight_pareto_winter.png",
        fig_title="04. Winter model: reward-weight Pareto (seed mean)",
    )

    # 05~07
    for seed_no in sorted(df["seed_no"].dropna().unique()):
        plot_seed_pareto(df, int(seed_no))

    # 08~09
    plot_mean_all_vs_seasonal(
        agg,
        season_model="Summer",
        tests=COOLING_TESTS,
        filename="08_mean_all_vs_summer.png",
        fig_title="08. Cooling: All vs Summer (seed mean)",
    )

    plot_mean_all_vs_seasonal(
        agg,
        season_model="Winter",
        tests=HEATING_TESTS,
        filename="09_mean_all_vs_winter.png",
        fig_title="09. Heating: All vs Winter (seed mean)",
    )

    # tables
    save_summary_tables(df, agg)

    print("Created files:")
    for p in sorted(OUTDIR.iterdir()):
        print(" ", p.name)


if __name__ == "__main__":
    main()