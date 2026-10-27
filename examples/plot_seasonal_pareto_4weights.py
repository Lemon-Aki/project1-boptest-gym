"""
plot_seasonal_pareto_4weights.py

Seasonal reward-weight validation Pareto plots.

Input
-----
boptest_seasonal_weights_8_13_16_26_109_pareto.xlsx

Sheets
------
Mean_Pareto
Seed_Pareto

Compared weights
----------------
Summer:
  8:1, 13:1, 16:1, 26:1
  Tests: Peak Cool, Typical Cool

Winter:
  8:1, 16:1, 26:1, 109:1
  Tests: Peak Heat, Typical Heat

Outputs
-------
1) 5-seed mean raw Pareto plots
2) 5-seed mean normalized Pareto plots
3) Seed1~Seed5 raw Pareto 2x2 plots
4) Seed1~Seed5 normalized Pareto 2x2 plots
5) Pareto summary CSVs

Notes
-----
- cost_tot and tdis_tot are both minimized.
- Mean plots use only the 5-seed mean points; no SD error bars.
- Normalization is performed separately within each Model × Test
  across the four compared weight candidates.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = Path(
    "boptest_seasonal_weights_8_13_16_26_109_pareto.xlsx"
)

OUTPUT_DIR = Path(
    "seasonal_pareto_4weights_output"
)
OUTPUT_DIR.mkdir(exist_ok=True)

DPI = 300

COMPARISONS = [
    ("Summer", "Peak Cool"),
    ("Summer", "Typical Cool"),
    ("Winter", "Peak Heat"),
    ("Winter", "Typical Heat"),
]


# ============================================================
# 1. Helpers
# ============================================================

def find_header(path, sheet_name, required):
    preview = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=None,
        nrows=10,
    )

    for i in range(len(preview)):
        vals = {
            str(v).strip()
            for v in preview.iloc[i].dropna().tolist()
        }
        if required.issubset(vals):
            return i

    raise ValueError(
        f"{sheet_name} 시트의 header를 찾지 못했습니다."
    )


def pareto_mask(points):
    """
    points[:,0] = Cost
    points[:,1] = Tdis
    Both minimized.
    """
    n = len(points)
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            if (
                np.all(points[j] <= points[i])
                and np.any(points[j] < points[i])
            ):
                mask[i] = False
                break

    return mask


def normalize(sub, cost_col, tdis_col):
    out = sub.copy()

    for raw, norm in [
        (cost_col, "Cost norm"),
        (tdis_col, "Tdis norm"),
    ]:
        lo = out[raw].min()
        hi = out[raw].max()

        if np.isclose(lo, hi):
            out[norm] = 0.0
        else:
            out[norm] = (
                out[raw] - lo
            ) / (hi - lo)

    return out


def marker_for_weight(weight):
    return {
        8: "o",
        13: "^",
        16: "s",
        26: "D",
        109: "P",
    }.get(int(weight), "o")


def annotate(ax, sub, xcol, ycol):
    offsets = [
        (7, 7),
        (7, -14),
        (-55, 9),
        (-60, -16),
    ]

    for i, (_, r) in enumerate(
        sub.sort_values("Weight").iterrows()
    ):
        dx, dy = offsets[i % len(offsets)]

        ax.annotate(
            f'{int(r["Weight"])}:1',
            (r[xcol], r[ycol]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
        )


def draw_front(ax, sub, xcol, ycol):
    front = (
        sub[sub["Pareto"]]
        .sort_values(xcol)
    )

    if len(front) >= 2:
        ax.plot(
            front[xcol],
            front[ycol],
            linestyle="--",
            linewidth=1.6,
            label="Empirical Pareto front",
        )


# ============================================================
# 2. Read workbook
# ============================================================

mean_required = {
    "Model","Test","Setting","Weight",
    "Cost mean","Tdis mean","Pareto?"
}

seed_required = {
    "Model","Test","Seed","Numeric Seed",
    "Setting","Weight","Cost","Tdis","Pareto?"
}

mean_header = find_header(
    INPUT_FILE,
    "Mean_Pareto",
    mean_required,
)

seed_header = find_header(
    INPUT_FILE,
    "Seed_Pareto",
    seed_required,
)

mean_df = pd.read_excel(
    INPUT_FILE,
    sheet_name="Mean_Pareto",
    header=mean_header,
)

seed_df = pd.read_excel(
    INPUT_FILE,
    sheet_name="Seed_Pareto",
    header=seed_header,
)

mean_df.columns = [
    str(c).strip()
    for c in mean_df.columns
]

seed_df.columns = [
    str(c).strip()
    for c in seed_df.columns
]

for c in [
    "Weight",
    "Cost mean",
    "Tdis mean",
]:
    mean_df[c] = pd.to_numeric(
        mean_df[c],
        errors="coerce",
    )

for c in [
    "Seed",
    "Numeric Seed",
    "Weight",
    "Cost",
    "Tdis",
]:
    seed_df[c] = pd.to_numeric(
        seed_df[c],
        errors="coerce",
    )

mean_df = mean_df.dropna(
    subset=[
        "Model","Test","Weight",
        "Cost mean","Tdis mean"
    ]
)

seed_df = seed_df.dropna(
    subset=[
        "Model","Test","Seed","Weight",
        "Cost","Tdis"
    ]
)

seed_df["Seed"] = seed_df["Seed"].astype(int)


# ============================================================
# 3. Ensure Pareto flags are recalculated in Python
# ============================================================

def get_mean_points(model, test):
    sub = mean_df[
        (mean_df["Model"] == model)
        & (mean_df["Test"] == test)
    ].copy()

    if len(sub) != 4:
        raise ValueError(
            f"{model}/{test}: "
            f"4 weight points expected, found {len(sub)}."
        )

    sub["Pareto"] = pareto_mask(
        sub[
            ["Cost mean","Tdis mean"]
        ].to_numpy()
    )

    return sub.reset_index(drop=True)


def get_seed_points(model, test, seed):
    sub = seed_df[
        (seed_df["Model"] == model)
        & (seed_df["Test"] == test)
        & (seed_df["Seed"] == seed)
    ].copy()

    if len(sub) != 4:
        raise ValueError(
            f"{model}/{test}/Seed{seed}: "
            f"4 weight points expected, found {len(sub)}."
        )

    sub["Pareto"] = pareto_mask(
        sub[
            ["Cost","Tdis"]
        ].to_numpy()
    )

    return sub.reset_index(drop=True)


# ============================================================
# 4. Mean raw plots
# ============================================================

def draw_mean_raw(ax, model, test):
    sub = get_mean_points(
        model,
        test,
    )

    for _, r in sub.iterrows():
        ax.scatter(
            r["Cost mean"],
            r["Tdis mean"],
            marker=marker_for_weight(
                r["Weight"]
            ),
            s=92,
            label=f'{int(r["Weight"])}:1',
        )

    draw_front(
        ax,
        sub,
        "Cost mean",
        "Tdis mean",
    )

    annotate(
        ax,
        sub,
        "Cost mean",
        "Tdis mean",
    )

    ax.set_title(
        f"{model} — {test}"
    )
    ax.set_xlabel("cost_tot")
    ax.set_ylabel("tdis_tot")
    ax.grid(alpha=0.25)

    return sub


def save_mean_raw_single(model, test):
    fig, ax = plt.subplots(
        figsize=(7.3, 5.7)
    )

    sub = draw_mean_raw(
        ax,
        model,
        test,
    )

    handles, labels = (
        ax.get_legend_handles_labels()
    )
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    name = (
        f"mean_raw_"
        f"{model.lower()}_"
        f"{test.replace(' ','_').lower()}.png"
    )

    fig.savefig(
        OUTPUT_DIR / name,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return sub


# ============================================================
# 5. Mean normalized plots
# ============================================================

def draw_mean_norm(ax, model, test):
    sub = normalize(
        get_mean_points(
            model,
            test,
        ),
        "Cost mean",
        "Tdis mean",
    )

    for _, r in sub.iterrows():
        ax.scatter(
            r["Cost norm"],
            r["Tdis norm"],
            marker=marker_for_weight(
                r["Weight"]
            ),
            s=92,
            label=f'{int(r["Weight"])}:1',
        )

    draw_front(
        ax,
        sub,
        "Cost norm",
        "Tdis norm",
    )

    annotate(
        ax,
        sub,
        "Cost norm",
        "Tdis norm",
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_title(
        f"{model} — {test}"
    )
    ax.set_xlabel(
        "Normalized cost"
    )
    ax.set_ylabel(
        "Normalized tdis"
    )
    ax.grid(alpha=0.25)

    return sub


def save_mean_norm_single(model, test):
    fig, ax = plt.subplots(
        figsize=(6.2, 6.0)
    )

    sub = draw_mean_norm(
        ax,
        model,
        test,
    )

    handles, labels = (
        ax.get_legend_handles_labels()
    )
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    name = (
        f"mean_normalized_"
        f"{model.lower()}_"
        f"{test.replace(' ','_').lower()}.png"
    )

    fig.savefig(
        OUTPUT_DIR / name,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return sub


# ============================================================
# 6. Mean 2x2 figures
# ============================================================

def save_mean_2x2(normalized=False):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            (12,10)
            if normalized
            else (13,10)
        ),
    )

    for ax, (
        model,
        test,
    ) in zip(
        axes.ravel(),
        COMPARISONS,
    ):
        if normalized:
            draw_mean_norm(
                ax,
                model,
                test,
            )
        else:
            draw_mean_raw(
                ax,
                model,
                test,
            )

    fig.suptitle(
        (
            "Seasonal reward weights — "
            + (
                "Normalized"
                if normalized
                else "Raw"
            )
            + " cost–comfort Pareto "
            "(5-seed mean)"
        ),
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0,0,1,0.96]
    )

    out = (
        "mean_normalized_all_tests.png"
        if normalized
        else "mean_raw_all_tests.png"
    )

    fig.savefig(
        OUTPUT_DIR / out,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 7. Per-seed plots
# ============================================================

def draw_seed_raw(
    ax,
    model,
    test,
    seed,
):
    sub = get_seed_points(
        model,
        test,
        seed,
    )

    for _, r in sub.iterrows():
        ax.scatter(
            r["Cost"],
            r["Tdis"],
            marker=marker_for_weight(
                r["Weight"]
            ),
            s=82,
            label=f'{int(r["Weight"])}:1',
        )

    draw_front(
        ax,
        sub,
        "Cost",
        "Tdis",
    )

    annotate(
        ax,
        sub,
        "Cost",
        "Tdis",
    )

    ax.set_title(
        f"{model} — {test}"
    )
    ax.set_xlabel("cost_tot")
    ax.set_ylabel("tdis_tot")
    ax.grid(alpha=0.25)

    return sub


def draw_seed_norm(
    ax,
    model,
    test,
    seed,
):
    sub = normalize(
        get_seed_points(
            model,
            test,
            seed,
        ),
        "Cost",
        "Tdis",
    )

    for _, r in sub.iterrows():
        ax.scatter(
            r["Cost norm"],
            r["Tdis norm"],
            marker=marker_for_weight(
                r["Weight"]
            ),
            s=82,
            label=f'{int(r["Weight"])}:1',
        )

    draw_front(
        ax,
        sub,
        "Cost norm",
        "Tdis norm",
    )

    annotate(
        ax,
        sub,
        "Cost norm",
        "Tdis norm",
    )

    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(-0.05,1.05)
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_title(
        f"{model} — {test}"
    )
    ax.set_xlabel(
        "Normalized cost"
    )
    ax.set_ylabel(
        "Normalized tdis"
    )
    ax.grid(alpha=0.25)

    return sub


def save_seed_2x2(
    seed,
    normalized=False,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            (12,10)
            if normalized
            else (13,10)
        ),
    )

    for ax, (
        model,
        test,
    ) in zip(
        axes.ravel(),
        COMPARISONS,
    ):
        if normalized:
            draw_seed_norm(
                ax,
                model,
                test,
                seed,
            )
        else:
            draw_seed_raw(
                ax,
                model,
                test,
                seed,
            )

    fig.suptitle(
        (
            f"Seed {seed} — "
            + (
                "Normalized"
                if normalized
                else "Raw"
            )
            + " seasonal reward-weight Pareto"
        ),
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0,0,1,0.96]
    )

    seed_dir = (
        OUTPUT_DIR
        / f"seed{seed}"
    )
    seed_dir.mkdir(exist_ok=True)

    out = (
        f"seed{seed}_normalized_all_tests.png"
        if normalized
        else f"seed{seed}_raw_all_tests.png"
    )

    fig.savefig(
        seed_dir / out,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 8. Run
# ============================================================

mean_raw = []
mean_norm = []

for model, test in COMPARISONS:
    mean_raw.append(
        save_mean_raw_single(
            model,
            test,
        )
    )
    mean_norm.append(
        save_mean_norm_single(
            model,
            test,
        )
    )

save_mean_2x2(
    normalized=False,
)
save_mean_2x2(
    normalized=True,
)

for seed in range(1,6):
    save_seed_2x2(
        seed,
        normalized=False,
    )
    save_seed_2x2(
        seed,
        normalized=True,
    )


# ============================================================
# 9. Save summary CSV
# ============================================================

mean_raw_df = pd.concat(
    mean_raw,
    ignore_index=True,
)

mean_norm_df = pd.concat(
    mean_norm,
    ignore_index=True,
)

mean_raw_df.to_csv(
    OUTPUT_DIR
    / "mean_pareto_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

mean_norm_df.to_csv(
    OUTPUT_DIR
    / "mean_pareto_normalized_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# Seed Pareto frequency
freq_rows = []

for model, test in COMPARISONS:
    weights = sorted(
        seed_df[
            (seed_df["Model"] == model)
            & (seed_df["Test"] == test)
        ]["Weight"].unique()
    )

    counts = {
        int(w): 0
        for w in weights
    }

    for seed in range(1,6):
        sub = get_seed_points(
            model,
            test,
            seed,
        )

        for _, r in sub[
            sub["Pareto"]
        ].iterrows():
            counts[int(r["Weight"])] += 1

    for weight, count in counts.items():
        freq_rows.append({
            "Model": model,
            "Test": test,
            "Weight": weight,
            "Pareto count": count,
            "Pareto frequency": count / 5,
        })

pd.DataFrame(
    freq_rows
).to_csv(
    OUTPUT_DIR
    / "seed_pareto_frequency.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 10. Console summary
# ============================================================

print(
    "\n=== 5-seed mean empirical Pareto sets ==="
)

for model, test in COMPARISONS:
    sub = get_mean_points(
        model,
        test,
    )

    print(
        f"\n[{model} / {test}]"
    )

    for _, r in sub[
        sub["Pareto"]
    ].sort_values(
        "Cost mean"
    ).iterrows():

        print(
            f'  {int(r["Weight"])}:1  '
            f'cost={r["Cost mean"]:.6f}, '
            f'tdis={r["Tdis mean"]:.6f}'
        )


print("\nDone.")
print(
    "Output directory:",
    OUTPUT_DIR.resolve(),
)
