
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path("boptest_reward_weights_8_16_PI_seed1-5_comparison.xlsx")
SHEET_NAME = "Seed_Comparison"
OUTPUT_DIR = Path("pareto_by_seed_8_16_pi_output")
OUTPUT_DIR.mkdir(exist_ok=True)
DPI = 300

SEEDS = [1, 2, 3, 4, 5]
COMPARISONS = [
    ("Peak Cool", "Summer"),
    ("Typical Cool", "Summer"),
    ("Peak Heat", "Winter"),
    ("Typical Heat", "Winter"),
]

WEIGHTS = {
    "All": {"8:1": 8, "16:1": 16, "PI": 26},
    "Summer": {"8:1": 8, "16:1": 16, "PI": 13},
    "Winter": {"8:1": 8, "16:1": 16, "PI": 109},
}

def load_seed_data(path):
    required = {
        "Test","Model","Seed","Numeric Seed",
        "8 Cost","16 Cost","PI Cost",
        "8 Tdis","16 Tdis","PI Tdis",
    }
    preview = pd.read_excel(path, sheet_name=SHEET_NAME, header=None, nrows=10)
    header_row = None
    for i in range(len(preview)):
        vals = {str(v).strip() for v in preview.iloc[i].dropna().tolist()}
        if required.issubset(vals):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Seed_Comparison 시트의 실제 헤더 행을 찾지 못했습니다.")

    df = pd.read_excel(path, sheet_name=SHEET_NAME, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필요 컬럼 누락: {sorted(missing)}")

    num_cols = ["Seed","Numeric Seed","8 Cost","16 Cost","PI Cost","8 Tdis","16 Tdis","PI Tdis"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=list(required))
    df["Seed"] = df["Seed"].astype(int)
    df["Numeric Seed"] = df["Numeric Seed"].astype(int)
    return df

def pareto_mask(points):
    n = len(points)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                mask[i] = False
                break
    return mask

df = load_seed_data(INPUT_FILE)

def get_seed_test_points(seed, test_name, seasonal_model):
    src = df[
        (df["Seed"] == seed)
        & (df["Test"] == test_name)
        & (df["Model"].isin(["All", seasonal_model]))
    ].copy()

    if len(src) != 2:
        raise ValueError(
            f"Seed {seed}, {test_name}: All + {seasonal_model} 2행을 기대했지만 {len(src)}행입니다."
        )

    rows = []
    for _, r in src.iterrows():
        model = r["Model"]
        for setting, ccol, dcol in [
            ("8:1", "8 Cost", "8 Tdis"),
            ("16:1", "16 Cost", "16 Tdis"),
            ("PI", "PI Cost", "PI Tdis"),
        ]:
            weight = WEIGHTS[model][setting]
            label = f"{model} PI {weight}:1" if setting == "PI" else f"{model} {setting}"
            rows.append({
                "Seed": seed,
                "Numeric Seed": int(r["Numeric Seed"]),
                "Test": test_name,
                "Model": model,
                "Setting": setting,
                "Weight": weight,
                "Cost": float(r[ccol]),
                "Tdis": float(r[dcol]),
                "label": label,
            })

    out = pd.DataFrame(rows)
    out["Pareto"] = pareto_mask(out[["Cost","Tdis"]].to_numpy())
    return out

def normalize(sub):
    out = sub.copy()
    for raw, norm in [("Cost","Cost_norm"),("Tdis","Tdis_norm")]:
        lo, hi = out[raw].min(), out[raw].max()
        out[norm] = 0.0 if np.isclose(lo, hi) else (out[raw]-lo)/(hi-lo)
    return out

def marker(setting):
    return {"8:1":"o","16:1":"s","PI":"^"}[setting]

def annotate(ax, sub, xcol, ycol):
    offsets = [(7,7),(7,-14),(-84,7),(7,16),(-94,-14),(7,-22)]
    for i,(_,r) in enumerate(sub.iterrows()):
        dx,dy = offsets[i % len(offsets)]
        ax.annotate(
            r["label"], (r[xcol], r[ycol]),
            xytext=(dx,dy), textcoords="offset points", fontsize=7.5
        )

def draw_front(ax, sub, xcol, ycol):
    front = sub[sub["Pareto"]].sort_values(xcol)
    if len(front) >= 2:
        ax.plot(
            front[xcol], front[ycol],
            linestyle="--", linewidth=1.6,
            label="Empirical Pareto front"
        )

def draw_raw(ax, sub, title):
    for _,r in sub.iterrows():
        ax.scatter(r["Cost"], r["Tdis"], marker=marker(r["Setting"]), s=78, label=r["label"])
    draw_front(ax, sub, "Cost", "Tdis")
    annotate(ax, sub, "Cost", "Tdis")
    ax.set_title(title)
    ax.set_xlabel("cost_tot")
    ax.set_ylabel("tdis_tot")
    ax.grid(alpha=0.25)

def draw_norm(ax, sub, title):
    for _,r in sub.iterrows():
        ax.scatter(r["Cost_norm"], r["Tdis_norm"], marker=marker(r["Setting"]), s=78, label=r["label"])
    draw_front(ax, sub, "Cost_norm", "Tdis_norm")
    annotate(ax, sub, "Cost_norm", "Tdis_norm")
    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(-0.05,1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("Normalized cost")
    ax.set_ylabel("Normalized tdis")
    ax.grid(alpha=0.25)

def save_single_raw(seed, test_name, seasonal_model):
    sub = get_seed_test_points(seed, test_name, seasonal_model)
    fig, ax = plt.subplots(figsize=(7.5,5.8))
    draw_raw(ax, sub, f"Seed {seed} — {test_name}: Cost–Comfort Pareto")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7.2, loc="best")
    fig.tight_layout()
    d = OUTPUT_DIR / f"seed{seed}"
    d.mkdir(exist_ok=True)
    fig.savefig(d / f"raw_{test_name.replace(' ','_').lower()}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return sub

def save_single_norm(seed, test_name, seasonal_model):
    sub = normalize(get_seed_test_points(seed, test_name, seasonal_model))
    fig, ax = plt.subplots(figsize=(6.4,6.0))
    draw_norm(ax, sub, f"Seed {seed} — {test_name}: Normalized Pareto")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7.0, loc="best")
    fig.tight_layout()
    d = OUTPUT_DIR / f"seed{seed}"
    d.mkdir(exist_ok=True)
    fig.savefig(d / f"normalized_{test_name.replace(' ','_').lower()}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return sub

def save_2x2(seed, normalized=False):
    fig, axes = plt.subplots(2,2, figsize=(12 if normalized else 13,10))
    for ax,(test_name,seasonal_model) in zip(axes.ravel(), COMPARISONS):
        sub = get_seed_test_points(seed, test_name, seasonal_model)
        if normalized:
            sub = normalize(sub)
            draw_norm(ax, sub, test_name)
        else:
            draw_raw(ax, sub, test_name)
    fig.suptitle(
        f"Seed {seed}: 8:1 vs 16:1 vs PI-derived — "
        + ("Normalized" if normalized else "Raw")
        + " Cost–Comfort Pareto",
        fontsize=15
    )
    fig.tight_layout(rect=[0,0,1,0.96])
    d = OUTPUT_DIR / f"seed{seed}"
    d.mkdir(exist_ok=True)
    fname = f"seed{seed}_{'normalized' if normalized else 'raw'}_all_tests.png"
    fig.savefig(d / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

all_raw = []
all_norm = []

for seed in SEEDS:
    print(f"\n===== Seed {seed} =====")
    for test_name, seasonal_model in COMPARISONS:
        raw_sub = save_single_raw(seed, test_name, seasonal_model)
        norm_sub = save_single_norm(seed, test_name, seasonal_model)
        all_raw.append(raw_sub)
        all_norm.append(norm_sub)

        front = raw_sub[raw_sub["Pareto"]]
        print(f"[{test_name}]")
        for _,r in front.iterrows():
            print(f"  {r['label']}: cost={r['Cost']:.6f}, tdis={r['Tdis']:.6f}")

    save_2x2(seed, normalized=False)
    save_2x2(seed, normalized=True)

raw_summary = pd.concat(all_raw, ignore_index=True)
norm_summary = pd.concat(all_norm, ignore_index=True)

raw_summary.to_csv(
    OUTPUT_DIR / "pareto_by_seed_raw_summary.csv",
    index=False, encoding="utf-8-sig"
)
norm_summary.to_csv(
    OUTPUT_DIR / "pareto_by_seed_normalized_summary.csv",
    index=False, encoding="utf-8-sig"
)

freq = (
    raw_summary
    .groupby(["Test","Model","Setting"], as_index=False)
    .agg(
        Pareto_count=("Pareto","sum"),
        Seed_count=("Seed","nunique"),
    )
)
freq["Pareto_frequency"] = freq["Pareto_count"] / freq["Seed_count"]
freq.to_csv(
    OUTPUT_DIR / "pareto_frequency_by_setting.csv",
    index=False, encoding="utf-8-sig"
)

print("\nDone.")
print("Output directory:", OUTPUT_DIR.resolve())
