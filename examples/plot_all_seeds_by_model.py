"""
plot_all_seeds_by_model.py

목적
----
각 모델(All / Summer / Winter)에 대해 5개 seed가 cost_tot - tdis_tot 평면에서
어디에 찍히는지 한 번에 확인한다.

특히:
- All 모델: Peak/Typical Cool + Peak/Typical Heat 모두 그림
- All 모델: Cooling 2개 / Heating 2개를 따로 묶은 figure도 생성
- Summer 모델: Peak/Typical Cool
- Winter 모델: Peak/Typical Heat

각 test/model 그림에는
  8:1, 16:1, PI-derived
각 reward setting의 Seed1~Seed5를 모두 표시한다.

이 그림의 목적은 Pareto front 자체보다는
"어느 seed가 다른 seed와 멀리 떨어져 있는가?"를 보는 것이다.

입력 Excel
----------
boptest_reward_weights_8_16_PI_seed1-5_comparison.xlsx

사용 시트
---------
Seed_Comparison

출력
----
seed_spread_by_model_output/
  all/
    all_peak_cool.png
    all_typical_cool.png
    all_peak_heat.png
    all_typical_heat.png
    all_cooling_2panel.png
    all_heating_2panel.png
    all_all_tests_2x2.png

  summer/
    summer_peak_cool.png
    summer_typical_cool.png
    summer_cooling_2panel.png

  winter/
    winter_peak_heat.png
    winter_typical_heat.png
    winter_heating_2panel.png

  all_seed_points_long.csv
  seed_spread_summary.csv

필요 패키지
-----------
pip install pandas numpy matplotlib openpyxl
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 설정
# ============================================================

INPUT_FILE = Path(
    "boptest_reward_weights_8_16_PI_seed1-5_comparison.xlsx"
)

SHEET_NAME = "Seed_Comparison"

OUTPUT_DIR = Path("seed_spread_by_model_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DPI = 300

TESTS_BY_MODEL = {
    "All": [
        "Peak Cool",
        "Typical Cool",
        "Peak Heat",
        "Typical Heat",
    ],
    "Summer": [
        "Peak Cool",
        "Typical Cool",
    ],
    "Winter": [
        "Peak Heat",
        "Typical Heat",
    ],
}

WEIGHTS = {
    "All": {
        "8:1": 8,
        "16:1": 16,
        "PI": 26,
    },
    "Summer": {
        "8:1": 8,
        "16:1": 16,
        "PI": 13,
    },
    "Winter": {
        "8:1": 8,
        "16:1": 16,
        "PI": 109,
    },
}

SETTING_ORDER = ["8:1", "16:1", "PI"]


# ============================================================
# 2. Excel 읽기
# ============================================================

def load_seed_comparison(path: Path) -> pd.DataFrame:
    required = {
        "Test",
        "Model",
        "Seed",
        "Numeric Seed",
        "8 Cost",
        "16 Cost",
        "PI Cost",
        "8 Tdis",
        "16 Tdis",
        "PI Tdis",
    }

    preview = pd.read_excel(
        path,
        sheet_name=SHEET_NAME,
        header=None,
        nrows=10,
    )

    header_row = None

    for i in range(len(preview)):
        row_values = {
            str(v).strip()
            for v in preview.iloc[i].dropna().tolist()
        }

        if required.issubset(row_values):
            header_row = i
            break

    if header_row is None:
        raise ValueError(
            f"{path.name}의 {SHEET_NAME} 시트에서 "
            "실제 header 행을 찾지 못했습니다."
        )

    df = pd.read_excel(
        path,
        sheet_name=SHEET_NAME,
        header=header_row,
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"필요한 컬럼이 없습니다: {sorted(missing)}"
        )

    numeric_cols = [
        "Seed",
        "Numeric Seed",
        "8 Cost",
        "16 Cost",
        "PI Cost",
        "8 Tdis",
        "16 Tdis",
        "PI Tdis",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=list(required)
    )

    df["Seed"] = df["Seed"].astype(int)
    df["Numeric Seed"] = df["Numeric Seed"].astype(int)

    return df


wide_df = load_seed_comparison(INPUT_FILE)


# ============================================================
# 3. Long format 변환
# ============================================================

def to_long_format(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in wide.iterrows():
        model = r["Model"]

        if model not in WEIGHTS:
            continue

        for setting, cost_col, tdis_col in [
            ("8:1", "8 Cost", "8 Tdis"),
            ("16:1", "16 Cost", "16 Tdis"),
            ("PI", "PI Cost", "PI Tdis"),
        ]:
            rows.append({
                "Test": r["Test"],
                "Model": model,
                "Seed": int(r["Seed"]),
                "Numeric Seed": int(r["Numeric Seed"]),
                "Setting": setting,
                "Weight": WEIGHTS[model][setting],
                "Cost": float(r[cost_col]),
                "Tdis": float(r[tdis_col]),
            })

    out = pd.DataFrame(rows)

    out["Label"] = out.apply(
        lambda r: (
            f'PI {int(r["Weight"])}:1'
            if r["Setting"] == "PI"
            else r["Setting"]
        ),
        axis=1,
    )

    return out


df = to_long_format(wide_df)


# ============================================================
# 4. seed spread 요약
# ============================================================

def build_spread_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    각 Model/Test/Setting별로 5개 seed의 범위와 SD를 정리한다.
    'outlier 판정'이 아니라 seed variation의 크기를 보기 위한 summary.
    """
    rows = []

    for (
        model,
        test,
        setting,
    ), g in data.groupby(
        ["Model", "Test", "Setting"]
    ):

        rows.append({
            "Model": model,
            "Test": test,
            "Setting": setting,
            "Weight": int(g["Weight"].iloc[0]),
            "Seed count": g["Seed"].nunique(),

            "Cost mean": g["Cost"].mean(),
            "Cost SD": g["Cost"].std(ddof=1),
            "Cost min": g["Cost"].min(),
            "Cost max": g["Cost"].max(),
            "Cost range": g["Cost"].max() - g["Cost"].min(),

            "Tdis mean": g["Tdis"].mean(),
            "Tdis SD": g["Tdis"].std(ddof=1),
            "Tdis min": g["Tdis"].min(),
            "Tdis max": g["Tdis"].max(),
            "Tdis range": g["Tdis"].max() - g["Tdis"].min(),

            "Max Tdis seed": int(
                g.loc[g["Tdis"].idxmax(), "Seed"]
            ),
            "Min Tdis seed": int(
                g.loc[g["Tdis"].idxmin(), "Seed"]
            ),
        })

    return pd.DataFrame(rows)


spread_summary = build_spread_summary(df)


# ============================================================
# 5. 그래프 유틸리티
# ============================================================

def marker_for_setting(setting: str) -> str:
    return {
        "8:1": "o",
        "16:1": "s",
        "PI": "^",
    }[setting]


def label_for_setting(
    model: str,
    setting: str,
    weight: int,
) -> str:
    if setting == "PI":
        return f"PI {weight}:1"

    return setting


def draw_seed_scatter(
    ax,
    model: str,
    test_name: str,
    show_setting_centroid: bool = True,
):
    """
    하나의 model/test에서
    3 reward setting × 5 seed = 15개 점을 모두 표시한다.

    같은 setting의 seed 5개는 같은 marker를 사용한다.
    각 점 옆에는 S1~S5를 표시한다.

    setting 평균점은 큰 X marker로 표시할 수 있다.
    """
    sub = df[
        (df["Model"] == model)
        & (df["Test"] == test_name)
    ].copy()

    if len(sub) != 15:
        raise ValueError(
            f"{model} / {test_name}: "
            f"15개 점을 기대했지만 {len(sub)}개입니다."
        )

    for setting in SETTING_ORDER:
        g = sub[
            sub["Setting"] == setting
        ].sort_values("Seed")

        weight = int(g["Weight"].iloc[0])
        series_label = label_for_setting(
            model,
            setting,
            weight,
        )

        ax.scatter(
            g["Cost"],
            g["Tdis"],
            marker=marker_for_setting(setting),
            s=72,
            label=series_label,
        )

        # seed 번호를 각 점 옆에 표기
        for _, r in g.iterrows():
            ax.annotate(
                f'S{int(r["Seed"])}',
                (r["Cost"], r["Tdis"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )

        # 각 reward setting의 5-seed 평균 위치
        if show_setting_centroid:
            ax.scatter(
                g["Cost"].mean(),
                g["Tdis"].mean(),
                marker="X",
                s=125,
                label=f"{series_label} mean",
            )

    ax.set_title(
        f"{model} — {test_name}"
    )
    ax.set_xlabel("cost_tot")
    ax.set_ylabel("tdis_tot")
    ax.grid(alpha=0.25)


def dedup_legend(ax, fontsize=7.5):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=fontsize,
        loc="best",
    )


# ============================================================
# 6. 개별 Model/Test figure
# ============================================================

def plot_single_test(
    model: str,
    test_name: str,
):
    fig, ax = plt.subplots(
        figsize=(7.6, 5.8)
    )

    draw_seed_scatter(
        ax,
        model,
        test_name,
        show_setting_centroid=True,
    )

    dedup_legend(
        ax,
        fontsize=7.2,
    )

    fig.tight_layout()

    model_dir = (
        OUTPUT_DIR
        / model.lower()
    )
    model_dir.mkdir(exist_ok=True)

    filename = (
        f"{model.lower()}_"
        f"{test_name.replace(' ', '_').lower()}.png"
    )

    fig.savefig(
        model_dir / filename,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 7. All: 4 test 전체 2x2
# ============================================================

def plot_all_model_2x2():
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
    )

    tests = [
        "Peak Cool",
        "Typical Cool",
        "Peak Heat",
        "Typical Heat",
    ]

    for ax, test in zip(
        axes.ravel(),
        tests,
    ):
        draw_seed_scatter(
            ax,
            "All",
            test,
            show_setting_centroid=True,
        )

    fig.suptitle(
        "All model — Seed distribution "
        "for 8:1 / 16:1 / PI 26:1",
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    out_dir = OUTPUT_DIR / "all"
    out_dir.mkdir(exist_ok=True)

    fig.savefig(
        out_dir / "all_all_tests_2x2.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 8. All: Cooling / Heating 별도 2-panel
# ============================================================

def plot_two_panel(
    model: str,
    tests: list,
    filename: str,
    title: str,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.3),
    )

    for ax, test in zip(
        axes,
        tests,
    ):
        draw_seed_scatter(
            ax,
            model,
            test,
            show_setting_centroid=True,
        )

    fig.suptitle(
        title,
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    out_dir = (
        OUTPUT_DIR
        / model.lower()
    )
    out_dir.mkdir(exist_ok=True)

    fig.savefig(
        out_dir / filename,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 9. 실행
# ============================================================

for model, tests in TESTS_BY_MODEL.items():
    for test in tests:
        plot_single_test(
            model,
            test,
        )


# All 전체 + Cooling / Heating
plot_all_model_2x2()

plot_two_panel(
    "All",
    ["Peak Cool", "Typical Cool"],
    "all_cooling_2panel.png",
    "All model — Cooling tests",
)

plot_two_panel(
    "All",
    ["Peak Heat", "Typical Heat"],
    "all_heating_2panel.png",
    "All model — Heating tests",
)


# Summer cooling
plot_two_panel(
    "Summer",
    ["Peak Cool", "Typical Cool"],
    "summer_cooling_2panel.png",
    "Summer model — Cooling tests",
)


# Winter heating
plot_two_panel(
    "Winter",
    ["Peak Heat", "Typical Heat"],
    "winter_heating_2panel.png",
    "Winter model — Heating tests",
)


# ============================================================
# 10. CSV 저장
# ============================================================

df.sort_values(
    [
        "Model",
        "Test",
        "Setting",
        "Seed",
    ]
).to_csv(
    OUTPUT_DIR
    / "all_seed_points_long.csv",
    index=False,
    encoding="utf-8-sig",
)

spread_summary.sort_values(
    [
        "Model",
        "Test",
        "Setting",
    ]
).to_csv(
    OUTPUT_DIR
    / "seed_spread_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 11. 콘솔 요약
# ============================================================

print("\n=== Largest Tdis by model/test/setting ===")

for _, r in spread_summary.sort_values(
    "Tdis range",
    ascending=False,
).iterrows():

    print(
        f'{r["Model"]:>6} | '
        f'{r["Test"]:<13} | '
        f'{r["Setting"]:<4} | '
        f'Tdis range={r["Tdis range"]:.6f} | '
        f'max seed=S{int(r["Max Tdis seed"])}'
    )


print("\nDone.")
print(
    "Output directory:",
    OUTPUT_DIR.resolve(),
)
