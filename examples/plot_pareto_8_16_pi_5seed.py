"""
plot_pareto_8_16_pi_5seed.py

5개 seed 평균을 사용해 8:1 / 16:1 / PI-derived reward 설정을
cost_tot - tdis_tot Pareto 그래프로 비교한다.

입력 Excel
----------
boptest_reward_weights_8_16_PI_seed1-5_comparison.xlsx

사용 시트
---------
Mean_Pareto

비교 정책
---------
All:
  - 8:1
  - 16:1
  - PI-derived 26:1

Summer:
  - 8:1
  - 16:1
  - PI-derived 13:1

Winter:
  - 8:1
  - 16:1
  - PI-derived 109:1

생성 결과
---------
1) 원본 KPI Pareto plot
   - x = cost_tot mean
   - y = tdis_tot mean
   - SD error bar 없음
   - 5-seed mean 점만 표시

2) 0~1 Min-Max normalized Pareto plot
   - 각 테스트의 6개 평균 정책을 기준으로
     cost / tdis를 각각 0~1로 scaling
   - x/y 동일 scale
   - 왼쪽 아래가 좋음

3) 2x2 통합 figure
   - raw
   - normalized

4) Pareto summary CSV

필요 패키지
-----------
pip install pandas numpy matplotlib openpyxl
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 사용자 설정
# ============================================================

INPUT_FILE = Path(
    "boptest_reward_weights_8_16_PI_seed1-5_comparison.xlsx"
)

SHEET_NAME = "Mean_Pareto"

OUTPUT_DIR = Path("pareto_8_16_pi_5seed_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DPI = 300

COMPARISONS = [
    ("Peak Cool", "Summer"),
    ("Typical Cool", "Summer"),
    ("Peak Heat", "Winter"),
    ("Typical Heat", "Winter"),
]


# ============================================================
# 2. Excel 읽기
# ============================================================

def load_mean_pareto_data(path: Path) -> pd.DataFrame:
    """
    Mean_Pareto 시트를 읽는다.

    시트 상단에 제목/빈 행이 있으므로
    실제 header row를 자동 탐색한다.
    """
    required = {
        "Test",
        "Model",
        "Setting",
        "Weight",
        "Cost mean",
        "Tdis mean",
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
            f"필요한 컬럼이 없습니다: {sorted(missing)}\n"
            f"현재 컬럼: {list(df.columns)}"
        )

    df = df.copy()

    df["Weight"] = pd.to_numeric(
        df["Weight"],
        errors="coerce",
    )
    df["Cost mean"] = pd.to_numeric(
        df["Cost mean"],
        errors="coerce",
    )
    df["Tdis mean"] = pd.to_numeric(
        df["Tdis mean"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Test",
            "Model",
            "Setting",
            "Weight",
            "Cost mean",
            "Tdis mean",
        ]
    )

    return df


df = load_mean_pareto_data(INPUT_FILE)


# ============================================================
# 3. 표시 이름
# ============================================================

def make_label(row) -> str:
    """
    PI 설정은 실제 모델별 weight까지 보이도록 표시.
    """
    if row["Setting"] == "PI":
        return (
            f'{row["Model"]} PI {int(row["Weight"])}:1'
        )

    return (
        f'{row["Model"]} {row["Setting"]}'
    )


# ============================================================
# 4. Pareto 판정
# ============================================================

def pareto_mask(points: np.ndarray) -> np.ndarray:
    """
    points[:, 0] = cost
    points[:, 1] = tdis

    두 objective 모두 최소화.

    True = tested policies 가운데 non-dominated point
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


# ============================================================
# 5. 테스트별 6개 정책 가져오기
# ============================================================

def get_test_points(
    test_name: str,
    seasonal_model: str,
) -> pd.DataFrame:
    """
    한 test에서 아래 6개 mean point를 가져온다.

      All 8:1
      All 16:1
      All PI 26:1

      Summer/Winter 8:1
      Summer/Winter 16:1
      Summer PI 13:1 또는 Winter PI 109:1
    """
    sub = df[
        (df["Test"] == test_name)
        & (df["Model"].isin(["All", seasonal_model]))
        & (df["Setting"].isin(["8:1", "16:1", "PI"]))
    ].copy()

    if len(sub) != 6:
        raise ValueError(
            f"{test_name}: 6개 정책을 기대했지만 "
            f"{len(sub)}개를 찾았습니다.\n"
            f"{sub[['Test','Model','Setting','Weight','Cost mean','Tdis mean']]}"
        )

    sub["label"] = sub.apply(
        make_label,
        axis=1,
    )

    pts = sub[
        ["Cost mean", "Tdis mean"]
    ].to_numpy()

    # 해당 test의 6개 정책 전체를 대상으로 판정
    sub["pareto"] = pareto_mask(pts)

    return sub.reset_index(drop=True)


# ============================================================
# 6. 0~1 Min-Max 정규화
# ============================================================

def minmax_normalize(
    sub: pd.DataFrame
) -> pd.DataFrame:
    """
    각 test의 6개 mean point를 기준으로
    cost와 tdis를 각각 0~1로 변환.

    0 = 해당 test에서 가장 좋은 값
    1 = 해당 test에서 가장 나쁜 값

    양의 affine transform이므로
    Pareto dominance 관계는 바뀌지 않는다.
    """
    out = sub.copy()

    for raw_col, norm_col in [
        ("Cost mean", "Cost norm"),
        ("Tdis mean", "Tdis norm"),
    ]:
        lo = out[raw_col].min()
        hi = out[raw_col].max()

        if np.isclose(hi, lo):
            out[norm_col] = 0.0
        else:
            out[norm_col] = (
                (out[raw_col] - lo)
                / (hi - lo)
            )

    return out


# ============================================================
# 7. 그래프 보조 함수
# ============================================================

def marker_for_setting(setting: str) -> str:
    """
    reward 설정별 marker 모양.
    """
    return {
        "8:1": "o",
        "16:1": "s",
        "PI": "^",
    }[setting]


def annotation_offset(i: int):
    """
    label 겹침을 줄이기 위한 offset.
    """
    offsets = [
        (7, 7),
        (7, -14),
        (-80, 7),
        (7, 16),
        (-90, -14),
        (7, -22),
    ]

    return offsets[i % len(offsets)]


def annotate_points(
    ax,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
):
    for i, (_, r) in enumerate(
        sub.iterrows()
    ):
        dx, dy = annotation_offset(i)

        ax.annotate(
            r["label"],
            (r[x_col], r[y_col]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
        )


def draw_empirical_front(
    ax,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
):
    """
    non-dominated mean points만 cost 순으로 연결.
    """
    front = (
        sub[sub["pareto"]]
        .sort_values(x_col)
    )

    if len(front) >= 2:
        ax.plot(
            front[x_col],
            front[y_col],
            linestyle="--",
            linewidth=1.6,
            label="Empirical Pareto front",
        )


# ============================================================
# 8. Raw Pareto
# ============================================================

def plot_raw(
    test_name: str,
    seasonal_model: str,
):
    sub = get_test_points(
        test_name,
        seasonal_model,
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 5.8)
    )

    for _, r in sub.iterrows():
        ax.scatter(
            r["Cost mean"],
            r["Tdis mean"],
            marker=marker_for_setting(
                r["Setting"]
            ),
            s=90,
            label=r["label"],
        )

    draw_empirical_front(
        ax,
        sub,
        "Cost mean",
        "Tdis mean",
    )

    annotate_points(
        ax,
        sub,
        "Cost mean",
        "Tdis mean",
    )

    ax.set_title(
        f"{test_name}: Cost–Comfort Pareto "
        "(5-seed mean)"
    )
    ax.set_xlabel("cost_tot")
    ax.set_ylabel("tdis_tot")
    ax.grid(alpha=0.25)

    handles, labels = (
        ax.get_legend_handles_labels()
    )
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=7.3,
        loc="best",
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        / f"pareto_raw_{test_name.replace(' ', '_').lower()}.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return sub


# ============================================================
# 9. Normalized Pareto
# ============================================================

def plot_normalized(
    test_name: str,
    seasonal_model: str,
):
    sub = minmax_normalize(
        get_test_points(
            test_name,
            seasonal_model,
        )
    )

    fig, ax = plt.subplots(
        figsize=(6.4, 6.0)
    )

    for _, r in sub.iterrows():
        ax.scatter(
            r["Cost norm"],
            r["Tdis norm"],
            marker=marker_for_setting(
                r["Setting"]
            ),
            s=90,
            label=r["label"],
        )

    draw_empirical_front(
        ax,
        sub,
        "Cost norm",
        "Tdis norm",
    )

    annotate_points(
        ax,
        sub,
        "Cost norm",
        "Tdis norm",
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # x/y 축을 동일한 시각적 scale로 표시
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_title(
        f"{test_name}: Normalized Pareto "
        "(5-seed mean)"
    )
    ax.set_xlabel(
        "Normalized cost "
        "(0 = best, 1 = worst)"
    )
    ax.set_ylabel(
        "Normalized tdis "
        "(0 = best, 1 = worst)"
    )
    ax.grid(alpha=0.25)

    handles, labels = (
        ax.get_legend_handles_labels()
    )
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=7.1,
        loc="best",
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        / f"pareto_normalized_{test_name.replace(' ', '_').lower()}.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return sub


# ============================================================
# 10. 2x2 raw 통합 figure
# ============================================================

def plot_all_raw():
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
    )

    for ax, (
        test_name,
        seasonal_model,
    ) in zip(
        axes.ravel(),
        COMPARISONS,
    ):
        sub = get_test_points(
            test_name,
            seasonal_model,
        )

        for _, r in sub.iterrows():
            ax.scatter(
                r["Cost mean"],
                r["Tdis mean"],
                marker=marker_for_setting(
                    r["Setting"]
                ),
                s=72,
                label=r["label"],
            )

        draw_empirical_front(
            ax,
            sub,
            "Cost mean",
            "Tdis mean",
        )

        annotate_points(
            ax,
            sub,
            "Cost mean",
            "Tdis mean",
        )

        ax.set_title(test_name)
        ax.set_xlabel("cost_tot")
        ax.set_ylabel("tdis_tot")
        ax.grid(alpha=0.25)

    fig.suptitle(
        "8:1 vs 16:1 vs PI-derived "
        "— Raw Cost–Comfort Pareto "
        "(5-seed mean)",
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    output = (
        OUTPUT_DIR
        / "pareto_raw_all_tests.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 11. 2x2 normalized 통합 figure
# ============================================================

def plot_all_normalized():
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    for ax, (
        test_name,
        seasonal_model,
    ) in zip(
        axes.ravel(),
        COMPARISONS,
    ):
        sub = minmax_normalize(
            get_test_points(
                test_name,
                seasonal_model,
            )
        )

        for _, r in sub.iterrows():
            ax.scatter(
                r["Cost norm"],
                r["Tdis norm"],
                marker=marker_for_setting(
                    r["Setting"]
                ),
                s=72,
                label=r["label"],
            )

        draw_empirical_front(
            ax,
            sub,
            "Cost norm",
            "Tdis norm",
        )

        annotate_points(
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

        ax.set_title(test_name)
        ax.set_xlabel("Normalized cost")
        ax.set_ylabel("Normalized tdis")
        ax.grid(alpha=0.25)

    fig.suptitle(
        "8:1 vs 16:1 vs PI-derived "
        "— Normalized Cost–Comfort Pareto "
        "(5-seed mean)",
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    output = (
        OUTPUT_DIR
        / "pareto_normalized_all_tests.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 12. 실행
# ============================================================

raw_tables = []
norm_tables = []

for (
    test_name,
    seasonal_model,
) in COMPARISONS:

    raw_sub = plot_raw(
        test_name,
        seasonal_model,
    )

    norm_sub = plot_normalized(
        test_name,
        seasonal_model,
    )

    raw_tables.append(raw_sub)
    norm_tables.append(norm_sub)


plot_all_raw()
plot_all_normalized()


# ============================================================
# 13. CSV 저장
# ============================================================

raw_summary = pd.concat(
    raw_tables,
    ignore_index=True,
)

norm_summary = pd.concat(
    norm_tables,
    ignore_index=True,
)

raw_summary.to_csv(
    OUTPUT_DIR
    / "pareto_8_16_pi_raw_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

norm_summary.to_csv(
    OUTPUT_DIR
    / "pareto_8_16_pi_normalized_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. 콘솔에 Pareto set 출력
# ============================================================

print(
    "\n=== 5-seed mean empirical Pareto set ==="
)

for (
    test_name,
    seasonal_model,
) in COMPARISONS:

    sub = get_test_points(
        test_name,
        seasonal_model,
    )

    front = sub[sub["pareto"]]

    print(f"\n[{test_name}]")

    for _, r in front.iterrows():
        print(
            f"  {r['label']}: "
            f"cost={r['Cost mean']:.6f}, "
            f"tdis={r['Tdis mean']:.6f}"
        )


print("\nDone.")
print(
    f"Output directory: "
    f"{OUTPUT_DIR.resolve()}"
)
