"""
plot_pareto_seed1_8_16_pi.py

Seed1만 사용하여 다음 reward 설정을 모두 비교하는
cost_tot - tdis_tot empirical Pareto 그래프를 생성한다.

비교 설정
---------
All:
  - 8:1
  - 16:1
  - PI-scale 26:1

Summer:
  - 8:1
  - 16:1
  - PI-scale 13:1

Winter:
  - 8:1
  - 16:1
  - PI-scale 109:1

입력 Excel
----------
boptest_reward_weight_pareto_8_16_PI.xlsx

입력 시트
---------
Pareto_seed1_8_16_PI

생성 결과
---------
1) 원본 KPI Pareto plot
   - cost_tot vs tdis_tot
   - Seed1이므로 error bar 없음

2) 0~1 Min-Max normalized Pareto plot
   - 각 테스트별 6개 정책을 기준으로 cost와 tdis를 각각 정규화
   - x/y 모두 0~1
   - 왼쪽 아래가 좋음

3) 2x2 통합 그림
   - raw
   - normalized

4) seed1 Pareto summary CSV

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

INPUT_FILE = Path("boptest_reward_weight_pareto_8_16_PI.xlsx")
SHEET_NAME = "Pareto_seed1_8_16_PI"

OUTPUT_DIR = Path("pareto_seed1_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DPI = 300

# 논문 핵심 테스트와 해당 계절 모델
COMPARISONS = [
    ("Peak Cool", "Summer"),
    ("Typical Cool", "Summer"),
    ("Peak Heat", "Winter"),
    ("Typical Heat", "Winter"),
]


# ============================================================
# 2. Excel 읽기
# ============================================================

def load_seed1_pareto_data(path: Path) -> pd.DataFrame:
    """
    Pareto_seed1_8_16_PI 시트를 읽는다.

    시트 앞부분에 제목/빈 행이 있으므로 실제 헤더 행을 자동 탐색한다.
    """
    required = {
        "Test",
        "Model",
        "Reward setting",
        "Training weight",
        "Cost",
        "Tdis",
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
            f"{path.name}의 {SHEET_NAME} 시트에서 실제 헤더 행을 찾지 못했습니다."
        )

    df = pd.read_excel(
        path,
        sheet_name=SHEET_NAME,
        header=header_row,
    )

    df.columns = [str(c).strip() for c in df.columns]

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"필요한 컬럼이 없습니다: {sorted(missing)}\n"
            f"현재 컬럼: {list(df.columns)}"
        )

    df = df.copy()

    df["Training weight"] = pd.to_numeric(
        df["Training weight"],
        errors="coerce",
    )
    df["Cost"] = pd.to_numeric(
        df["Cost"],
        errors="coerce",
    )
    df["Tdis"] = pd.to_numeric(
        df["Tdis"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Test",
            "Model",
            "Reward setting",
            "Training weight",
            "Cost",
            "Tdis",
        ]
    )

    return df


df = load_seed1_pareto_data(INPUT_FILE)


# ============================================================
# 3. Pareto 판정
# ============================================================

def pareto_mask(points: np.ndarray) -> np.ndarray:
    """
    points[:, 0] = Cost
    points[:, 1] = Tdis

    두 목적 모두 minimize.

    True:
        다른 tested policy에 의해 지배되지 않는 점
        (= empirical Pareto-optimal among tested policies)
    """
    n = len(points)
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # j가 i보다 두 축에서 같거나 좋고,
            # 적어도 하나에서 strict하게 좋으면 i는 dominated.
            if (
                np.all(points[j] <= points[i])
                and np.any(points[j] < points[i])
            ):
                mask[i] = False
                break

    return mask


# ============================================================
# 4. 테스트별 데이터 추출
# ============================================================

def get_test_points(
    test_name: str,
    seasonal_model: str,
) -> pd.DataFrame:
    """
    한 테스트에서 아래 6개 정책을 가져온다.

    All:
      8:1 / 16:1 / PI-scale 26:1

    Seasonal:
      Summer: 8:1 / 16:1 / PI-scale 13:1
      또는
      Winter: 8:1 / 16:1 / PI-scale 109:1
    """
    sub = df[
        (df["Test"] == test_name)
        & (df["Model"].isin(["All", seasonal_model]))
    ].copy()

    # 2 models × 3 reward settings = 6 points
    if len(sub) != 6:
        raise ValueError(
            f"{test_name}: 6개 정책을 기대했지만 {len(sub)}개를 찾았습니다.\n"
            f"{sub[['Test','Model','Reward setting','Cost','Tdis']]}"
        )

    sub["label"] = (
        sub["Model"].astype(str)
        + " "
        + sub["Reward setting"].astype(str)
    )

    # 여기서는 "모델 내부"가 아니라
    # 해당 테스트의 6개 정책 전체를 대상으로 Pareto 판정.
    points = sub[["Cost", "Tdis"]].to_numpy()

    sub["global_pareto"] = pareto_mask(points)

    return sub.reset_index(drop=True)


# ============================================================
# 5. 0~1 Min-Max scaling
# ============================================================

def minmax_normalize(sub: pd.DataFrame) -> pd.DataFrame:
    """
    한 테스트의 6개 점을 기준으로
    Cost, Tdis 각각 0~1 Min-Max scaling.

    0 = 해당 테스트에서 관측된 가장 좋은 값
    1 = 해당 테스트에서 관측된 가장 나쁜 값

    양의 affine transformation이므로
    Pareto dominance 관계는 변하지 않는다.
    """
    out = sub.copy()

    for raw_col, norm_col in [
        ("Cost", "Cost_norm"),
        ("Tdis", "Tdis_norm"),
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
# 6. 그래프 보조 함수
# ============================================================

def marker_for_model(model: str) -> str:
    """
    All과 Seasonal 모델을 marker 모양으로 구분.
    """
    return "o" if model == "All" else "s"


def annotation_offset(i: int):
    """
    가까운 점의 label이 완전히 겹치는 것을 조금 줄이기 위한 offset.
    """
    offsets = [
        (7, 7),
        (7, -14),
        (-72, 7),
        (7, 16),
        (-82, -14),
        (7, -22),
    ]
    return offsets[i % len(offsets)]


def annotate_points(
    ax,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
):
    for i, (_, r) in enumerate(sub.iterrows()):
        dx, dy = annotation_offset(i)

        ax.annotate(
            r["label"],
            (r[x_col], r[y_col]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
        )


def draw_empirical_pareto_front(
    ax,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
):
    """
    6개 tested policy 중 global non-dominated points만 연결.

    주의:
      이것은 전체 최적화 공간의 theoretical Pareto frontier가 아니라
      'tested policies among empirical Pareto front'임.
    """
    front = (
        sub[sub["global_pareto"]]
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


def draw_model_trajectory(
    ax,
    sub: pd.DataFrame,
    model: str,
    x_col: str,
    y_col: str,
):
    """
    같은 모델이 reward weight에 따라 어떻게 이동하는지
    Training weight 오름차순으로 얇은 점선으로 연결.

    예:
      All: 8 -> 16 -> 26
      Summer: 8 -> 13 -> 16
      Winter: 8 -> 16 -> 109
    """
    m = (
        sub[sub["Model"] == model]
        .sort_values("Training weight")
    )

    ax.plot(
        m[x_col],
        m[y_col],
        linestyle=":",
        linewidth=1.0,
        alpha=0.55,
    )


# ============================================================
# 7. Raw KPI Pareto plot
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
        figsize=(7.4, 5.8)
    )

    # Seed1이므로 error bar 없음.
    for _, r in sub.iterrows():
        ax.scatter(
            r["Cost"],
            r["Tdis"],
            marker=marker_for_model(r["Model"]),
            s=90,
            label=r["label"],
        )

    # reward 설정 변화 방향을 보기 위한 같은-model 연결선
    draw_model_trajectory(
        ax,
        sub,
        "All",
        "Cost",
        "Tdis",
    )
    draw_model_trajectory(
        ax,
        sub,
        seasonal_model,
        "Cost",
        "Tdis",
    )

    # 6개 정책 전체 기준 empirical Pareto front
    draw_empirical_pareto_front(
        ax,
        sub,
        "Cost",
        "Tdis",
    )

    annotate_points(
        ax,
        sub,
        "Cost",
        "Tdis",
    )

    ax.set_title(
        f"{test_name}: Seed1 Cost–Comfort Pareto"
    )
    ax.set_xlabel("cost_tot")
    ax.set_ylabel("tdis_tot")
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=7.5,
        loc="best",
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        / f"seed1_pareto_raw_{test_name.replace(' ', '_').lower()}.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return sub


# ============================================================
# 8. 0~1 normalized Pareto plot
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
            r["Cost_norm"],
            r["Tdis_norm"],
            marker=marker_for_model(r["Model"]),
            s=90,
            label=r["label"],
        )

    draw_model_trajectory(
        ax,
        sub,
        "All",
        "Cost_norm",
        "Tdis_norm",
    )
    draw_model_trajectory(
        ax,
        sub,
        seasonal_model,
        "Cost_norm",
        "Tdis_norm",
    )

    draw_empirical_pareto_front(
        ax,
        sub,
        "Cost_norm",
        "Tdis_norm",
    )

    annotate_points(
        ax,
        sub,
        "Cost_norm",
        "Tdis_norm",
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # x/y를 같은 시각적 scale로 표시.
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_title(
        f"{test_name}: Seed1 Normalized Pareto"
    )
    ax.set_xlabel(
        "Normalized cost (0 = best, 1 = worst)"
    )
    ax.set_ylabel(
        "Normalized tdis (0 = best, 1 = worst)"
    )
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=7.2,
        loc="best",
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        / f"seed1_pareto_normalized_{test_name.replace(' ', '_').lower()}.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return sub


# ============================================================
# 9. 2x2 통합 그림
# ============================================================

def plot_all_raw():
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
    )

    for ax, (test_name, seasonal_model) in zip(
        axes.ravel(),
        COMPARISONS,
    ):
        sub = get_test_points(
            test_name,
            seasonal_model,
        )

        for _, r in sub.iterrows():
            ax.scatter(
                r["Cost"],
                r["Tdis"],
                marker=marker_for_model(r["Model"]),
                s=72,
                label=r["label"],
            )

        draw_model_trajectory(
            ax,
            sub,
            "All",
            "Cost",
            "Tdis",
        )
        draw_model_trajectory(
            ax,
            sub,
            seasonal_model,
            "Cost",
            "Tdis",
        )

        draw_empirical_pareto_front(
            ax,
            sub,
            "Cost",
            "Tdis",
        )

        annotate_points(
            ax,
            sub,
            "Cost",
            "Tdis",
        )

        ax.set_title(test_name)
        ax.set_xlabel("cost_tot")
        ax.set_ylabel("tdis_tot")
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Seed1: 8:1 vs 16:1 vs PI-scale — Raw Cost–Comfort Pareto",
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    output = (
        OUTPUT_DIR
        / "seed1_pareto_raw_all_tests.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_all_normalized():
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    for ax, (test_name, seasonal_model) in zip(
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
                r["Cost_norm"],
                r["Tdis_norm"],
                marker=marker_for_model(r["Model"]),
                s=72,
                label=r["label"],
            )

        draw_model_trajectory(
            ax,
            sub,
            "All",
            "Cost_norm",
            "Tdis_norm",
        )
        draw_model_trajectory(
            ax,
            sub,
            seasonal_model,
            "Cost_norm",
            "Tdis_norm",
        )

        draw_empirical_pareto_front(
            ax,
            sub,
            "Cost_norm",
            "Tdis_norm",
        )

        annotate_points(
            ax,
            sub,
            "Cost_norm",
            "Tdis_norm",
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
        "Seed1: 8:1 vs 16:1 vs PI-scale — Normalized Cost–Comfort Pareto",
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    output = (
        OUTPUT_DIR
        / "seed1_pareto_normalized_all_tests.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 10. 실행 및 CSV 저장
# ============================================================

raw_tables = []
normalized_tables = []

for test_name, seasonal_model in COMPARISONS:
    raw_sub = plot_raw(
        test_name,
        seasonal_model,
    )

    norm_sub = plot_normalized(
        test_name,
        seasonal_model,
    )

    raw_tables.append(raw_sub)
    normalized_tables.append(norm_sub)

plot_all_raw()
plot_all_normalized()


raw_summary = pd.concat(
    raw_tables,
    ignore_index=True,
)

normalized_summary = pd.concat(
    normalized_tables,
    ignore_index=True,
)

raw_summary.to_csv(
    OUTPUT_DIR / "seed1_pareto_raw_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

normalized_summary.to_csv(
    OUTPUT_DIR / "seed1_pareto_normalized_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 11. 콘솔 요약
# ============================================================

print("\n=== Seed1 empirical Pareto-optimal policies ===")

for test_name, seasonal_model in COMPARISONS:
    sub = get_test_points(
        test_name,
        seasonal_model,
    )

    front = sub[sub["global_pareto"]]

    print(f"\n[{test_name}]")

    for _, r in front.iterrows():
        print(
            f"  {r['label']}: "
            f"cost={r['Cost']:.6f}, "
            f"tdis={r['Tdis']:.6f}"
        )


print("\nDone.")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
