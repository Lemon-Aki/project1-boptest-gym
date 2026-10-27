"""
plot_pareto_8vs16.py

BOPTEST SAC 결과에서 8:1 vs 16:1 reward-weight 정책의
cost_tot - tdis_tot Pareto 그래프를 생성한다.

입력 Excel:
  1) boptest_dynamic_8to1_seed1-5_analysis.xlsx
  2) boptest_dynamic_16to1_seed1-5_analysis.xlsx

두 파일 모두 'Raw_Results' 시트에서 데이터를 읽는다.

생성 결과:
  - raw Pareto plot (5-seed mean only; no SD error bars)
  - 0~1 Min-Max normalized Pareto plot
  - 요약 CSV
  - 그래프 PNG

필요 패키지:
    pip install pandas numpy matplotlib openpyxl
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 사용자 설정
# ============================================================

FILE_8 = Path("boptest_dynamic_8to1_seed1-5_analysis.xlsx")
FILE_16 = Path("boptest_dynamic_16to1_seed1-5_analysis.xlsx")

SHEET_NAME = "Raw_Results"

OUTPUT_DIR = Path("pareto_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 논문 핵심 비교:
# Summer는 냉방, Winter는 난방
COMPARISONS = [
    ("Peak Cool", "Summer"),
    ("Typical Cool", "Summer"),
    ("Peak Heat", "Winter"),
    ("Typical Heat", "Winter"),
]

# 그림 dpi
DPI = 300


# ============================================================
# 2. 데이터 읽기
# ============================================================

def load_result(path: Path, weight: int) -> pd.DataFrame:
    """
    분석 Excel의 Raw_Results 시트를 읽는다.

    이 Excel은
      1행: 제목
      2행: 빈 행
      3행: 실제 컬럼명
    구조이므로 기본 header=0으로 읽으면 컬럼명이 틀어진다.

    아래 코드는 앞부분을 먼저 읽어서 실제 헤더 행을 자동 탐색한다.
    """
    required = {
        "Seed", "Model", "Test",
        "cost_tot", "tdis_tot"
    }

    # 헤더 없이 앞부분을 읽어 실제 header row 탐색
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
            f"{path.name}의 {SHEET_NAME} 시트에서 실제 헤더 행을 찾지 못했습니다. "
            f"필요 컬럼: {sorted(required)}"
        )

    # 찾은 행을 실제 header로 다시 읽음
    df = pd.read_excel(
        path,
        sheet_name=SHEET_NAME,
        header=header_row,
    )

    # 혹시 공백이 들어간 경우 대비
    df.columns = [str(c).strip() for c in df.columns]

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name}에 필요한 컬럼이 없습니다: {sorted(missing)}\n"
            f"현재 컬럼: {list(df.columns)}"
        )

    df = df.copy()

    # 완전히 빈 행 제거
    df = df.dropna(
        subset=["Seed", "Model", "Test", "cost_tot", "tdis_tot"],
        how="all",
    )

    df["Weight"] = weight

    # 숫자형 변환
    df["Seed"] = pd.to_numeric(df["Seed"], errors="coerce")
    df["cost_tot"] = pd.to_numeric(df["cost_tot"], errors="coerce")
    df["tdis_tot"] = pd.to_numeric(df["tdis_tot"], errors="coerce")

    # 필요한 값이 없는 행 제거
    df = df.dropna(
        subset=["Seed", "Model", "Test", "cost_tot", "tdis_tot"]
    )

    return df


df8 = load_result(FILE_8, 8)
df16 = load_result(FILE_16, 16)

df = pd.concat([df8, df16], ignore_index=True)


# ============================================================
# 3. 5-seed 평균/표준편차 계산
# ============================================================

summary = (
    df.groupby(["Test", "Model", "Weight"], as_index=False)
      .agg(
          cost_mean=("cost_tot", "mean"),
          cost_sd=("cost_tot", "std"),
          tdis_mean=("tdis_tot", "mean"),
          tdis_sd=("tdis_tot", "std"),
          n_seed=("Seed", "nunique"),
      )
)


# ============================================================
# 4. Pareto 판정
# ============================================================

def pareto_mask(costs: np.ndarray) -> np.ndarray:
    """
    costs[:, 0] = cost
    costs[:, 1] = tdis

    두 목적함수 모두 minimize.
    True = non-dominated (Pareto-optimal among tested points)
    """
    n = len(costs)
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # j가 i보다 두 목적에서 같거나 좋고,
            # 적어도 하나에서는 strict하게 좋으면 i는 dominated
            if (
                np.all(costs[j] <= costs[i])
                and np.any(costs[j] < costs[i])
            ):
                mask[i] = False
                break

    return mask


# ============================================================
# 5. 테스트별 데이터 구성
# ============================================================

def get_test_points(test_name: str, seasonal_model: str) -> pd.DataFrame:
    """
    한 테스트에서 다음 네 점을 가져온다.

      All 8:1
      All 16:1
      Seasonal 8:1
      Seasonal 16:1
    """
    sub = summary[
        (summary["Test"] == test_name)
        & (summary["Model"].isin(["All", seasonal_model]))
        & (summary["Weight"].isin([8, 16]))
    ].copy()

    expected = 4
    if len(sub) != expected:
        raise ValueError(
            f"{test_name}: expected {expected} points, got {len(sub)}"
        )

    sub["label"] = (
        sub["Model"].astype(str)
        + " "
        + sub["Weight"].astype(str)
        + ":1"
    )

    pts = sub[["cost_mean", "tdis_mean"]].to_numpy()
    sub["pareto"] = pareto_mask(pts)

    return sub.reset_index(drop=True)


# ============================================================
# 6. 0~1 Min-Max 정규화
# ============================================================

def minmax_normalize(sub: pd.DataFrame) -> pd.DataFrame:
    """
    각 테스트의 4개 점 안에서 cost와 tdis를 각각 0~1로 정규화.

    0 = 그 테스트에서 관측된 최솟값
    1 = 그 테스트에서 관측된 최댓값

    양의 affine transformation이므로
    Pareto dominance 관계는 변하지 않는다.
    """
    out = sub.copy()

    for raw_col, norm_col in [
        ("cost_mean", "cost_norm"),
        ("tdis_mean", "tdis_norm"),
    ]:
        lo = out[raw_col].min()
        hi = out[raw_col].max()

        if np.isclose(hi, lo):
            out[norm_col] = 0.0
        else:
            out[norm_col] = (out[raw_col] - lo) / (hi - lo)

    return out


# ============================================================
# 7. 그래프 유틸리티
# ============================================================

def annotate_points(ax, sub, xcol, ycol):
    """
    각 점 옆에 정책 label 표시.
    """
    for _, r in sub.iterrows():
        ax.annotate(
            r["label"],
            (r[xcol], r[ycol]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )


def draw_pareto_front(ax, sub, xcol, ycol):
    """
    현재 테스트의 non-dominated point만 x축 기준 정렬 후 연결.
    """
    front = sub[sub["pareto"]].sort_values(xcol)

    if len(front) >= 2:
        ax.plot(
            front[xcol],
            front[ycol],
            linestyle="--",
            linewidth=1.5,
            label="Empirical Pareto front",
        )


# ============================================================
# 8. Raw Pareto plot
# ============================================================

def plot_raw(test_name: str, seasonal_model: str):
    """
    원본 KPI 값(cost_tot, tdis_tot)을 사용한 Pareto 그래프.

    각 점은 5개 seed의 평균값만 표시한다.
    SD는 그래프에 그리지 않지만 summary CSV에는 그대로 저장된다.
    """
    sub = get_test_points(test_name, seasonal_model)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    # 정책별 5-seed 평균점만 표시
    for _, r in sub.iterrows():
        marker = "o" if r["Model"] == "All" else "s"

        ax.scatter(
            r["cost_mean"],
            r["tdis_mean"],
            marker=marker,
            s=90,
            label=r["label"],
        )

    # 평균점 기준 empirical Pareto front
    draw_pareto_front(ax, sub, "cost_mean", "tdis_mean")
    annotate_points(ax, sub, "cost_mean", "tdis_mean")

    ax.set_title(f"{test_name}: Cost–Comfort Pareto (5-seed mean)")
    ax.set_xlabel("cost_tot")
    ax.set_ylabel("tdis_tot")
    ax.grid(alpha=0.25)

    # legend 중복 제거
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(
        uniq.values(),
        uniq.keys(),
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    out = OUTPUT_DIR / f"pareto_raw_{test_name.replace(' ', '_').lower()}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return sub


# ============================================================
# 9. 0~1 Normalized Pareto plot
# ============================================================

def plot_normalized(test_name: str, seasonal_model: str):
    raw_sub = get_test_points(test_name, seasonal_model)
    sub = minmax_normalize(raw_sub)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    for _, r in sub.iterrows():
        marker = "o" if r["Model"] == "All" else "s"

        ax.scatter(
            r["cost_norm"],
            r["tdis_norm"],
            marker=marker,
            s=85,
            label=r["label"],
        )

    draw_pareto_front(ax, sub, "cost_norm", "tdis_norm")
    annotate_points(ax, sub, "cost_norm", "tdis_norm")

    # 동일한 0~1 스케일
    # annotation이 잘리지 않도록 약간 여유만 둔다.
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # 시각적으로 x/y 동일 스케일
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"{test_name}: Normalized Cost–Comfort Pareto")
    ax.set_xlabel("Normalized cost (0 = best, 1 = worst)")
    ax.set_ylabel("Normalized tdis (0 = best, 1 = worst)")
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(
        uniq.values(),
        uniq.keys(),
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    out = OUTPUT_DIR / f"pareto_normalized_{test_name.replace(' ', '_').lower()}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return sub


# ============================================================
# 10. 2x2 figure: 논문/발표용
# ============================================================

def plot_all_normalized():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (test_name, seasonal_model) in zip(
        axes.ravel(),
        COMPARISONS
    ):
        sub = minmax_normalize(
            get_test_points(test_name, seasonal_model)
        )

        for _, r in sub.iterrows():
            marker = "o" if r["Model"] == "All" else "s"

            ax.scatter(
                r["cost_norm"],
                r["tdis_norm"],
                marker=marker,
                s=75,
                label=r["label"],
            )

        draw_pareto_front(
            ax,
            sub,
            "cost_norm",
            "tdis_norm",
        )
        annotate_points(
            ax,
            sub,
            "cost_norm",
            "tdis_norm",
        )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(test_name)
        ax.set_xlabel("Normalized cost")
        ax.set_ylabel("Normalized tdis")
        ax.grid(alpha=0.25)

    # 공통 legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))

    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        "8:1 vs 16:1 — Normalized Cost–Comfort Pareto",
        fontsize=15,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    out = OUTPUT_DIR / "pareto_normalized_all_tests.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 11. 실행
# ============================================================

all_normalized = []

for test_name, seasonal_model in COMPARISONS:
    plot_raw(test_name, seasonal_model)

    norm = plot_normalized(
        test_name,
        seasonal_model
    )

    all_normalized.append(norm)

plot_all_normalized()


# ============================================================
# 12. CSV 저장
# ============================================================

summary.to_csv(
    OUTPUT_DIR / "pareto_8vs16_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

normalized_df = pd.concat(
    all_normalized,
    ignore_index=True,
)

normalized_df.to_csv(
    OUTPUT_DIR / "pareto_8vs16_normalized.csv",
    index=False,
    encoding="utf-8-sig",
)


print("Done.")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
