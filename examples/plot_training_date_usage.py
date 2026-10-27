from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


# =====================================================================
# 사용자가 수정할 부분
# =====================================================================

# seed 폴더들이 들어 있는 상위 폴더
ROOT_DIR = Path(r"/mnt/d/dev-code/boptest-gym/examples/agents")

# 이번 재학습에 사용한 seed 값
SEEDS = [123456, 199580, 173827, 101291, 362507]

# 학습 데이터 종류
MODELS = ["all", "sum", "win"]


# =====================================================================
# 폴더 및 실험 설정
# =====================================================================

FOLDER_NAME = "SAC_D_3e05_air_seoul_{model}_34_16-1_{seed}_300k_4_logdir"
CSV_NAME = "episode_rewards.csv"
OUTPUT_NAME = "training_date_usage.png"

YEAR_START = pd.Timestamp("2021-01-01")
YEAR_END = pd.Timestamp("2021-12-31")
EPISODE_LENGTH = pd.Timedelta(days=7)


def draw_graph(model, seed, csv_path, output_path=None):
    """reward CSV 하나의 날짜별 학습 사용 횟수를 그래프로 저장합니다."""

    df = pd.read_csv(csv_path)

    if "start_time_sec" not in df.columns:
        raise ValueError("CSV에 start_time_sec 열이 없습니다.")

    # 1월 1일부터 12월 31일까지 날짜별 사용 횟수를 0으로 초기화
    all_dates = pd.date_range(YEAR_START, YEAR_END, freq="D")
    daily_count = pd.Series(0, index=all_dates, dtype=int)

    for start_time_sec in df["start_time_sec"]:
        episode_start = YEAR_START + pd.to_timedelta(start_time_sec, unit="s")
        episode_end = episode_start + EPISODE_LENGTH

        # 에피소드가 조금이라도 사용한 모든 날짜
        first_date = episode_start.normalize()
        last_date = (episode_end - pd.Timedelta(nanoseconds=1)).normalize()

        used_dates = pd.date_range(first_date, last_date, freq="D")
        used_dates = used_dates.intersection(all_dates)

        # 이 에피소드가 사용한 각 날짜의 y값을 1씩 증가
        daily_count.loc[used_dates] += 1

    if output_path is None:
        output_path = csv_path.parent / OUTPUT_NAME

    # 날짜별 사용 횟수이므로 막대그래프로 표시
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(
        daily_count.index,
        daily_count.values,
        width=1.0,
        color="#4C78A8",
        edgecolor="none",
    )

    ax.set_title(
        f"Training Episode Coverage by Date - {model.upper()} / Seed {seed}",
        fontsize=16,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of episodes using the date", fontsize=12)

    # x축을 1월부터 12월까지 월 단위로 표시
    month_starts = pd.date_range(YEAR_START, YEAR_END, freq="MS")
    ax.set_xlim(
        YEAR_START - pd.Timedelta(hours=12),
        YEAR_END + pd.Timedelta(hours=12),
    )
    ax.set_xticks(month_starts)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # y축 눈금을 정수로 표시
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return len(df), daily_count, output_path


def main():
    success_count = 0
    error_count = 0

    for model in MODELS:
        for seed in SEEDS:
            folder = ROOT_DIR / FOLDER_NAME.format(model=model, seed=seed)
            csv_path = folder / CSV_NAME

            print("=" * 70)
            print(f"Model: {model} / Seed: {seed}")
            print(f"CSV  : {csv_path}")

            if not csv_path.is_file():
                print("[ERROR] episode_rewards.csv 파일을 찾을 수 없습니다.")
                error_count += 1
                continue

            try:
                episode_count, daily_count, output_path = draw_graph(
                    model, seed, csv_path
                )
            except (ValueError, OSError) as error:
                print(f"[ERROR] {error}")
                error_count += 1
                continue

            print(f"전체 에피소드 수       : {episode_count}개")
            print(f"학습에 사용된 날짜     : {(daily_count > 0).sum()}일")
            print(f"하루 최대 사용 횟수    : {daily_count.max()}회")
            print(f"[완료] 그래프 저장 경로: {output_path}")
            success_count += 1

    print()
    print("=" * 70)
    print("최종 결과")
    print("=" * 70)
    print(f"그래프 생성 완료: {success_count}개 seed")
    print(f"오류             : {error_count}개 seed")


if __name__ == "__main__":
    main()