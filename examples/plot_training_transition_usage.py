from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================
# 사용자가 수정할 부분
# =====================================================================

# seed 폴더들이 들어 있는 상위 폴더
ROOT_DIR = Path(r"/mnt/d/dev-code/boptest-gym/examples/agents")

# 이번 재학습에 사용한 seed 값
SEEDS = [123456, 199580, 173827, 101291, 362507]

# 폴더명에서 바뀌는 학습 데이터 종류
MODELS = ["all", "sum", "win"]

# 결과를 저장할 폴더
OUTPUT_DIR = ROOT_DIR / "training_transition_usage"

# 그래프에 개별 seed 선도 함께 표시할지 여부
SHOW_INDIVIDUAL_SEEDS = True


# =====================================================================
# 현재 실험 설정
# =====================================================================

FOLDER_NAME = "SAC_D_3e05_air_seoul_{model}_34_16-1_{seed}_300k_4_logdir"
CSV_NAME = "episode_rewards.csv"

YEAR = 2021
YEAR_START = pd.Timestamp(f"{YEAR}-01-01")
YEAR_END = pd.Timestamp(f"{YEAR}-12-31")
SECONDS_PER_DAY = 24 * 60 * 60

# 실제 학습 설정: 15분 간격, RL 학습 에피소드 7일
STEP_SECONDS = 15 * 60
EPISODE_DAYS = 7
STEPS_PER_EPISODE = EPISODE_DAYS * SECONDS_PER_DAY // STEP_SECONDS  # 672

# 끝 시각은 포함하지 않는 구간 [시작, 끝)입니다.
NO_TRAIN_PERIODS = [
    ("Peak Heat", pd.Timestamp("2021-01-22"), pd.Timestamp("2021-01-29")),
    ("Typical Heat", pd.Timestamp("2021-04-03"), pd.Timestamp("2021-04-10")),
    ("Typical Cool", pd.Timestamp("2021-06-22"), pd.Timestamp("2021-06-29")),
    ("Peak Cool", pd.Timestamp("2021-09-16"), pd.Timestamp("2021-09-23")),
    ("Mix", pd.Timestamp("2021-11-21"), pd.Timestamp("2021-11-28")),
]


def read_start_times(csv_path):
    """episode_rewards.csv에서 완료된 에피소드의 시작 초를 읽습니다."""

    df = pd.read_csv(csv_path, usecols=["start_time_sec"])
    start_times = pd.to_numeric(df["start_time_sec"], errors="coerce")

    if start_times.isna().any():
        bad_rows = (start_times.isna()).sum()
        raise ValueError(f"start_time_sec에 숫자가 아닌 값이 {bad_rows}개 있습니다.")

    values = start_times.to_numpy(dtype=np.float64)
    if not np.all(values == np.floor(values)):
        raise ValueError("start_time_sec에는 정수 초가 들어 있어야 합니다.")

    values = values.astype(np.int64)
    if np.any(values < 0):
        raise ValueError("start_time_sec에 음수가 있습니다.")

    return values


def count_daily_transitions(start_times):
    """
    각 transition의 시작 시각이 속한 날짜를 기준으로 365일 횟수를 셉니다.

    모든 transition을 큰 표로 만들지 않습니다. 에피소드 하나의 672개 날짜
    인덱스만 계산한 뒤 즉시 365일 카운터에 더합니다.
    """

    number_of_days = (pd.Timestamp(f"{YEAR + 1}-01-01") - YEAR_START).days
    daily_counts = np.zeros(number_of_days, dtype=np.int64)
    transition_offsets = np.arange(STEPS_PER_EPISODE, dtype=np.int64) * STEP_SECONDS
    outside_year_count = 0

    for episode_start in start_times:
        transition_seconds = episode_start + transition_offsets
        day_indices = transition_seconds // SECONDS_PER_DAY

        in_year = (day_indices >= 0) & (day_indices < number_of_days)
        valid_indices = day_indices[in_year]

        daily_counts += np.bincount(
            valid_indices,
            minlength=number_of_days,
        )[:number_of_days]
        outside_year_count += int((~in_year).sum())

    return daily_counts, outside_year_count


def make_protected_mask(all_dates):
    """학습 금지 기간에 해당하는 날짜 마스크를 만듭니다."""

    mask = np.zeros(len(all_dates), dtype=bool)
    for _, period_start, period_end in NO_TRAIN_PERIODS:
        mask |= (all_dates >= period_start) & (all_dates < period_end)
    return mask


def save_model_results(model, all_dates, seed_counts):
    """모델 하나의 날짜별 CSV와 평균±표준편차 그래프를 저장합니다."""

    seed_numbers = list(seed_counts.keys())
    count_matrix = np.vstack([seed_counts[seed] for seed in seed_numbers])

    daily_mean = count_matrix.mean(axis=0)
    if len(seed_numbers) >= 2:
        daily_std = count_matrix.std(axis=0, ddof=1)  # 표본 표준편차
    else:
        daily_std = np.zeros(len(all_dates), dtype=float)

    result_df = pd.DataFrame({"date": all_dates})
    for seed in seed_numbers:
        result_df[f"seed_{seed}"] = seed_counts[seed]
    result_df["mean_across_seeds"] = daily_mean
    result_df["std_across_seeds"] = daily_std

    csv_output = OUTPUT_DIR / f"transition_usage_{model}.csv"
    result_df.to_csv(csv_output, index=False, date_format="%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(16, 6))

    if SHOW_INDIVIDUAL_SEEDS:
        for seed in seed_numbers:
            ax.plot(
                all_dates,
                seed_counts[seed],
                linewidth=0.8,
                alpha=0.35,
                label=f"Seed {seed}",
            )

    lower = np.maximum(daily_mean - daily_std, 0)
    upper = daily_mean + daily_std
    ax.fill_between(
        all_dates,
        lower,
        upper,
        color="#4C78A8",
        alpha=0.22,
        label="Mean ± 1 SD",
    )
    ax.plot(
        all_dates,
        daily_mean,
        color="#1F4E79",
        linewidth=2.0,
        label="Mean across seeds",
    )

    # 학습 금지 기간을 연한 빨간색으로 표시합니다.
    for index, (_, period_start, period_end) in enumerate(NO_TRAIN_PERIODS):
        ax.axvspan(
            period_start,
            period_end,
            color="#E45756",
            alpha=0.12,
            label="No-train period" if index == 0 else None,
        )

    ax.set_title(
        f"Daily Training Transitions - {model.upper()} "
        f"({len(seed_numbers)} seeds)",
        fontsize=15,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of 15-minute transitions")
    ax.set_xlim(
        YEAR_START - pd.Timedelta(hours=12),
        YEAR_END + pd.Timedelta(hours=12),
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False)

    fig.tight_layout()
    png_output = OUTPUT_DIR / f"transition_usage_{model}.png"
    fig.savefig(png_output, dpi=200)
    plt.close(fig)

    return csv_output, png_output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dates = pd.date_range(YEAR_START, YEAR_END, freq="D")
    protected_mask = make_protected_mask(all_dates)
    run_summary = []

    for model in MODELS:
        print()
        print("=" * 70)
        print(f"Model: {model}")
        print("=" * 70)

        seed_counts = {}

        for seed in SEEDS:
            folder = ROOT_DIR / FOLDER_NAME.format(model=model, seed=seed)
            csv_path = folder / CSV_NAME

            print(f"Seed {seed}: {csv_path}")

            if not csv_path.is_file():
                print("  [ERROR] episode_rewards.csv 파일을 찾을 수 없습니다.")
                continue

            try:
                start_times = read_start_times(csv_path)
                daily_counts, outside_year_count = count_daily_transitions(
                    start_times
                )
            except (ValueError, OSError) as error:
                print(f"  [ERROR] {error}")
                continue

            expected_count = len(start_times) * STEPS_PER_EPISODE
            counted_in_year = int(daily_counts.sum())
            protected_count = int(daily_counts[protected_mask].sum())
            seed_counts[seed] = daily_counts

            run_summary.append(
                {
                    "model": model,
                    "seed": seed,
                    "completed_episodes": len(start_times),
                    "expected_transitions": expected_count,
                    "transitions_in_year": counted_in_year,
                    "transitions_outside_year": outside_year_count,
                    "transitions_in_no_train_periods": protected_count,
                }
            )

            print(f"  완료 에피소드          : {len(start_times):,}개")
            print(f"  연도 내 transition     : {counted_in_year:,}개")
            print(f"  연도 밖 transition     : {outside_year_count:,}개")
            print(f"  학습 금지 기간 사용량  : {protected_count:,}개")

            if counted_in_year + outside_year_count != expected_count:
                print("  [WARNING] transition 합계가 예상값과 다릅니다.")
            if protected_count > 0:
                print("  [WARNING] 학습 금지 기간과 겹치는 transition이 있습니다.")

        if not seed_counts:
            print(f"[ERROR] {model}에서 읽을 수 있는 seed가 없습니다.")
            continue

        csv_output, png_output = save_model_results(
            model,
            all_dates,
            seed_counts,
        )
        print(f"[완료] 날짜별 CSV : {csv_output}")
        print(f"[완료] 그래프     : {png_output}")

    summary_output = OUTPUT_DIR / "transition_usage_run_summary.csv"
    pd.DataFrame(run_summary).to_csv(summary_output, index=False)

    print()
    print("=" * 70)
    print(f"전체 실행 요약: {summary_output}")
    print("모델별 CSV는 365행뿐이며, 개별 transition 행은 저장하지 않습니다.")


if __name__ == "__main__":
    main()
