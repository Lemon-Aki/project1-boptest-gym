from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# 사용자가 수정할 부분
# =====================================================================

# seed 폴더들이 들어 있는 상위 폴더
ROOT_DIR = Path(r"/mnt/d/dev-code/boptest-gym/examples/agents")

# 이번 재학습에 사용한 seed 값
SEEDS = [123456, 199580, 173827, 101291, 362507]

# 결과 저장 폴더
OUTPUT_DIR = ROOT_DIR / "training_transition_usage"


# =====================================================================
# 현재 실험 설정
# =====================================================================

MODELS = ["all", "sum", "win"]
FOLDER_NAME = "SAC_D_3e05_air_seoul_{model}_34_16-1_{seed}_300k_4_logdir"
CSV_NAME = "episode_rewards.csv"

YEAR = 2021
YEAR_START = pd.Timestamp("2021-01-01")
YEAR_END = pd.Timestamp("2021-12-31")
SECONDS_PER_DAY = 24 * 60 * 60

# 실제 학습 설정: 15분 간격, RL 학습 에피소드 7일
STEP_SECONDS = 15 * 60
EPISODE_DAYS = 7
STEPS_PER_EPISODE = EPISODE_DAYS * SECONDS_PER_DAY // STEP_SECONDS  # 672

# 계절 구간은 끝 날짜를 포함하지 않는 [시작, 끝) 형식입니다.
SEASON_PERIODS = {
    # Summer: 2021-05-22 ~ 2021-09-29
    "sum": [
        (pd.Timestamp("2021-05-22"), pd.Timestamp("2021-09-30")),
    ],
    # Winter: 2021-01-01 ~ 2021-04-14, 2021-12-04 ~ 2021-12-31
    "win": [
        (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-04-15")),
        (pd.Timestamp("2021-12-04"), pd.Timestamp("2022-01-01")),
    ],
}

SEASON_LABELS = {
    "sum": "Summer",
    "win": "Winter",
}

# 학습에 사용하면 안 되는 7일 보호구간, 끝 날짜는 포함하지 않습니다.
NO_TRAIN_PERIODS = [
    (pd.Timestamp("2021-01-22"), pd.Timestamp("2021-01-29")),
    (pd.Timestamp("2021-04-03"), pd.Timestamp("2021-04-10")),
    (pd.Timestamp("2021-06-22"), pd.Timestamp("2021-06-29")),
    (pd.Timestamp("2021-09-16"), pd.Timestamp("2021-09-23")),
    (pd.Timestamp("2021-11-21"), pd.Timestamp("2021-11-28")),
]


def read_start_times(csv_path):
    """episode_rewards.csv에서 완료 에피소드의 시작 초를 읽습니다."""

    df = pd.read_csv(csv_path, usecols=["start_time_sec"])
    values = pd.to_numeric(df["start_time_sec"], errors="coerce")

    if values.isna().any():
        raise ValueError("start_time_sec에 숫자가 아닌 값이 있습니다.")

    values = values.to_numpy(dtype=np.float64)
    if not np.all(values == np.floor(values)):
        raise ValueError("start_time_sec에는 정수 초가 들어 있어야 합니다.")

    values = values.astype(np.int64)
    if np.any(values < 0):
        raise ValueError("start_time_sec에 음수가 있습니다.")

    return values


def count_daily_transitions(start_times):
    """각 transition의 시작 시각을 기준으로 날짜별 횟수를 계산합니다."""

    number_of_days = len(pd.date_range(YEAR_START, YEAR_END, freq="D"))
    daily_counts = np.zeros(number_of_days, dtype=np.int64)
    offsets = np.arange(STEPS_PER_EPISODE, dtype=np.int64) * STEP_SECONDS

    for episode_start in start_times:
        day_indices = (episode_start + offsets) // SECONDS_PER_DAY
        valid = (day_indices >= 0) & (day_indices < number_of_days)
        daily_counts += np.bincount(
            day_indices[valid],
            minlength=number_of_days,
        )[:number_of_days]

    return daily_counts


def make_period_mask(all_dates, periods):
    """여러 개의 [시작, 끝) 기간에 해당하는 날짜 마스크를 만듭니다."""

    mask = np.zeros(len(all_dates), dtype=bool)
    for period_start, period_end in periods:
        mask |= (all_dates >= period_start) & (all_dates < period_end)
    return mask


def make_valid_season_mask(all_dates, season_model):
    """계절 구간에서 테스트 보호구간을 제외한 날짜를 반환합니다."""

    season_mask = make_period_mask(all_dates, SEASON_PERIODS[season_model])
    protected_mask = make_period_mask(all_dates, NO_TRAIN_PERIODS)
    return season_mask & ~protected_mask


def sample_std(series):
    """2개 이상이면 표본 표준편차, 아니면 NaN을 반환합니다."""

    if len(series) < 2:
        return np.nan
    return float(series.std(ddof=1))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dates = pd.date_range(YEAR_START, YEAR_END, freq="D")
    result_rows = []

    # seed를 반복하고, 각 seed 안에서 all/sum/win 폴더를 반복합니다.
    for seed in SEEDS:
        print()
        print("=" * 72)
        print(f"Seed: {seed}")
        print("=" * 72)

        daily_by_model = {}
        episodes_by_model = {}

        for model in MODELS:
            folder = ROOT_DIR / FOLDER_NAME.format(model=model, seed=seed)
            csv_path = folder / CSV_NAME
            print(f"{model.upper():>3}: {csv_path}")

            if not csv_path.is_file():
                print("     [ERROR] episode_rewards.csv 파일을 찾을 수 없습니다.")
                continue

            try:
                start_times = read_start_times(csv_path)
                daily_counts = count_daily_transitions(start_times)
            except (ValueError, OSError) as error:
                print(f"     [ERROR] {error}")
                continue

            daily_by_model[model] = daily_counts
            episodes_by_model[model] = len(start_times)
            print(
                f"     완료 에피소드 {len(start_times):,}개, "
                f"집계 transition {daily_counts.sum():,}개"
            )

        if "all" not in daily_by_model:
            print("[SKIP] All 파일이 없으므로 이 seed의 비율을 계산할 수 없습니다.")
            continue

        # All 대 Summer, All 대 Winter를 차례로 비교합니다.
        for season_model in ["sum", "win"]:
            if season_model not in daily_by_model:
                print(
                    f"[SKIP] {season_model.upper()} 파일이 없어 "
                    "해당 계절 비교를 생략합니다."
                )
                continue

            season_label = SEASON_LABELS[season_model]
            valid_mask = make_valid_season_mask(all_dates, season_model)
            valid_days = int(valid_mask.sum())

            all_target_count = int(daily_by_model["all"][valid_mask].sum())
            seasonal_target_count = int(
                daily_by_model[season_model][valid_mask].sum()
            )
            seasonal_outside_count = int(
                daily_by_model[season_model][~valid_mask].sum()
            )

            if all_target_count == 0:
                exposure_ratio = np.nan
            else:
                exposure_ratio = seasonal_target_count / all_target_count

            all_mean_per_day = all_target_count / valid_days
            seasonal_mean_per_day = seasonal_target_count / valid_days

            result_rows.append(
                {
                    "seed": seed,
                    "comparison": f"{season_model}_vs_all",
                    "season": season_label,
                    "valid_target_days": valid_days,
                    "all_completed_episodes": episodes_by_model["all"],
                    "seasonal_completed_episodes": episodes_by_model[season_model],
                    "all_target_transitions": all_target_count,
                    "seasonal_target_transitions": seasonal_target_count,
                    "all_mean_transitions_per_target_day": all_mean_per_day,
                    "seasonal_mean_transitions_per_target_day": seasonal_mean_per_day,
                    "seasonal_to_all_exposure_ratio": exposure_ratio,
                    "seasonal_transitions_outside_valid_target": seasonal_outside_count,
                }
            )

            print(f"\n[{season_label} exposure comparison]")
            print(f"  유효 계절 날짜 수              : {valid_days}일")
            print(f"  All의 해당 계절 transition     : {all_target_count:,}개")
            print(f"  {season_model.upper()} transition              : {seasonal_target_count:,}개")
            print(f"  All 날짜당 평균                : {all_mean_per_day:,.2f}개")
            print(
                f"  {season_model.upper()} 날짜당 평균                : "
                f"{seasonal_mean_per_day:,.2f}개"
            )
            print(f"  실제 노출비 ({season_model.upper()} / All)       : {exposure_ratio:.4f}배")

            if seasonal_outside_count > 0:
                print(
                    "  [WARNING] 계절 모델에서 유효 계절 밖 또는 보호구간의 "
                    f"transition이 {seasonal_outside_count:,}개 발견됐습니다."
                )

    if not result_rows:
        print("\n[ERROR] 계산 가능한 All-계절 모델 쌍이 없습니다.")
        return

    by_seed_df = pd.DataFrame(result_rows)
    by_seed_output = OUTPUT_DIR / "transition_exposure_ratio_by_seed.csv"
    by_seed_df.to_csv(by_seed_output, index=False)

    summary_rows = []
    for comparison, group in by_seed_df.groupby("comparison", sort=False):
        ratio = group["seasonal_to_all_exposure_ratio"].dropna()
        all_total = group["all_target_transitions"]
        seasonal_total = group["seasonal_target_transitions"]
        all_daily = group["all_mean_transitions_per_target_day"]
        seasonal_daily = group["seasonal_mean_transitions_per_target_day"]

        summary_rows.append(
            {
                "comparison": comparison,
                "season": group["season"].iloc[0],
                "number_of_seeds": len(group),
                "valid_target_days": int(group["valid_target_days"].iloc[0]),
                "all_target_transitions_mean": all_total.mean(),
                "all_target_transitions_std": sample_std(all_total),
                "seasonal_target_transitions_mean": seasonal_total.mean(),
                "seasonal_target_transitions_std": sample_std(seasonal_total),
                "all_mean_per_day_across_seeds": all_daily.mean(),
                "all_mean_per_day_std_across_seeds": sample_std(all_daily),
                "seasonal_mean_per_day_across_seeds": seasonal_daily.mean(),
                "seasonal_mean_per_day_std_across_seeds": sample_std(
                    seasonal_daily
                ),
                "exposure_ratio_mean": ratio.mean(),
                "exposure_ratio_std": sample_std(ratio),
                "ratio_of_mean_transition_totals": (
                    seasonal_total.mean() / all_total.mean()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_output = OUTPUT_DIR / "transition_exposure_ratio_summary.csv"
    summary_df.to_csv(summary_output, index=False)

    print()
    print("=" * 72)
    print("5개 seed 요약")
    print("=" * 72)
    for _, row in summary_df.iterrows():
        print(
            f"{row['season']}: 계절 모델 / All 실제 노출비 = "
            f"{row['exposure_ratio_mean']:.4f} ± "
            f"{row['exposure_ratio_std']:.4f} "
            f"(n={int(row['number_of_seeds'])})"
        )

    print(f"\nSeed별 결과 : {by_seed_output}")
    print(f"평균±표준편차: {summary_output}")


if __name__ == "__main__":
    main()