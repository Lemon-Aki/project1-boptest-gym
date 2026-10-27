from pathlib import Path
import pandas as pd


# =====================================================================
# 사용자가 수정할 부분
# =====================================================================

# seed 폴더들이 들어 있는 상위 폴더
ROOT_DIR = Path(r"/mnt/d/dev-code/boptest-gym/examples/agents")

# 이번 재학습에 사용한 seed 값
SEEDS = [123456, 199580, 173827, 101291, 362507]


# =====================================================================
# 현재 파일 및 실험 설정
# =====================================================================

FOLDER_NAME = "SAC_D_3e05_air_seoul_win_34_16-1_{seed}_300k_4_logdir"
CSV_NAME = "episode_rewards.csv"

YEAR_START = pd.Timestamp("2021-01-01")
EPISODE_LENGTH = pd.Timedelta(days=7)

# 학습에 사용하면 안 되는 1주 보호구간
NO_TRAIN_PERIODS = [
    ("Peak Heat",    pd.Timestamp("2021-01-22"), pd.Timestamp("2021-01-29")),
    ("Typical Heat", pd.Timestamp("2021-04-03"), pd.Timestamp("2021-04-10")),
    ("Peak Cool",    pd.Timestamp("2021-09-16"), pd.Timestamp("2021-09-23")),
    ("Typical Cool", pd.Timestamp("2021-06-22"), pd.Timestamp("2021-06-29")),
    ("Mix",          pd.Timestamp("2021-11-21"), pd.Timestamp("2021-11-28")),
]


pass_count = 0
fail_count = 0
error_count = 0

for seed in SEEDS:
    folder = ROOT_DIR / FOLDER_NAME.format(seed=seed)
    csv_path = folder / CSV_NAME

    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"CSV : {csv_path}")

    if not csv_path.is_file():
        print("[ERROR] episode_rewards.csv 파일을 찾을 수 없습니다.")
        error_count += 1
        continue

    df = pd.read_csv(csv_path)

    if "start_time_sec" not in df.columns:
        print("[ERROR] CSV에 start_time_sec 열이 없습니다.")
        error_count += 1
        continue

    # 각 에피소드의 7일 학습 시작·종료 시각 계산
    df["train_start"] = (
        YEAR_START
        + pd.to_timedelta(df["start_time_sec"], unit="s")
    )
    df["train_end"] = (
        df["train_start"]
        + EPISODE_LENGTH
    )

    seed_violations = 0

    for period_name, no_train_start, no_train_end in NO_TRAIN_PERIODS:

        # 학습기간과 금지기간이 겹치는 에피소드 선택
        overlap = (
            (df["train_start"] < no_train_end)
            & (no_train_start < df["train_end"])
        )

        overlap_rows = df.loc[overlap]
        seed_violations += len(overlap_rows)

        for _, row in overlap_rows.iterrows():
            print(
                f"[중복] Episode {int(row['episode'])}"
                f" / {period_name}"
            )
            print(
                f"       학습기간: "
                f"{row['train_start']} ~ {row['train_end']}"
            )
            print(
                f"       금지기간: "
                f"{no_train_start} ~ {no_train_end}"
            )

    print(f"전체 에피소드: {len(df)}개")

    if seed_violations == 0:
        print(
            "[PASS] 학습 금지기간과 겹치는 "
            "에피소드가 없습니다."
        )
        pass_count += 1
    else:
        print(
            f"[FAIL] 학습 금지기간과 겹치는 경우가 "
            f"{seed_violations}개 있습니다."
        )
        fail_count += 1


print()
print("=" * 70)
print("최종 결과")
print("=" * 70)
print(f"PASS  : {pass_count}개 seed")
print(f"FAIL  : {fail_count}개 seed")
print(f"ERROR : {error_count}개 seed")