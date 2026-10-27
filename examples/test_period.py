from datetime import datetime, timedelta
import random

DAY = 24 * 3600
YEAR = 365 * DAY

# ============================================================
# 현재 train_RL.py 설정 그대로
# ============================================================

day = 7

REFERENCE_DAYS = {
    "Peak Heat": 24,
    "Typical Heat": 95,
    "Peak Cool": 261,
    "Typical Cool": 175,
    "Mix": 327,
}

start_time_tests = {
    name: (reference_day - day) * DAY
    for name, reference_day in REFERENCE_DAYS.items()
}

# 학습 episode = 7일
max_episode_length = 7 * DAY

# Case D
regressive_period = 6 * 3600
predictive_period = 24 * 3600

bgn_year_margin = regressive_period
end_year_margin = max_episode_length + predictive_period


# ============================================================
# 최종적으로 사용할 TEST exclusion
# 기준일 -3일 ~ 기준일 +4일
# ============================================================

test_exclusions = []

for name, start_time_test in start_time_tests.items():

    # start_time_test = 기준일 - 7일
    reference_time = start_time_test + day * DAY

    exclude_start = reference_time - 3 * DAY
    exclude_end   = reference_time + 4 * DAY

    test_exclusions.append(
        (name, exclude_start, exclude_end)
    )


# ============================================================
# 날짜 출력
# ============================================================

def sec_to_date(sec):
    return datetime(2021, 1, 1) + timedelta(seconds=sec)


print("=" * 70)
print("FINAL TEST EXCLUSIONS")
print("=" * 70)

for name, start, end in test_exclusions:
    print(
        f"{name:15s}: "
        f"{sec_to_date(start).strftime('%Y-%m-%d %H:%M')} ~ "
        f"{sec_to_date(end).strftime('%Y-%m-%d %H:%M')}"
    )


# 환경에서 사용할 형식으로 변환
base_excluding_periods = [
    (start, end)
    for _, start, end in test_exclusions
]


# ============================================================
# All / Summer / Winter exclusion
# ============================================================

def get_excluding_periods(model_type):

    periods = list(base_excluding_periods)

    if model_type == "all":
        pass

    elif model_type == "summer":

        # Summer 허용:
        # day 141 ~ day 271
        periods.append(
            (0, 141 * DAY)
        )

        periods.append(
            (272 * DAY, 365 * DAY)
        )

    elif model_type == "winter":

        # Winter 허용:
        # day 0~103 + day 337~364
        periods.append(
            (104 * DAY, 337 * DAY)
        )

    else:
        raise ValueError(model_type)

    return periods


# ============================================================
# 실제 BoptestGymEnv와 동일한 overlap 검사
# ============================================================

def overlaps(episode, period):

    return (
        episode[0] < period[1]
        and period[0] < episode[1]
    )


def is_valid_episode(start_time, excluding_periods):

    episode = (
        start_time,
        start_time + max_episode_length
    )

    for period in excluding_periods:

        if overlaps(episode, period):
            return False

    return True


# ============================================================
# random start time 선택
# 실제 환경 방식과 동일
# ============================================================

def find_start_time(rng, excluding_periods):

    for _ in range(10000):

        start_time = rng.randint(
            int(bgn_year_margin),
            int(YEAR - end_year_margin)
        )

        if is_valid_episode(
            start_time,
            excluding_periods
        ):
            return start_time

    raise RuntimeError(
        "No valid start_time found."
    )


# ============================================================
# 10,000 episode 검증
# ============================================================

def check_model(model_type, n_samples=10000):

    excluding_periods = get_excluding_periods(
        model_type
    )

    rng = random.Random(123456)

    violations = 0
    starts = []

    for _ in range(n_samples):

        start_time = find_start_time(
            rng,
            excluding_periods
        )

        starts.append(start_time)

        episode = (
            start_time,
            start_time + max_episode_length
        )

        # 다시 한 번 독립적으로 검사
        for period in excluding_periods:

            if overlaps(episode, period):
                violations += 1

                print("\nERROR")

                print(
                    "Episode:",
                    sec_to_date(episode[0]),
                    "~",
                    sec_to_date(episode[1])
                )

                print(
                    "Excluded:",
                    sec_to_date(period[0]),
                    "~",
                    sec_to_date(period[1])
                )

    print("\n" + "=" * 70)
    print(f"{model_type.upper()} MODEL CHECK")
    print("=" * 70)

    print(f"검사 episode 수 : {n_samples}")
    print(f"위반 episode 수 : {violations}")

    if violations == 0:
        print("PASS")
    else:
        print("FAIL")

    print("\n샘플 시작일 10개")

    for start_time in starts[:10]:

        print(
            f"{sec_to_date(start_time)}"
            f" ~ "
            f"{sec_to_date(start_time + max_episode_length)}"
        )


# ============================================================
# 실행
# ============================================================

check_model("all", 10000)
check_model("summer", 10000)
check_model("winter", 10000)