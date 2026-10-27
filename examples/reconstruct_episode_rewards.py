import base64
from collections import deque
import io
import json
from pathlib import Path
import pickle
import random
import shutil
import zipfile

import numpy as np
import pandas as pd


# =====================================================================
# 사용자가 수정할 부분
# =====================================================================

# all/sum/win 및 seed 폴더들이 들어 있는 상위 폴더
ROOT_DIR = Path(r"/mnt/d/dev-code/boptest-gym/examples/agents")

# 이번 재학습에 사용한 seed
SEEDS = [123456, 199580, 173827, 101291, 362507]

# 자동으로 반복할 학습 범위
MODELS = ["all", "sum", "win"]

# 복구본을 한곳에 추가로 모을 폴더
COLLECTED_DIR = ROOT_DIR / "reconstructed_episode_rewards"


# =====================================================================
# 현재 실험 설정
# =====================================================================

FOLDER_NAME = "SAC_D_3e05_air_seoul_{model}_34_16-1_{seed}_300k_4_logdir"
ORIGINAL_REWARD_NAME = "episode_rewards.csv"

# training monitor 백업이 있다면 monitor.csv보다 앞에 적어주세요.
MONITOR_CANDIDATES = ["monitor_train.csv", "monitor.csv"]

# 학습 종료 직후 저장된 최종 모델 ZIP 후보
FINAL_MODEL_CANDIDATES = ["last_model.zip"]

# 완전 복구되면 원본을 이 이름으로 한 번 백업한 뒤 episode_rewards.csv를 교체합니다.
BACKUP_REWARD_NAME = "episode_rewards_before_reconstruction.csv"

# 완전 복구에 실패하면 원본을 덮어쓰지 않고 이 이름으로만 저장합니다.
PARTIAL_OUTPUT_NAME = "episode_rewards_reconstructed_PARTIAL.csv"
SUMMARY_NAME = "reconstruction_summary.csv"

YEAR_START = pd.Timestamp("2021-01-01")
SECONDS_PER_DAY = 24 * 60 * 60

TOTAL_TRAINING_TIMESTEPS = 300_000
STEP_SECONDS = 15 * 60
EPISODE_DAYS = 7
STEPS_PER_EPISODE = EPISODE_DAYS * SECONDS_PER_DAY // STEP_SECONDS  # 672
EXPECTED_COMPLETED_EPISODES = TOTAL_TRAINING_TIMESTEPS // STEPS_PER_EPISODE  # 446

# BoptestGymEnv 설정
REGRESSIVE_PERIOD = 6 * 60 * 60
PREDICTIVE_PERIOD = 24 * 60 * 60
YEAR_SECONDS = 365 * SECONDS_PER_DAY
END_YEAR_MARGIN = EPISODE_DAYS * SECONDS_PER_DAY + PREDICTIVE_PERIOD

# monitor.csv reward는 저장 정밀도가 다를 수 있으므로 작은 오차를 허용합니다.
REWARD_ATOL = 1e-5

# 시작시각 재생 결과가 원본과 맞지 않으면 잘못된 복구를 막기 위해 저장하지 않습니다.
REQUIRE_EXACT_START_MATCH = True


def get_excluding_periods(model):
    """학습 당시 사용한 테스트 보호구간과 계절 제외구간을 반환합니다."""

    reference_days = [24, 95, 261, 175, 327]
    periods = [
        (
            (reference_day - 3) * SECONDS_PER_DAY,
            (reference_day + 4) * SECONDS_PER_DAY,
        )
        for reference_day in reference_days
    ]

    if model == "sum":
        # Summer 유지: day 141~271, 즉 [141, 272)
        periods.extend(
            [
                (0, 141 * SECONDS_PER_DAY),
                (272 * SECONDS_PER_DAY, 365 * SECONDS_PER_DAY),
            ]
        )
    elif model == "win":
        # Winter 유지: day 0~103 및 337~364
        periods.append((104 * SECONDS_PER_DAY, 337 * SECONDS_PER_DAY))
    elif model != "all":
        raise ValueError(f"알 수 없는 model입니다: {model}")

    return periods


def generate_episode_start_times(model, seed, episode_count):
    """학습 코드의 random.randint와 rejection sampling을 그대로 재생합니다."""

    rng = random.Random(seed)
    excluding_periods = get_excluding_periods(model)
    minimum_start = REGRESSIVE_PERIOD
    maximum_start = YEAR_SECONDS - END_YEAR_MARGIN
    episode_length = EPISODE_DAYS * SECONDS_PER_DAY
    start_times = []

    while len(start_times) < episode_count:
        for _ in range(10_000):
            start_time = rng.randint(minimum_start, maximum_start)
            episode_end = start_time + episode_length

            overlaps = any(
                start_time < period_end and period_start < episode_end
                for period_start, period_end in excluding_periods
            )

            if not overlaps:
                start_times.append(start_time)
                break
        else:
            raise RuntimeError("유효한 episode 시작시각을 찾지 못했습니다.")

    return np.asarray(start_times, dtype=np.int64)


def read_original_episode_rewards(csv_path):
    """기존 episode_rewards.csv를 읽고 필수 열을 검증합니다."""

    df = pd.read_csv(csv_path)
    required_columns = {"reward", "start_time_sec"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing_columns)}")

    rewards = pd.to_numeric(df["reward"], errors="coerce")
    start_times = pd.to_numeric(df["start_time_sec"], errors="coerce")

    if rewards.isna().any():
        raise ValueError("reward 열에 숫자가 아닌 값이 있습니다.")
    if start_times.isna().any():
        raise ValueError("start_time_sec 열에 숫자가 아닌 값이 있습니다.")

    start_values = start_times.to_numpy(dtype=float)
    if not np.all(start_values == np.floor(start_values)):
        raise ValueError("start_time_sec에는 정수 초가 들어 있어야 합니다.")

    return (
        df,
        rewards.to_numpy(dtype=float),
        start_values.astype(np.int64),
    )


def find_start_alignment(logged_starts, generated_starts):
    """기존 시작시각이 생성된 몇 번째 에피소드부터인지 찾습니다."""

    possible_offsets = []
    maximum_offset = len(generated_starts) - len(logged_starts)

    for offset in range(maximum_offset + 1):
        candidate = generated_starts[offset : offset + len(logged_starts)]
        if np.array_equal(logged_starts, candidate):
            possible_offsets.append(offset)

    if len(possible_offsets) == 1:
        return possible_offsets[0]
    if not possible_offsets:
        return None

    raise ValueError(f"시작시각 정렬 위치가 여러 개입니다: {possible_offsets}")


def read_monitor_file(monitor_path):
    """Stable-Baselines3 monitor.csv를 읽습니다."""

    monitor_df = pd.read_csv(monitor_path, comment="#")
    required_columns = {"r", "l"}
    if not required_columns.issubset(monitor_df.columns):
        raise ValueError("monitor.csv에 r 또는 l 열이 없습니다.")

    rewards = pd.to_numeric(monitor_df["r"], errors="coerce")
    lengths = pd.to_numeric(monitor_df["l"], errors="coerce")
    valid = rewards.notna() & lengths.notna()

    monitor_df = monitor_df.loc[valid].copy()
    monitor_df["r"] = rewards.loc[valid].astype(float)
    monitor_df["l"] = lengths.loc[valid].astype(int)

    # 테스트로 덮였거나 추가된 14일 episode를 제외하고 7일 학습 episode만 선택
    return monitor_df.loc[monitor_df["l"] == STEPS_PER_EPISODE].reset_index(
        drop=True
    )


def find_monitor_rewards(folder, original_rewards):
    """
    원래 reward 배열과 앞부분이 일치하는 training monitor를 찾습니다.

    monitor가 테스트 결과로 덮였으면 reward 순서가 일치하지 않으므로 자동 거부합니다.
    """

    messages = []

    for monitor_name in MONITOR_CANDIDATES:
        monitor_path = folder / monitor_name
        if not monitor_path.is_file():
            messages.append(f"{monitor_name}: 파일 없음")
            continue

        try:
            monitor_df = read_monitor_file(monitor_path)
        except (ValueError, OSError) as error:
            messages.append(f"{monitor_name}: 읽기 실패 ({error})")
            continue

        monitor_rewards = monitor_df["r"].to_numpy(dtype=float)
        original_count = len(original_rewards)

        if len(monitor_rewards) < original_count:
            messages.append(
                f"{monitor_name}: 7일 episode가 {len(monitor_rewards)}개뿐임"
            )
            continue

        matching_offsets = []
        for offset in range(len(monitor_rewards) - original_count + 1):
            segment = monitor_rewards[offset : offset + original_count]
            if np.allclose(
                segment,
                original_rewards,
                rtol=0.0,
                atol=REWARD_ATOL,
            ):
                matching_offsets.append(offset)

        if len(matching_offsets) != 1:
            messages.append(
                f"{monitor_name}: 기존 reward 순서와 고유하게 일치하지 않음"
            )
            continue

        offset = matching_offsets[0]
        available_rewards = monitor_rewards[offset:]
        messages.append(
            f"{monitor_name}: training reward 일치, "
            f"offset={offset}, available={len(available_rewards)}"
        )
        return available_rewards, monitor_name, "; ".join(messages)

    return None, None, "; ".join(messages)


class RestrictedModelUnpickler(pickle.Unpickler):
    """SB3 data 안에서 collections.deque만 허용하는 제한된 unpickler입니다."""

    def find_class(self, module, name):
        if module == "collections" and name == "deque":
            return deque
        raise pickle.UnpicklingError(
            f"허용되지 않은 pickle global입니다: {module}.{name}"
        )


def read_model_episode_buffer(model_path, expected_seed):
    """last_model.zip의 data에서 최근 episode의 r/l/t만 안전하게 읽습니다."""

    with zipfile.ZipFile(model_path) as archive:
        if "data" not in archive.namelist():
            raise ValueError("모델 ZIP에 data 파일이 없습니다.")

        data = json.loads(archive.read("data").decode("utf-8"))

    model_seed = data.get("seed")
    if model_seed is not None and int(model_seed) != int(expected_seed):
        raise ValueError(
            f"모델 seed가 폴더 seed와 다릅니다: {model_seed} != {expected_seed}"
        )

    num_timesteps = int(data.get("num_timesteps", -1))
    episode_num = int(data.get("_episode_num", -1))
    if num_timesteps != TOTAL_TRAINING_TIMESTEPS:
        raise ValueError(
            f"모델 timestep이 예상값과 다릅니다: "
            f"{num_timesteps} != {TOTAL_TRAINING_TIMESTEPS}"
        )
    if episode_num != EXPECTED_COMPLETED_EPISODES:
        raise ValueError(
            f"모델 완료 episode 수가 예상값과 다릅니다: "
            f"{episode_num} != {EXPECTED_COMPLETED_EPISODES}"
        )

    buffer_entry = data.get("ep_info_buffer")
    if not isinstance(buffer_entry, dict) or ":serialized:" not in buffer_entry:
        raise ValueError("모델 ZIP에 직렬화된 ep_info_buffer가 없습니다.")

    encoded = buffer_entry[":serialized:"]
    raw_buffer = base64.b64decode(encoded)
    episode_buffer = RestrictedModelUnpickler(io.BytesIO(raw_buffer)).load()

    if not isinstance(episode_buffer, deque) or len(episode_buffer) == 0:
        raise ValueError("ep_info_buffer가 비어 있거나 deque가 아닙니다.")

    rewards = []
    lengths = []
    for item in episode_buffer:
        if not isinstance(item, dict) or not {"r", "l"}.issubset(item):
            raise ValueError("ep_info_buffer 항목에 r 또는 l이 없습니다.")
        rewards.append(float(item["r"]))
        lengths.append(int(item["l"]))

    if any(length != STEPS_PER_EPISODE for length in lengths):
        raise ValueError("ep_info_buffer에 7일 학습 episode가 아닌 항목이 있습니다.")

    rewards = np.asarray(rewards, dtype=float)
    first_episode = episode_num - len(rewards) + 1
    episode_numbers = np.arange(first_episode, episode_num + 1, dtype=int)

    return episode_numbers, rewards


def find_model_buffer_rewards(folder, seed, original_rewards):
    """기존 reward와 겹치는 구간을 검증한 뒤 모델 ZIP의 최근 reward를 읽습니다."""

    messages = []

    for model_name in FINAL_MODEL_CANDIDATES:
        model_path = folder / model_name
        if not model_path.is_file():
            messages.append(f"{model_name}: 파일 없음")
            continue

        try:
            episode_numbers, buffer_rewards = read_model_episode_buffer(
                model_path,
                seed,
            )
        except (ValueError, OSError, zipfile.BadZipFile, pickle.UnpicklingError) as error:
            messages.append(f"{model_name}: 읽기 실패 ({error})")
            continue

        original_count = len(original_rewards)
        overlap = episode_numbers <= original_count
        if not overlap.any():
            messages.append(f"{model_name}: 기존 reward와 검증할 겹침 구간 없음")
            continue

        overlap_episode_numbers = episode_numbers[overlap]
        original_overlap = original_rewards[overlap_episode_numbers - 1]
        buffer_overlap = buffer_rewards[overlap]

        if not np.allclose(
            original_overlap,
            buffer_overlap,
            rtol=0.0,
            atol=REWARD_ATOL,
        ):
            messages.append(
                f"{model_name}: 기존 reward와 최근 buffer reward가 일치하지 않음"
            )
            continue

        messages.append(
            f"{model_name}: 최근 {len(buffer_rewards)}개 reward 검증 완료, "
            f"episode {episode_numbers[0]}~{episode_numbers[-1]}"
        )
        return episode_numbers, buffer_rewards, model_name, "; ".join(messages)

    return None, None, None, "; ".join(messages)


def resolve_all_rewards(folder, seed, original_rewards):
    """원본 → monitor → last_model.zip 순서로 446개 reward를 복구합니다."""

    rewards = np.full(EXPECTED_COMPLETED_EPISODES, np.nan, dtype=float)
    sources = np.full(EXPECTED_COMPLETED_EPISODES, "missing", dtype=object)

    original_count = min(len(original_rewards), EXPECTED_COMPLETED_EPISODES)
    rewards[:original_count] = original_rewards[:original_count]
    sources[:original_count] = ORIGINAL_REWARD_NAME

    monitor_rewards, monitor_name, monitor_message = find_monitor_rewards(
        folder,
        original_rewards,
    )
    if monitor_rewards is not None:
        usable_count = min(len(monitor_rewards), EXPECTED_COMPLETED_EPISODES)
        rewards[:usable_count] = monitor_rewards[:usable_count]
        sources[:usable_count] = monitor_name

    (
        model_episode_numbers,
        model_rewards,
        model_name,
        model_message,
    ) = find_model_buffer_rewards(folder, seed, original_rewards)

    if model_rewards is not None:
        indices = model_episode_numbers - 1

        # monitor와 model ZIP이 모두 있으면 겹치는 값도 서로 검증합니다.
        already_available = ~np.isnan(rewards[indices])
        if already_available.any() and not np.allclose(
            rewards[indices][already_available],
            model_rewards[already_available],
            rtol=0.0,
            atol=REWARD_ATOL,
        ):
            raise ValueError("monitor/원본 reward와 last_model.zip reward가 충돌합니다.")

        missing = np.isnan(rewards[indices])
        rewards[indices[missing]] = model_rewards[missing]
        sources[indices[missing]] = model_name

    return (
        rewards,
        sources,
        monitor_name,
        monitor_message,
        model_name,
        model_message,
    )


def build_reconstructed_dataframe(
    model,
    seed,
    generated_starts,
    rewards,
    reward_sources,
):
    """복원한 446개 시작시각과 reward를 하나의 표로 만듭니다."""

    episode_count = len(generated_starts)

    start_datetime = YEAR_START + pd.to_timedelta(generated_starts, unit="s")

    return pd.DataFrame(
        {
            "episode": np.arange(1, episode_count + 1, dtype=int),
            "reward": rewards,
            "reward_available": ~np.isnan(rewards),
            "reward_source": reward_sources,
            "start_time_sec": generated_starts,
            "start_time_source": "regenerated_from_seed",
            "start_day": generated_starts // SECONDS_PER_DAY,
            "start_datetime": start_datetime,
            "start_month": start_datetime.month,
            "start_date": start_datetime.strftime("%m-%d"),
            "start_hour": start_datetime.hour,
            "model": model,
            "seed": seed,
        }
    )


def process_one_folder(model, seed):
    """실험 폴더 하나를 검사하고 완전 복구된 episode_rewards.csv를 저장합니다."""

    folder = ROOT_DIR / FOLDER_NAME.format(model=model, seed=seed)
    original_path = folder / ORIGINAL_REWARD_NAME

    print("=" * 76)
    print(f"Model: {model} / Seed: {seed}")
    print(f"Folder: {folder}")

    if not original_path.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {original_path}")

    _, original_rewards, logged_starts = read_original_episode_rewards(
        original_path
    )

    if len(original_rewards) > EXPECTED_COMPLETED_EPISODES:
        raise ValueError(
            f"기존 reward가 예상 완료 episode 수보다 많습니다: "
            f"{len(original_rewards)} > {EXPECTED_COMPLETED_EPISODES}"
        )

    generated_starts = generate_episode_start_times(
        model,
        seed,
        EXPECTED_COMPLETED_EPISODES,
    )
    start_offset = find_start_alignment(logged_starts, generated_starts)

    if start_offset is None and REQUIRE_EXACT_START_MATCH:
        raise ValueError(
            "기존 시작시각이 seed 재생 결과와 일치하지 않습니다. "
            "학습 설정 또는 seed를 확인하세요."
        )

    (
        resolved_rewards,
        reward_sources,
        monitor_name,
        monitor_message,
        final_model_name,
        final_model_message,
    ) = resolve_all_rewards(
        folder,
        seed,
        original_rewards,
    )

    reconstructed_df = build_reconstructed_dataframe(
        model,
        seed,
        generated_starts,
        resolved_rewards,
        reward_sources,
    )

    missing_reward_count = int((~reconstructed_df["reward_available"]).sum())
    complete = missing_reward_count == 0

    backup_output = folder / BACKUP_REWARD_NAME

    if complete:
        # 원본은 최초 실행 때 한 번만 백업합니다.
        if not backup_output.is_file():
            shutil.copy2(original_path, backup_output)

        local_output = original_path
        collected_output = (
            COLLECTED_DIR / f"{model}_seed_{seed}_episode_rewards.csv"
        )
        reconstructed_df.to_csv(local_output, index=False)
        reconstructed_df.to_csv(collected_output, index=False)
    else:
        # 완전 복구가 아니면 원본 episode_rewards.csv를 절대 덮어쓰지 않습니다.
        local_output = folder / PARTIAL_OUTPUT_NAME
        collected_output = (
            COLLECTED_DIR
            / f"{model}_seed_{seed}_{PARTIAL_OUTPUT_NAME}"
        )
        reconstructed_df.to_csv(local_output, index=False)
        reconstructed_df.to_csv(collected_output, index=False)

    print(f"기존 행 수             : {len(original_rewards)}")
    print(f"복구본 행 수           : {len(reconstructed_df)}")
    print(f"시작시각 정렬 offset   : {start_offset}")
    print(f"monitor 검사           : {monitor_message}")
    print(f"last_model 검사        : {final_model_message}")
    print(f"누락 reward            : {missing_reward_count}")
    print(f"복구 상태              : {'COMPLETE' if complete else 'PARTIAL'}")
    if complete:
        print(f"원본 백업              : {backup_output}")
    else:
        print("[주의] 완전 복구 실패로 원본 episode_rewards.csv는 유지했습니다.")
    print(f"폴더 내 저장           : {local_output}")
    print(f"통합 폴더 저장         : {collected_output}")

    return {
        "model": model,
        "seed": seed,
        "original_rows": len(original_rewards),
        "reconstructed_rows": len(reconstructed_df),
        "start_alignment_offset": start_offset,
        "monitor_used": monitor_name or "",
        "final_model_used": final_model_name or "",
        "reward_rows_available": int(
            reconstructed_df["reward_available"].sum()
        ),
        "missing_reward_rows": missing_reward_count,
        "status": "COMPLETE" if complete else "PARTIAL",
        "local_output": str(local_output),
        "collected_output": str(collected_output),
        "monitor_check": monitor_message,
        "final_model_check": final_model_message,
    }


def main():
    COLLECTED_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    # all/sum/win과 모든 seed를 자동으로 반복합니다.
    for model in MODELS:
        for seed in SEEDS:
            try:
                summary_rows.append(process_one_folder(model, seed))
            except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
                print(f"[ERROR] {error}")
                summary_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "status": "ERROR",
                        "error": str(error),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = COLLECTED_DIR / SUMMARY_NAME
    summary_df.to_csv(summary_path, index=False)

    print()
    print("=" * 76)
    print("최종 요약")
    print("=" * 76)
    if "status" in summary_df.columns:
        print(summary_df["status"].value_counts(dropna=False).to_string())
    print(f"요약 CSV: {summary_path}")
    print("COMPLETE는 monitor 또는 last_model.zip으로 모든 reward가 복구된 경우입니다.")
    print("PARTIAL은 원본 episode_rewards.csv를 덮어쓰지 않고 별도 파일로 저장됩니다.")


if __name__ == "__main__":
    main()