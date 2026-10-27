"""
Evaluate a trained SAC controller on the hottest Seoul day (2021-08-08)
and on the 7-day window centered around it (2021-08-05 ~ 2021-08-11).

This script DOES NOT train a model.
It only:
  1) rebuilds the same bestest_air Case-D environment used for training,
  2) loads one existing SAC checkpoint,
  3) runs deterministic evaluation on fixed dates,
  4) saves 15-minute step logs, KPI JSON files, and separate temperature/action plots.

Run this file from the boptest-gym project environment while BOPTEST is running.
"""

from collections import OrderedDict
from pathlib import Path
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from stable_baselines3 import SAC

from boptestGymEnv import (
    BoptestGymEnv,
    NormalizedActionWrapper,
    NormalizedObservationWrapper,
)
from testing import utilities


# ============================================================================
# USER SETTINGS: normally you only need to edit this block
# ============================================================================
URL = "http://127.0.0.1"

# Model to evaluate
MODEL_VARIANT = "sum"       # "all" or "summer" (Aug 8 is a cooling test)
SEED_NO = 199580
MODEL_NAME = "model_105000"  # e.g. model_100000, model_105000, model_300000

# This value is used only to reconstruct the training-folder name.
# It must match the training script's folder naming convention.
TRAINING_TIMESTEPS = 3e5
CASE = "D"
ALGORITHM = "SAC"

# If your agent folder name differs from the automatic pattern below,
# put the full folder path here. Otherwise leave it as None.
MODEL_DIR_OVERRIDE = None

# Fixed evaluation dates
HOTTEST_DAY = pd.Timestamp("2021-08-08 00:00:00")
WEEK_START = pd.Timestamp("2021-08-05 00:00:00")

# Keep the same evaluation protocol used in the original script.
WARMUP_DAYS = 7
STEP_PERIOD = 900  # 15 minutes

# Reward definition used when the current models were trained/tested.
COST_WEIGHT = 16.0

# The BOPTEST year is represented by seconds from Jan 1.
BASE_DATE = pd.Timestamp("2021-01-01 00:00:00")

# Create separate temperature and action figures for each test window.
MAKE_PLOTS = True


# ============================================================================
# Helpers
# ============================================================================
def timestamp_to_seconds(ts: pd.Timestamp) -> int:
    """Convert a 2021 calendar timestamp to BOPTEST seconds from Jan 1."""
    return int((ts - BASE_DATE).total_seconds())


def resolve_model_dir() -> Path:
    """Return the directory containing the trained model checkpoint."""
    if MODEL_DIR_OVERRIDE is not None:
        path = Path(MODEL_DIR_OVERRIDE).expanduser().resolve()
    else:
        folder = (
            f"{ALGORITHM}_{CASE}_{TRAINING_TIMESTEPS:.0e}_"
            f"air_seoul_{MODEL_VARIANT}_34_16-1_{SEED_NO}_300k_4_logdir"
        ).replace("+", "")
        path = Path(utilities.get_root_path()) / "examples" / "agents" / folder

    if not path.exists():
        raise FileNotFoundError(
            "Model directory was not found:\n"
            f"  {path}\n\n"
            "Change MODEL_VARIANT / SEED_NO / TRAINING_TIMESTEPS, or set "
            "MODEL_DIR_OVERRIDE to the exact directory."
        )
    return path


def resolve_model_file(model_dir: Path) -> Path:
    """Find MODEL_NAME with or without the .zip suffix."""
    p = model_dir / MODEL_NAME
    if p.exists():
        return p
    p_zip = model_dir / f"{MODEL_NAME}.zip"
    if p_zip.exists():
        return p_zip
    raise FileNotFoundError(
        f"Checkpoint '{MODEL_NAME}' was not found in:\n  {model_dir}"
    )


def build_output_dir() -> Path:
    """Use a separate directory so training logs/monitor.csv are never overwritten."""
    out = (
        Path(utilities.get_root_path())
        / "examples"
        / "aug8_rl_analysis"
        / f"{MODEL_VARIANT}_{SEED_NO}_{MODEL_NAME}"
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


# ============================================================================
# Environment: same action/state/scenario definition as train_RL(7).py Case D
# ============================================================================
class BoptestGymEnvCustomReward(BoptestGymEnv):
    def get_reward(self):
        kpis = requests.get(
            f"{self.url}/kpi/{self.testid}"
        ).json()["payload"]

        objective_integrand = (
            kpis["cost_tot"] * COST_WEIGHT + kpis["tdis_tot"]
        )
        reward = -(objective_integrand - self.objective_integrand)
        self.objective_integrand = objective_integrand
        return reward


def make_env(output_dir: Path):
    """Create the exact bestest_air Case-D environment used by the model."""
    actions = ["fcu_oveFan_u", "fcu_oveTSup_u"]

    observations = OrderedDict([
        ("time", (0, 604800)),
        ("zon_reaTRooAir_y", (280.0, 310.0)),
        ("zon_reaCO2RooAir_y", (400.0, 2000.0)),
        ("TDryBul", (250.0, 315.0)),
        ("HDirNor", (0.0, 1000.0)),
        ("PriceElectricPowerHighlyDynamic", (-0.4, 0.4)),
        ("LowerSetp[1]", (280.0, 310.0)),
        ("UpperSetp[1]", (280.0, 310.0)),
    ])

    env = BoptestGymEnvCustomReward(
        url=URL,
        testcase="bestest_air",
        actions=actions,
        observations=observations,
        predictive_period=24 * 3600,
        regressive_period=6 * 3600,
        scenario={"electricity_price": "highly_dynamic"},
        random_start_time=True,   # overwritten before each fixed test
        excluding_periods=[],
        max_episode_length=7 * 24 * 3600,
        warmup_period=WARMUP_DAYS * 24 * 3600,
        step_period=STEP_PERIOD,
        render_episodes=False,
        log_dir=str(output_dir),
    )

    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)
    return env


# ============================================================================
# Fixed-date deterministic evaluation
# ============================================================================
def choose_result_points(base_env):
    """Choose available BOPTEST points while tolerating naming differences."""
    available = set(base_env.all_measurement_vars.keys()) | set(base_env.all_input_vars.keys())

    candidates = OrderedDict([
        ("Room_K", ["zon_reaTRooAir_y"]),
        ("Outdoor_K", ["weaSta_reaWeaTDryBul_y", "zon_weaSta_reaWeaTDryBul_y"]),
        ("HeatSet_K", ["reaTSetHea_y", "con_oveTSetHea_y"]),
        ("CoolSet_K", ["reaTSetCoo_y", "con_oveTSetCoo_y"]),
        ("CO2_ppm", ["zon_reaCO2RooAir_y"]),
        ("PFan_W", ["fcu_reaPFan_y"]),
        ("PCool_W", ["fcu_reaPCoo_y"]),
        ("PHeat_W", ["fcu_reaPHea_y"]),
        ("FlowSup", ["fcu_reaFloSup_y"]),
    ])

    selected = OrderedDict()
    for logical_name, names in candidates.items():
        for name in names:
            if name in available:
                selected[logical_name] = name
                break

    return selected


def decode_action(env, base_env, action):
    """Decode NormalizedActionWrapper [-1, 1] actions to physical BOPTEST values."""
    action_flat = np.asarray(action, dtype=float).flatten()

    env_low = np.asarray(env.action_space.low, dtype=float).flatten()
    env_high = np.asarray(env.action_space.high, dtype=float).flatten()
    base_low = np.asarray(base_env.lower_act_bounds, dtype=float).flatten()
    base_high = np.asarray(base_env.upper_act_bounds, dtype=float).flatten()

    is_normalized = (
        len(action_flat) == len(base_low)
        and np.allclose(env_low, -1.0)
        and np.allclose(env_high, 1.0)
    )

    if is_normalized:
        actual = base_low + (action_flat + 1.0) * 0.5 * (base_high - base_low)
    else:
        actual = action_flat

    decoded = {}
    for i, name in enumerate(base_env.actions):
        if i >= len(actual):
            continue
        decoded[f"{name}_raw_normalized"] = float(action_flat[i])
        decoded[f"{name}_actual"] = float(actual[i])

        if "TSup" in name or "TSet" in name:
            decoded[f"{name}_actual_C"] = float(actual[i] - 273.15)

    return decoded


def query_result_at_time(base_env, selected_points, sim_time):
    """Get selected BOPTEST values at one simulation time."""
    if not selected_points:
        return {}

    point_names = list(selected_points.values())
    response = requests.put(
        f"{base_env.url}/results/{base_env.testid}",
        json={
            "point_names": point_names,
            "start_time": sim_time,
            "final_time": sim_time,
        },
    ).json()

    payload = response.get("payload", {})
    result = {}

    for logical_name, point_name in selected_points.items():
        values = payload.get(point_name, [])
        if len(values) > 0:
            result[logical_name] = values[-1]

    # Convenience Celsius columns
    if "Room_K" in result:
        result["Room_C"] = result["Room_K"] - 273.15
    if "Outdoor_K" in result:
        result["Outdoor_C"] = result["Outdoor_K"] - 273.15
    if "HeatSet_K" in result:
        result["HeatSet_C"] = result["HeatSet_K"] - 273.15
    if "CoolSet_K" in result:
        result["CoolSet_C"] = result["CoolSet_K"] - 273.15

    return result



def add_comfort_setpoint_fallback(result_dict, timestamp):
    """
    Ensure HeatSet_C / CoolSet_C are always available for plotting.

    In this bestest_air Seoul case, the thermostat comfort schedule is:
      occupied   08:00 <= time < 18:00 : heating 21°C, cooling 24°C
      unoccupied otherwise             : heating 15°C, cooling 30°C

    BOPTEST's /results endpoint does not always expose the internal
    con_oveTSetHea_y / con_oveTSetCoo_y signals as measurement points.
    If they were successfully queried, keep the queried values.
    Otherwise fill them from the known test-case schedule.
    """
    ts = pd.Timestamp(timestamp)
    occupied = (ts.hour >= 8) and (ts.hour < 18)

    if "HeatSet_C" not in result_dict or pd.isna(result_dict.get("HeatSet_C")):
        result_dict["HeatSet_C"] = 21.0 if occupied else 15.0
        result_dict["HeatSet_source"] = "schedule_fallback"
    else:
        result_dict["HeatSet_source"] = "boptest_result"

    if "CoolSet_C" not in result_dict or pd.isna(result_dict.get("CoolSet_C")):
        result_dict["CoolSet_C"] = 24.0 if occupied else 30.0
        result_dict["CoolSet_source"] = "schedule_fallback"
    else:
        result_dict["CoolSet_source"] = "boptest_result"

    return result_dict


def run_fixed_test(env, model, name, eval_start, eval_days, output_dir):
    """
    Run one deterministic fixed-date evaluation.

    BOPTEST warmup is applied BEFORE eval_start, so:
      - hottest_day: warmup Aug 1~7, evaluation Aug 8
      - hottest_week: warmup Jul 29~Aug 4, evaluation Aug 5~11
    """
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    start_time = timestamp_to_seconds(eval_start)
    episode_length = int(eval_days * 24 * 3600)
    warmup_period = int(WARMUP_DAYS * 24 * 3600)

    base_env.random_start_time = False
    base_env.start_time = start_time
    base_env.max_episode_length = episode_length
    base_env.warmup_period = warmup_period

    # Reset first so the test instance and its metadata are fully initialized.
    obs, _ = env.reset()
    selected_points = choose_result_points(base_env)

    print("\n" + "=" * 78)
    print(f"Running: {name}")
    print(f"Evaluation start : {eval_start}")
    print(f"Evaluation length: {eval_days} day(s)")
    print(f"Warmup           : {WARMUP_DAYS} day(s) before evaluation start")
    print("Logged BOPTEST points:")
    for logical, physical in selected_points.items():
        print(f"  {logical:12s} <- {physical}")

    rows = []
    rewards = []
    step_idx = 0
    done = False

    while not done:
        # Deterministic policy evaluation
        action, _ = model.predict(obs, deterministic=True)
        decoded_action = decode_action(env, base_env, action)

        # Action is applied over [decision_time, result_time].
        decision_time = start_time + step_idx * base_env.step_period

        obs, reward, terminated, truncated, _ = env.step(action)
        result_time = decision_time + base_env.step_period

        kpis_now = base_env.get_kpis()
        boptest_values = query_result_at_time(base_env, selected_points, result_time)

        result_datetime = BASE_DATE + pd.to_timedelta(result_time, unit="s")

        # con_oveTSetHea_y / con_oveTSetCoo_y are internal signals and may not
        # be exposed through BOPTEST /results. In that case, restore the known
        # bestest_air comfort schedule so the temperature plot always contains
        # the same heating/cooling setpoints as the PI baseline.
        boptest_values = add_comfort_setpoint_fallback(
            boptest_values,
            result_datetime,
        )

        row = {
            "step": step_idx,
            "decision_time_s": decision_time,
            "time_s": result_time,
            "datetime": result_datetime,
            "reward": float(reward),
            "cost_tot": kpis_now.get("cost_tot"),
            "tdis_tot": kpis_now.get("tdis_tot"),
            "idis_tot": kpis_now.get("idis_tot"),
            "ener_tot": kpis_now.get("ener_tot"),
        }
        row.update(decoded_action)
        row.update(boptest_values)

        rows.append(row)
        rewards.append(float(reward))

        step_idx += 1
        done = bool(terminated or truncated)

    final_kpis = base_env.get_kpis()
    df = pd.DataFrame(rows)

    # Diagnostic: these should now always exist.
    print("Setpoint columns in step log:")
    print("  HeatSet_C:", "HeatSet_C" in df.columns)
    print("  CoolSet_C:", "CoolSet_C" in df.columns)
    if "HeatSet_source" in df.columns:
        print("  HeatSet sources:", df["HeatSet_source"].value_counts(dropna=False).to_dict())
    if "CoolSet_source" in df.columns:
        print("  CoolSet sources:", df["CoolSet_source"].value_counts(dropna=False).to_dict())

    csv_path = output_dir / f"{name}_step_log.csv"
    kpi_path = output_dir / f"{name}_kpis.json"

    df.to_csv(csv_path, index=False)
    with open(kpi_path, "w", encoding="utf-8") as f:
        json.dump(final_kpis, f, indent=2)

    print(f"Total reward: {sum(rewards):.6f}")
    print("Final KPIs:")
    for key, value in final_kpis.items():
        print(f"  {key}: {value}")
    print(f"Saved step log: {csv_path}")
    print(f"Saved KPI JSON: {kpi_path}")

    base_env.random_start_time = True
    return df, final_kpis


# ============================================================================
# Plotting
#   1) Temperature figure:
#      Outdoor / Room / Heating setpoint / Cooling setpoint
#   2) Action figure:
#      RL fan action / RL supply-air-temperature action
#
# The two RL actions have different units, so the action plot uses two y-axes:
#   left  = supply-air temperature action (°C)
#   right = fan action (0-1)
# ============================================================================
def _mark_outdoor_peak(ax, df):
    """Mark the outdoor-temperature peak if Outdoor_C is available."""
    if "Outdoor_C" in df.columns and df["Outdoor_C"].notna().any():
        peak_i = df["Outdoor_C"].idxmax()
        peak_t = pd.to_datetime(df.loc[peak_i, "datetime"])
        peak_v = float(df.loc[peak_i, "Outdoor_C"])
        peak_line = ax.axvline(
            peak_t,
            linestyle=":",
            linewidth=1.5,
            label=f"Outdoor peak ({peak_v:.2f}°C)",
        )
        return peak_line
    return None


def plot_temperatures(df, title, output_path):
    """
    Plot only:
      - outdoor temperature
      - room temperature
      - heating setpoint
      - cooling setpoint
    """
    if df.empty:
        print(f"Skip temperature plot (empty dataframe): {title}")
        return

    x = pd.to_datetime(df["datetime"])

    fig, ax = plt.subplots(figsize=(15, 6))
    lines = []

    series = [
        ("Outdoor_C", "Outdoor temperature"),
        ("Room_C", "Room temperature"),
        ("CoolSet_C", "Cooling setpoint"),
        ("HeatSet_C", "Heating setpoint"),
    ]

    for col, label in series:
        if col in df.columns and df[col].notna().any():
            line, = ax.plot(x, df[col], label=label)
            lines.append(line)

    peak_line = _mark_outdoor_peak(ax, df)
    if peak_line is not None:
        lines.append(peak_line)

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)

    if lines:
        ax.legend(lines, [line.get_label() for line in lines], loc="best")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved temperature plot: {output_path}")


def plot_actions(df, title, output_path):
    """
    Plot the two physical RL actions in one figure:
      - fcu_oveTSup_u_actual_C : supply-air-temperature action (°C), left axis
      - fcu_oveFan_u_actual    : fan action (0-1), right axis
    """
    if df.empty:
        print(f"Skip action plot (empty dataframe): {title}")
        return

    x = pd.to_datetime(df["datetime"])

    tsup_col = "fcu_oveTSup_u_actual_C"
    fan_col = "fcu_oveFan_u_actual"

    has_tsup = tsup_col in df.columns and df[tsup_col].notna().any()
    has_fan = fan_col in df.columns and df[fan_col].notna().any()

    if not has_tsup and not has_fan:
        print(
            "Skip action plot: decoded physical action columns were not found.\n"
            f"Expected: {tsup_col}, {fan_col}"
        )
        return

    fig, ax_tsup = plt.subplots(figsize=(15, 6))
    lines = []

    if has_tsup:
        tsup_line, = ax_tsup.plot(
            x,
            df[tsup_col],
            linewidth=1.8,
            label="RL supply-air-temperature action",
        )
        lines.append(tsup_line)

    ax_tsup.set_xlabel("Time")
    ax_tsup.set_ylabel("Supply-air-temperature action (°C)")
    ax_tsup.grid(True, alpha=0.25)

    ax_fan = ax_tsup.twinx()

    if has_fan:
        fan_line, = ax_fan.plot(
            x,
            df[fan_col],
            linestyle="--",
            linewidth=1.8,
            label="RL fan action",
        )
        lines.append(fan_line)

    ax_fan.set_ylabel("Fan action (0–1)")
    ax_fan.set_ylim(-0.05, 1.05)

    ax_tsup.set_title(title)

    if lines:
        ax_tsup.legend(lines, [line.get_label() for line in lines], loc="best")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved action plot: {output_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    random.seed(SEED_NO)
    np.random.seed(SEED_NO)

    model_dir = resolve_model_dir()
    model_file = resolve_model_file(model_dir)
    output_dir = build_output_dir()

    print("Model configuration")
    print(f"  variant   : {MODEL_VARIANT}")
    print(f"  seed      : {SEED_NO}")
    print(f"  checkpoint: {MODEL_NAME}")
    print(f"  model file: {model_file}")
    print(f"  output dir: {output_dir}")

    # Load only. No training is performed in this script.
    model = SAC.load(str(model_file))
    env = make_env(output_dir)

    # 1) Aug 8 only. Warmup is Aug 1~7.
    day_df, day_kpis = run_fixed_test(
        env=env,
        model=model,
        name="aug08_day",
        eval_start=HOTTEST_DAY,
        eval_days=1,
        output_dir=output_dir,
    )

    # 2) Aug 5~11. Warmup is Jul 29~Aug 4.
    week_df, week_kpis = run_fixed_test(
        env=env,
        model=model,
        name="aug05_11_week",
        eval_start=WEEK_START,
        eval_days=7,
        output_dir=output_dir,
    )

    if MAKE_PLOTS:
        # Aug 8: temperature/state plot
        plot_temperatures(
            day_df,
            f"Seoul RL temperatures on Aug 8 - {MODEL_VARIANT}, seed {SEED_NO}, {MODEL_NAME}",
            output_dir / "RL_aug08_day_temperatures.png",
        )

        # Aug 8: two RL actions in one figure
        plot_actions(
            day_df,
            f"Seoul RL actions on Aug 8 - {MODEL_VARIANT}, seed {SEED_NO}, {MODEL_NAME}",
            output_dir / "RL_aug08_day_actions.png",
        )

        # Aug 5-11: temperature/state plot
        plot_temperatures(
            week_df,
            f"Seoul RL temperatures Aug 5-11 - {MODEL_VARIANT}, seed {SEED_NO}, {MODEL_NAME}",
            output_dir / "RL_aug05_11_week_temperatures.png",
        )

        # Aug 5-11: two RL actions in one figure
        plot_actions(
            week_df,
            f"Seoul RL actions Aug 5-11 - {MODEL_VARIANT}, seed {SEED_NO}, {MODEL_NAME}",
            output_dir / "RL_aug05_11_week_actions.png",
        )

    # One concise summary file for later PI-vs-RL comparison.
    summary = {
        "model_variant": MODEL_VARIANT,
        "seed": SEED_NO,
        "model_name": MODEL_NAME,
        "model_file": str(model_file),
        "day_test": {
            "start": str(HOTTEST_DAY),
            "days": 1,
            "warmup_days": WARMUP_DAYS,
            "total_reward": float(day_df["reward"].sum()) if not day_df.empty else None,
            "kpis": day_kpis,
        },
        "week_test": {
            "start": str(WEEK_START),
            "days": 7,
            "warmup_days": WARMUP_DAYS,
            "total_reward": float(week_df["reward"].sum()) if not week_df.empty else None,
            "kpis": week_kpis,
        },
    }

    with open(output_dir / "RL_aug8_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print("The training directory was only READ; all new files were written to:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()