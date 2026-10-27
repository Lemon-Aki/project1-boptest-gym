#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot training reward convergence from a ZIP file containing multiple episode_rewards.csv files.

Purpose
-------
- Read 3 or 5 seed CSV files directly from one ZIP archive.
- Plot each seed as a separate line.
- Plot a thicker seed-mean line.
- NO SD band is drawn.
- Also create a checkpoint stability plot and CSV summary.

Expected ZIP contents
---------------------
The ZIP may contain files anywhere inside folders, as long as filenames end with:
    episode_rewards.csv
or contain:
    episode_rewards

Required CSV columns
--------------------
episode, reward

Optional columns
----------------
start_time_sec, start_day, start_datetime, ...

Usage
-----
python reward_convergence_from_zip.py \
    --zip_file ./all_16_seed1-3.zip \
    --output_dir ./all_convergence

Optional:
    --window 20
    --steps_per_episode 672
    --checkpoints 50000 75000 100000 125000 150000 175000 200000 225000 250000 275000 300000
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CHECKPOINTS = [
    50_000, 75_000, 100_000, 125_000,
    150_000, 175_000, 200_000, 225_000,
    250_000, 275_000, 300_000,
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("./convergence_outputs"))
    p.add_argument("--steps_per_episode", type=int, default=672)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--checkpoints", type=int, nargs="*", default=DEFAULT_CHECKPOINTS)
    p.add_argument(
        "--title",
        type=str,
        default="Training reward convergence",
        help="Base title used in output figures",
    )
    return p.parse_args()


def natural_key(s):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def seed_label_from_name(name, index):
    """
    Try to infer a meaningful seed label from path/name.
    Falls back to Seed 1, Seed 2, ...
    """
    # Common actual seed values used in these experiments
    known = ["123456", "199580", "173827", "101291", "362507"]
    for k in known:
        if k in name:
            return f"Seed {k}"

    m = re.search(r"seed[_\- ]?(\d+)", name, flags=re.I)
    if m:
        return f"Seed {m.group(1)}"

    return f"Seed {index}"


def read_logs_from_zip(zip_path: Path, steps_per_episode: int):
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    logs = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            n for n in zf.namelist()
            if not n.endswith("/")
            and n.lower().endswith(".csv")
            and "episode_rewards" in Path(n).name.lower()
        ]

        if len(members) < 2:
            raise RuntimeError(
                f"Only {len(members)} episode_rewards CSV file(s) found in ZIP. "
                "At least 2 are recommended."
            )

        members = sorted(members, key=natural_key)

        for i, member in enumerate(members, start=1):
            raw = zf.read(member)
            df = pd.read_csv(io.BytesIO(raw))

            required = {"episode", "reward"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{member}: missing columns {sorted(missing)}")

            df = df.sort_values("episode").reset_index(drop=True)
            df["training_step"] = df["episode"].astype(int) * steps_per_episode

            label = seed_label_from_name(member, i)

            # Avoid duplicate labels if names do not expose seed.
            original_label = label
            suffix = 2
            while label in logs:
                label = f"{original_label} ({suffix})"
                suffix += 1

            logs[label] = {
                "member": member,
                "df": df,
            }

    return logs


def add_rolling(logs, window):
    for item in logs.values():
        df = item["df"]
        df["reward_rolling"] = df["reward"].rolling(
            window=window,
            min_periods=max(3, window // 4),
        ).mean()


def make_aligned(logs):
    merged = None

    for label, item in logs.items():
        df = item["df"][["episode", "training_step", "reward_rolling"]].copy()
        df = df.rename(columns={"reward_rolling": label})

        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df,
                on=["episode", "training_step"],
                how="inner",
            )

    seed_cols = list(logs.keys())
    merged["Seed mean"] = merged[seed_cols].mean(axis=1)
    return merged, seed_cols


def checkpoint_summary(logs, checkpoints, window):
    rows = []

    for cp in sorted(checkpoints):
        row = {
            "checkpoint": cp,
            "checkpoint_k": cp / 1000.0,
        }

        values = []

        for label, item in logs.items():
            df = item["df"]
            eligible = df[df["training_step"] <= cp]

            if eligible.empty:
                value = np.nan
            else:
                value = eligible.tail(window)["reward"].mean()

            row[label] = value
            if np.isfinite(value):
                values.append(value)

        row["Seed mean"] = np.mean(values) if values else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def plot_full_convergence(aligned, seed_cols, output_dir, title, window):
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Matplotlib default color cycle gives each seed a distinct color.
    for seed in seed_cols:
        ax.plot(
            aligned["training_step"] / 1000.0,
            aligned[seed],
            linewidth=1.25,
            alpha=0.8,
            label=seed,
        )

    ax.plot(
        aligned["training_step"] / 1000.0,
        aligned["Seed mean"],
        linewidth=3.0,
        label="Seed mean",
    )

    ax.set_title(f"{title} ({window}-episode rolling mean)")
    ax.set_xlabel("Training steps (k)")
    ax.set_ylabel("Episode reward")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()

    out = output_dir / "reward_convergence_all_seeds.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_checkpoint_stability(summary, seed_cols, output_dir, title, window):
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    for seed in seed_cols:
        ax.plot(
            summary["checkpoint_k"],
            summary[seed],
            marker="o",
            linewidth=1.25,
            alpha=0.8,
            label=seed,
        )

    ax.plot(
        summary["checkpoint_k"],
        summary["Seed mean"],
        marker="o",
        linewidth=3.0,
        label="Seed mean",
    )

    ax.set_title(f"{title} — checkpoint stability (trailing {window} episodes)")
    ax.set_xlabel("Checkpoint (k steps)")
    ax.set_ylabel("Trailing-window episode reward")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()

    out = output_dir / "checkpoint_stability_all_seeds.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logs = read_logs_from_zip(
        args.zip_file,
        steps_per_episode=args.steps_per_episode,
    )
    add_rolling(logs, args.window)

    print("===== Files loaded from ZIP =====")
    for label, item in logs.items():
        print(f"{label}: {item['member']}")

    aligned, seed_cols = make_aligned(logs)

    summary = checkpoint_summary(
        logs,
        checkpoints=args.checkpoints,
        window=args.window,
    )

    csv_out = args.output_dir / "checkpoint_reward_summary.csv"
    summary.to_csv(csv_out, index=False, encoding="utf-8-sig")

    fig1 = plot_full_convergence(
        aligned,
        seed_cols,
        args.output_dir,
        args.title,
        args.window,
    )

    fig2 = plot_checkpoint_stability(
        summary,
        seed_cols,
        args.output_dir,
        args.title,
        args.window,
    )

    print("\n===== Outputs =====")
    print(fig1)
    print(fig2)
    print(csv_out)

    print("\n===== Checkpoint seed mean =====")
    print(
        summary[["checkpoint_k", "Seed mean"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
