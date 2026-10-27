import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

txt_path = "./examples/txtfile/spring_fall2.txt"
window = 20  # moving average window

episodes = []
rewards = []
'''
with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        m = re.search(r"\[Episode\s+(\d+)\]\s+reward:\s+(-?\d+(?:\.\d+)?)", line)
        if m:
            episodes.append(int(m.group(1)))
            rewards.append(float(m.group(2)))

df = pd.DataFrame({
    "episode": episodes,
    "reward": rewards
})
'''

'''
# 저장도 같이 하고 싶으면
#df.to_csv("./examples/txtfile/spring_fall2.csv", index=False)
csv_path = './examples/txtfile/spring_fall2.csv'

df = pd.read_csv(csv_path)

# moving average
df["moving_avg"] = df["reward"].rolling(window=window, min_periods=1).mean()

plt.figure(figsize=(12, 5))
plt.plot(df["episode"], df["reward"], alpha=0.35, label="Episode reward")
plt.plot(df["episode"], df["moving_avg"], linewidth=2, label=f"Moving average ({window})")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Curve - spring_fall2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''

"""
import os
import json
import pandas as pd

dir_path = "examples/agents/SAC_D_5e05_spr_fall2_logdir/results_tests_last_model_highly_dynamic"


rows = []

for file in os.listdir(dir_path):
    if file.startswith("kpis_") and file.endswith(".json"):
        file_path = os.path.join(dir_path, file)

        with open(file_path, "r") as f:
            data = json.load(f)

        # JSON → DataFrame (한 줄)
        df = pd.DataFrame([data])

        # CSV 파일명 동일하게 생성
        csv_name = file.replace(".json", ".csv")
        csv_path = os.path.join(dir_path, csv_name)

        df.to_csv(csv_path, index=False)

        print(f"✅ 변환 완료: {csv_name}")
"""

'''
import pandas as pd

# 🔧 파일 경로
file_path = "examples/agents/SAC_D_5e05_logdir/results_tests_last_model_highly_dynamic/"

# CSV 읽기
df = pd.read_csv(file_path + "step_log1_108_all.csv",skiprows=1)

# datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'])
df_eval = df
# -----------------------------
# 2️⃣ 누적 KPI 보정
# -----------------------------
df_eval['tdis_eval'] = df_eval['tdis_tot'] - df_eval['tdis_tot'].iloc[0]
df_eval['cost_eval'] = df_eval['cost_tot'] - df_eval['cost_tot'].iloc[0]

# -----------------------------
# 3️⃣ step 단위 변화량 (선택)
# -----------------------------
df_eval['tdis_diff'] = df_eval['tdis_tot'].diff().fillna(0)
df_eval['cost_diff'] = df_eval['cost_tot'].diff().fillna(0)

# -----------------------------
# 4️⃣ overheating / underheating 분석
# -----------------------------
df_eval['overheating'] = df_eval['Tzone_C'] > df_eval['Tset_cooling_C']
df_eval['underheating'] = df_eval['Tzone_C'] < df_eval['Tset_heating_C']

# -----------------------------
# 5️⃣ 결과 요약
# -----------------------------
tdis_total = df_eval['tdis_diff'].sum()
cost_total = df_eval['cost_diff'].sum()
reward_total = df_eval['reward'].sum()

over_count = df_eval['overheating'].sum()
under_count = df_eval['underheating'].sum()

print("===== Evaluation Summary (Last 7 Days) =====")
print(f"Total tdis: {tdis_total:.3f}")
print(f"Total cost: {cost_total:.3f}")
print(f"Total reward: {reward_total:.3f}")
print()
print(f"Overheating count: {over_count}")
print(f"Underheating count: {under_count}")

# -----------------------------
# 6️⃣ CSV 저장 (옵션)
# -----------------------------
df_eval.to_csv(file_path+"eval_processed_108_all.csv", index=False)
print("\n✅ eval_processed.csv 저장 완료")

'''
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. CSV 파일 로드
df = pd.read_csv('examples/agents/SAC_D_5e05_air_all_logdir/weather.csv')

# 2. 15분 단위 시각 데이터 생성
# 데이터 개수만큼 2026-01-01 00:00부터 15분 간격으로 생성
start_time = '2026-01-01 00:00:00'
time_index = pd.date_range(start=start_time, periods=len(df), freq='15T')
df['Timestamp'] = time_index

# 3. 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(df['Timestamp'], df['외기온도(℃)'], color='tab:blue', linewidth=1)

# x축 설정: 월별로 표시되도록 포맷팅
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# 그래프 꾸미기
plt.title('Outdoor Temperature Variation (15min interval)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Temperature (Celsius)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 그래프 출력
plt.show()
'''
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# CSV 읽기
df = pd.read_csv("examples/agents/SAC_D_5e05_air_all_logdir/weather.csv")

# 15분 간격 datetime 생성
df["datetime"] = pd.date_range(
    start="2022-01-01 00:00:00",
    periods=len(df),
    freq="15min"
)

# 테스트 기간 정의
periods = {
    "Peak Heat\nDec 1-Dec 15": ("2022-12-01", "2022-12-15"),
    "Typical Heat\nFeb 14-Feb 28": ("2022-02-14", "2022-02-28"),
    "Peak Cool\nOct 10-Oct 24": ("2022-10-10", "2022-10-24"),
    "Typical Cool\nMay 27-Jun 10": ("2022-05-27", "2022-06-10"),
    "Mix\nJan 15-Jan 29": ("2022-01-15", "2022-01-29"),
}

plt.figure(figsize=(14, 6))
ax = plt.gca()

# 외기온도 그래프
ax.plot(df["datetime"], df["외기온도(℃)"], linewidth=0.8)

# 테스트 기간 음영 표시
for label, (start, end) in periods.items():
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    ax.axvspan(start, end, alpha=0.18)
    
    # 라벨 위치
    mid = start + (end - start) / 2
    ax.text(
        mid,
        ax.get_ylim()[1] - 2,
        label,
        ha="center",
        va="top",
        fontsize=9,
        rotation=90
    )
# x축 월 단위 표시
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

ax.set_title("Outdoor Temperature Variation")
ax.set_xlabel("Month")
ax.set_ylabel("Temperature (Celsius)")
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

'''

'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {
    "Test": ["Peak Heat", "Typical Heat", "Peak Cool", "Typical Cool", "Mix"],
    "All-year": [-7.454, -3.605, -1.721, -0.984, -3.144],
    "Heating-dominant": [-3.588, -2.816, -2.180, -1.089, -2.927],
    "Cooling-dominant": [-48.616, -14.689, -1.729, -0.959, -22.910],
}

df = pd.DataFrame(data)

# Convert reward to objective value
for col in ["All-year", "Heating-dominant", "Cooling-dominant"]:
    df[col] = -df[col]

x = np.arange(len(df["Test"]))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5))

ax.bar(x - width, df["All-year"], width, label="All-year")
ax.bar(x, df["Heating-dominant"], width, label="Winter data")
ax.bar(x + width, df["Cooling-dominant"], width, label="Summer data")

ax.set_xticks(x)
ax.set_xticklabels(df["Test"])
ax.set_ylabel("-Total reward")
ax.set_title("Comparison of Test Result")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

'''
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tests = ["Peak Heat", "Typical Heat", "Peak Cool", "Typical Cool", "Mix"]

tdis = pd.DataFrame({
    "Test": tests,
    "All-year": [5.101770939721115, 2.1292019074943784, 0.1479369310811999, 0.015538457436674473, 1.771147513727228],
    "Heating-dominant": [2.1862859103070553, 1.7811920780747594, 0.354722603570201, 0.054984197521880177, 1.960832223452205],
    "Cooling-dominant": [47.547390239039615, 13.857891456908183, 0.22300987682452936, 0.0032890529648341273, 22.00875364851878],
})

cost = pd.DataFrame({
    "Test": tests,
    "All-year": [0.1470343887466186, 0.09226586886920463, 0.09832714558248995, 0.0604999931798978, 0.08578030490610919],
    "Heating-dominant": [0.08762885701407935, 0.06466922269200255, 0.11411095551107837, 0.06459887955654077, 0.06039918589683693],
    "Cooling-dominant": [0.06681362095626732, 0.051941207183429566, 0.09413049740065907, 0.05972045193377733, 0.05658793444757565],
})

cool_tests = ["Peak Cool", "Typical Cool"]

tdis_cool = tdis[tdis["Test"].isin(cool_tests)].reset_index(drop=True)
cost_cool = cost[cost["Test"].isin(cool_tests)].reset_index(drop=True)


def grouped_bar(df, ylabel, title):
    x = np.arange(len(df["Test"]))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, df["All-year"], width, label="All-year")
    ax.bar(x, df["Heating-dominant"], width, label="Heating-dominant")
    ax.bar(x + width, df["Cooling-dominant"], width, label="Cooling-dominant")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Test"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

grouped_bar(tdis, "Thermal discomfort, tdis_tot", "Comparison of Thermal Discomfort")
grouped_bar(cost, "Operational cost, cost_tot", "Comparison of Operational Cost")


grouped_bar(tdis_cool, "Thermal discomfort, tdis_tot", "Thermal Discomfort in Cooling Tests")
grouped_bar(cost_cool, "Operational cost, cost_tot", "Operational Cost in Cooling Tests")

'''


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# =========================
# 1. 파일 경로 설정
# =========================
weather_csv_path = "/mnt/d/dev-code/boptest-0.8.0/testcases/bestest_air/models/Resources/weather.csv"   # 서울 weather.csv 경로로 수정

# =========================
# 2. 서울 test day 설정
#    (확인된 서울 days.json 기준)
# =========================
test_days = {
    "Mix": 35,
    "Peak Heat": 24,
    "Typical Cool": 77,
    "Typical Heat": 95,
    "Peak Cool": 261,
}

# BOPTEST에서는 start_time_tests = (day - 7) * 24 * 3600 로 사용했으므로
# 각 테스트는 중심일 기준 2주(14일) 구간으로 표시
test_windows = {}
for name, center_day in test_days.items():
    start_sec = (center_day - 7) * 24 * 3600
    start_dt = datetime(2021, 1, 1) + timedelta(seconds=start_sec)
    end_dt = start_dt + timedelta(days=14)
    test_windows[name] = (start_dt, end_dt)

# =========================
# 3. weather.csv 읽기
# =========================
df = pd.read_csv(weather_csv_path)

# time(sec) -> datetime
df["datetime"] = pd.to_datetime("2021-01-01") + pd.to_timedelta(df["time"], unit="s")

# TDryBul [K] -> Celsius
df["TDryBul_C"] = df["TDryBul"] - 273.15

# =========================
# 4. 라벨 순서 정렬 (원하면 순서 조정 가능)
# =========================
plot_order = ["Mix", "Peak Heat", "Typical Cool", "Typical Heat", "Peak Cool"]

# =========================
# 5. 그래프 그리기
# =========================
plt.figure(figsize=(18, 7))
plt.plot(df["datetime"], df["TDryBul_C"], linewidth=1)

# 테스트 기간 음영 + 라벨
for name in plot_order:
    start_dt, end_dt = test_windows[name]
    plt.axvspan(start_dt, end_dt, alpha=0.18)

    label = f"{name}\n{start_dt.strftime('%b %d')}–{end_dt.strftime('%b %d')}"
    mid_dt = start_dt + (end_dt - start_dt) / 2

    plt.text(
        mid_dt,
        df["TDryBul_C"].max() - 1,
        label,
        rotation=90,
        ha="center",
        va="top",
        fontsize=13
    )

# =========================
# 6. 축/제목 설정
# =========================
plt.title("Outdoor Temperature (Seoul)", fontsize=18)
plt.xlabel("Month", fontsize=15)
plt.ylabel("Temperature (Celsius)", fontsize=15)
plt.grid(True, linestyle="--", alpha=0.5)

# 축 눈금 글씨 크기
plt.tick_params(axis="both", labelsize=13)

# 월 단위 표시
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))

plt.tight_layout()
plt.show()