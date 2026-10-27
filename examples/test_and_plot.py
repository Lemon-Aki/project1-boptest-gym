'''
Common functionality to test and plot an agent

'''

import matplotlib.pyplot as plt
from scipy import interpolate
from gymnasium.core import Wrapper
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
import json
import os


def test_agent(env, model, start_time, episode_length, warmup_period,
               log_dir=os.getcwd(), model_name='last_model', 
               save_to_file=False, plot=False, save_step_log=True):
    ''' Test model agent in env.
    
    '''
        
    # Set a fixed start time
    if isinstance(env,Wrapper): 
        env.unwrapped.random_start_time   = False
        env.unwrapped.start_time          = start_time
        env.unwrapped.max_episode_length  = episode_length
        env.unwrapped.warmup_period       = warmup_period
    else:
        env.random_start_time   = False
        env.start_time          = start_time
        env.max_episode_length  = episode_length
        env.warmup_period       = warmup_period
    
    # Reset environment
    obs, _ = env.reset()
    
    # Simulation loop
    done = False
    observations = [obs]
    actions = []
    rewards = []
    print('Simulating...')

    ###
    step_logs = []
    step_idx = 0
    points_to_log = [
        'reaTZon_y',
        'reaTSetHea_y',
        'reaTSetCoo_y',
        'oveHeaPumY_u',
        'weaSta_reaWeaTDryBul_y',
        'weaSta_reaWeaHDirNor_y'
    ]
    ###

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        ###
        kpis_now = env.get_kpis()

        sim_time = env.unwrapped.start_time + step_idx * env.unwrapped.step_period
        datetime = pd.Timestamp('2022/1/1') + pd.Timedelta(sim_time, 's')

        action_flat = np.array(action).flatten()
        obs_flat = np.array(obs).flatten()

        # 정규화 action
        action_norm = float(np.array(action).flatten()[0])
        # heatpump 실제 action 복원: [-1, 1] -> [0, 1]
        action_real = (action_norm + 1.0) / 2.0

        # 실제 시뮬레이션 출력값 조회
        res = requests.put(
            '{0}/results/{1}'.format(env.unwrapped.url, env.unwrapped.testid),
            json={
                'point_names': points_to_log,
                'start_time': sim_time,
                'final_time': sim_time
            }
        ).json()['payload']


        row = {
            'step': step_idx,
            'datetime': datetime,
            'reward': reward,
            'action_norm': action_norm,
            'action_real_estimated': action_real,
            'cost_tot': kpis_now.get('cost_tot'),
            'tdis_tot': kpis_now.get('tdis_tot'),
        }

        
        for p in points_to_log:
            if p in res and len(res[p]) > 0:
                val = res[p][-1]

                if p == 'reaTZon_y':
                    row['Tzone_C'] = val - 273.15

                elif p == 'reaTSetHea_y':
                    row['Tset_heating_C'] = val - 273.15

                elif p == 'reaTSetCoo_y':
                    row['Tset_cooling_C'] = val - 273.15

                elif p == 'weaSta_reaWeaTDryBul_y':
                    row['Tout_C'] = val - 273.15

                else:
                    row[p] = val


        step_logs.append(row)
        ###

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        #obs = obs
        step_idx += 1

        done = (terminated or truncated)

    ###
    kpis = env.get_kpis()

    if save_step_log:
        scenario = env.unwrapped.scenario if isinstance(env, Wrapper) else env.scenario

        dir_name = os.path.join(
            log_dir,
            'results_tests_' + model_name + '_' + scenario['electricity_price']
        )
        os.makedirs(dir_name, exist_ok=True)

        df_step = pd.DataFrame(step_logs)
        df_step.to_csv(
            os.path.join(dir_name, 'step_log1_{}.csv'.format(str(int(start_time/3600/24)))),
            index=False
        )
    ###
    
    if save_to_file:
        os.makedirs(os.path.join(log_dir, 'results_tests_'+model_name+'_'+env.scenario['electricity_price']), exist_ok=True)
        with open(os.path.join(log_dir, 'results_tests_'+model_name+'_'+env.scenario['electricity_price'], 'kpis_{}.json'.format(str(int(start_time/3600/24)))), 'w') as f:
            json.dump(kpis, f)
    
    if plot:
        plot_results(env, rewards, save_to_file=save_to_file, log_dir=log_dir, model_name=model_name)
    
    # Back to random start time, just in case we're testing in the loop
    if isinstance(env,Wrapper): 
        env.unwrapped.random_start_time = True
    else:
        env.random_start_time = True
    
    return observations, actions, rewards, kpis

### air 전용 테스트
def test_agent_air(env, model, start_time, episode_length, warmup_period,
                   log_dir=os.getcwd(), model_name='last_model',
                   save_to_file=False, plot=False, save_step_log=True):
    """Test trained RL agent for BOPTEST bestest_air case."""

    # Wrapper 처리
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    # Set fixed test period
    base_env.random_start_time = False
    base_env.start_time = start_time
    base_env.max_episode_length = episode_length
    base_env.warmup_period = warmup_period

    # Reset environment
    obs, _ = env.reset()

    done = False
    observations = [obs]
    actions = []
    rewards = []

    step_logs = []
    step_idx = 0

    print("Simulating bestest_air...")

    # bestest_air 중심 로그 변수
    candidate_points_to_log = [
        # indoor condition
        "zon_reaTRooAir_y",
        "zon_reaCO2RooAir_y",

        # comfort setpoints
        "reaTSetHea_y",
        "reaTSetCoo_y",

        # air case control inputs
        "con_oveTSetHea_u",
        "con_oveTSetCoo_u",
        "fcu_oveFan_u",

        # weather
        "weaSta_reaWeaTDryBul_y",
        "weaSta_reaWeaHDirNor_y",
        "weaSta_reaWeaHGloHor_y",
        "weaSta_reaWeaRelHum_y",
    ]

    # 실제 test case에 존재하는 point만 사용
    available_points = set(base_env.all_measurement_vars.keys()) | set(base_env.all_input_vars.keys())
    points_to_log = [p for p in candidate_points_to_log if p in available_points]

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        kpis_now = base_env.get_kpis()

        sim_time = base_env.start_time + step_idx * base_env.step_period
        datetime = pd.Timestamp("2022/1/1") + pd.Timedelta(sim_time, "s")

        action_flat = np.array(action).flatten()

        row = {
            "step": step_idx,
            "time": sim_time,
            "datetime": datetime,
            "reward": float(reward),
            "cost_tot": kpis_now.get("cost_tot"),
            "tdis_tot": kpis_now.get("tdis_tot"),
            "idis_tot": kpis_now.get("idis_tot"),
            "ener_tot": kpis_now.get("ener_tot"),
        }

        # action logging
        for i, a in enumerate(action_flat):
            row[f"action_raw_{i}"] = float(a)

        # NormalizedActionWrapper를 쓴 경우 실제 BOPTEST action 값 복원
        try:
            env_low = np.array(env.action_space.low).flatten()
            env_high = np.array(env.action_space.high).flatten()
            base_low = np.array(base_env.lower_act_bounds).flatten()
            base_high = np.array(base_env.upper_act_bounds).flatten()

            is_normalized_action = (
                np.allclose(env_low, -1.0) and
                np.allclose(env_high, 1.0) and
                len(base_low) == len(action_flat)
            )

            if is_normalized_action:
                action_real = base_low + (action_flat + 1.0) * 0.5 * (base_high - base_low)
            else:
                action_real = action_flat

            for i, act_name in enumerate(base_env.actions):
                if i < len(action_real):
                    row[f"{act_name}_actual"] = float(action_real[i])

                    # setpoint 계열이면 섭씨 변환도 저장
                    if "TSet" in act_name:
                        row[f"{act_name}_actual_C"] = float(action_real[i] - 273.15)

        except Exception as e:
            row["action_decode_error"] = str(e)

        # BOPTEST result 조회
        if len(points_to_log) > 0:
            res = requests.put(
                "{0}/results/{1}".format(base_env.url, base_env.testid),
                json={
                    "point_names": points_to_log,
                    "start_time": sim_time,
                    "final_time": sim_time,
                }
            ).json()["payload"]

            for p in points_to_log:
                if p in res and len(res[p]) > 0:
                    val = res[p][-1]

                    if p == "zon_reaTRooAir_y":
                        row["Tzone_C"] = val - 273.15

                    elif p == "reaTSetHea_y":
                        row["Tset_heating_C"] = val - 273.15

                    elif p == "reaTSetCoo_y":
                        row["Tset_cooling_C"] = val - 273.15

                    elif p == "con_oveTSetHea_u":
                        row["con_oveTSetHea_u_K"] = val
                        row["con_oveTSetHea_u_C"] = val - 273.15

                    elif p == "con_oveTSetCoo_u":
                        row["con_oveTSetCoo_u_K"] = val
                        row["con_oveTSetCoo_u_C"] = val - 273.15

                    elif p == "weaSta_reaWeaTDryBul_y":
                        row["Tout_C"] = val - 273.15

                    elif p == "weaSta_reaWeaRelHum_y":
                        row["RH_out"] = val

                    else:
                        row[p] = val

        step_logs.append(row)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        step_idx += 1
        done = terminated or truncated

    # Final KPI
    kpis = base_env.get_kpis()
    scenario = base_env.scenario

    # 저장 경로
    dir_name = os.path.join(
        log_dir,
        "results_tests_" + model_name + "_" + scenario["electricity_price"]
    )
    os.makedirs(dir_name, exist_ok=True)

    if save_step_log:
        df_step = pd.DataFrame(step_logs)
        df_step.to_csv(
            os.path.join(
                dir_name,
                "step_log_air_{}.csv".format(str(int(start_time / 3600 / 24)))
            ),
            index=False
        )

    if save_to_file:
        with open(
            os.path.join(
                dir_name,
                "kpis_air_{}.json".format(str(int(start_time / 3600 / 24)))
            ),
            "w"
        ) as f:
            json.dump(kpis, f, indent=4)

    # 일단 air plot은 별도 구현 전까지 끄는 걸 추천
    if plot:
        print("plot=True was requested, but test_agent_air does not call heatpump plot_results.")
        print("Use saved step_log_air_*.csv for plotting bestest_air results.")

    # 혹시 다음 학습/테스트에 영향 안 주도록 복구
    base_env.random_start_time = True

    return observations, actions, rewards, kpis

###

def plot_results(env, rewards, points=['reaTZon_y','reaTSetHea_y','reaTSetCoo_y','oveHeaPumY_u',
                                       'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'],
                 log_dir=os.getcwd(), model_name='last_model', save_to_file=False):
    

    if points is None:
        points = list(env.all_measurement_vars.keys()) + \
                 list(env.all_input_vars.keys())
        
    # Retrieve all simulation data
    # We use env.start_time+1 to ensure that we don't return the last 
    # point from the initialization period to don't confuse it with 
    # actions taken by the agent in a previous episode. 
    res = requests.put('{0}/results/{1}'.format(env.url, env.testid), 
                        json={'point_names':points,
                              'start_time':env.start_time+1, 
                              'final_time':3.1536e7}).json()['payload']

    df = pd.DataFrame(res)
    df = create_datetime_index(df)
    df.dropna(axis=0, inplace=True)
    scenario = env.scenario

    if save_to_file:
        df.to_csv(os.path.join(log_dir, 'results_tests_'+model_name+'_'+scenario['electricity_price'], 
                  'results_sim_{}.csv'.format(str(int(res['time'][0]/3600/24)))))
    
    # Project rewards into results index
    rewards_time_days = np.arange(df['time'][0], 
                                  env.start_time+env.max_episode_length,
                                  env.step_period)/3600./24.
    f = interpolate.interp1d(rewards_time_days, rewards, kind='zero',
                             fill_value='extrapolate')
    res_time_days = np.array(df['time'])/3600./24.
    rewards_reindexed = f(res_time_days)
    
    if not plt.get_fignums():
        # no window(s) are open, so open a new window. 
        _, axs = plt.subplots(4, sharex=True, figsize=(8,6))
    else:
        # There is a window open, so get current figure. 
        # Combine this with plt.ion(), plt.figure()
        fig = plt.gcf()
        axs = fig.subplots(nrows=4, ncols=1, sharex=True)
            
    x_time = df.index.to_pydatetime()

    axs[0].plot(x_time, df['reaTZon_y']  -273.15, color='darkorange',   linestyle='-', linewidth=1, label='_nolegend_')
    axs[0].plot(x_time, df['reaTSetHea_y'] -273.15, color='gray',       linewidth=1, label='Comfort setp.')
    axs[0].plot(x_time, df['reaTSetCoo_y'] -273.15, color='gray',       linewidth=1, label='_nolegend_')
    axs[0].set_yticks(np.arange(15, 31, 5))
    axs[0].set_ylabel('Operative\ntemperature\n($^\circ$C)')
    
    axs[1].plot(x_time, df['oveHeaPumY_u'],   color='darkorange',     linestyle='-', linewidth=1, label='_nolegend_')
    axs[1].set_ylabel('Heat pump\nmodulation\nsignal\n( - )')
    
    axs[2].plot(x_time, rewards_reindexed, 'b', linewidth=1, label='rewards')
    axs[2].set_ylabel('Rewards\n(-)')
    
    axs[3].plot(x_time, df['weaSta_reaWeaTDryBul_y'] - 273.15, color='royalblue', linestyle='-', linewidth=1, label='_nolegend_')
    axs[3].set_ylabel('Ambient\ntemperature\n($^\circ$C)')
    axs[3].set_yticks(np.arange(-5, 16, 5))
    axt = axs[3].twinx()
    
    axt.plot(x_time, df['weaSta_reaWeaHDirNor_y'], color='gold', linestyle='-', linewidth=1, label='$\dot{Q}_rad$')
    axt.set_ylabel('Solar\nirradiation\n($W$)')
    
    axs[3].plot([],[], color='darkorange',  linestyle='-', linewidth=1, label='RL')
    axs[3].plot([],[], color='dimgray',     linestyle='dotted', linewidth=1, label='Price')
    axs[3].plot([],[], color='royalblue',   linestyle='-', linewidth=1, label='$T_a$')
    axs[3].plot([],[], color='gold',        linestyle='-', linewidth=1, label='$\dot{Q}_{rad}$')
    axs[3].legend(fancybox=True, ncol=6, bbox_to_anchor=(1.06, -0.3)) 
    
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.tight_layout()
    
    if save_to_file:
        dir_name = os.path.join(log_dir, 'results_tests_'+model_name+'_'+scenario['electricity_price'])
        fil_name = os.path.join(dir_name,'results_sim_{}.pdf'.format(str(int(res['time'][0]/3600/24))))
        os.makedirs(dir_name, exist_ok=True)
        plt.savefig(fil_name, bbox_inches='tight')
    
    if not save_to_file:
        # showing and saving to file are incompatible
        plt.pause(0.001)
        plt.show()  

    
def reindex(df, interval=60, start=None, stop=None):
    '''
    Define the index. Make sure last point is included if 
    possible. If interval is not an exact divisor of stop,
    the closest possible point under stop will be the end 
    point in order to keep interval unchanged among index.
    
    ''' 
    
    if start is None:
        start = df['time'][0]
    if stop is None:
        stop  = df['time'][-1]  
    index = np.arange(start,stop+0.1,interval).astype(int)
    df_reindexed = df.reindex(index)
    
    # Avoid duplicates from FMU simulation. Duplicates lead to 
    # extrapolation errors
    df.drop_duplicates('time',inplace=True)
    
    for key in df_reindexed.keys():
        # Use linear interpolation 
        f = interpolate.interp1d(df['time'], df[key], kind='linear',
                                 fill_value='extrapolate')
        df_reindexed.loc[:,key] = f(index)
        
    return df_reindexed


def create_datetime_index(df):
    '''
    Create a datetime index for the data
    
    '''
    
    datetime = []
    for t in df['time']:
        datetime.append(pd.Timestamp('2023/1/1') + pd.Timedelta(t,'s'))
    df['datetime'] = datetime
    df.set_index('datetime', inplace=True)
    
    return df
    
    
