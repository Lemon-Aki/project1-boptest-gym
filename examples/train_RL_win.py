'''
Module to train and test a RL agent for the bestest_hydronic_heatpump 
case. This case needs to be deployed to run this script.  
To load the ExpertDataset it's needed to comment the first line in stable_baselines\gail\_init_.py
from stable_baselines3.gail.model import GAIL
If further issues are encountered related to the np.ndarrais for pretraining, it may happen that
numpy is installed twice. Check:
https://stackoverflow.com/questions/54943168/problem-with-tensorflow-tf-sessionrun-wrapper-expected-all-values-in-input-dic

'''

from boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper, \
    NormalizedObservationWrapper, SaveAndTestCallback, DiscretizedActionWrapper
# from stable_baselines3.gail import ExpertDataset
from stable_baselines3 import A2C, SAC, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from examples.test_and_plot import test_agent, test_agent_air
from collections import OrderedDict
from testing import utilities
import requests
import random
import os
import numpy as np
import pandas as pd


url = 'http://127.0.0.1'

# Seed for random starting times of episodes
#random seeds = 123456, 199580, 173827, 101291, 362507
#random seeds = 265329, 344628, 140915, 46611, 390794
#random seeds = 209140, 137587, 31216, 98057, 17431


day = 7 #start_time_tests 7

def train_RL(algorithm           = 'SAC',
             #start_time_tests    = [(23-7)*24*3600, (115-7)*24*3600], 
             ###
             #start_time_tests = [(24-day)*24*3600,  # peak heat                
             #                    (95-day)*24*3600,   # typical heat                
             #                    (261-day)*24*3600,  # peak cool                
             #                    (175-day)*24*3600,  # typical cool  
             #                    #(181-7)*24*3600,    
             #                    #(212-7)*24*3600,           
             #                    (327-day)*24*3600    # mix                
             #                    ],
             #denver
             #start_time_tests = [(341-day)*24*3600,  # peak heat                
             #                    (322-day)*24*3600,   # typical heat                
             #                    (289-day)*24*3600,  # peak cool                
             #                    (153-day)*24*3600,  # typical cool  
             #                    #(181-7)*24*3600,    
             #                    #(212-7)*24*3600,           
             #                    (21-day)*24*3600    # mix                
             #                    ],
             #denver_new
             start_time_tests = [(341-day)*24*3600,  # peak heat                
                                 (322-day)*24*3600,   # typical heat                
                                 (244-day)*24*3600,  # peak cool                
                                 (188-day)*24*3600,  # typical cool  
                                 #(181-7)*24*3600,    
                                 #(212-7)*24*3600,           
                                 (318-day)*24*3600    # mix                
                                 ],
             ### #24, 95, 261, 77, 35  -> 24, 95, 261, 175, 327
             episode_length_test = 14*24*3600,
             warmup_period       = 1*24*3600,
             max_episode_length  = 7*24*3600,
             mode                = 'train',
             case                = 'simple',
             training_timesteps  = 3e5,
             render              = False,
             expert_traj         = None, 
             model_name          = 'last_model',
             ###
             seed_no = 1):
    '''Method to train (or load a pre-trained) A2C agent. Testing periods 
    have to be introduced already here to not use these during training. 
    
    Parameters
    ----------
    start_time_tests : list of integers
        Time in seconds from the beginning of the year that will be used 
        for testing. These periods should be excluded in the training 
        process. By default the peak and typical heat periods for the 
        bestest hydronic case with a heat pump are used. 
    episode_length_test : integer
        Number of seconds indicating the length of the testing periods. By
        default two weeks are reserved for testing. 
    mode : string
        Either train, load, or empty.
    case : string
        Case to be tested.
    training_timesteps : integer
        Total number of timesteps used for training
    render : boolean
        If true, it renders every episode while training.
    expert_traj : string
        Path to expert trajectory in .npz format. If not None, the agent 
        will be pretrained using behavior cloning with these data. 
    model_name : string
        Name of the model to be saved or loaded. 
            
    '''
    
    excluding_periods = []
    for start_time_test in start_time_tests:
        ###
        #excluding_periods.append((start_time_test,
        #                          start_time_test+episode_length_test))
        ###

        #exclude_start = start_time_test - 3*24*3600
        #exclude_end = start_time_test + 4*24*3600

        #excluding_periods.append((exclude_start, exclude_end))


        # start_time_test = 기준일 - 7일
        reference_time = start_time_test + day * 24 * 3600

        # 기준일 주변 7일을 training에서 제외
        exclude_start = reference_time - 3 * 24 * 3600
        exclude_end   = reference_time + 4 * 24 * 3600

        excluding_periods.append((exclude_start, exclude_end))


    # Summer period (from June 21st till September 22nd). 
    # Excluded since no heating during this period (nothing to learn).
    #excluding_periods.append((173*24*3600, 266*24*3600))  
    ###
    # winter = Jan 1 ~ Feb 28, Dec 1 ~ Dec 31
    # non-winter = Mar 1 ~ Nov 30
    #excluding_periods.append((59*24*3600, 334*24*3600))
    
    # Spring: day 59~173
    # Fall:   day 266~334
    # 제외: day 0~59, day 173~266, day 334~365
    #excluding_periods.append((0, 59*24*3600))
    #excluding_periods.append((173*24*3600, 266*24*3600))
    #excluding_periods.append((334*24*3600, 365*24*3600))
    
    # Spring: day 59~173
    #excluding_periods.append((0, 59*24*3600))
    #excluding_periods.append((173*24*3600, 365*24*3600))
    ###

    ### air period 
    #Heating season:  Nov 19 ~ Mar 2
    #Spring shoulder: Mar 3  ~ Jun 6
    #Cooling season:  Jun 7  ~ Sep 14
    #Fall shoulder:   Sep 15 ~ Nov 18
    # # heating only
    #excluding_periods.append((61 * 24 * 3600, 322 * 24 * 3600))

    # spring only
    #excluding_periods.append((0 * 24 * 3600, 61 * 24 * 3600))
    #excluding_periods.append((157 * 24 * 3600, 365 * 24 * 3600))

    # cooling only 
    #excluding_periods.append((0 * 24 * 3600, 157 * 24 * 3600))
    #excluding_periods.append((257 * 24 * 3600, 365 * 24 * 3600))

    # cooling only 변경 day 120 ~ day 300 5/1~10/27
    #excluding_periods.append((0 * 24 * 3600, 120 * 24 * 3600))
    #excluding_periods.append((300 * 24 * 3600, 365 * 24 * 3600))

    # cooooooling only Jun 7 ~ Sep 14 day 157 ~ day 257
    #excluding_periods.append((0 * 24 * 3600, 157 * 24 * 3600))
    #excluding_periods.append((257 * 24 * 3600, 365 * 24 * 3600))

    # fall only
    #excluding_periods.append((0 * 24 * 3600, 257 * 24 * 3600))
    #excluding_periods.append((322 * 24 * 3600, 365 * 24 * 3600))
    ###
    

    # air_seoul_summer_34_16-1 5월 22일∼9월 29일 141~271
    #excluding_periods.append((0 * 24 * 3600, 141 * 24 * 3600))
    #excluding_periods.append((272 * 24 * 3600, 365 * 24 * 3600))

    # air_seoul_winter_34_16-1 12월 4일~4월 14일 0~104, 337~365
    #excluding_periods.append((104 * 24 * 3600, 337 * 24 * 3600))
    
    # air_denver_winter: 1월 1일~3월 2일 + 11월 19일~12월 31일
    # day 0~60 + day 322~364
    excluding_periods.append((61 * 24 * 3600, 322 * 24 * 3600))

    # Create a log directory
    #air/seoul-denver/all-sum-win/action1234/reward비율/seed1~5/100k~500k/testno
    log_dir = os.path.join(utilities.get_root_path(), 'examples', 
        'agents', '{}_{}_{:.0e}_air_denver_win_34_76-1_{}_300k_5_logdir'.format(algorithm,case,training_timesteps,seed_no))
    log_dir = log_dir.replace('+', '')
    os.makedirs(log_dir, exist_ok=True)
    
    # Redefine reward function
    class BoptestGymEnvCustomReward(BoptestGymEnv):
        '''Define a custom reward for this building
        
        '''
        def get_reward(self):
            '''Custom reward function
            
            '''
            
            # Compute BOPTEST core kpis
            kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
            
            # Calculate objective integrand function at this point
            #objective_integrand = kpis['cost_tot']*12.*16. + 100*kpis['tdis_tot']
            ### air
            #objective_integrand = kpis['cost_tot']*12.*16. + 100*kpis['tdis_tot']
            objective_integrand = kpis['cost_tot']*76. + kpis['tdis_tot']
            ###
            
            # Compute reward
            reward = -(objective_integrand - self.objective_integrand)
            
            self.objective_integrand = objective_integrand
            #print("kpis['cost_tot']", kpis['cost_tot'] , "['tdis_tot']", kpis['tdis_tot'], reward)
            
            return reward
    
    #flag = 'a'
    flag = 'b'

    if flag == 'a':
        testcase = 'bestest_hydronic_heat_pump'
        actions = ['oveHeaPumY_u']
        observations = OrderedDict([('time',(0,604800)),
                                                     ('reaTZon_y',(280.,310.)),
                                                     ('TDryBul',(265,303)),
                                                     ('HDirNor',(0,862)),
                                                     ('InternalGainsRad[1]',(0,219)),
                                                     ('PriceElectricPowerHighlyDynamic',(-0.4,0.4)),
                                                     ('LowerSetp[1]',(280.,310.)),
                                                     ('UpperSetp[1]',(280.,310.))])

    elif flag == 'b':
        testcase = 'bestest_air'
        #actions = ['con_oveTSetHea_u', 'con_oveTSetCoo_u']
        #actions = ['con_oveTSetHea_u', 'con_oveTSetCoo_u', 'fcu_oveFan_u']
        actions = ['fcu_oveFan_u', 'fcu_oveTSup_u'] #12~40도
        observations=OrderedDict([
                            ('time', (0, 604800)),
                            ('zon_reaTRooAir_y', (280., 310.)),
                            ('zon_reaCO2RooAir_y', (400., 2000.)),
                            ('TDryBul', (250., 315.)),
                            ('HDirNor', (0., 1000.)),
                            #('PriceElectricPowerHighlyDynamic', (-0.4, 0.4)),
                            ('PriceElectricPowerDynamic', (-0.4, 0.4)),
                            ('LowerSetp[1]', (280., 310.)),
                            ('UpperSetp[1]', (280., 310.))
                        ])

        
    if case == 'simple':
        env = BoptestGymEnvCustomReward(
                            url                   = url,
                            testcase              = testcase,
                            actions               = actions,
                            observations          = OrderedDict([('reaTZon_y',(280.,310.))]), 
                            random_start_time     = True,
                            excluding_periods     = excluding_periods,
                            max_episode_length    = max_episode_length,
                            warmup_period         = warmup_period,
                            step_period           = 900,
                            render_episodes       = render,
                            log_dir               = log_dir)
    if case == 'A':
        env = BoptestGymEnvCustomReward(
                            url                   = url,
                            testcase              = testcase,
                            actions               = actions,
                            observations          = OrderedDict([('time',(0,604800)),
                                                     ('reaTZon_y',(280.,310.)),
                                                     ('PriceElectricPowerHighlyDynamic',(-0.4,0.4))]), 
                            scenario              = {'electricity_price':'highly_dynamic'},
                            predictive_period     = 0, 
                            random_start_time     = True,
                            excluding_periods     = excluding_periods,
                            max_episode_length    = max_episode_length,
                            warmup_period         = warmup_period,
                            step_period           = 900,
                            render_episodes       = render,
                            log_dir               = log_dir)
    if case == 'B':
        env = BoptestGymEnvCustomReward(
                            url                   = url,
                            testcase              = testcase,
                            actions               = actions,
                            observations          = OrderedDict([('time',(0,604800)),
                                                     ('reaTZon_y',(280.,310.)),
                                                     ('PriceElectricPowerHighlyDynamic',(-0.4,0.4)),
                                                     ('LowerSetp[1]',(280.,310.)),
                                                     ('UpperSetp[1]',(280.,310.))]), 
                            predictive_period     = 0, 
                            scenario              = {'electricity_price':'highly_dynamic'},
                            random_start_time     = True,
                            excluding_periods     = excluding_periods,
                            max_episode_length    = max_episode_length,
                            warmup_period         = warmup_period,
                            step_period           = 900,
                            render_episodes       = render,
                            log_dir               = log_dir)
    if case == 'C':
        env = BoptestGymEnvCustomReward(
                            url                   = url,
                            testcase              = testcase,
                            actions               = actions,
                            observations          = OrderedDict([('time',(0,604800)),
                                                     ('reaTZon_y',(280.,310.)),
                                                     ('PriceElectricPowerHighlyDynamic',(-0.4,0.4)),
                                                     ('LowerSetp[1]',(280.,310.)),
                                                     ('UpperSetp[1]',(280.,310.))]), 
                            predictive_period     = 3*3600, 
                            scenario              = {'electricity_price':'highly_dynamic'},
                            random_start_time     = True,
                            excluding_periods     = excluding_periods,
                            max_episode_length    = max_episode_length,
                            warmup_period         = warmup_period,
                            step_period           = 1800,
                            render_episodes       = render,
                            log_dir               = log_dir)
        
    if case == 'D':
        env = BoptestGymEnvCustomReward(
                            url                   = url,
                            testcase              = testcase,
                            actions               = actions,
                            observations          = observations, 
                            predictive_period     = 24*3600, 
                            regressive_period     = 6*3600, 
                            #scenario              = {'electricity_price':'highly_dynamic'},
                            scenario              = {'electricity_price':'dynamic'},
                            random_start_time     = True,
                            excluding_periods     = excluding_periods,
                            max_episode_length    = max_episode_length,
                            warmup_period         = warmup_period,
                            step_period           = 900,
                            render_episodes       = render,
                            log_dir               = log_dir)
    
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)  
    
    # Modify the environment to include the callback
    env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))
    
    if mode == 'train': 
        
        # Define RL agent
        if 'SAC' in algorithm:
            model = SAC('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed, 
                        learning_rate=5e-5, batch_size=96, ent_coef='auto',
                        buffer_size=365*96, learning_starts=96, train_freq=1,
                        tensorboard_log=log_dir)
    
        elif 'A2C' in algorithm:
            model = A2C('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed, 
                        learning_rate=1e-6, n_steps=4, ent_coef=0,
                        tensorboard_log=log_dir)
            
        elif 'DQN' in algorithm:
            env = DiscretizedActionWrapper(env,n_bins_act=10)
            model = DQN('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed, 
                        learning_rate=5e-4, batch_size=24, 
                        buffer_size=365*24, learning_starts=24, train_freq=1,
                        tensorboard_log=log_dir)
        
        #if expert_traj is not None:
            # Do not shuffle (randomize) to obtain deterministic result
        #    dataset = ExpertDataset(expert_path=expert_traj, randomize=False,
        #                            traj_limitation=1, batch_size=96)
        #    model.pretrain(dataset, n_epochs=1000)
        
        # Create the callback test and save the agent while training
        callback = SaveAndTestCallback(env, check_freq=1e10, save_freq=5e3,
                                       log_dir=log_dir, test=False)
        
        # set up logger
        new_logger = configure(log_dir, ['csv'])
        model.set_logger(new_logger)

        # Main training loop
        model.learn(total_timesteps=int(training_timesteps), callback=callback)
        # Save the agent
        model.save(os.path.join(log_dir,model_name))

        ###
        np.save(
            os.path.join(log_dir, "episode_rewards.npy"),
            callback.episode_rewards
        )

        np.save(
            os.path.join(log_dir, "episode_start_times.npy"),
            callback.episode_start_times
        )

        # Reward와 시작 시각의 개수가 같은지 확인
        if len(callback.episode_rewards) != len(callback.episode_start_times):
            raise RuntimeError(
                "Episode reward 개수와 start_time 개수가 다릅니다: "
                f"{len(callback.episode_rewards)} rewards, "
                f"{len(callback.episode_start_times)} start times"
            )

        start_time_sec = np.asarray(
            callback.episode_start_times,
            dtype=np.int64
        )

        # 0-based day index: 1월 1일 = day 0
        start_day = start_time_sec // (24 * 3600)

        # 표시용 날짜와 시각
        start_datetime = (
            pd.Timestamp("2021-01-01")
            + pd.to_timedelta(start_time_sec, unit="s")
        )

        df = pd.DataFrame({
            "episode": np.arange(
                1,
                len(callback.episode_rewards) + 1
            ),
            "reward": callback.episode_rewards,
            "start_time_sec": start_time_sec,
            "start_day": start_day,
            "start_datetime": start_datetime,
            "start_month": start_datetime.month,
            "start_date": start_datetime.strftime("%m-%d"),
            "start_hour": start_datetime.hour,
        })

        df.to_csv(
            os.path.join(log_dir, "episode_rewards.csv"),
            index=False
        )
        ###
        
    elif mode == 'load':
        # Load the trained agent
        if 'SAC' in algorithm:
            model = SAC.load(os.path.join(log_dir,model_name))
        elif 'A2C' in algorithm:
            model = A2C.load(os.path.join(log_dir,model_name))
        elif 'DQN' in algorithm:
            env = DiscretizedActionWrapper(env,n_bins_act=10)
            model = DQN.load(os.path.join(log_dir,model_name))
            
    elif mode == 'empty':
        model = None
    
    else:
        raise ValueError('mode should be either train, load, or empty')
    
    return env, model, start_time_tests, log_dir
        
def test_peak(env, model, start_time_tests, episode_length_test, 
              warmup_period_test, log_dir=os.getcwd(), model_name='last_model', 
              save_to_file=False, plot=False):
    ''' Perform test in peak heat period (February). 
    
    '''

    #observations, actions, rewards, kpis = test_agent(env, model, 
    observations, actions, rewards, kpis = test_agent_air(env, model, 
                                                      start_time=start_time_tests[0], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      log_dir=log_dir, model_name=model_name,
                                                      save_to_file=save_to_file,
                                                      plot=plot)
    return observations, actions, rewards, kpis

def test_typi(env, model, start_time_tests, episode_length_test, 
              warmup_period_test, log_dir=os.getcwd(), model_name='last_model', 
              save_to_file=False, plot=False):
    ''' Perform test in typical heat period (November)
    
    '''

    #observations, actions, rewards, kpis = test_agent(env, model, 
    observations, actions, rewards, kpis = test_agent_air(env, model, 
                                                      start_time=start_time_tests[1], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      log_dir=log_dir, model_name=model_name,
                                                      save_to_file=save_to_file,
                                                      plot=plot)
    return observations, actions, rewards, kpis


def test_peak_cool(env, model, start_time_tests, episode_length_test, 
              warmup_period_test, log_dir=os.getcwd(), model_name='last_model', 
              save_to_file=False, plot=False):
    ''' Perform test in typical heat period (November)
    
    '''

    observations, actions, rewards, kpis = test_agent_air(env, model, 
                                                      start_time=start_time_tests[2], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      log_dir=log_dir, model_name=model_name,
                                                      save_to_file=save_to_file,
                                                      plot=plot)
    return observations, actions, rewards, kpis


def test_typi_cool(env, model, start_time_tests, episode_length_test, 
              warmup_period_test, log_dir=os.getcwd(), model_name='last_model', 
              save_to_file=False, plot=False):
    ''' Perform test in typical heat period (November)
    
    '''

    observations, actions, rewards, kpis = test_agent_air(env, model, 
                                                      start_time=start_time_tests[3], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      log_dir=log_dir, model_name=model_name,
                                                      save_to_file=save_to_file,
                                                      plot=plot)
    return observations, actions, rewards, kpis


def test_mix(env, model, start_time_tests, episode_length_test, 
              warmup_period_test, log_dir=os.getcwd(), model_name='last_model', 
              save_to_file=False, plot=False):
    ''' Perform test in typical heat period (November)
    
    '''

    observations, actions, rewards, kpis = test_agent_air(env, model, 
                                                      start_time=start_time_tests[4], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      log_dir=log_dir, model_name=model_name,
                                                      save_to_file=save_to_file,
                                                      plot=plot)
    return observations, actions, rewards, kpis


if __name__ == "__main__":
    render = True
    plot = not render # Plot does not work together with render

    ###
    model_name='last_model'
    #model_name='model_150000'
    ###

    #123456, 199580, 173827, 101291, 362507
    for seed_no in [123456, 199580, 173827]:

        #for chk_point in [80000,150000,200000,220000,250000,300000]:
        #for chk_point in [100000]:
            #model_name = f'model_{chk_point}'
            
        seed = seed_no
        random.seed(seed)
        #env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', mode='train', case='A', training_timesteps=0, render=render)
        #env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', mode='load', case='B', training_timesteps=3e5, render=render)
        #env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', mode='load', case='C', training_timesteps=3e5, render=render)
        env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', mode='train', case='D', training_timesteps=3e5, render=render, model_name=model_name,seed_no=seed_no)
        print(log_dir,model_name)
    
        #env, model, start_time_tests, log_dir = train_RL(algorithm='A2C', mode='train', case='simple', training_timesteps=1, render=render, expert_traj=os.path.join('trajectories','expert_traj_cont_28.npz'))
        
        warmup_period_test  = 7*24*3600
        episode_length_test = 14*24*3600
        save_to_file = True
        #render = False
        #plot = False

        
        obs_peak, act_peak, rew_peak, kpis_peak = test_peak(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, model_name, save_to_file, plot)
        obs_typi, act_typi, rew_typi, kpis_typi = test_typi(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, model_name, save_to_file, plot)
        obs_cool, act_cool, rew_cool, kpis_cool = test_peak_cool(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, model_name, save_to_file, plot)
        obs_tyco, act_tyco, rew_tyco, kpis_tyco = test_typi_cool(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, model_name, save_to_file, plot)
        obs_mix,  act_mix,  rew_mix,  kpis_mix  = test_mix(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, model_name, save_to_file, plot)
        
        '''
        '''
        print("\n===== Peak Heat Test KPIs =====")
        for k, v in kpis_peak.items():
            print(f"{k}: {v}")

        print("\n===== Typical Heat Test KPIs =====")
        for k, v in kpis_typi.items():
            print(f"{k}: {v}")

        
        print("\n===== Peak Cool Test KPIs =====")
        for k, v in kpis_cool.items():
            print(f"{k}: {v}")

        print("\n===== Typical Cool Test KPIs =====")
        for k, v in kpis_tyco.items():
            print(f"{k}: {v}")

        '''
        '''
        print("\n===== Mix Test KPIs =====")
        for k, v in kpis_mix.items():
            print(f"{k}: {v}")

        print("\n===== Test Rewards =====")
        print(f"Peak heat total reward: {sum(rew_peak):.3f}")
        print(f"Typical heat total reward: {sum(rew_typi):.3f}")
        print(f"Peak cool total reward: {sum(rew_cool):.3f}")
        print(f"Typical cool total reward: {sum(rew_tyco):.3f}")
        print(f"Mix total reward: {sum(rew_mix):.3f}")