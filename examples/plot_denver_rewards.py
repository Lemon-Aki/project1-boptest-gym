from pathlib import Path
from io import BytesIO
import re, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
INPUTS = {'all': BASE/'test_result_data/최종 사용 데이터들/episode_rewards 파일(denver)/denver_all_episode_rewards.zip',
          'sum': BASE/'test_result_data/최종 사용 데이터들/episode_rewards 파일(denver)/denver_summer_episode_rewards.zip',
          'win': BASE/'test_result_data/최종 사용 데이터들/episode_rewards 파일(denver)/denver_winter_episode_rewards.zip'}
OUTPUT = BASE/'denver_reward_graphs'
OUTPUT.mkdir(exist_ok=True)
STEPS_PER_EPISODE = 672  # 14 days x 24 hours x 2 steps/hour
ROLLING_WINDOW = 25
ZOOM_START = 75_000

records=[]
for model, path in INPUTS.items():
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.csv'): continue
            found=re.search(r'_34_(\d+-1)_(\d+)_300k_', name)
            if found is None:
                print('SKIPPED unrecognized filename:', name)
                continue
            weight, seed = found.groups()
            df=pd.read_csv(BytesIO(archive.read(name)))
            df['episode']=pd.to_numeric(df['episode'],errors='coerce')
            df['reward']=pd.to_numeric(df['reward'],errors='coerce')
            df=df.dropna(subset=['episode','reward']).sort_values('episode')
            df['step']=df['episode']*STEPS_PER_EPISODE
            df['model']=model; df['weight']=weight; df['seed']=seed
            records.append(df[['model','weight','seed','episode','step','reward']])
data=pd.concat(records,ignore_index=True)
data.to_csv(OUTPUT/'reward_data_combined.csv',index=False)

# Plot each model in a 2x2 grid (four reward weightings); each panel shows seeds and the across-seed mean.
for model in INPUTS:
    weights=sorted(data.loc[data.model==model,'weight'].unique(),key=lambda w:int(w.split('-')[0]))
    for start, suffix in [(0,'full'),(ZOOM_START,'after_75k')]:
        fig,axs=plt.subplots(2,2,figsize=(15,9),sharex=True)
        for ax,weight in zip(axs.flat, weights):
            part=data[(data.model==model)&(data.weight==weight)]
            seed_smoothed=[]
            for seed,grp in part.groupby('seed'):
                grp=grp.sort_values('step')
                smooth=grp.reward.rolling(ROLLING_WINDOW,min_periods=1).mean()
                ax.plot(grp.step,smooth,alpha=.45,lw=1.2,label=f'Seed {seed}')
                seed_smoothed.append(pd.Series(smooth.to_numpy(),index=grp.step.to_numpy(),name=seed))
            mean=pd.concat(seed_smoothed,axis=1).mean(axis=1).sort_index()
            ax.plot(mean.index,mean.values,'k--',linewidth=2.3,label='Seed mean')
            ax.set_title(f'{model.upper()} | Cost:Tdis = {weight.replace("-", ":")}')
            ax.grid(alpha=.25)
            ax.set_xlabel('Training steps');ax.set_ylabel('Episode reward (rolling mean)')
            ax.set_xlim(start,300000)
            ax.legend(fontsize=8)
        fig.suptitle(f'Denver {model.upper()} | Reward learning curves | {suffix}',fontsize=15)
        fig.tight_layout()
        fig.savefig(OUTPUT/f'{model}_{suffix}.png',dpi=170,bbox_inches='tight')
        plt.close(fig)

# Smoothed mean reward for 3 models under common reward weightings.
for weight in ['8-1','16-1','26-1']:
    fig,ax=plt.subplots(figsize=(11,5))
    for model in INPUTS:
        part=data[(data.model==model)&(data.weight==weight)]
        means=[]
        for seed,grp in part.groupby('seed'):
            grp=grp.sort_values('step')
            means.append(pd.Series(grp.reward.rolling(ROLLING_WINDOW,min_periods=1).mean().to_numpy(),index=grp.step.to_numpy()))
        avg=pd.concat(means,axis=1).mean(axis=1).sort_index()
        ax.plot(avg.index,avg.values,label=model.upper())
    ax.set_xlim(0,300000);ax.set_xlabel('Training steps');ax.set_ylabel('Episode reward (smoothed seed mean)')
    ax.set_title(f'Denver | Common weight {weight.replace("-", ":")} | NOT identical training distributions')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(OUTPUT/f'compare_models_{weight}.png',dpi=170,bbox_inches='tight')
    plt.close(fig)

# Summarize late-phase reward means and drift without claiming policy convergence.
rows=[]
for (model,weight,seed),part in data.groupby(['model','weight','seed']):
    part=part.sort_values('step')
    def mean_for(lo,hi):
        x=part[(part.step>lo)&(part.step<=hi)].reward
        return float(x.mean()) if len(x) else np.nan
    rows.append({'model':model,'weight':weight,'seed':seed,'episode_count':len(part),
                 'max_step_estimate':int(part.step.max()),
                 'mean_reward_150k_200k':mean_for(150000,200000),
                 'mean_reward_200k_250k':mean_for(200000,250000),
                 'mean_reward_250k_300k':mean_for(250000,300000)})
summary=pd.DataFrame(rows)
summary.to_csv(OUTPUT/'late_training_summary.csv',index=False)
print('Saved to',OUTPUT)
print(summary.to_string(index=False))
