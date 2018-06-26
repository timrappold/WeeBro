
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import lib


# % matplotlib inline

list_ = lib.load_pickle('../data/live_stream_predictions/total_stream.pkl')

df = pd.DataFrame(list_, columns=['Timestamp','Prediction'])

df['Hour'] = df.Timestamp.apply(lambda t: t.hour)
df['Minute'] = df.Timestamp.apply(lambda t: t.minute)
df['Day'] = df.Timestamp.apply(lambda t: t.day)
df['Month'] = df.Timestamp.apply(lambda t: t.month)


dfg = df.groupby(['Month', 'Day', 'Hour', 'Minute'], as_index=False
                 )['Prediction'].sum()

dfg['Min_cried'] = dfg['Prediction'] >= 2
dfg['Hour_frac'] = dfg['Hour'] + dfg['Minute']/60

dfg_hr = dfg.groupby(['Month', 'Day', 'Hour'], as_index=False)['Min_cried'].sum()

merge_frame = pd.DataFrame(
    np.concatenate(([np.ones((24,), dtype=int)*6],
                    [np.ones((24,), dtype=int)*22],
                    [np.arange(24)]), axis=0).transpose(),
    columns=['Month', 'Day', 'Hour'])


pivoted = pd.merge(dfg_hr,
                   merge_frame,
                   how='outer',
                   on=['Month', 'Day', 'Hour']).pivot('Day', 'Hour', 'Min_cried').fillna(0)


#sns.set_context(context='talk')


#sns.heatmap(pivoted, annot=True, fmt="g", linewidths=.5, cmap='Dark2_r')

with sns.plotting_context("talk", rc={'fontname':'Helvetica'}):
    #rc = {'fontname': 'Helvetica'}
    plt.figure(figsize=(12, 8))

    plt.title('Minutes cried')

    g = sns.heatmap(pivoted, annot=True, fmt="g", linewidths=5,
                    cmap=sns.cubehelix_palette(5))

    g.set_yticklabels(['Fri', 'Sat', 'Sun', ], rotation=0)

    g.set_xticklabels(list(range(12)) + ['noon'] + list(range(1, 12)),
                      rotation=0)

    g.set_xlabel('Time of Day')

    fig = g.get_figure()

    fig.savefig('../docs/heatmap.png')


