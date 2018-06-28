
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import lib


def plot_heatmap(save_filename=None):

    list_ = lib.load_pickle('../data/live_stream_predictions/total_stream.pkl')

    df = pd.DataFrame(list_, columns=['Timestamp', 'Prediction'])

    df['Hour'] = df.Timestamp.apply(lambda t: t.hour)
    df['Minute'] = df.Timestamp.apply(lambda t: t.minute)
    df['Day'] = df.Timestamp.apply(lambda t: t.day)
    df['Month'] = df.Timestamp.apply(lambda t: t.month)

    dfg = (df.groupby(['Month', 'Day', 'Hour', 'Minute'], as_index=False)
           ['Prediction'].sum())

    dfg['Min_cried'] = dfg['Prediction'] >= 2
    dfg['Hour_frac'] = dfg['Hour'] + dfg['Minute']/60

    dfg_hr = dfg.groupby(['Month', 'Day', 'Hour'],
                         as_index=False)['Min_cried'].sum()

    merge_frame = pd.DataFrame(np.concatenate(([np.ones((24,), dtype=int)*6],
                                               [np.ones((24,), dtype=int)*22],
                                               [np.arange(24)]), axis=0)
                               .transpose(),
                               columns=['Month', 'Day', 'Hour'])

    pivoted = (pd.merge(dfg_hr,
                        merge_frame,
                        how='outer',
                        on=['Month', 'Day', 'Hour'])
               .pivot('Day', 'Hour', 'Min_cried')
               .fillna(0))

    labels = (pivoted[pivoted > 3]
              .fillna(0)  # filter
              .astype(int)
              .astype(str)
              .replace('0', '')
              .values
              )

    # labels = (pivoted[pivoted > 3]   # filter
    #           .astype(str)
    #           .replace('nan', '', regex=True)
    #           .replace('.0', '', regex=True)
    #           .values
    #           )

    with sns.plotting_context("poster"):
        plt.figure(figsize=(20, 6))

        # plt.title('Minutes cried')

        g = sns.heatmap(pivoted,
                        annot=labels,
                        fmt="",
                        linewidths=5,
                        cmap=sns.cubehelix_palette(5),
                        cbar_kws=dict(use_gridspec=False, location="top"))

        g.set_yticklabels(['Fri', 'Sat', 'Sun', 'Mon', 'Tue', 'Wed', ]
                          , rotation=0, fontsize=25)

        g.set_xticks(range(25))  # 25 to label the interstices.
        g.set_xticklabels((['mid-\nnight\n']
                           + [str(t)+'a' for t in list(range(1, 12))]
                           + ['noon']
                           + [str(t)+'p' for t in list(range(1, 12))]
                           + ['mid-\nnight\n']),
                          rotation=0, fontsize=20,)

        g.set_xlabel('Time of Day')

        fig = g.get_figure()

        if save_filename is not None:
            fig.savefig('../docs/' + save_filename)
