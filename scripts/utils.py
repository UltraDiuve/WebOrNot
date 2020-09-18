import numpy as np


def compute_zscores(df, grouper_fields, indicators, suffix='_clt_zscore'):
    '''
    Function that computes z-scores on a dataset based on groups
    '''
    grouped = df.groupby(grouper_fields, observed=True)
    for indicator in indicators:
        mean_ds = grouped[indicator].transform('mean')
        std_ds = grouped[indicator].transform('std')
        df[indicator + suffix] = (((df[indicator] - mean_ds) / std_ds)
                                  .replace([np.inf, -np.inf], np.nan)
                                  .fillna(0.))
        del(mean_ds)
        del(std_ds)
        print(f'{indicator} done!')
    print('ALL DONE!')

