import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


composite_indicators_dict = {
    'PMVK': ['brutrevenue', 'weight'],
    'marginperkg': ['margin', 'weight'],
    'marginpercent': ['margin', 'brutrevenue'],
    'lineweight': ['weight', 'linecount'],
}

libs = {
    'margin': 'Marge (€)',
    'brutrevenue': 'CA brut (€)',
    'weight': 'Tonnage (kg)',
    'linecount': 'Nombre de lignes',
    'marginperkg': 'Marge au kilo (€/kg)',
    'marginpercent': 'Marge % (%)',
    'lineweight': 'Poids de la ligne (kg)',
    'PMVK': 'PMVK (€/kg)',
    'origin': 'Canal de commande',
    'origin2': 'Canal de commande',
    'margin_clt_zscore': 'Marge (€) - z-score',
    'brutrevenue_clt_zscore': 'CA brut (€) - z-score',
    'weight_clt_zscore': 'Tonnage (kg) - z-score',
    'linecount_clt_zscore': 'Nombre de lignes - z-score',
    'marginperkg_clt_zscore': 'Marge au kilo (€/kg) - z-score',
    'marginpercent_clt_zscore': 'Marge % (%) - z-score',
    'lineweight_clt_zscore': 'Poids de la ligne (kg) - z-score',
    'PMVK_clt_zscore': 'PMVK (€/kg) - z-score',
    'TV': 'Télévente',
    'VR': 'Vente route',
    'WEB': 'e-commerce',
    'EDI': 'EDI',
    '2BRE': '2BRE - Episaveurs Bretagne',
    '1ALO': '1ALO - PassionFroid Est',
    '1LRO': '1LRO - PassionFroid Languedoc-Roussillon',
    '1SOU': '1SOU - PassionFroid Sud-Ouest',
}


def lib(code):
    '''
    Function that returns the long description of a given code

    From `libs` dictionary. If the code is not in this dictionary, it returns
    the initial code.
    '''
    if code in libs:
        return(libs[code])
    else:
        return(code)


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
    return(df)


def process_df(df,
               orders_doctypes=['ZC10'],
               avoirs_doctypes=['ZA01', 'ZA02'],
               indicators=['margin', 'brutrevenue', 'weight'],
               grouper_fields=['orgacom', 'date', 'client', 'material'],
               ):
    before_processing = df[indicators].sum()
    mask_ZC = df.doctype.isin(orders_doctypes)
    mask_ZA = df.doctype.isin(avoirs_doctypes)
    raw_avoirs = df.loc[mask_ZA, grouper_fields + indicators].copy()
    avoirs = raw_avoirs.groupby(grouper_fields, observed=True).sum()
    mask_dup_ZC = (df.loc[mask_ZC]
                     .duplicated(grouper_fields, keep=False)
                     .rename('_duplicated'))
    mask_dup_ZC = mask_dup_ZC.reindex(df.index, fill_value=False)
    df = (df.merge(mask_dup_ZC, how='left', left_index=True, right_index=True))
    to_update = (
        df.loc[~df._duplicated & mask_ZC, grouper_fields + indicators]
        .merge(avoirs,
               how='inner',
               left_on=grouper_fields,
               right_index=True,
               validate='1:1')
    )
    for indicator in indicators:
        to_update[indicator] = (to_update[indicator + '_x'] +
                                to_update[indicator + '_y'])
    to_update = to_update.loc[(to_update.weight >= 0) &
                              (to_update.brutrevenue >= 0)]
    to_update.drop(columns=[indicator + '_x' for indicator in indicators] +
                           [indicator + '_y' for indicator in indicators],
                   inplace=True)
    mask_to_del = (
        df.set_index(grouper_fields)
          .index.isin(to_update.set_index(grouper_fields).index)
    )
    df = df.loc[~mask_to_del | ~df.doctype.isin(avoirs_doctypes)]
    merged = df.merge(to_update, on=grouper_fields, how='left', indicator=True)
    del(to_update)
    merged_mask_ZC = merged.doctype.isin(orders_doctypes)
    for indicator in indicators:
        merged.loc[:, indicator] = (
            merged[indicator + '_x'].where(merged[indicator + '_y'].isna() |
                                           ~merged_mask_ZC,
                                           merged[indicator + '_y'])
        )
        merged = merged.drop(columns=[indicator + '_x', indicator + '_y'])
    merged = merged.drop(columns='_merge')
    df = merged
    del(merged)
    after_processing = df[indicators].sum()
    delta = after_processing - before_processing
    print(f'Evolution des indicateurs pendant le traitement : \n{delta}')
    if max(delta.max(), abs(delta.min())) > .1:
        raise RuntimeError('Something bad happened during avoir processing')
    return(df)


def divide_round_up(n, d):
    return(int((n + (d - 1))/d))


def to_coords(i, nrows):
    return(i % nrows, i // nrows)


def compute_distribution(data=None,
                         indicators=None,
                         x=None,
                         hue=None,
                         percentile_selection=.99,
                         IQR_factor_selection=3.,
                         IQR_factor_plot=1.5,
                         ):
    lev_count = 1
    if x and hue:
        data = data.groupby([x, hue])
        lev_count = 3
    elif x:
        data = data.groupby(x)
        lev_count = 2
    lev_order = list(range(lev_count))
    lev_order = lev_order[-1:] + lev_order[:-1]

    stats = (data[indicators].describe(percentiles=[1 - percentile_selection,
                                                    .25,
                                                    .50,
                                                    .75,
                                                    percentile_selection])
                             .T
                             .unstack(0)
                             .reorder_levels(lev_order, axis=1)
                             .sort_index(axis=1)
             )

    # IQR = interquartile range (brute)
    stats.loc['IQR'] = stats.iloc[7] - stats.iloc[5]
    # minimum_selection => le "minimum" au sens de
    # 1er quartile - IQR * facteur de sélection
    stats.loc['minimum_selection'] = (stats.iloc[5] -
                                      IQR_factor_selection * stats.loc['IQR'])
    # maximum_selection => le "maximum" au sens de
    # 3eme quartile + IQR * facteur de sélection
    stats.loc['maximum_selection'] = (stats.iloc[7] +
                                      IQR_factor_selection * stats.loc['IQR'])
    # min_plot_selection, max_plot_selection =>
    # on prend au plus large entre la percentile selection et l'IQR selection
    stats.loc['min_plot_selection'] = stats.iloc[[4, 11]].min()
    stats.loc['max_plot_selection'] = stats.iloc[[8, 12]].max()
    # max_plot_range, min_plot_range =>
    # pour tracer les violons, uniquement de l'affichage
    if IQR_factor_plot is not None:
        stats.loc['minimum_plot_range'] = (stats.iloc[5] -
                                           IQR_factor_plot * stats.loc['IQR'])
        stats.loc['maximum_plot_range'] = (stats.iloc[7] +
                                           IQR_factor_plot * stats.loc['IQR'])
    return(stats)


def plot_distrib(data=None,
                 filter=None,
                 indicators=None,
                 ncols=1,
                 x=None,
                 order=None,
                 hue=None,
                 hue_order=None,
                 kind='violin',
                 percentile_selection=.99,
                 IQR_factor_selection=3.,
                 IQR_factor_plot=1.5,
                 show_means=False,
                 plot_kwargs=None,
                 ):
    '''
    Function that plot the distribution of some indicators (boxplot or violin)

    This function returns the figure and axes list in a `plt.subplots` fashion
    for further customization of the output (e.g. changing axis limits, labels,
    font sizes, ...)
    '''
    # filter the input dataset
    if filter is not None:
        data = data.reset_index().loc[filter.array]

    # convert grouping fields to categorical data type
    if x is not None:
        data[x] = data[x].astype(pd.CategoricalDtype(order, ordered=True))
    if hue is not None:
        data[hue] = data[hue].astype(pd.CategoricalDtype(hue_order,
                                                         ordered=True)
                                     )

    # compute means **before** filtering extreme data points
    if show_means:
        groupers = []
        if x:
            groupers.append(x)
        if hue:
            groupers.append(hue)

        means = (data.groupby(groupers, observed=True)
                     .size()
                     .rename('size')
                     .to_frame()
                 )
        means = means.join(
            data[groupers + indicators].groupby(groupers, observed=True).mean()
            )
        for indicator in indicators:
            if indicator in composite_indicators_dict:
                components = (data.loc[:, groupers +
                                       composite_indicators_dict[indicator]]
                                  .groupby(groupers)
                                  .sum()
                              )
                means[indicator] = (
                    components[composite_indicators_dict[indicator][0]] /
                    components[composite_indicators_dict[indicator][1]]
                    )
        # compute means abscissas
        means = means.reset_index()
        means['abscissa'] = (means[x].apply(lambda x: order.index(x))
                                     .astype('float'))
        if hue is not None:
            means['hue_num'] = (means[hue].apply(lambda x: hue_order.index(x))
                                          .astype('float'))
            width = 1 / (len(hue_order) + 1)
            means['hue_offset'] = (means['hue_num'] * width -
                                   (len(hue_order) - 1) * width / 2
                                   )
            means['abscissa'] = means['abscissa'] + means['hue_offset']
        means = means.set_index(groupers)

    # compute the distribution for further use
    stats = compute_distribution(data=data,
                                 indicators=indicators,
                                 x=x,
                                 hue=hue,
                                 percentile_selection=percentile_selection,
                                 IQR_factor_selection=IQR_factor_selection,
                                 IQR_factor_plot=IQR_factor_plot,
                                 )

    # if plot filter is set up, then compute it
    agg_dict = dict()
    if IQR_factor_plot is not None:
        agg_dict = {**agg_dict,
                    'minimum_plot_range': 'min',
                    'maximum_plot_range': 'max',
                    }
    agg_dict = {**agg_dict,
                'min_plot_selection': 'min',
                'max_plot_selection': 'max',
                }
    # if boxplot and selection parameters are provided, ignore them and
    # raise a warning
    if kind == 'boxplot' and (percentile_selection is not None or
                              IQR_factor_selection is not None):
        warnings.warn('Selection parameters have been given for'
                      'a boxplot. They will be ignored.')

    plot_ranges = stats.T.groupby(level=0, axis=0).agg(agg_dict)

    # the plotting part
    nrows = divide_round_up(len(indicators), ncols)
    fig, axs = plt.subplots(nrows=nrows,
                            ncols=ncols,
                            figsize=(15, 8 * nrows),
                            squeeze=False,
                            )
    if plot_kwargs is None:
        plot_kwargs = dict()
    if kind == 'violin':
        defaults = {
            'inner': 'quartile',
            'cut': 0.,
        }
    if kind == 'boxplot':
        defaults = {
            'whis': 0.,
            'showfliers': False,
        }
    plot_kwargs = {**defaults, **plot_kwargs}

    for i, indicator in enumerate(indicators):
        ax = axs[to_coords(i, nrows)]
        if kind == 'violin':
            cols = [indicator]
            if x:
                cols.append(x)
            if hue:
                cols.append(hue)
            to_plot = data.loc[(data[indicator] >=
                                plot_ranges.loc[indicator,
                                                'min_plot_selection']) &
                               (data[indicator] <=
                                plot_ranges.loc[indicator,
                                                'max_plot_selection']),
                               cols
                               ]
            sns.violinplot(data=to_plot,
                           y=indicator,
                           x=x,
                           hue=hue,
                           ax=ax,
                           **plot_kwargs,
                           )
        if kind == 'boxplot':
            sns.boxplot(data=data,
                        y=indicator,
                        x=x,
                        hue=hue,
                        ax=ax,
                        **plot_kwargs,
                        )
        if show_means:
            sns.scatterplot(data=means,
                            x='abscissa',
                            y=indicator,
                            marker='s',
                            color='k',
                            ax=ax)

        # Customize labels
        ax.set_xlabel(lib(x), fontsize=14)
        ax.set_ylabel(lib(indicator), fontsize=14)
        ticklabels = [lib(label.get_text())
                      for label in ax.get_xticklabels()]
        ax.set_xticklabels(ticklabels, fontsize=12)

        if IQR_factor_plot is not None:
            ax.set_ylim(plot_ranges.loc[indicator, 'minimum_plot_range'],
                        plot_ranges.loc[indicator, 'maximum_plot_range'])

    return(fig, axs)
