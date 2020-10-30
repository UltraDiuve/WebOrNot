import numpy as np
import warnings
import pandas as pd
from pandas import IndexSlice as idx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import seaborn as sns
from pathlib import Path
from functools import partial
from collections import namedtuple
from IPython.display import display
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FactorRange, Range1d, Span, Button
from bokeh.models import TextInput, Div
from bokeh.models.widgets import Select, DatePicker, CheckboxGroup
from bokeh.models.annotations import BoxAnnotation
from bokeh.transform import factor_cmap
from bokeh.layouts import row, column
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models.tickers import MonthsTicker
from math import pi
from datetime import date
from datetime import datetime
now = datetime.now


mcolorpalette = list(mcolor.TABLEAU_COLORS.values())

# formula is:
# composite indicator = component[0] / component[1]
composite_indicators_dict = {
    'PMVK': ['brutrevenue', 'weight'],
    'marginperkg': ['margin', 'weight'],
    'marginpercent': ['margin', 'brutrevenue'],
    'lineweight': ['weight', 'linecount'],
    'lineperorder': ['linecount', 'ordercount'],
}

libs = {
    'orgacom': 'Succursale',
    'margin': 'Marge (€)',
    'brutrevenue': 'CA brut (€)',
    'weight': 'Tonnage (kg)',
    'linecount': 'Nombre de lignes',
    'marginperkg': 'Marge au kilo (€/kg)',
    'marginpercent': 'Marge % (%)',
    'lineweight': 'Poids de la ligne (kg)',
    'lineperorder': 'Nombre de lignes par commande',
    'PMVK': 'PMVK (€/kg)',
    'size': 'Nb commandes',
    'origin': 'Canal de commande',
    'origin2': 'Canal de commande',
    'brutrevenue_glob': 'CA brut total (€)',
    'margin_glob': 'Marge totale (€)',
    'weight_glob': 'Tonnage total (kg)',
    'linecount_glob': 'Nombre de lignes total',
    'origin2_lib': 'Canal de commande',
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
    'PPF': 'PassionFroid',
    'PES': 'EpiSaveurs',
    '2BRE': '2BRE - Episaveurs Bretagne',
    '1ALO': '1ALO - PassionFroid Est',
    '1LRO': '1LRO - PassionFroid Languedoc-Roussillon',
    '1SOU': '1SOU - PassionFroid Sud-Ouest',
    'seg1': 'Segmentation niveau 1',
    'seg2': 'Segmentation niveau 2',
    'seg3': 'Segmentation niveau 3',
    'seg4': 'Segmentation niveau 4',
    'seg1_lib': 'Segmentation niveau 1',
    'seg2_lib': 'Segmentation niveau 2',
    'seg3_lib': 'Segmentation niveau 3',
    'seg4_lib': 'Segmentation niveau 4',
}

formats = {
    'weight': '{:.2f} kg',
    'margin': '{:.2f} €',
    'brutrevenue': '{:.2f} €',
    'linecount': '{:.2f}',
    'PMVK': '{:.2f} €/kg',
    'marginperkg': '{:.2f} €/kg',
    'marginpercent': '{:.2%}',
    'lineweight': '{:.2f} kg',
    'lineperorder': '{:.2f}',
    'weight_clt_zscore': '{:.3f}',
    'margin_clt_zscore': '{:.3f}',
    'brutrevenue_clt_zscore': '{:.3f}',
    'linecount_clt_zscore': '{:.3f}',
    'PMVK_clt_zscore': '{:.3f}',
    'marginperkg_clt_zscore': '{:.3f}',
    'marginpercent_clt_zscore': '{:.3f}',
    'lineweight_clt_zscore': '{:.3f}',
}

colormaps = {
    'seg1': {
        'Z1': 'blue',
        'Z2': 'red',
        'Z3': 'green',
        'Z4': 'orange',
    },
    'seg3': {
        'ZI': mcolorpalette[4],
        'ZJ': mcolorpalette[5],
        'ZK': mcolorpalette[6],
        'ZL': 'orange',
    },
    'origin2': {
        'TV': mcolorpalette[0],
        'VR': mcolorpalette[1],
        'WEB': mcolorpalette[2],
        'EDI': mcolorpalette[3],
    }
}

labeled_bins = namedtuple('labeled_bins', ['labels', 'bin_limits'])
def_bins = labeled_bins(
    labels=['no_web', 'adopt', 'exploit', 'exclusive'],
    bin_limits=[0., .2, .5, .9, 1.001],
)
bin_colors = {
    'no_web': 'green',
    'inactive': 'black',
    'adopt': 'yellow',
    'exploit': 'pink',
    'exclusive': 'red',
}

# The lines below might require improvement (with __file__ or smth else)
path = Path('..') / 'data' / 'libelles_segments.csv'
lib_seg = pd.read_csv(path,
                      sep=';',
                      encoding='latin1',
                      header=None,
                      names=['level', 'code', 'designation'],
                      index_col=['level', 'code']
                      )
transco = dict()
for i in range(1, 7):
    transco['seg' + str(i)] = (
        lib_seg.loc[idx[i, :, :]]
               .reset_index()
               .set_index('code')['designation']
    )


def seg_lib(code, level):
    '''
    Returns the designation of segment with code code and at level level
    '''
    try:
        return(transco[level][code])
    except KeyError:
        return(transco['seg' + str(level)][code])


def domain_lib(code, domain):
    '''
    Returns the designation of code code in domain domain
    '''
    if domain[:3] == 'seg':
        return(seg_lib(code, domain))
    else:
        raise KeyError(f'Unknown domain: {domain}')


def lib(code, domain=None):
    '''
    Function that returns the long description of a given code

    From `libs` dictionary. If the code is not in this dictionary, it returns
    the initial code.
    '''
    if domain is not None:
        try:
            return(domain_lib(code, domain))
        except KeyError:
            pass
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

    # definitions:
    # stats.iloc[4] =  1% (example, 1 - percentile selection)
    # stats.iloc[5] = 25%
    # stats.iloc[7] = 75%
    # stats.iloc[8] = 99% (example, percentile selection)
    # IQR = interquartile range (brut)
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
        stats.loc['minimum_plot_range'] = pd.concat([
            stats.loc['minimum_plot_range'], stats.loc['min']
        ], axis=1).max(axis=1)

        stats.loc['maximum_plot_range'] = (stats.iloc[7] +
                                           IQR_factor_plot * stats.loc['IQR'])
        stats.loc['maximum_plot_range'] = pd.concat([
            stats.loc['maximum_plot_range'], stats.loc['max']
        ], axis=1).min(axis=1)
    return(stats)


def compute_composite_indicators(data=None,
                                 indicator_defs=composite_indicators_dict):
    df = pd.DataFrame()
    for composite_indicator, components in indicator_defs.items():
        try:
            df[composite_indicator] = (
                data[components[0]] / data[components[1]]
            )
        except KeyError:
            pass
    return(pd.concat([data, df], axis=1))


def compute_means(data=None,
                  groupers=None,
                  indicators=None,
                  ):
    '''
    Computes size of groups and correct means for composite indicators
    '''
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
    return(means)


def pretty_means(data=None,
                 groupers=None,
                 indicators=None,
                 formats=formats,
                 translate=['columns'],
                 ):
    formatted = compute_means(data=data,
                              groupers=groupers,
                              indicators=indicators,
                              )
    formatted = formatted.reset_index()
    for to_translate in translate:
        try:
            fun = partial(lib, domain=to_translate)
            formatted[to_translate] = formatted[to_translate].map(fun)
        except KeyError:
            pass
    if 'columns' in translate:
        formats = {libs[col_code]: format_
                   for col_code, format_ in formats.items()}
        formatted = formatted.rename(libs, axis=1)

    formatted = (
        formatted.style
                 .format(formats)
                 .hide_index()
    )
    display(formatted)


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
                 IQR_factor_plot=None,
                 show_means=False,
                 plot_kwargs=None,
                 translate=True,
                 fontsizes_kwargs=None,
                 ):
    '''
    Function that plot the distribution of some indicators (boxplot or violin)

    This function returns the figure and axes list in a `plt.subplots` fashion
    for further customization of the output (e.g. changing axis limits, labels,
    font sizes, ...)
    translate : if True, translate every code to long labels. Else, pass an
    iterable of 'x', 'xaxis' or 'indicator'. 'hue' not yet implemented.
    '''
    # filter the input dataset
    if filter is not None:
        data = data.reset_index().loc[filter.array]

    # if orders are not given, infer them from data
    if x is not None and order is None:
        order = list(data[x].unique())
    if hue is not None and hue_order is None:
        hue_order = list(data[hue].unique())

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

        means = compute_means(data=data,
                              groupers=groupers,
                              indicators=indicators
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
        warnings.warn('Selection parameters have been given for '
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
        fontsizes_def = {
            'xaxis': 14,
            'indicator': 14,
            'x': 12,
        }
        if fontsizes_kwargs is None:
            fontsizes_kwargs = dict()
        fontsizes_kwargs = {**fontsizes_def, **fontsizes_kwargs}

        if translate == True or 'xaxis' in translate:  # noqa: E712
            ax.set_xlabel(lib(x), fontsize=fontsizes_kwargs['xaxis'])
        else:
            ax.set_xlabel(x, fontsize=fontsizes_kwargs['xaxis'])

        if translate == True or 'indicator' in translate:  # noqa: E712
            ax.set_ylabel(lib(indicator),
                          fontsize=fontsizes_kwargs['indicator'])
        else:
            ax.set_ylabel(indicator, fontsize=fontsizes_kwargs['indicator'])

        if translate == True or 'x' in translate:  # noqa: E712
            ticklabels = [lib(label.get_text(), domain=x)
                          for label in ax.get_xticklabels()]
        else:
            ticklabels = [label.get_text()
                          for label in ax.get_xticklabels()]
        ax.set_xticklabels(ticklabels, fontsize=fontsizes_kwargs['x'])

        if IQR_factor_plot is not None:
            ax.set_ylim(plot_ranges.loc[indicator, 'minimum_plot_range'],
                        plot_ranges.loc[indicator, 'maximum_plot_range'])

    return(fig, axs)


def bk_histo_seg(doc,
                 source_df=None,
                 segs=None,
                 filters=None,
                 filters_exclude=None,
                 plot_kwargs=None,
                 ):
    '''
    Bokeh server app that enables to draw stacked bar plot on segmentation
    '''

    if plot_kwargs is None:
        plot_kwargs = dict()

    # define controls
    # select: choose indicator from list
    indicator_map = {
        'Marge (€)': 'margin',
        'CA brut (€)': 'brutrevenue',
        'Tonnage (kg)': 'weight',
    }
    select = Select(title="Indicateur",
                    options=list(indicator_map),
                    value=list(indicator_map)[1])
    # datepickers : filter data on date
    min_date, max_date = date(2017, 7, 3), date(2020, 8, 30)
    min_def_date, max_def_date = date(2019, 1, 1), date(2019, 12, 31)
    datepickers = [
        DatePicker(title='Date de début',
                   value=min_def_date,
                   min_date=min_date,
                   max_date=max_date),
        DatePicker(title='Date de fin',
                   value=max_def_date,
                   min_date=min_date,
                   max_date=max_date),
    ]

    controls = [select, *datepickers]

    # compute data source
    origins = ['TV', 'VR', 'WEB', 'EDI']

    def compute_indicator(df, indicator):
        temp = (
            df.groupby(['origin2'] + segs + ['orgacom'],
                       observed=True,)[indicator]
              .sum()
              .unstack('origin2', fill_value=0.)
              .reindex(columns=origins)
              .reset_index()
        )
        for seg in segs:
            temp[seg] = temp[seg].map(transco[seg])
        temp = temp.set_index(segs + ['orgacom'])
        return(temp)

    def select_data():
        date_range = [datepickers[0].value, datepickers[1].value]
        selected = source_df.loc[
            (source_df.date >= pd.to_datetime(date_range[0])) &
            (source_df.date <= pd.to_datetime(date_range[1]))
        ]
        for attribute, filter_values in filters.items():
            selected = selected.loc[
                selected[attribute].isin(filter_values)
            ]
        for attribute, filter_values in filters_exclude.items():
            selected = selected.loc[
                ~selected[attribute].isin(filter_values)
            ]
        return(selected)

    def update():
        indicator = indicator_map[select.value]
        df = select_data()
        grouped = compute_indicator(df, indicator)
        source.data = ColumnDataSource.from_df(grouped)

    for control in controls:
        control.on_change('value', lambda attr, old, new: update())

    df = compute_indicator(select_data(), indicator_map[select.value])
    source = ColumnDataSource(data=df)

    p = figure(
        x_range=FactorRange(*list(df.index)),
        plot_width=900,
        **plot_kwargs,
        )
    p.vbar_stack(df.columns,
                 x='_'.join(segs) + '_orgacom',
                 source=source,
                 width=.9,
                 color=list(mcolor.TABLEAU_COLORS.values())[:len(df.columns)],
                 legend_label=list(df.columns),
                 )
    p.xaxis.major_label_orientation = 1
    p.yaxis.formatter = NumeralTickFormatter(format="0")

    doc.add_root(column(select, row(*datepickers), p))


def bk_bubbles(doc, data=None, filters=None):
    max_size = 50
    # line_width = 2.5
    plot_indicators = ['brutrevenue', 'margin', 'weight', 'linecount']
    select_indicators = (
        [indicator + '_glob' for indicator in plot_indicators] +
        plot_indicators +
        [
            'PMVK',
            'marginperkg',
            'marginpercent',
            'lineweight',
        ]
    )
    plot_analysis_axes = ['seg3', 'origin2']
    hover_fields = [
        'margin',
        'brutrevenue',
        'marginperkg',
        'marginpercent',
        'lineperorder',
    ]
    axis_formatters = {indicator: '0' for indicator in select_indicators}
    axis_formatters = {
        **axis_formatters,
        **{
            'marginperkg': '0.0',
            'marginpercent': '0%',
        },
        }
    # jsformats_tooltips = {
    #         'seg3_lib': '',
    #         'origin2_lib': '',
    #         'margin': '{0.00 a}€',
    #         'weight': '{0.00 a}kg',
    #         'brutrevenue': '{0.00 a}€',
    #         'marginperkg': '{0.000} €/kg',
    #         'marginpercent': '{0.00 %}',
    #         'lineperorder': '{0.00}',
    # }

    # x = 'weight'
    # y = 'margin'
    # line_color = 'seg3_c'
    # fill_color = 'origin2_c'
    # size = 'weight_s'

    # Widgets definition
    # axes customization
    select_x = Select(
        title="Axe x",
        options=list(zip(
            select_indicators,
            list(map(lib, select_indicators))
        )),
        value=select_indicators[0],
        width=200,
        sizing_mode='stretch_width',
        )
    select_y = Select(
        title="Axe y",
        options=list(zip(
            select_indicators,
            list(map(lib, select_indicators))
        )),
        value=select_indicators[1],
        width=200,
        sizing_mode='stretch_width',
        )
    select_size = Select(
        title="Taille des bulles",
        options=list(zip(
            ['constant'] + select_indicators,
            ['Constante'] + list(map(lib, select_indicators))
        )),
        value=select_indicators[2],
        width=200,
        sizing_mode='stretch_width',
        )
    axes_widgets = [select_x, select_y, select_size]
    for control in axes_widgets:
        control.on_change('value', lambda attr, old, new: update_CDS())
    axes_widgets = column(*axes_widgets)

    running = False

    def change_group(old, new, which_control):
        nonlocal running
        if running:
            return
        running = True
        if which_control == 'bubble':
            other = select_line
        if which_control == 'line':
            other = select_bubble
        if new == other.value:
            other.value = old
            update_CDS()
        else:
            update_dataframe()
        running = False

    # grouping customization
    select_bubble = Select(
        title="Couleur des bulles",
        options=list(zip(
            plot_analysis_axes,
            list(map(lib, plot_analysis_axes))
        )),
        value=plot_analysis_axes[1],
        width=200,
        sizing_mode='stretch_width',
        )
    select_line = Select(
        title="Groupes de bulles",
        options=[
            ('None', 'Aucun'),
            *list(zip(
                plot_analysis_axes,
                list(map(lib, plot_analysis_axes))))
        ],
        value=plot_analysis_axes[0],
        width=200,
        sizing_mode='stretch_width',
        )
    groups_widgets = [select_bubble, select_line]
    select_bubble.on_change(
        'value',
        lambda attr, old, new: change_group(old, new, 'bubble')
        )
    select_line.on_change(
        'value',
        lambda attr, old, new: change_group(old, new, 'line')
        )
    # for control in groups_widgets:
    #     control.on_change('value', lambda attr, old, new: update_dataframe())
    groups_widgets = column(groups_widgets)

    # filter customization
    filter_labels = ['Uniquement RHD', "Retirer O'Tacos"]
    select_filters = CheckboxGroup(labels=filter_labels,
                                   active=[0],
                                   )
    min_date, max_date = date(2017, 7, 3), date(2020, 8, 30)
    min_def_date, max_def_date = date(2019, 1, 1), date(2019, 12, 31)
    datepickers = [
        DatePicker(title='Date de début',
                   value=min_def_date,
                   min_date=min_date,
                   max_date=max_date,
                   width=150,
                   sizing_mode='stretch_width',
                   ),
        DatePicker(title='Date de fin',
                   value=max_def_date,
                   min_date=min_date,
                   max_date=max_date,
                   width=150,
                   sizing_mode='stretch_width',
                   ),
    ]
    filter_widgets = column(select_filters, row(*datepickers))

    def select_data():
        # aggregate data
        groupers = [select_bubble.value]
        if select_line.value != 'None':
            groupers.append(select_line.value)
        to_plot = (
            data
            .loc[filters]
            .groupby(groupers, observed=True)[plot_indicators]
            .sum()
        )
        to_plot = to_plot.rename(
            {indicator: indicator + '_glob' for indicator in plot_indicators},
            axis=1,
        )
        sizes = (
            data
            .loc[filters]
            .groupby(groupers, observed=True)
            .size()
            .rename('ordercount')
        )
        to_plot = to_plot.join(sizes).reset_index()
        del(sizes)
        for indicator in plot_indicators:
            try:
                to_plot[indicator] = (
                    to_plot[indicator + '_glob'] / to_plot['ordercount']
                )
            except KeyError:
                pass

        # compute composite indicators
        # for indicator, components in utils.composite_indicators_dict:
        to_plot = compute_composite_indicators(
            to_plot,
            indicator_defs={
                **composite_indicators_dict,
                **{'lineperorder': ['linecount_glob', 'ordercount']},
            }
        )

        # compute designations
        for code in groupers:
            if (len(code) == 4) & (code[:3] == 'seg'):
                fun = partial(lib, domain='seg' + code[3])
            else:
                fun = lib
            to_plot[code + '_lib'] = to_plot[code].apply(fun)

        # compute sizes
        numeric_cols = (to_plot.select_dtypes(include=[np.number]).columns
                        .tolist())
        for indicator in numeric_cols:
            to_plot[indicator + '_s'] = to_plot[indicator] ** 0.5
            to_plot[indicator + '_s'] = (to_plot[indicator + '_s'] * max_size /
                                         to_plot[indicator + '_s'].max())
        to_plot['constant_s'] = 20

        # compute colors
        for axis, colors in colormaps.items():
            try:
                to_plot[axis + '_c'] = to_plot[axis].map(colors)
            except KeyError:
                pass
        return(to_plot)

    source_cols = dict(
        x=[],
        y=[],
        line_color=[],
        fill_color=[],
        size=[],
        hover_field1=[],
        hover_field2=[],
        fill_lib=[],
    )
    hover_fields_cols = {hover_field: [] for hover_field in hover_fields}
    source_cols = {**source_cols, **hover_fields_cols}
    source = ColumnDataSource(data=source_cols)
    line_CDS = ColumnDataSource(dict(
        xs=[],
        ys=[],
        color=[],
        line_lib=[],
    ))
    to_plot = select_data()
    with pd.option_context('display.max_columns', None):
        display(to_plot)

    def update_CDS():
        global to_plot
        if select_line.value != 'None':
            to_plot = to_plot.sort_values([select_line.value,
                                           select_x.value])
            # line_color = to_plot[select_line.value + '_c']
            line_axes = to_plot[select_line.value].unique()
            line_CDS.data = dict(
                xs=[to_plot.loc[to_plot[select_line.value] == axis,
                                select_x.value]
                    for axis in line_axes],
                ys=[to_plot.loc[to_plot[select_line.value] == axis,
                                select_y.value]
                    for axis in line_axes],
                color=[colormaps[select_line.value][axis]
                       for axis in line_axes],
                line_lib=[lib(axis, domain=select_line.value)
                          for axis in line_axes],
                )
        else:
            to_plot = to_plot.sort_values(select_x.value)
            # line_color = 'blue'
            line_CDS.data = dict(
                xs=[],
                ys=[],
                color=[],
                line_lib=[],
            )
        source_data = dict(
            x=to_plot[select_x.value],
            y=to_plot[select_y.value],
            fill_color=to_plot[select_bubble.value + '_c'],
            size=to_plot[select_size.value + '_s'],
            fill_lib=to_plot[select_bubble.value + '_lib'],
            # hover_field1=to_plot[select_line],
            # hover_field2=to_plot[select_bubble],
        )
        if select_line.value != 'None':
            source_data = {
                **source_data,
                **{'line_color': to_plot[select_line.value + '_c']},
                }
        source.data = source_data
        update_plot()

    def update_dataframe():
        global to_plot
        to_plot = select_data()
        update_CDS()

    p = figure(plot_height=500, plot_width=800)
    p.multi_line(
        xs='xs',
        ys='ys',
        line_color='color',
        line_width=2,
        source=line_CDS,
        legend_field='line_lib',
        )
    p.circle(
        source=source,
        x='x',
        y='y',
        size='size',  # size
        fill_color='fill_color',
        line_color='line_color',
        line_width=.3,
        legend_field='fill_lib',
        )
    p.legend.location = 'bottom_right'
    # tooltips_fields = [
    #     'seg3_lib',
    #     'origin2_lib',
    #     'margin',
    #     'brutrevenue',
    #     'marginperkg',
    #     'marginpercent',
    #     'lineperorder',
    # ]
    # tooltips = [(lib(field), '@' + field + jsformats_tooltips[field])
    #             for field in tooltips_fields]
    # hover = HoverTool(
    #     renderers=[circles],
    #     tooltips=tooltips,
    # )
    # p.add_tools(hover)

    def update_plot():
        nonlocal p
        p.x_range.start = 0
        p.y_range.start = 0
        p.yaxis.formatter = NumeralTickFormatter(
            format=axis_formatters[select_y.value]
            )
        p.xaxis.formatter = NumeralTickFormatter(
            format=axis_formatters[select_x.value]
            )
        p.xaxis.axis_label = lib(select_x.value)
        p.yaxis.axis_label = lib(select_y.value)
        title = (
            f'{lib(select_y.value)} en fonction de {lib(select_x.value)} par '
            f'{lib(select_bubble.value)}'
        )
        if select_line.value != 'None':
            title += f' et {lib(select_line.value)}'
        p.title.text = title

    update_dataframe()

    doc.add_root(
        column(
            row(axes_widgets,
                groups_widgets,
                filter_widgets,
                sizing_mode='stretch_width'),
            p,
            )
        )


def bk_detail(
    doc,
    in_data=None,
    order_data=None,
    oc=None,
    client=None,
    bins=def_bins,
    groupers=['orgacom', 'client'],
    inactive_duration=20,
    indicator_status='brutrevenue',
    origin='WEB',
    indicator_perf='margin',
    bin_colors=bin_colors,
    inactive_roll_mode='stitch',
    clt_data=None,
):
    if groupers is None:
        groupers = ['orgacom', 'client']
    roll_parms = dict(
        window=20,
        center=True,
        win_type='triang',
        min_periods=1,
    )

    source = ColumnDataSource()
    hist_source = ColumnDataSource()
    data = pd.DataFrame()
    status_updates = pd.DataFrame()
    boxes = list()

    def update_CDS():
        nonlocal boxes
        for box in boxes:
            box.visible = False
        boxes = list()
        for row_ in list(status_updates.itertuples()):
            if not pd.isnull(row_.end_date):
                end_date = row_.end_date
            else:
                end_date = None
            boxes.append(
                BoxAnnotation(
                    left=row_.date,
                    right=end_date,
                    fill_color=bin_colors[row_.status],
                    # fill_alpha=1.,
                    level="underlay",
                )
            )
        for box in boxes:
            p_dens.add_layout(box)
            p_hist.add_layout(box)
            p_perf.add_layout(box)
        source.data = ColumnDataSource.from_df(data.reset_index())
        hist_source.data = ColumnDataSource.from_df(order_data.reset_index())

    def update_dataframe():
        nonlocal data
        nonlocal status_updates
        data = day_orders_pipe(
            data=in_data,
            inactive_duration=inactive_duration,
            indicator_status=indicator_status,
            origin=origin,
            indicator_perf=indicator_perf,
            groupers=groupers,
            roll_parms=roll_parms,
            inactive_roll_mode=inactive_roll_mode,
            bins=bins,
        )
        status_updates = get_first_rupture_from_group(
            df=(
                data
                .reset_index()
                .loc[:, groupers + ['date', 'status']]
                .droplevel(1, axis=1)
            ),
            groupers=groupers,
            order_keys=['date'],
            targets=['status'],
        )
        status_updates = compute_end_date(
            data=status_updates,
            groupers=groupers,
            )
        update_CDS()

    def acquire_input():
        nonlocal inactive_duration
        inactive_duration = int(inactive_duration_input.value)
        roll_parms['window'] = int(window_input.value)
        update_dataframe()

    # client information widget
    dates = order_data.index.get_level_values(2)
    delta = (dates.max() - dates.min()).days
    month_rev = order_data.brutrevenue.sum() * 30 / delta
    week_freq = len(order_data) * 7 / delta
    client_info = Div(
        text=create_client_info(clt_data=clt_data, other_data={
            'month_revenue': f'{month_rev:.2f} €/mois',
            'week_freq': f'{week_freq:.2f} com./sem.',
        }),
    )

    # widgets definition
    inactive_duration_input = TextInput(
        value=str(inactive_duration),
        title="Durée pour inactivité :",
    )
    window_input = TextInput(
        value=str(roll_parms['window']),
        title="Fenêtre de moyenne glissante :",
    )
    button = Button(label="Go !", button_type="primary", width=20)
    button.on_click(acquire_input)
    widgets = column(inactive_duration_input, window_input, button)

    # figures definitions
    TOOLS = "pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"
    p_hist = figure(
        x_axis_type="datetime",
        title='Détail pour ' + oc + ' / ' + client,
        plot_height=280,
        plot_width=600,
        toolbar_location=None,
        )
    p_hist.y_range.start = 0
    p_hist.xaxis.ticker = MonthsTicker(months=list(range(1, 13)))
    p_hist.xaxis.major_label_orientation = pi / 2
    p_dens = figure(
        x_range=p_hist.x_range,
        x_axis_type="datetime",
        plot_width=p_hist.plot_width,
        plot_height=280,
        toolbar_location=None,
    )
    p_dens.yaxis.formatter = NumeralTickFormatter(format='0 %')
    p_dens.y_range = Range1d(-.1, 1.1)
    p_perf = figure(
        x_range=p_hist.x_range,
        x_axis_type="datetime",
        plot_width=p_hist.plot_width,
        plot_height=280,
        toolbar_location=None,
    )
    p_perf.y_range.start = 0
    for plot in [p_hist, p_dens, p_perf]:
        plot.xaxis.ticker = MonthsTicker(months=list(range(1, 13)))
        plot.xaxis.major_label_orientation = pi / 2
    p_compare = figure(
        plot_width=500,
        plot_height=500,
        x_range=[-.05, 1.05],
        tools=TOOLS,
    )
    p_compare.xaxis.formatter = NumeralTickFormatter(format='0 %')

    acquire_input()

    # glyphs definitions
    for val in bins.bin_limits:
        p_dens.add_layout(Span(
            location=val,
            dimension='width',
            line_dash='dashed',
            line_color='red',
            line_alpha=.5,
            ))
    p_hist.line(
        x='date',
        y='brutrevenue',
        line_width=.2,
        line_color='black',
        source=hist_source,
    )
    p_hist.circle(
        x='date',
        y='brutrevenue',
        color=factor_cmap(
            'origin2',
            list(colormaps['origin2'].values()),
            list(colormaps['origin2'])),
        source=hist_source,
        size=5.,
        )
    p_dens.line(
        x='date_',
        y='WEB_percentage_',
        source=source,
        line_width=1.,
        color=colormaps['origin2']['WEB'],
        )
    p_perf.line(
        x='date_',
        y='margin_rolled_total',
        source=source,
    )
    p_compare.circle(
        source=source,
        x='WEB_percentage_',
        y='margin_rolled_total',
        color='black',
        fill_alpha=.5,
        line_alpha=.5,
    )

    doc.add_root(
        row(
            column(client_info, widgets),
            column(p_hist, p_dens, p_perf),
            column(p_compare),
            )
    )


def create_client_info(
    clt_data=None,
    other_data=None,
):
    html = "<h1>Informations client</h1>"
    html += "<h2>Généralités</h2>"
    html += "<ul>"
    html += f"<li><b>Code</b> : </b>{clt_data.name[1]}</li>"
    data_to_show = {
        'nom': 'Libellé client',
        'postalcode': 'Code postal',
    }
    for code, lib in data_to_show.items():
        try:
            html += f"<li><b>{lib} : </b> {clt_data[code]}</li>"
        except KeyError:
            pass
    html += "</ul>"
    html += "<h2>Segmentation</h2>"
    html += "<ul>"
    data_to_show = {
        'seg1_lib': 'Segment 1',
        'seg2_lib': 'Segment 2',
        'seg3_lib': 'Segment 3',
        'seg4_lib': 'Segment 4',
        'cat_lib': 'Catégorie',
        'sscat_lib': 'Sous-catégorie',
    }
    for code, lib in data_to_show.items():
        try:
            html += f"<li><b>{lib} : </b> {clt_data[code]}</li>"
        except KeyError:
            pass
    html += "</ul>"
    html += "<h2>Hiérarchie</h2>"
    html += "<ul>"
    data_to_show = {
        'hier4': 'Hiérarchie 4',
        'hier3': 'Hiérarchie 3',
        'hier2': 'Hiérarchie 2',
        'hier1': 'Hiérarchie 1',
    }
    for code, lib in data_to_show.items():
        html += (
            f"<li><b>{lib} : </b> {clt_data[code]} - "
            f"{clt_data[code + '_lib']}</li>"
        )
    html += "</ul>"
    html += "<h2>Informations quantitatives</h2>"
    html += "<ul>"
    data_to_show = {
        'month_revenue': 'CA brut / mois',
        'week_freq': "Commandes / semaine",
    }
    for code, lib in data_to_show.items():
        try:
            html += (
                f"<li><b>{lib} : </b> {other_data[code]}"
            )
        except KeyError:
            pass
    html += "</ul>"
    return(html)


def compute_end_date(
    data=None,
    groupers=None,
    begin_field='date',
    end_field_name='end_date',
):
    same_grp = (data[groupers].shift(-1) == data[groupers]).all(axis=1)
    end_dates = (
        data[begin_field]
        .shift(-1)
        .where(same_grp, None)
        .rename('end_date')
        .to_frame()
    )
    # end_dates.columns = pd.MultiIndex.from_product([['end_date'], ['']])
    return(data.join(end_dates))


def compute_rolling_percentage(
    data=None,
    window=None,
    win_type=None,
    indicator=None,
    axis=None,
    roll_kwargs=None,
    groupers=None,
):
    '''
    data is a dataframe, with multiindex on columns.
    Level 0 must be the indicator, Level 1 the axis on which to compute the
    percentage.
    returns a series, indexed like the rows of data, which is the "rolling
    percentage" of the axis.
    '''
    if indicator not in data.columns.get_level_values(0):
        raise RuntimeError(f"Indicator '{indicator}' not in first level of "
                           f"columns index.")

    if axis not in data.columns.get_level_values(1):
        raise RuntimeError(f"Axis '{axis}' not in second level of "
                           f"columns index.")

    if (indicator, 'total') not in data.columns:
        df = data.loc[:, indicator].copy()
        df['total'] = df.sum(axis=1)
        df = df[[axis, 'total']]
    else:
        df = data.loc[:, idx[indicator, [axis, 'total']]].copy()
        df.columns = df.columns.droplevel(level=0)

    if roll_kwargs is None:
        roll_kwargs = dict()

    df = df.groupby(groupers).apply(
        lambda x: x.rolling(
            window,
            win_type=win_type,
            min_periods=1,
            **roll_kwargs,
            ).sum()
    )

    return((df[axis] / df['total']).fillna(0))


def compute_stat_from_percentage(
    ds=None,
    mode='cut',
    bins=def_bins,
):
    '''
    Returns a series with bins from an input series

    bins are a named tuple with attributes label and bin limits
    '''
    if mode != 'cut':
        raise NotImplementedError(f'mode :{mode} not yet implemented)')
    return(
        pd.cut(
            ds,
            bins=bins.bin_limits,
            labels=bins.labels,
            right=False,
        )
    )


def mask_successive_values(
    ds=None,
    value=None,
    count=None,
    groupers=None,
):
    '''
    Returns the initial series with consecutive values
    '''
    data = ds.rename('target').reset_index()
    diffs = (
        data[groupers + ['target']].shift() != data[groupers + ['target']]
    )
    diffs = diffs.any(axis=1).cumsum()
    diffs = diffs.map(diffs.value_counts() >= count)
    if value is not None:
        diffs = (diffs & (data['target'] == value))
    diffs.index = ds.index
    return(diffs)


def get_first_rupture_from_group(
    df=None,
    groupers=None,
    order_keys=None,
    targets=None,
):
    '''
    Return first rupture from a dataframe, among groups

    df is a dataframe
    other parameters are lists of fields, from columns.
    Those fields must not be part of the index, use reset_index prior to
    calling this function
    Any rupture on a target field is considered a rupture (i.e. if only a
    single field from target is different it will appear as rupture)
    '''
    df = df.sort_values(groupers + order_keys)
    same_group = (df[groupers] == df[groupers].shift()).all(axis=1)
    same_target = (df[targets] == df[targets].shift()).all(axis=1)
    return(
        df.loc[
            ~same_group |
            (same_group & ~same_target)
        ]
    )


def day_orders_pipe(
    data=None,
    inactive_duration=None,
    indicator_status=None,
    origin=None,
    indicator_perf=None,
    groupers=['orgacom', 'client'],
    roll_parms=None,
    inactive_roll_mode='stitch',
    bins=def_bins,
):
    '''
    Whole pipeline to treat day_orders formatted dataframe for plotting

    day_order format is:
    - dataframe
    - row indexed by: groupers (orgacom & client) and date
    - with dense dates (i.e. all business days are in the df)
    - column indexed by:
      - lev0: 'indicators' the indicators for each row
      - lev1: 'origin2' the canal of origin
    Not a constraint, but with my way of computing, usually each row will have
    a single indicator populated.
    Optionnaly, the column origin2 may have a 'total' precomputed.
    '''
    # Step 1: compute total if not in dataframe columns
    indicators = list(set([indicator_status, indicator_perf]))
    try:
        df = data.loc[:, indicators].copy()
    except KeyError:
        raise RuntimeError(f"Indicator '{indicators}' has not been "
                           f"found in first level of dataframe column "
                           f"multiindex.")
    if (
        (indicator_status, 'total') not in data.columns or
        (indicator_perf, 'total') not in data.columns
            ):
        start = now()
        print(f'{start}: Computing totals')
        df = df.join(
            pd.concat(
                [df[indicators].groupby('indicators', axis=1).sum()],
                keys=['total'],
                axis=1
                ).swaplevel(axis=1)
        )
        df = df.sort_index(axis=1)
        print(f'{now()}: Done! Elapsed: {now() - start}')

    # Step 2: compute inactive periods
    if 'inactive' not in df.columns:
        start = now()
        print(f'{start}: Computing inactive periods')
        df['inactive'] = mask_successive_values(
            ds=df[(indicator_status, 'total')],
            groupers=groupers,
            count=inactive_duration,
            value=0.,
        )
        print(f'{now()}: Done! Elapsed: {now() - start}')

    # Step 3: compute rolling indicators
    start = now()
    print(f'{start}: Computing rolling indicators')
    if roll_parms is None:
        roll_parms = dict(
            center=False,
            win_type='triang',
            window=100,
            min_periods=15,
        )
    if inactive_roll_mode not in {'stitch', 'ignore', 'zero'}:
        raise ValueError(
            f"inactive_roll_mode must be 'stitch', 'ignore' or 'zero'. Got "
            f"{inactive_roll_mode} instead."
            )
    if indicator_status == indicator_perf:
        cols = [
            (indicator_status, origin),
            (indicator_status, 'total'),
            ]
    else:
        cols = [
            (indicator_status, origin),
            (indicator_status, 'total'),
            (indicator_perf, 'total'),
            ]
    if inactive_roll_mode == 'stitch':
        scope = df.loc[~df.inactive, [*cols, ('inactive', '')]]
    else:
        scope = df.loc[:, [*cols, ('inactive', '')]]
    mask_value = 0. if inactive_roll_mode == 'zero' else np.nan
    rolled = (
        scope
        .where(~scope.inactive, mask_value)
        .loc[:, cols]
        .groupby(groupers, observed=True)
        .apply(
            lambda x: x.rolling(**roll_parms).mean()
            )
    )
    if inactive_roll_mode == 'stitch':
        rolled = rolled.reindex(df.index, method='ffill')
    df = df.join(rolled, rsuffix='_rolled')
    del(rolled)
    print(f'{now()}: Done! Elapsed: {now() - start}')

    # Step 4: compute statuses
    start = now()
    print(f'{start}: Computing percentage and statuses')
    df[origin + '_percentage'] = (
        df.loc[:, (indicator_status + '_rolled', origin)] /
        df.loc[:, (indicator_status + '_rolled', 'total')]
    )
    df['status'] = compute_stat_from_percentage(
        ds=df[origin + '_percentage'],
        mode='cut',
        bins=bins,
    ).fillna(method='ffill').astype('str').where(~df.inactive, 'inactive')
    print(f'{now()}: Done! Elapsed: {now() - start}')

    return(df)
