import numpy as np
import warnings
import pandas as pd
from pandas import IndexSlice as idx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from functools import partial
from collections import namedtuple
from itertools import product
from IPython.display import display
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FactorRange, Range1d, Span, Button
from bokeh.models import TextInput, Div, LabelSet
from bokeh.models.widgets import Select, DatePicker, CheckboxGroup
from bokeh.models.annotations import BoxAnnotation
from bokeh.transform import factor_cmap
from bokeh.layouts import row, column
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models.tickers import MonthsTicker
import holoviews as hv
from holoviews import opts  # dim
from holoviews.operation.timeseries import rolling
import panel as pn
import param
from math import pi, cos, sin, acos
from datetime import date
from datetime import datetime
import datetime as dt
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
    'branch': 'Branche',
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
    'main_origin': 'Canal de commande',
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
    'brutrevenue_perbusday': 'CA brut (€/j.o.)',
    'margin_perbusday': 'Marge (€/j.o.)',
    'weight_perbusday': 'Tonnage (kg/j.o.)',
    'linecount_perbusday': 'Nombre de lignes (/j.o.)',
    'ordercount_perbusday': 'Nombre de commandes (/j.o.)',
    'marches_publics_seg': 'Type marché (segmentation)',
    'marches_publics_ddar': 'Type marché (hiérarchie DDAR)',
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
    '1RAA': '1RAA - PassionFroid Rhône-Alpes Auvergne',
    '2RAA': '2RAA - EpiSaveurs Rhône-Alpes Auvergne',
    '1PAC': "1PAC - PassionFroid Provence-Alpes Côte d'Azur",
    '1PSU': '1PSU - PassionFroid Ile de France Sud',
    '1CAP': "1CAP - PassionFroid Centrale d'Achats",
    '1CTR': '1CTR - PassionFroid Centre',
    '2CTR': '2CTR - EpiSaveurs Centre',
    '1BFC': '1BFC - PassionFroid Bourgogne Franche-Comté',
    '1OUE': '1OUE - PassionFroid Ouest',
    '2EST': '2EST - EpiSaveurs Est',
    '2NOR': '2NOR - EpiSaveurs Nord',
    '2IDF': '2IDF - EpiSaveurs Ile de France',
    '1NCH': '1NCH - PassionFroid Nord Champagne',
    '1LXF': '1LXF - PassionFroid LuxFrais',
    '2SES': '2SES - EpiSaveurs Sud-Est',
    '1PNO': '1PNO - PassionFroid Ile de France Nord',
    '2MPY': '2MPY - EpiSaveurs Midi-Pyrénées',
    '2SOU': '2SOU - EpiSaveurs Aquitaine',
    '1EXP': '1EXP - PassionFroid Ile de France Export',
    '2CAE': "2CAE - EpiSaveurs Centrale d'achats",
    'seg1': 'Segmentation niveau 1',
    'seg2': 'Segmentation niveau 2',
    'seg3': 'Segmentation niveau 3',
    'seg4': 'Segmentation niveau 4',
    'seg1_lib': 'Segmentation niveau 1',
    'seg2_lib': 'Segmentation niveau 2',
    'seg3_lib': 'Segmentation niveau 3',
    'seg4_lib': 'Segmentation niveau 4',
}

suc_libs_inv = {
    libs[k]: k for k in [
        '1RAA',
        '2RAA',
        '1PAC',
        '1PSU',
        '1LRO',
        '1CAP',
        '1CTR',
        '2CTR',
        '1BFC',
        '1OUE',
        '2EST',
        '2NOR',
        '1ALO',
        '2IDF',
        '1NCH',
        '1LXF',
        '2SES',
        '1PNO',
        '2BRE',
        '2MPY',
        '1SOU',
        '2SOU',
        '1EXP',
        '2CAE',
    ]
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

plot_yaxis_fmts = {
    'marginpercent': mtick.PercentFormatter(xmax=1),
}

seg3_dict = {
    'ZK': 'RCI',
    'ZL': 'RCS',
    'ZI': 'RCA',
    'ZJ': 'RCC',
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
    },
    'main_origin': {
        'TV': mcolorpalette[0],
        'VR': mcolorpalette[1],
        'WEB': mcolorpalette[2],
        'EDI': mcolorpalette[3],
    },
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

# The lines below requires that the data and persist folders be on the same
# level like the directory containing this script.
path = Path(__file__).parents[0] / '..' / 'data' / 'libelles_segments.csv'
lib_seg = pd.read_csv(path,
                      sep=';',
                      encoding='latin1',
                      header=None,
                      names=['level', 'code', 'designation'],
                      index_col=['level', 'code']
                      )
persist_path = Path(__file__).parents[0] / '..' / 'persist'

transco = dict()
for i in range(1, 7):
    transco['seg' + str(i)] = (
        lib_seg.loc[idx[i, :, :]]
               .reset_index()
               .set_index('code')['designation']
    )
transco['marches_publics_seg'] = {
    True: "Marchés publics",
    False: "Hors marchés publics",
}
transco['marches_publics_ddar'] = {
    True: "Marchés publics",
    False: "Hors marchés publics",
}

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


def process_df(
    df,
    orders_doctypes=['ZC10'],
    avoirs_doctypes=['ZA01', 'ZA02'],
    indicators=['margin', 'brutrevenue', 'weight'],
    grouper_fields=['orgacom', 'date', 'client', 'material'],
    debug=False,
):
    init_index = df.index.names
    init_cols = df.columns
    before_processing = df[indicators].sum()

    df = df.reset_index()

    mask_ZC = df.doctype.isin(orders_doctypes)
    mask_ZA = df.doctype.isin(avoirs_doctypes)
    raw_avoirs = df.loc[mask_ZA, grouper_fields + indicators]
    avoirs = raw_avoirs.groupby(grouper_fields, observed=True).sum()
    mask_dup_ZC = (df.loc[mask_ZC]
                     .duplicated(grouper_fields, keep=False)
                     .rename('_duplicated'))
    # mask_dup_ZC = mask_dup_ZC.reindex(df.index, fill_value=False)
    df = df.merge(
        mask_dup_ZC,
        how='left',
        left_index=True,
        right_index=True)
    df['_duplicated'] = df['_duplicated'].fillna(False)
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
    if debug:
        display(df)
        display(to_update)
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
    merged = (
        merged
        .drop(columns=['_merge', '_duplicated'])
    )
    try:
        merged = (
            merged
            .set_index(init_index)
        )
    except KeyError:  # case initial index did not have name
        pass
    merged = (
        merged
        [init_cols]  # reorder columns
        .sort_index()
    )
    df = merged
    del(merged)
    after_processing = df[indicators].sum()
    delta = after_processing - before_processing
    print(f'Evolution des indicateurs pendant le traitement : \n{delta}')
    if max(delta.max(), abs(delta.min())) > .001:
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

    stats = (
        data[indicators]
        .describe(
            percentiles=[
                1 - percentile_selection,
                .25,
                .50,
                .75,
                percentile_selection
            ]
        )
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


def plot_distrib(
    data=None,
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
    debug=False,
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
        data[hue] = data[hue].astype(
            pd.CategoricalDtype(
                hue_order,
                ordered=True,
            )
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
    if debug:
        display(data)
    stats = compute_distribution(
        data=data,
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

        if translate == True or 'legend' in translate:  # noqa: E712
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            labels = map(partial(lib, domain=hue), labels)
            ax.legend(handles, labels)

        if indicator in plot_yaxis_fmts:
            ax.yaxis.set_major_formatter(plot_yaxis_fmts[indicator])

        if IQR_factor_plot is not None:
            ax.set_ylim(plot_ranges.loc[indicator, 'minimum_plot_range'],
                        plot_ranges.loc[indicator, 'maximum_plot_range'])

    return(fig, axs)


def bk_histo_seg(
    doc,
    source_df=None,
    segs=None,
    stack_axis='origin2',
    x_dimension='orgacom',
    filters=None,
    filters_exclude=None,
    plot_kwargs=None,
    plot_kwargs_dict=None,
    max_text_len=15,
    range_padding=0.,
):
    '''
    Bokeh server app that enables to draw stacked bar plot on segmentation
    '''

    if plot_kwargs is None:
        plot_kwargs = dict()
    if plot_kwargs_dict is None:
        plot_kwargs_dict = dict()

    # some default values for plot_kwargs
    plot_kwargs = {
        **{
            'plot_width': 900,
        },
        **plot_kwargs
    }

    # some default values for plot_kwargs_dict
    plot_kwargs_dict = {
        **{
            ('xaxis', 'major_label_orientation'): 1,
            ('yaxis', 'formatter'): NumeralTickFormatter(format='0'),
        },
        **plot_kwargs_dict
    }

    # define controls
    # select: choose indicator from list
    indicator_map = {
        'Marge (€)': 'margin',
        'CA brut (€)': 'brutrevenue',
        'Tonnage (kg)': 'weight',
    }
    select = Select(
        title="Indicateur",
        options=list(indicator_map),
        value=list(indicator_map)[1]
    )
    # datepickers : filter data on date
    min_date, max_date = date(2017, 7, 3), date(2020, 8, 30)
    min_def_date, max_def_date = date(2019, 1, 1), date(2019, 12, 31)
    datepickers = [
        DatePicker(
            title='Date de début',
            value=min_def_date,
            min_date=min_date,
            max_date=max_date
        ),
        DatePicker(
            title='Date de fin',
            value=max_def_date,
            min_date=min_date,
            max_date=max_date
        ),
    ]

    controls = [select, *datepickers]

    # compute data source
    origins = ['TV', 'VR', 'WEB', 'EDI']

    def compute_indicator(df, indicator):
        temp = (
            df.groupby([stack_axis] + segs + [x_dimension],
                       observed=True,)[indicator]
              .sum()
              .unstack(stack_axis, fill_value=0.)
              .reindex(columns=origins)
              .reset_index()
        )
        for seg in segs:
            truncated_transco = {
                k: v[:max_text_len] for k, v in transco[seg].items()
            }
            temp[seg] = temp[seg].map(truncated_transco)
        temp = temp.set_index(segs + [x_dimension])
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
        x_range=FactorRange(*list(df.index), range_padding=range_padding),
        **plot_kwargs,
    )
    p.vbar_stack(
        df.columns,
        x='_'.join(segs + [x_dimension]),
        source=source,
        width=.9,
        color=list(mcolor.TABLEAU_COLORS.values())[:len(df.columns)],
        legend_label=list(df.columns),
    )

    def _unpack_plot_kwargs_dict(target, dict_):
        for attr_list, val in dict_.items():
            cur_target = target
            for attr in attr_list[:-1]:
                cur_target = cur_target.__getattribute__(attr)
            cur_target.__setattr__(attr_list[-1], val)

    _unpack_plot_kwargs_dict(p, plot_kwargs_dict)
    doc.add_root(column(select, row(*datepickers), p))


def bk_bubbles(
    doc,
    data=None,
    filters=None,
    plot_analysis_axes=None,
    debug=False,
):
    max_size = 50
    # line_width = 2.5
    plot_indicators = ['brutrevenue', 'margin', 'weight', 'linecount']
    composite_indicators = ['PMVK', 'marginperkg', 'marginpercent',
                            'lineweight']
    select_indicators = (
        [indicator + '_glob' for indicator in plot_indicators] +
        plot_indicators +
        composite_indicators
    )
    for indicator in plot_indicators + composite_indicators:
        if indicator + '_clt_zscore' in data.columns:
            select_indicators.append(indicator + '_clt_zscore')
    if not plot_analysis_axes:
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
            'brutrevenue_clt_zscore': '0.00',
            'margin_clt_zscore': '0.00',
            'weight_clt_zscore': '0.00',
            'linecount_clt_zscore': '0.00',
            'lineweight_clt_zscore': '0.00',
            'marginperkg_clt_zscore': '0.00',
            'marginpercent_clt_zscore': '0.00',
            'PMVK_clt_zscore': '0.00',
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
            me = select_bubble
            other = select_line
        if which_control == 'line':
            me = select_line
            other = select_bubble
            if old == 'None' and new == other.value:
                me.value = 'None'
                running = False
                return
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

    def select_data(
        debug=False,
    ):
        # aggregate data
        groupers = [select_bubble.value]
        if select_line.value != 'None':
            groupers.append(select_line.value)
        if filters is not None:
            filtered_data = data.loc[filters]
        else:
            filtered_data = data
        to_plot = (
            filtered_data
            .groupby(groupers, observed=True)[plot_indicators]
            .sum()
        )
        to_plot = to_plot.rename(
            {indicator: indicator + '_glob' for indicator in plot_indicators},
            axis=1,
        )
        sizes = (
            filtered_data
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
        to_plot = compute_composite_indicators(
            to_plot,
            indicator_defs={
                **composite_indicators_dict,
                **{'lineperorder': ['linecount_glob', 'ordercount']},
            }
        )

        # compute z-score indicators
        for indicator in plot_indicators + composite_indicators:
            if indicator + '_clt_zscore' in data.columns:
                ds = (
                    filtered_data
                    .groupby(groupers, observed=True)
                    [indicator + '_clt_zscore']
                    .mean()
                )
                to_plot = to_plot.set_index(groupers).join(ds).reset_index()

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
        if debug:
            display(to_plot)
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
    to_plot = select_data(debug=debug)
    # with pd.option_context('display.max_columns', None):
    #     display(to_plot)

    def update_CDS():
        global to_plot
        nonlocal source
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
            line_colors = to_plot[select_line.value + '_c']
        else:
            line_colors = [None] * len(to_plot)

        source_data = {
            **source_data,
            **{'line_color': line_colors},
            }
        source.data = source_data
        update_plot()

    def update_dataframe():
        global to_plot
        to_plot = select_data(debug=debug)
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
        nonlocal to_plot
        if to_plot[select_x.value].min() >= 0:
            # to fix, does not work: updating x_range.start does not work after
            # the first time it has been done...
            p.x_range.start = 0
        else:
            p.x_range.start = to_plot[select_x.value].min()
        if to_plot[select_y.value].min() >= 0:
            p.y_range.start = 0
        else:
            p.y_range.start = to_plot[select_y.value].min()
        p.xaxis.formatter = NumeralTickFormatter(
            format=axis_formatters[select_x.value]
            )
        p.yaxis.formatter = NumeralTickFormatter(
            format=axis_formatters[select_y.value]
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
                # filter_widgets, # filters not implemented yet.
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
        window=75,
        center=True,
        win_type='triang',
        min_periods=5,
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
    month_freq = len(order_data) * 30 / delta
    client_info = Div(
        text=create_client_info(clt_data=clt_data, other_data={
            'month_revenue': f'{month_rev:.2f} €/mois',
            'month_freq': f'{month_freq:.2f} com./mois',
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
    # TOOLS = "pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"
    p_hist = figure(
        x_axis_type="datetime",
        title='Historique de commande du client ' + oc + ' / ' + client,
        plot_height=280,
        plot_width=800,
        toolbar_location=None,
        )
    p_hist.yaxis.axis_label = "CA brut (€)"
    p_hist.y_range.start = 0
    p_hist.xaxis.ticker = MonthsTicker(months=list(range(1, 13)))
    p_hist.xaxis.major_label_orientation = pi / 2
    p_dens = figure(
        x_range=p_hist.x_range,
        x_axis_type="datetime",
        title="Proportion de CA Web",
        plot_width=p_hist.plot_width,
        plot_height=280,
        toolbar_location=None,
    )
    p_dens.yaxis.formatter = NumeralTickFormatter(format='0 %')
    p_dens.y_range = Range1d(-.1, 1.1)
    p_dens.yaxis.axis_label = "Pourcentage de CA Web (%)"
    p_perf = figure(
        x_range=p_hist.x_range,
        x_axis_type="datetime",
        plot_width=p_hist.plot_width,
        plot_height=280,
        toolbar_location=None,
        title="Marge moyenne par jour"
    )
    p_perf.y_range.start = 0
    for plot in [p_hist, p_dens, p_perf]:
        plot.xaxis.ticker = MonthsTicker(months=list(range(1, 13)))
        plot.xaxis.major_label_orientation = pi / 2
    p_perf.yaxis.formatter = NumeralTickFormatter(format="0")
    p_perf.yaxis.axis_label = "Marge moyenne par jour (€/jour)"

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
        legend_field='origin2',
        )
    p_hist.legend.location = 'top_right'

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

    doc.add_root(
        row(
            column(client_info, widgets),
            column(p_hist, p_dens, p_perf),
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
        'month_freq': "Commandes / mois",
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


def dedup(list_):
    return(list(dict.fromkeys(list_)))


def compute_grouped(
    input_df=None,
    atomic_keys=None,
    population_keys=None,
    subatomic_keys=None,
    keys_to_expand=None,
    expand_categories=None,
):
    indicator_df = (
        input_df
        .groupby(
            dedup(population_keys + subatomic_keys),
            observed=True,
            dropna=False,
        )
        .sum()
    )
    size_df = (
        input_df
        .loc[:, dedup(atomic_keys + population_keys)]
        .drop_duplicates()
        .groupby(population_keys, observed=True)
        .size()
        .fillna(0.)
        .astype('int')
        .rename('size')
    )
    # reorder index levels to be consistent with input
    aligned = tuple(map(
        lambda x: x.reorder_levels(dedup(population_keys + subatomic_keys)),
        size_df.align(indicator_df),
        ))
    ret = aligned[1].div(aligned[0], axis=0).join(aligned[0]).reset_index()

    if keys_to_expand:
        vals_to_expand = {
            key: list(input_df[key].unique()) for key in keys_to_expand
        }
        if not expand_categories:
            expand_categories = dict()
        vals_to_expand = {**vals_to_expand, **expand_categories}
        new_idx = pd.MultiIndex.from_product(
            list(vals_to_expand.values()),
            names=list(vals_to_expand.keys()),
        )
        nan_df = pd.DataFrame(
            [np.nan] * len(new_idx),
            index=new_idx,
            columns=['dummy'],
        )
        ret = (
            ret
            .set_index(keys_to_expand)
            .join(nan_df, how='outer')
            .drop('dummy', axis=1)
        )

    return(ret)


def compute_scope(
    in_df=None,
    field=None,
    values=None,
    atomic_keys=None,
    status_names=None,
    target_field=None,
):
    temp = (
        in_df
        .loc[~pd.isna(in_df[field]), atomic_keys + [field]]
        .drop_duplicates()
        .assign(in_period=True)
        .set_index(atomic_keys + [field])
        .unstack(field, fill_value=False)
        .astype('bool')
    )
    temp.columns = [
        '_'.join(index_) for index_ in temp.columns.to_flat_index()
        ]
    atoms = in_df[atomic_keys].drop_duplicates()
    temp = atoms.merge(
        temp,
        how='left',
        on=atomic_keys
        ).set_index(atomic_keys).fillna(False)
    temp[target_field] = None
    temp.loc[
        temp[('in_period_' + values[0])] &
        temp[('in_period_' + values[1])],
        target_field] = status_names[0]
    temp.loc[
        temp[('in_period_' + values[0])] &
        ~temp[('in_period_' + values[1])],
        target_field] = status_names[1]
    temp.loc[
        ~temp[('in_period_' + values[0])] &
        temp[('in_period_' + values[1])],
        target_field] = status_names[2]
    temp.loc[
        ~temp[('in_period_' + values[0])] &
        ~temp[('in_period_' + values[1])],
        target_field] = status_names[3]
    return(temp[target_field])


pn.param.Param._mapping.update(
    {param.CalendarDate: pn.widgets.DatePicker}
)


class WebProgressShow(param.Parameterized):

    orgacom = param.ObjectSelector(
        objects=suc_libs_inv,
        default='1ALO',
        label='Succursale',
    )

    seg3 = param.ObjectSelector(
        objects={
            'Restauration commerciale indépendante': 'ZK',
            'Restauration commerciale structurée': 'ZL',
            'Restauration collective autogérée': 'ZI',
            'Restauration collective concédée': 'ZJ',
        },
        default='ZK',
        label='Segment 3',
    )

    indicator = param.ObjectSelector(
        objects={
            'CA brut': 'brutrevenue',
            'Marge': 'margin',
            'Tonnage': 'weight',
            'Nb lignes': 'linecount',
        },
        default='margin',
        label='Indicateur',
    )

    rolling_window = param.Integer(default=10, bounds=(1, 50))

    d1period1 = param.CalendarDate(
        dt.date(2019, 1, 1),
        bounds=(dt.date(2017, 1, 1), dt.date(2021, 2, 1)),
        label='Période 1 - Date 1',
    )

    d2period1 = param.CalendarDate(
        dt.date(2019, 2, 28),
        bounds=(dt.date(2017, 1, 1), dt.date(2021, 2, 1)),
        label='Période 1 - Date 2',
    )

    d1period2 = param.CalendarDate(
        dt.date(2020, 1, 1),
        bounds=(dt.date(2017, 1, 1), dt.date(2021, 2, 1)),
        label='Période 2 - Date 1',
    )

    d2period2 = param.CalendarDate(
        dt.date(2020, 2, 29),
        bounds=(dt.date(2017, 1, 1), dt.date(2021, 2, 1)),
        label='Période 2 - Date 2',
    )

    computation_locker = param.Boolean(
        default=False,
        label='Verrouillage des calculs',
    )

    indicator_cut = param.ObjectSelector(
        objects={
            'CA brut': 'brutrevenue',
            'Marge': 'margin',
            'Tonnage': 'weight',
            'Nb lignes': 'linecount',
            'Nb commandes': 'ordercount'
        },
        default='brutrevenue',
        label='Indicateur pour classification client',
    )

    canal_cut = param.ObjectSelector(
        objects=['TV', 'VR', 'WEB', 'EDI'],
        default='WEB',
        label='Canal pour classification client',
    )

    threshold_cut = param.Integer(
        default=15,
        bounds=(0, 100),
        step=5,
        label='Seuil pour classification client',
    )

    evol_per_client = param.Boolean(
        default=True,
        label="Par client",
    )

    indicator_grid = param.ObjectSelector(
        objects={
            'CA brut': 'brutrevenue',
            'Marge': 'margin',
            'Tonnage': 'weight',
            'Nombre de lignes': 'linecount',
            'Nombre de commandes': 'ordercount',
            },
        default='margin',
        label='Indicateur',
    )

    perbusday_grid = param.Boolean(
        default=True,
        label='Par jour ouvré',
    )

    def __init__(
        self,
        orders_path=None,
        clt_path=None,
        filters_include=None,
    ):
        super().__init__()
        self.filters_include = filters_include
        self.group_names = {
            (True, True): 'Fidèles',
            (True, False): 'Abandonneurs',
            (False, True): 'Adopteurs',
            (False, False): 'Ignoreurs',
        }
        self.dfs = dict()
        self.dfs['orders'] = pd.read_pickle(orders_path)
        if self.filters_include:
            idx_init = self.dfs['orders'].index.names
            self.dfs['orders'] = self.dfs['orders'].reset_index()
            for field, values in self.filters_include.items():
                try:
                    self.dfs['orders'] = (
                        self.dfs['orders'].loc[
                            lambda x: x[field].isin(values)
                        ]
                    )
                except KeyError:
                    print('Something is wrong with filters')
                    raise
            self.dfs['orders'] = self.dfs['orders'].set_index(idx_init)
        # rename main_origin column as origin2 if necessary
        if 'origin2' not in self.dfs['orders'].columns:
            self.dfs['orders'] = (
                self.dfs['orders']
                .rename({'main_origin': 'origin2'}, axis=1)
            )
        self.dfs['orders'] = self.dfs['orders'].reset_index()
        self.dfs['orders'].date = self.dfs['orders'].date.dt.date

        self.dfs['clt'] = pd.read_pickle(clt_path)
        self.origins = self.dfs['orders']['origin2'].unique()
        self.compute_period_df()

        # Refactoring needed here!
        self.dfs['orgacom_date'] = (
            self.dfs['orders']
            .groupby(['orgacom', 'date'], observed=True)
            .sum()
        )
        self.dfs['orgacom_origin2_date'] = (
            self.dfs['orders']
            .groupby(['orgacom', 'origin2', 'date'], observed=True)
            .sum()
        )

        # compute list of orgacoms in data
        # to be refactored: how can I update orgacom dropdown list during
        # __init__?
        self.ocs = (
            self.dfs['orders']['orgacom'].unique()
        )
        self.param.orgacom.objects = list(self.ocs)
        # self.orgacom = self.ocs[0]

    def curve_indicator(
        self,
        data,
        *,
        orgacom=None,
        indicator=None,
        spans=None,
    ):
        # spans should be a 2-tuple (period 1 and period 2) of
        # 2-tuple of dates (start and end dates)
        df = data.loc[self.orgacom]
        holo = hv.HoloMap(
            {
                origin: hv.Curve(
                    df.loc[origin],
                    kdims=('date', 'La Date'),
                    vdims=hv.Dimension(indicator, label=lib(indicator)),
                ).opts(color=colormaps['origin2'][origin])
                for origin in df.index.get_level_values(0).unique()
            },
            kdims=('origin2', 'Canal de commande')
        )
        span1 = hv.VSpan(*spans[0])
        span2 = hv.VSpan(*spans[1])
        return(
            rolling(
                holo.overlay('origin2'),
                rolling_window=self.rolling_window,
                window_type='triang').opts(
                    opts.Curve(),
                )
            * span1
            * span2
        )

    @param.depends(
        'd1period1',
        'd2period1',
        'd1period2',
        'd2period2',
        'computation_locker',
        watch=True)
    def compute_period_df(self):
        # should be put elsewhere...
        atomic_keys = ['orgacom', 'client']
        intrapop_keys = ['origin2', 'period']
        busdays = {
            'P1': np.busday_count(self.d1period1, self.d2period1),
            'P2': np.busday_count(self.d1period2, self.d2period2),
        }

        if self.computation_locker:
            return()

        self.dfs['orders'] = self.dfs['orders'].set_index('date').sort_index()
        self.dfs['orders']['period'] = None
        self.dfs['orders'].loc[self.d1period1:self.d2period1, 'period'] = 'P1'
        self.dfs['orders'].loc[self.d1period2:self.d2period2, 'period'] = 'P2'
        self.dfs['orders'] = self.dfs['orders'].reset_index()
        self.dfs['scope'] = compute_scope(
            self.dfs['orders'],
            field='period',
            values=['P1', 'P2'],
            atomic_keys=['orgacom', 'client'],
            status_names=['in_scope', 'lost', 'new', 'out_of_scope'],
            target_field='scope',
        )
        self.dfs['scope_orgacom_seg3'] = (
            self.dfs['scope']
            .to_frame()
            .join(self.dfs['clt']['seg3'])
            .groupby(['orgacom', 'seg3', 'scope'])
            .size()
            .unstack('scope', fill_value=0)
            .sort_index()
        )

        # we create the venn panel attributes here rather than in __init__
        # (depends on scope_orgacom_seg3 for initial values).
        if not hasattr(self, 'venn_bokeh_pane'):
            self.create_venn_panel()

        self.dfs['scope_orders'] = (
            self.dfs['orders']
            .set_index(['orgacom', 'client'])
            .join(self.dfs['scope'])
            .loc[lambda x: (x.scope == 'in_scope') & ~pd.isna(x.period)]
            .reset_index()
            .drop('scope', axis=1)
        )
        self.dfs['atom_intrapop'] = (
            self.dfs['scope_orders']
            .groupby(atomic_keys + intrapop_keys)
            .agg(
                margin=('margin', 'sum'),
                brutrevenue=('brutrevenue', 'sum'),
                weight=('weight', 'sum'),
                linecount=('linecount', 'sum'),
                ordercount=('weight', 'size'),
            )
            .reset_index()
        )
        self.dfs['atom_intrapop'] = self.dfs['atom_intrapop'].join(
            self.dfs['atom_intrapop']
            .select_dtypes('number')
            .add_suffix('_perbusday')
            .div(
                self.dfs['atom_intrapop']['period']
                .map(busdays),
                axis=0,
            )
        )
        # create statuses dataframe: all atomic keys x periods
        # only requires to have a clear scope, so do it early.
        # nasty way to do a cartesian product...
        self.dfs['statuses_template'] = (
            self.dfs['atom_intrapop'][atomic_keys]
            .drop_duplicates()
            .assign(dummy=1)
            .merge(
                pd.DataFrame({'dummy': [1, 1], 'period': ['P1', 'P2']}),
                on='dummy',
            )
            .drop('dummy', axis=1)
            .assign(status=False)
        )
        self.compute_discriminant()

    @param.depends('indicator_cut', watch=True)
    def compute_discriminant(self):
        # should be put elsewhere...
        atomic_keys = ['orgacom', 'client']

        self.dfs['atom_intrapop']['discriminant'] = (
            self.dfs['atom_intrapop'][self.indicator_cut] /
            (
                self.dfs['atom_intrapop']
                .groupby(atomic_keys + ['period'])[self.indicator_cut]
                .transform('sum')
            )
        )
        self.compute_statuses()

    @param.depends('canal_cut', 'threshold_cut', 'indicator_cut', watch=True)
    def compute_statuses(self):
        # should be put elsewhere...
        atomic_keys = ['orgacom', 'client']

        self.dfs['statuses'] = self.dfs['statuses_template'].copy()
        self.dfs['statuses'].loc[:, 'status'] = False
        status_index = (
            self.dfs['atom_intrapop']
            .loc[
                lambda x: (
                    (x['discriminant'] >= self.threshold_cut / 100) &
                    (x['origin2'] == self.canal_cut)
                ),
                atomic_keys + ['period']
            ]
            .drop_duplicates()
            .set_index(atomic_keys + ['period'])
            .index
        )
        self.dfs['statuses'] = (
            self.dfs['statuses'].set_index(atomic_keys + ['period'])
        )
        self.dfs['statuses'].loc[
            status_index,
            'status'
        ] = True
        # "Flatten" statuses dataframe
        self.dfs['statuses'] = (
            self.dfs['statuses']
            .unstack('period')
            .stack(level=0)
            .reset_index(-1, drop=True)
            .reset_index()
        )
        self.dfs['statuses']['group'] = (
            self.dfs['statuses']
            .apply(
                lambda x: self.group_names[(x['P1'], x['P2'])],
                axis=1,
            )
        )
        self.compute_grid_data()

    def compute_grid_data(self):
        self.dfs['grid_data'] = compute_grouped(
            self.dfs['atom_intrapop']
            .drop('discriminant', axis=1)
            .set_index(['orgacom', 'client'])
            .join(self.dfs['clt']['seg3'])
            .join(
                self.dfs['statuses']
                .set_index(['orgacom', 'client'])['group']
            )
            .reset_index(),
            atomic_keys=['orgacom', 'client'],
            population_keys=['orgacom', 'seg3', 'group'],
            subatomic_keys=['period', 'origin2'],
            keys_to_expand=['orgacom', 'seg3', 'group', 'period'],
        ).reset_index()
        self.compute_group_summaries()

    def compute_group_summaries(self):
        population_keys = ['orgacom', 'seg3']
        # sum all indicators except 'size' which is constant for every group
        # and for which we default to 'mean'
        agg_funcs = {
            **{col: 'sum' for col in (
                self.dfs['grid_data'].select_dtypes(include=np.number)
            )},
            **{'size': 'mean'},
        }

        data = (
            self.dfs['grid_data']
            .groupby(['group', 'period'] + population_keys)
            .agg(agg_funcs)
            .unstack('period', fill_value=0)
            .swaplevel(axis=1)
        )
        evo = pd.concat(
            [(data['P2'] - data['P1']) / data['P1']],
            axis=1,
            keys=['evo']
        )
        data = pd.concat([data, evo], axis=1).sort_index(axis=1)
        self.dfs['group_summaries'] = data

        indicators = {
            'size': 'Effectif',
            'brutrevenue': 'CA brut (€)',
            'brutrevenue_perbusday': 'CA brut (€/j.o.)',
            'margin': 'Marge (€)',
            'margin_perbusday': 'Marge (€/j.o.)',
            'weight': 'Tonnage (kg)',
            'weight_perbusday': 'Tonnage (kg/j.o)',
        }
        data = (
            self.dfs['group_summaries']
            .loc[
                idx[:, self.orgacom, self.seg3],
                idx[:, indicators]
            ]
            .T
            .unstack('period')
            .droplevel(population_keys, axis=1)
            .reindex(pd.Index(indicators))
            # .reset_index()
            .assign(long_indicator=lambda x: x.index.map(indicators))
            .set_index('long_indicator')
        )

        self.dfs['formatted_group_summaries'] = data

    def sankey(self, data):
        return(hv.Sankey(data))

    @param.depends(
        'orgacom',
        'seg3',
        'indicator',
        'rolling_window',
        'd1period1',
        'd2period1',
        'd1period2',
        'd2period2',
        watch=True,
    )
    def show_curve(self):
        data = self.dfs['orgacom_origin2_date']
        return(
            self.curve_indicator(
                data,
                orgacom=self.orgacom,
                indicator=self.indicator,
                spans=(
                    (self.d1period1, self.d2period1),
                    (self.d1period2, self.d2period2)
                )
            )
        )

    @param.depends(
        'orgacom',
        'seg3',
        'd1period1',
        'd2period1',
        'd1period2',
        'd2period2',
        'indicator_cut',
        'canal_cut',
        'threshold_cut'
    )
    def show_sankey(self):
        data = (
            self.dfs['statuses']
            .set_index(['orgacom', 'client'])
            .join(self.dfs['clt'].loc[:, 'seg3'])
            .loc[self.orgacom]
            .groupby(['P1', 'P2', 'seg3'])
            .size()
            .rename('size')
            .reset_index()
            .loc[lambda x: (x['seg3'] == self.seg3)]
            .groupby(['P1', 'P2'])
            .sum()['size']
        )

        sankey_input = []
        translation = {
            (0, True): 'WebP1',
            (0, False): 'noWebP1',
            (1, True): 'WebP2',
            (1, False): 'noWebP2',
        }

        for line in data.iteritems():
            nodes = list(enumerate(line[0]))
            if line[1] > 0:
                sankey_input.append(
                    (
                        translation[nodes[0]],
                        translation[nodes[1]],
                        line[1]
                    )
                )
        return(self.sankey(sankey_input))

    @param.depends(
        'orgacom',
        'seg3',
        'd1period1',
        'd2period1',
        'd1period2',
        'd2period2',
        'threshold_cut',
        'canal_cut',
        'indicator_cut',
        'indicator_grid',
        'perbusday_grid',
    )
    def grid(self):
        hv_ds = hv.Dataset(
            self.dfs['grid_data'],
            kdims=['period', 'origin2', 'group'],
        )
        self.inspect = hv_ds
        indicator = self.indicator_grid
        if self.perbusday_grid:
            indicator += '_perbusday'
        return(
            hv_ds
            .select(orgacom=self.orgacom, seg3=self.seg3)
            .to(
                hv.Bars,
                kdims=['period', 'origin2'],
                vdims=hv.Dimension(indicator, label=lib(indicator)),
                )
            .opts(stacked=True, cmap=colormaps['origin2'], show_legend=False)
            .layout(['group'])
        )

    @param.depends(
        'orgacom',
        'seg3',
        'd1period1',
        'd2period1',
        'd1period2',
        'd2period2',
        'threshold_cut',
        'canal_cut',
        'indicator_cut',
    )
    def group_summaries(self):
        population_keys = ['orgacom', 'seg3']

        indicators = {
            'size': 'Effectif',
            'brutrevenue': 'CA brut (€)',
            'brutrevenue_perbusday': 'CA brut (€/j.o.)',
            'margin': 'Marge (€)',
            'margin_perbusday': 'Marge (€/j.o.)',
            'weight': 'Tonnage (kg)',
            'weight_perbusday': 'Tonnage (kg/j.o)',
        }

        row = []
        data = (
            self.dfs['group_summaries']
            .loc[
                idx[:, self.orgacom, self.seg3],
                idx[:, indicators]
            ]
            .T
            .unstack('period')
            .droplevel(population_keys, axis=1)
            .reindex(pd.Index(indicators))
            # .reset_index()
            .assign(long_indicator=lambda x: x.index.map(indicators))
            .set_index('long_indicator')
        )

        for group_name in sorted(
            self.dfs['group_summaries'].reset_index('group').group.unique()
        ):
            html = (
                data.loc[:, group_name]
            ).to_html(
                formatters={
                    'P1': lambda x: f'{x:.0f}',
                    'P2': lambda x: f'{x:.0f}',
                    'evo': lambda x: f'{x:.2%}',
                },
                # justify='center',
                na_rep='-',
                index_names=False,
                table_id='my_table',
            )
            row.append(pn.pane.HTML(html))
        row2 = [pn.layout.HSpacer()]
        for pane in row[:-1]:
            row2.extend([pane, pn.layout.HSpacer(), pn.layout.HSpacer()])
        row2.extend([row[-1], pn.layout.HSpacer()])
        return(pn.Row(*row2))

    def group_summary_with_plot(self, group):
        barplot = self.barplot(group, self.indicator_grid)
        summary_table = self.group_summary(group)
        col = pn.Column(
            pn.Row(
                pn.layout.HSpacer(),
                pn.pane.HTML(f'<h3>{group}</h3>'),
                pn.layout.HSpacer(),
            ),
            barplot,
            pn.Row(
                pn.layout.HSpacer(),
                summary_table,
                pn.layout.HSpacer(),
            )
        )
        return(col)

    def holomap_barplot(self, indicator):
        hv_ds = hv.Dataset(
            self.dfs['grid_data'],
            kdims=['period', 'origin2', 'group'],
        )
        self.inspect = hv_ds
        indicator = self.indicator_grid
        if self.perbusday_grid:
            indicator += '_perbusday'
        return(
            hv_ds
            .select(orgacom=self.orgacom, seg3=self.seg3)
            .to(
                hv.Bars,
                kdims=['group', 'period', 'origin2'],
                vdims=[indicator],
                )
            # .opts(stacked=True, cmap=colormaps['origin2'], show_legend=False)
        )

    def barplot(self, group, indicator):
        hv_ds = hv.Dataset(
            self.dfs['grid_data'],
            kdims=['period', 'origin2', 'group'],
        ).select(group=group)
        self.inspect = hv_ds
        indicator = self.indicator_grid
        if self.perbusday_grid:
            indicator += '_perbusday'
        return(
            hv_ds
            .select(orgacom=self.orgacom, seg3=self.seg3)
            .to(
                hv.Bars,
                kdims=['period', 'origin2'],
                vdims=[indicator],
                )
            .opts(stacked=True, cmap=colormaps['origin2'], show_legend=False)
        )

    def group_summary(self, group):
        data = self.dfs['formatted_group_summaries']
        html = (
            data.loc[:, group]
        ).to_html(
            formatters={
                'P1': lambda x: f'{x:.0f}',
                'P2': lambda x: f'{x:.0f}',
                'evo': lambda x: f'{x:.2%}',
            },
            # justify='center',
            na_rep='-',
            index_names=False,
            table_id='my_table',
        )

        return(pn.pane.HTML(html))

    def group_summaries_row(self):
        groups = ['Adopteurs', 'Ignoreurs', 'Fidèles', 'Abandonneurs']
        row = pn.Row(
            *[
                self.group_summary_with_plot(group)
                for group in groups
            ]
        )
        return(row)

    @param.depends(
        'orgacom',
        'seg3',
        'd1period1',
        'd2period1',
        'd1period2',
        'd2period2',
        watch=True,
    )
    def update_venn(self):
        data = (
            self.dfs['scope_orgacom_seg3']
            .loc[idx[self.orgacom, self.seg3], :]
        )
        pop1 = data.loc['lost'] + data.loc['in_scope']
        pop2 = data.loc['new'] + data.loc['in_scope']
        common = data.loc['in_scope']
        self.venn_object.update_CDS(pop1, pop2, common)

    # @param.depends(
    #     'orgacom',
    #     'seg3',
    #     'd1period1',
    #     'd2period1',
    #     'd1period2',
    #     'd2period2',
    #     watch=True,
    # )
    # def venn_with_filters(self):
    #     # DEPRECATED! Now using a VennDiagram object (to enable linking of
    #     # panel widgets and bokeh plot)
    #     data = (
    #         self.dfs['scope_orgacom_seg3']
    #         .loc[idx[self.orgacom, self.seg3], :]
    #     )
    #     pop1 = data.loc['lost'] + data.loc['in_scope']
    #     pop2 = data.loc['new'] + data.loc['in_scope']
    #     common = data.loc['in_scope']
    #     return(venn_diagram(pop1, pop2, common))

    def create_venn_panel(self):
        data = (
            self.dfs['scope_orgacom_seg3']
            .loc[idx[self.orgacom, self.seg3], :]
        )
        pop1 = data.loc['lost'] + data.loc['in_scope']
        pop2 = data.loc['new'] + data.loc['in_scope']
        common = data.loc['in_scope']
        self.venn_object = VennDiagram(
            initial_pop1=pop1,
            initial_pop2=pop2,
            initial_common=common,
            fontsize='9pt',
        )
        self.venn_bokeh_pane = pn.pane.Bokeh(self.venn_object.f, height=320)


def webprogress_dashboard(
    orders_path=persist_path / 'orders.pkl',
    clt_path=persist_path / 'clt.pkl',
    **kwargs,
):

    webprogress = WebProgressShow(
        orders_path=orders_path,
        clt_path=clt_path,
        **kwargs,
    )

    parameters_width = 300

    # for debugging purpose
    background = None  # '#0000FF'

    dashboard = (
        pn.Column(
            pn.Row(
                webprogress.param.orgacom,
                webprogress.param.seg3,
                align='center',
            ),
            pn.Row(
                pn.Column(
                    pn.layout.VSpacer(background=background),
                    webprogress.param.indicator,
                    webprogress.param.rolling_window,
                    pn.Row(
                        webprogress.param.d1period1,
                        webprogress.param.d2period1,
                        sizing_mode='scale_width',
                    ),
                    pn.Row(
                        webprogress.param.d1period2,
                        webprogress.param.d2period2,
                        sizing_mode='scale_width',
                    ),
                    webprogress.param.computation_locker,
                    pn.layout.VSpacer(background=background),
                    width=parameters_width,
                    sizing_mode='stretch_height',
                ),
                pn.layout.HSpacer(background=background),
                hv.DynamicMap(webprogress.show_curve, cache_size=1).opts(
                    opts.Curve(width=600, framewise=True, axiswise=True),
                    opts.Overlay(legend_position='top', responsive=True),
                ),
                pn.layout.HSpacer(background=background),
                pn.Column(
                    pn.layout.VSpacer(background=background),
                    webprogress.venn_bokeh_pane,
                    pn.layout.VSpacer(background=background),
                ),
                pn.layout.HSpacer(background=background),
                max_width=1800,
                # height=500,
                # sizing_mode='scale_both',
            ),
            pn.layout.VSpacer(background=background, min_height=20),
            pn.Row(
                pn.Column(
                    pn.layout.VSpacer(background=background),
                    webprogress.param.canal_cut,
                    webprogress.param.indicator_cut,
                    webprogress.param.threshold_cut,
                    pn.layout.VSpacer(background=background),
                    width=parameters_width,
                    sizing_mode='stretch_height',
                ),
                pn.layout.HSpacer(background=background),
                hv.DynamicMap(webprogress.show_sankey).opts(
                    opts.Sankey(width=600, height=300, label_position='outer'),
                ),
                pn.layout.HSpacer(background=background),
            ),
            pn.layout.VSpacer(background=background, min_height=20),
            pn.Row(
                pn.Column(
                    pn.layout.VSpacer(background=background),
                    webprogress.param.indicator_grid,
                    webprogress.param.perbusday_grid,
                    pn.layout.VSpacer(background=background),
                    width=parameters_width,
                    sizing_mode='stretch_height',
                ),
                pn.Column(
                    webprogress.grid,
                    webprogress.group_summaries,
                ),
            ),
            pn.Row(
                webprogress.param.orgacom,
                webprogress.param.seg3,
                align='center',
            ),
        )
    )
    return(dashboard)


def under_cord_area(r, alpha):
    return(r ** 2 * (alpha - cos(alpha) * sin(alpha)))


def get_angles_from_lengths(r1, r2, d):
    try:
        alpha_1 = acos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
        alpha_2 = acos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
        return(alpha_1, alpha_2)
    except (ValueError, ZeroDivisionError):
        return(0., 0.)


def intersect_area(r1, r2, d):
    if d > (r1 + r2):
        return(0)
    if d <= abs(r1 - r2):
        return(pi * min(r1, r2)**2)
    alpha_1, alpha_2 = get_angles_from_lengths(r1, r2, d)
    area_1, area_2 = under_cord_area(r1, alpha_1), under_cord_area(r2, alpha_2)
    return(area_1 + area_2)


def bounds(r1, r2):
    return(abs(r1 - r2), r1 + r2)


def optim(
    func,
    bounds,
    target,
    sign,
    delta_frac=0.001,
    n_iter=50,
    debug=False
):
    # sign : +1 si fct croissante, -1 si décroissante
    # 1 calcule les bornes et le delta
    if sign == 1:
        min_, max_ = func(bounds[0]), func(bounds[1])
    else:
        min_, max_ = func(bounds[1]), func(bounds[0])
    delta = delta_frac * abs(max_ - min_)

    cpt = 0
    new_val, x = min_, bounds[0]
    low, high = bounds[0], bounds[1]
    while True:
        if debug:
            print(
                f'cpt: {cpt}',
                f'x: {x}',
                f'low: {low}',
                f'high: {high}',
                f'current: {new_val}',
                f'error: {abs(new_val - target)}',
            )
        x = (high + low) / 2
        last_val = new_val
        new_val = func(x)
        if sign * new_val > sign * target:
            high = x
        else:
            low = x
        if abs(new_val - last_val) < delta:
            break
        cpt += 1
        if cpt >= n_iter:
            break

    return(x)


class VennDiagram(object):

    def __init__(
        self,
        colors=['blue', 'red'],
        labels=('Perdus', 'Communs', 'Gagnés'),
        labels2=('Période 1', 'Période 2'),
        fontsize='10pt',
        initial_pop1=None,
        initial_pop2=None,
        initial_common=None,
        debug=False,
    ):
        self.colors = colors
        self.labels = labels
        self.labels2 = labels2
        self.fontsize = fontsize
        self.debug = debug

        self.f = figure(match_aspect=True, aspect_scale=1)

        # remove grid and axes and toolbar
        self.f.xgrid.visible = False
        self.f.ygrid.visible = False
        self.f.axis.visible = False
        self.f.plot_width = 320
        self.f.plot_height = 320
        self.f.toolbar_location = None

        # main circles
        self.circle_source = ColumnDataSource(dict(
            x=[0, 0],
            y=[0, 0],
            radius=[0, 0],
            color=colors,
        ))
        self.f.circle(
            x='x',
            y='y',
            radius='radius',
            alpha=.3,
            color='color',
            line_width=4,
            source=self.circle_source,
        )

        # invisible circle for border range update
        self.border_source = ColumnDataSource(dict(
            x=[],
            y=[],
        ))
        border_circles = self.f.circle(x='x', y='y', source=self.border_source)
        border_circles.visible = False

        # bottom annotations
        # area labels (bottom)
        self.area_labels_source = ColumnDataSource(dict(
            x=[],
            y=[],
            text=[],
        ))
        area_labels = LabelSet(
            source=self.area_labels_source,
            x='x',
            y='y',
            text='text',
            text_align='center',
            text_font_size=self.fontsize,
        )
        self.f.add_layout(area_labels)
        # area center small circles
        self.area_center_source = ColumnDataSource(dict(
            x=[],
            y=[],
        ))
        self.f.circle(
            x='x',
            y='y',
            size=5,
            color='black',
            source=self.area_center_source
        )
        # bottom annotations arrows
        self.bottom_arrows_source = ColumnDataSource(dict(
            xs=[],
            ys=[],
            line_color=[],
        ))
        self.f.multi_line(
            xs='xs',
            ys='ys',
            line_color='line_color',
            source=self.bottom_arrows_source,
        )

        # top annotations
        # target top annotation circles
        self.top_annotation_circles_source = ColumnDataSource(dict(
            x=[],
            y=[],
            color=[],
        ))
        self.f.circle(
            x='x',
            y='y',
            size=5,
            color='color',
            source=self.top_annotation_circles_source,
        )
        # top annotations texts
        self.circle_labels_source = ColumnDataSource(dict(
            x=[],
            y=[],
            text=[],
            color=[],
        ))
        labs2 = LabelSet(
            source=self.circle_labels_source,
            x='x',
            y='y',
            text_color='color',
            text='text',
            text_align='center',
            text_baseline='middle',
            render_mode='canvas',
            text_font_size=self.fontsize,
        )
        self.f.add_layout(labs2)
        # top annotations arrows
        self.top_arrows_source = ColumnDataSource(dict(
            xs=[],
            ys=[],
            line_color=[],
        ))
        self.f.multi_line(
            source=self.top_arrows_source,
            xs='xs',
            ys='ys',
            line_color='line_color',
        )

        if initial_common and initial_pop1 and initial_pop2:
            self.update_CDS(initial_pop1, initial_pop2, initial_common)

    def compute_values(self, pop1, pop2, common):

        self.pop1, self.pop2, self.common = pop1, pop2, common

        # computation of radii and distance
        self.r1, self.r2 = (pop1 / pi) ** .5, (pop2 / pi) ** .5
        self.common = common

        self.distance = optim(
            func=partial(intersect_area, self.r1, self.r2),
            bounds=bounds(self.r1, self.r2),
            target=self.common,
            sign=-1,
            delta_frac=.00001,
        )

    def update_CDS(self, pop1, pop2, common):
        # compute radii and distance
        self.compute_values(pop1, pop2, common)
        if self.debug:
            print(
                f'pop1: {self.pop1}\n'
                f'pop2: {self.pop2}\n'
                f'common: {self.common}\n'
                f'r1: {self.r1}\n'
                f'r2: {self.r2}\n'
                f'distance: {self.distance}\n'
            )

        # main circles datasource
        self.circle_source.data = dict(
            x=[0, self.distance],
            y=[0, 0],
            radius=[self.r1, self.r2],
            color=self.colors,
        )

        # helpers for ranges computation
        padding = max(self.r1, self.r2) * .1
        xmin, xmax = -self.r1 - padding, self.distance + self.r2 + padding
        xwidth = 2 * padding + self.r1 + self.r2 + self.distance
        ywidth = 6 * padding + 2 * max(self.r1, self.r2)
        size = max(xwidth, ywidth)
        xcenter = (-self.r1 + self.distance + self.r2) / 2
        ycenter = 0

        # update of border circles datasource
        self.border_source.data = dict(
            x=[xcenter - size / 2, xcenter + size / 2],
            y=[ycenter - size / 2, ycenter + size / 2],
        )

        # bottom annotations
        # labels for areas
        annotation_height = -12 * padding
        pop_counts = [
            self.pop1 - self.common,
            self.common,
            self.pop2 - self.common
        ]
        labels = [self.labels[i] + f' :\n{pop_counts[i]}' for i in range(3)]
        xlabels = np.linspace(xmin, xmax, 7)[1::2]
        self.area_labels_source.data = dict(
            text=labels,
            x=xlabels,
            y=[annotation_height] * len(xlabels),
        )
        # small circles in center of areas (targets of arrows)
        area_centers = [
            (-self.r1 + self.distance - self.r2)/2,
            (self.r1 + self.distance - self.r2)/2,
            (self.r1 + self.distance + self.r2) / 2,
        ]
        self.area_center_source.data = dict(
            x=area_centers,
            y=[0, 0, 0]
        )
        # bottom annotations arrows
        self.bottom_arrows_source.data = dict(
            xs=[[xlabels[i], area_centers[i]] for i in range(3)],
            ys=[[annotation_height + 1.5 * padding, 0] for _ in range(3)],
            line_color=['k'] * 3,
        )

        # top annotation
        # anchors for top annotations target circles
        alpha1, alpha2 = get_angles_from_lengths(
            self.r1, self.r2, self.distance)
        beta1, beta2 = (pi + alpha1) / 2, (pi + alpha2) / 2
        anchors_x = [
            cos(beta1) * self.r1,
            -cos(beta2) * self.r2 + self.distance
        ]
        anchors_y = [sin(beta1) * self.r1, sin(beta2) * self.r2]
        # top annotation circles
        self.top_annotation_circles_source.data = dict(
            x=anchors_x,
            y=anchors_y,
            color=self.colors,
        )
        # top annotations texts
        annotation_height2 = 12 * padding
        labels2 = [
            self.labels2[0] + f' :\n{self.pop1}',
            self.labels2[1] + f' :\n{self.pop2}',
            ]
        xlabels2 = [xlabels[0], xlabels[-1]]
        self.circle_labels_source.data = dict(
            text=labels2,
            x=xlabels2,
            y=[annotation_height2] * len(labels2),
            color=self.colors,
        )
        # top annotations arrows
        self.top_arrows_source.data = dict(
            xs=[[xlabels2[i], anchors_x[i]] for i in range(2)],
            ys=[
                [annotation_height2 - padding, anchors_y[i]]
                for i in range(2)
                ],
            line_color=self.colors,
        )


def venn_diagram(
    pop1,
    pop2,
    common,
    labels=('Perdus', 'Communs', 'Gagnés'),
    labels2=('Période 1', 'Période 2'),
    colors=('blue', 'red'),
    fontsize='10pt',
    debug=False,
):
    r1, r2 = (pop1 / pi) ** .5, (pop2 / pi) ** .5

    distance = optim(
        func=partial(intersect_area, r1, r2),
        bounds=bounds(r1, r2),
        target=common,
        sign=-1,
        delta_frac=.00001,
    )

    # screen dim
    padding = max(r1, r2) * .1
    xmin, xmax = -r1 - padding, distance + r2 + padding
    xwidth = 2 * padding + r1 + r2 + distance
    ywidth = 6 * padding + 2 * max(r1, r2)
    size = max(xwidth, ywidth)
    xcenter = (-r1 + distance + r2) / 2
    ycenter = (0)

    # labels for areas
    annotation_height = -12 * padding
    pop_counts = [
        pop1 - common,
        common,
        pop2 - common
    ]
    labels = [labels[i] + f' :\n{pop_counts[i]}' for i in range(3)]
    xlabels = np.linspace(xmin, xmax, 7)[1::2]
    area_centers = [
        (-r1 + distance - r2)/2,
        (r1 + distance - r2)/2,
        (r1 + distance + r2) / 2,
    ]

    # labels for circles
    annotation_height2 = 12 * padding
    labels2 = [labels2[0] + f' :\n{pop1}', labels2[1] + f' :\n{pop2}']
    xlabels2 = [xlabels[0], xlabels[-1]]

    # anchors for top annotations
    alpha1, alpha2 = get_angles_from_lengths(r1, r2, distance)
    beta1, beta2 = (pi + alpha1) / 2, (pi + alpha2) / 2
    anchors_x = [cos(beta1) * r1, -cos(beta2) * r2 + distance]
    anchors_y = [sin(beta1) * r1, sin(beta2) * r2]

    f = figure(match_aspect=True, aspect_scale=1)

    # main circles
    circle_source = ColumnDataSource(dict(
        x=[0, distance],
        y=[0, 0],
        radius=[r1, r2],
        color=colors,
    ))
    f.circle(
        x='x',
        y='y',
        radius='radius',
        alpha=.3,
        color='color',
        line_width=4,
        source=circle_source,
    )

    # bottom annotations
    # target bottom annotations circles
    f.circle(x=area_centers, y=[0, 0, 0], size=5, color='black')
    # bottom anontations texts
    labs = LabelSet(
        source=ColumnDataSource(
            dict(
                text=labels, x=xlabels, y=[annotation_height] * len(xlabels),
            ),
        ),
        x='x',
        y='y',
        text='text',
        text_align='center',
        text_font_size=fontsize,
    )
    f.add_layout(labs)
    # bottom annotations arrows
    f.multi_line(
        xs=[[xlabels[i], area_centers[i]] for i in range(3)],
        ys=[[annotation_height + 1.5 * padding, 0] for _ in range(3)],
        line_color=['k'] * 3,
    )

    # top annotations
    # target top annotation circles
    f.circle(x=anchors_x, y=anchors_y, size=5, color=colors)
    # top annotations texts
    labs2 = LabelSet(
        source=ColumnDataSource(
            dict(
                text=labels2,
                x=xlabels2,
                y=[annotation_height2] * len(labels2),
                color=colors,
            ),
        ),
        x='x',
        y='y',
        text_color='color',
        text='text',
        text_align='center',
        text_baseline='middle',
        render_mode='canvas',
        text_font_size=fontsize,
    )
    f.add_layout(labs2)
    # top annotations arrows
    f.multi_line(
        xs=[[xlabels2[i], anchors_x[i]] for i in range(2)],
        ys=[[annotation_height2 - padding, anchors_y[i]] for i in range(2)],
        line_color=colors,
    )

    # remove grid and axes
    f.xgrid.visible = False
    f.ygrid.visible = False
    f.axis.visible = False

    # figure dimensionss
    f.x_range.start = xcenter - size/2
    f.x_range.end = xcenter + size/2
    f.y_range.start = ycenter - size/2
    f.y_range.end = ycenter + size/2
    f.plot_width = 320
    f.plot_height = 320

    if debug:
        print(
            f'r1: {r1:.3}',
            f'r2: {r2:.3}',
            f'padding: {padding:.3}',
            f'annonation_height1: {annotation_height:.3}',
            f'annonation_height2: {annotation_height2:.3}',
            f'xwidth: {xwidth:.3}',
            f'ywidth: {ywidth:.3}',
            f'size: {size:.3}',
            f'axis_range_x: {f.x_range.start:.3} -> {f.x_range.end:.3} = '
            f'{f.x_range.end - f.x_range.start:.3}',
            f'axis_range_y: {f.y_range.start:.3} -> {f.y_range.end:.3} = '
            f'{f.y_range.end - f.y_range.start:.3}',
            f'plot_width x plot_height: {f.plot_width} x {f.plot_height}',
            sep='\n',
        )

    f.toolbar_location = None

    return(f)


class MarginAnalyzer(param.Parameterized):

    orgacom = param.ObjectSelector(
        objects=suc_libs_inv,
        default='1ALO',
        label='Succursale',
    )

    # seg3_l = param.ObjectSelector(
    #     objects={
    #         'Restauration commerciale indépendante': 'RCI',
    #         'Restauration commerciale structurée': 'RCS',
    #         'Restauration collective autogérée': 'RCA',
    #         'Restauration collective concédée': 'RCC',
    #     },
    #     default='RCI',
    #     label='Segment 3',
    # )

    date1 = param.CalendarDate(
        dt.date(2019, 1, 1),
        bounds=(dt.date(2017, 1, 1), dt.date(2021, 2, 1)),
        label='Date début',
    )

    date2 = param.CalendarDate(
        dt.date(2019, 2, 1),
        bounds=(dt.date(2017, 1, 1), dt.date(2021, 2, 1)),
        label='Date fin',
    )

    stop_cost = param.Number(
        0.,
        bounds=(0, 200),
        label='Coup de frein (€)',
        step=5.0,
        )

    prepa_cost = param.Number(
        0.,
        bounds=(0., 1.),
        label='Coûts logistiques (€/kg)',
        step=.001,
    )

    line_cost = param.Number(
        0.,
        bounds=(0., 1.),
        label='Coûts logistiques (€/ligne)',
        step=.001,
    )

    def __init__(
        self,
        orders_path=persist_path / 'small_orders.pkl',
        clt_path=persist_path / 'clt.pkl'
    ):
        super().__init__()
        self.orders = pd.read_pickle(orders_path)
        self.clt = pd.read_pickle(clt_path)
        self.clt['seg3_l'] = self.clt['seg3'].map(seg3_dict)
        self.datasource = self.datasource_from_filters(
            order_data=self.orders,
            clt_data=self.clt,
            orgacom=self.orgacom,
            min_date=self.date1,
            max_date=self.date2,
        )
        self.datasource = self.compute_adjusted_margin(
            data=self.datasource,
            stop_cost=self.stop_cost,
            prepa_cost=self.prepa_cost,
            line_cost=self.line_cost,
        )
        self.update_composite_indicators()
        self.init_CDS()
        self.create_bokeh_pane()
        self.update_CDS()

    def init_CDS(self):
        self.CDS = ColumnDataSource(dict(
            **{var: [] for var in self.datasource.columns},
            **{var: [] for var in ['x', 'y', 'size', 'fill_color']},
        ))

    def create_bokeh_pane(self):
        p = figure(plot_height=500, plot_width=800)
        p.circle(
            source=self.CDS,
            x='x',
            y='y',
            size='size',  # size
            fill_color='fill_color',
            alpha=.7,
            legend_field='seg3_l',
            line_width=0.,
            )
        p.legend.location = 'bottom_right'
        self.bokeh_pane = pn.pane.Bokeh(p)

    def datasource_from_filters(
        self,
        order_data=None,
        clt_data=None,
        orgacom=None,
        min_date=None,
        max_date=None,
    ):
        data = (
            self.orders
            .reset_index()
            .loc[
                lambda x:
                    (x['date'].dt.date >= min_date) &
                    (x['date'].dt.date <= max_date) &
                    (x.orgacom == orgacom)
            ]
            .groupby(['orgacom', 'client'], observed=True)
            .agg(
                margin=('margin', 'sum'),
                brutrevenue=('brutrevenue', 'sum'),
                weight=('weight', 'sum'),
                linecount=('linecount', 'sum'),
                ordercount=('weight', 'size'),
            )
            .join(clt_data[['seg3', 'seg3_l', 'hier4']])
            .loc[lambda x: ~pd.isna(x['seg3_l'])]
        )
        return(data)

    @param.depends('orgacom', 'date1', 'date2', watch=True)
    def update_datasource(self):
        self.datasource = self.datasource_from_filters(
            order_data=self.orders,
            clt_data=self.clt,
            orgacom=self.orgacom,
            min_date=self.date1,
            max_date=self.date2,
        )
        self.update_adjusted_margin()

    def compute_adjusted_margin(
        self,
        data=None,
        stop_cost=None,
        prepa_cost=None,
        line_cost=None,
        inplace=True,
    ):
        if not inplace:
            data = data.copy()
        data['stop_cost'] = data['ordercount'] * stop_cost
        data['prepa_cost'] = data['weight'] * prepa_cost
        data['line_cost'] = data['linecount'] * line_cost
        data['adjusted_margin'] = (
            self.datasource['margin']
            - self.datasource['stop_cost']
            - self.datasource['prepa_cost']
            - self.datasource['line_cost']
        )
        return(data)

    @param.depends('stop_cost', 'prepa_cost', 'line_cost', watch=True)
    def update_adjusted_margin(self):
        self.compute_adjusted_margin(
            data=self.datasource,
            stop_cost=self.stop_cost,
            prepa_cost=self.prepa_cost,
            line_cost=self.line_cost,
        )
        self.update_composite_indicators()
        self.update_CDS()

    def update_CDS(self):
        self.datasource['x'] = self.datasource['brutrevenue']
        self.datasource['y'] = self.datasource['adjusted_margin']
        self.datasource['size'] = 10
        self.datasource['fill_color'] = (
            self.datasource['seg3'].map(colormaps['seg3'])
        )
        self.CDS.data = ColumnDataSource.from_df(self.datasource)
        self.bokeh_pane.param.trigger('object')

    def update_composite_indicators(self):
        indicator_dict = dict(
            **composite_indicators_dict,
            **{
                    'adjusted_marginperkg': ['adjusted_margin', 'weight'],
                    'adjusted_marginpercent': [
                        'adjusted_margin',
                        'brutrevenue'
                    ],
            }
        )
        self.datasource = compute_composite_indicators(
            self.datasource,
            indicator_defs=indicator_dict,
            )


periods = {
    '6jan_fev': (
        (
            dt.date.fromisoformat('2019-01-01'),
            dt.date.fromisoformat('2019-02-28')
        ),
        (
            dt.date.fromisoformat('2020-01-01'),
            dt.date.fromisoformat('2020-02-29')
        )
    ),
    '5nov_dec': (
        (
            dt.date.fromisoformat('2018-11-01'),
            dt.date.fromisoformat('2018-12-31')
        ),
        (
            dt.date.fromisoformat('2019-11-01'),
            dt.date.fromisoformat('2019-12-31')
        )
    ),
    '4sep_oct': (
        (
            dt.date.fromisoformat('2018-09-01'),
            dt.date.fromisoformat('2018-10-31')
        ),
        (
            dt.date.fromisoformat('2019-09-01'),
            dt.date.fromisoformat('2019-10-31')
        )
    ),
    '3jui_aou': (
        (
            dt.date.fromisoformat('2018-07-01'),
            dt.date.fromisoformat('2018-08-31')
        ),
        (
            dt.date.fromisoformat('2019-07-01'),
            dt.date.fromisoformat('2019-08-31')
        )
    ),
    '2mai_jui': (
        (
            dt.date.fromisoformat('2018-05-01'),
            dt.date.fromisoformat('2018-06-30')
        ),
        (
            dt.date.fromisoformat('2019-05-01'),
            dt.date.fromisoformat('2019-06-30')
        )
    ),
    '1mar_avr': (
        (
            dt.date.fromisoformat('2018-03-01'),
            dt.date.fromisoformat('2018-04-30')
        ),
        (
            dt.date.fromisoformat('2019-03-01'),
            dt.date.fromisoformat('2019-04-30')
        )
    ),
}


class ComparativeWebprogress(param.Parameterized):

    # orgacom = param.ObjectSelector(
    #     objects={
    #         **suc_libs_inv,
    #         **{
    #             'Toutes': None,
    #         }
    #     },
    #     default='1ALO',
    #     label='Succursale',
    # )

    orgacom = param.ListSelector(
        default=['1ALO'],
        objects={
            k: suc_libs_inv[k]
            for k in sorted(suc_libs_inv.keys())
        },
        label='Succursale',
    )

    PF_button = param.Action(
        lambda x: x.sel_PF(),
        label='PassionFroid',
    )

    ES_button = param.Action(
        lambda x: x.sel_ES(),
        label='EpiSaveurs',
    )

    DEL_button = param.Action(
        lambda x: x.sel_DEL(),
        label='Effacer',
    )

    seg3_l = param.ObjectSelector(
        objects={
            'Restauration commerciale indépendante': 'RCI',
            'Restauration commerciale structurée': 'RCS',
            'Restauration collective autogérée': 'RCA',
            'Restauration collective concédée': 'RCC',
        },
        default='RCI',
        label='Segment 3',
    )

    # period_key = param.ObjectSelector(
    #     objects=list(periods.keys()) + [None],
    #     default='6jan_fev',
    #     label='Périodes à comparer',
    # )

    period_key = param.ListSelector(
        objects=list(periods.keys()),
        default=['6jan_fev'],
        label='Périodes à comparer',
    )

    period_all_button = param.Action(
        lambda x: x.sel_period('All'),
        label='Toutes',
    )

    period_clear_button = param.Action(
        lambda x: x.sel_period('None'),
        label='Effacer',
    )

    def __init__(
        self,
        periods=periods,
        orders_path=persist_path / 'orders.pkl',
        clt_path=persist_path / 'clt.pkl',
        **kwargs,
    ):
        print('Initialisation called')
        super().__init__()
        self.webprogress = WebProgressShow(
            orders_path=orders_path,
            clt_path=clt_path,
            **kwargs,
        )
        self.data_structure = {
            'population': [
                'orgacom',
                'seg3',
                'seg3_l',
                'group',
                'period_key',
            ],
            'pop_size': 'size',
            'observation': [
                'period',
                'origin2',
            ]
        }
        self.restitutions = {
            'bars': [
                'group',
                'period',
                'origin2',
            ],
            'table': [
                'group',
                'period',
            ],
        }
        self.accumulation_indicators = [
            'margin',
            'brutrevenue',
            'weight',
            'linecount',
            'ordercount',
            'margin_perbusday',
            'brutrevenue_perbusday',
            'weight_perbusday',
            'linecount_perbusday',
            'ordercount_perbusday',
        ]
        self.table_indicators = [
            'size',
            'weight_perbusday',
            'brutrevenue_perbusday',
            'margin_perbusday',
            'PMVK',
            'marginperkg',
            'marginpercent',
            'linecount_perbusday',
            'lineperorder',
        ]
        self.dimension_dict = {
            'weight_perbusday': {'label': 'Tonnage', 'unit': 'kg/j.o.'},
            'margin_perbusday': {'label': 'Marge', 'unit': '€/j.o.'},
            'brutrevenue_perbusday': {'label': 'CA brut', 'unit': '€/j.o.'},
            'linecount_perbusday': {'label': 'Nb lignes', 'unit': '/j.o.'},
        }
        self.composite_indicators_dict = composite_indicators_dict
        self.periods = periods
        self.init_grid_df()
        # self.init_formatted()
        self.update_from_selection()
        # self.update_bar_layout()  # Not required anymore...

    # @param.depends('PF_button', watch=True)
    def sel_PF(self):
        self.sel_ocs(criterion='PPF')

    # @param.depends('ES_button', watch=True)
    def sel_ES(self):
        self.sel_ocs(criterion='PES')

    # @param.depends('DEL_button', watch=True)
    def sel_DEL(self):
        self.sel_ocs(criterion='DEL')

    def sel_period(
        self,
        period=None
    ):
        if period == 'All':
            self.period_key = self.param.period_key.objects
        elif period == 'None':
            self.period_key = []

    def sel_ocs(
        self,
        criterion=None,
    ):
        if criterion == 'PPF':
            self.orgacom = [
                oc for oc in suc_libs_inv.values() if oc[0] == '1'
            ]
        elif criterion == 'PES':
            self.orgacom = [
                oc for oc in suc_libs_inv.values() if oc[0] == '2'
            ]
        elif criterion == 'DEL':
            self.orgacom = []

    def init_grid_df(self):
        grid_dfs = dict()
        for period_key, date_ranges in periods.items():
            self.webprogress.computation_locker = True
            print(period_key)
            self.webprogress.d1period1 = date_ranges[0][0]
            self.webprogress.d2period1 = date_ranges[0][1]
            self.webprogress.d1period2 = date_ranges[1][0]
            self.webprogress.d2period2 = date_ranges[1][1]
            self.webprogress.computation_locker = False
            grid_dfs[period_key] = self.webprogress.dfs['grid_data'].copy()
        for period_key, df in grid_dfs.items():
            df['period_key'] = period_key
        grid_df = pd.concat(list(grid_dfs.values()), axis=0)
        grid_df['seg3_l'] = grid_df.seg3.map(seg3_dict)
        grid_df = grid_df.sort_values(
            ['orgacom', 'seg3', 'group', 'period_key', 'period', 'origin2']
        )
        self.grid_df = grid_df

    def init_formatted(self):
        # DEPRECATED!
        formatted = (
            self.grid_df
            .loc[~pd.isna(self.grid_df.seg3_l)]
            .groupby([
                'orgacom',
                'seg3_l',
                'period_key',
                'group',
                'period',
                'size'
            ])
            .sum()
            .fillna(0.)
            .assign(
                margin_perkg=lambda x: x.margin / x.weight,
                margin_percent=lambda x: x.margin / x.brutrevenue,
                PMVK=lambda x: x.brutrevenue / x.weight,
            )
            .unstack('period')
            .swaplevel(axis=1).sort_index(axis=1)
            .pipe(ComparativeWebprogress.compute_evol)
            .swaplevel(axis=1).sort_index(axis=1)
            .reset_index('size')
            .reorder_levels(['group', 'period_key', 'orgacom', 'seg3_l'])
            .sort_index()
            .pipe(ComparativeWebprogress.compute_delta)
            .reorder_levels(['period_key', 'orgacom', 'seg3_l', 'group'])
            .sort_index()
            .pipe(ComparativeWebprogress.reorder_cols)
        )
        formatted.loc[idx[:, :, :, 'Comparaison'], ('size', '')] = None
        self.formatted = formatted

        agg_formatted = (
            self.grid_df.loc[~pd.isna(self.grid_df.seg3_l)]
            .groupby([
                'orgacom',
                'seg3_l',
                'period_key',
                'group',
                'period',
                'size',
            ])
            .sum()
            .reset_index('size')
            .assign(
                total_size=lambda x: x.groupby(
                    [
                        'orgacom',
                        'seg3_l',
                        'group',
                        'period',
                    ])['size'].transform('sum'),
                **{ind + suf:
                    (lambda x, indic=ind+suf:
                        (x[indic] * x['size'] / x['total_size']))
                    for ind in [
                        'margin',
                        'brutrevenue',
                        'weight',
                        'linecount',
                        'ordercount'
                    ]
                    for suf in ['', '_perbusday']}
                    )
            .drop('total_size', axis=1)
            .groupby(['orgacom', 'seg3_l', 'group', 'period'])
            .sum()
            .set_index('size', append=True)
            .assign(
                margin_perkg=lambda x: x.margin / x.weight,
                margin_percent=lambda x: x.margin / x.brutrevenue,
                PMVK=lambda x: x.brutrevenue / x.weight,
            )
            .unstack('period')
            .swaplevel(axis=1).sort_index(axis=1)
            .pipe(ComparativeWebprogress.compute_evol)
            .swaplevel(axis=1).sort_index(axis=1)
            .reset_index('size')
            .reorder_levels(['group', 'orgacom', 'seg3_l'])
            .sort_index()
            .pipe(ComparativeWebprogress.compute_delta)
            .reorder_levels(['orgacom', 'seg3_l', 'group'])
            .sort_index()
            .pipe(ComparativeWebprogress.reorder_cols)
        )
        agg_formatted.loc[idx[:, :, 'Comparaison'], ('size', '')] = None
        self.agg_formatted = agg_formatted

        oc_agg_formatted = (
            self.grid_df.loc[~pd.isna(self.grid_df.seg3_l)]
            .groupby([
                # 'orgacom',
                'seg3_l',
                'period_key',
                'group',
                'period',
                'size',
            ])
            .sum()
            .reset_index('size')
            .assign(
                total_size=lambda x: x.groupby(
                    [
                        # 'orgacom',
                        'seg3_l',
                        'group',
                        'period',
                    ])['size'].transform('sum'),
                **{ind + suf:
                    (lambda x, indic=ind+suf:
                        (x[indic] * x['size'] / x['total_size']))
                    for ind in [
                        'margin',
                        'brutrevenue',
                        'weight',
                        'linecount',
                        'ordercount'
                    ]
                    for suf in ['', '_perbusday']}
                    )
            .drop('total_size', axis=1)
            .groupby([
                # 'orgacom',
                'seg3_l',
                'group',
                'period'
            ])
            .sum()
            .set_index('size', append=True)
            .assign(
                margin_perkg=lambda x: x.margin / x.weight,
                margin_percent=lambda x: x.margin / x.brutrevenue,
                PMVK=lambda x: x.brutrevenue / x.weight,
            )
            .unstack('period')
            .swaplevel(axis=1).sort_index(axis=1)
            .pipe(ComparativeWebprogress.compute_evol)
            .swaplevel(axis=1).sort_index(axis=1)
            .reset_index('size')
            .reorder_levels([
                'group',
                # 'orgacom',
                'seg3_l'
            ])
            .sort_index()
            .pipe(ComparativeWebprogress.compute_delta)
            .reorder_levels([
                # 'orgacom',
                'seg3_l',
                'group'
            ])
            .sort_index()
            .pipe(ComparativeWebprogress.reorder_cols)
        )
        oc_agg_formatted.loc[idx[:, 'Comparaison'], ('size', '')] = None
        self.oc_agg_formatted = oc_agg_formatted

    @staticmethod
    def format_number(number):
        if float(number) > 0:
            style = '"color:green;"'
        elif number < 0:
            style = '"color:red;"'
        else:
            style = '"color:black;"'
        return(f'<b style={style}>{number:.2%}</b>')

    @staticmethod
    def compute_diff(
        df,
        index=None,
        diff_col=None,
        ref_val=None,
        new_val=None,
        target_val=None,
        append=True,
        ratio=False,
    ):
        # takes a df without index as input.
        # the values in the index should be unique (must they?)
        # if append == True, then the initial df with the new rows is provided
        # if ratio == True, then the comparison is divided by the reference
        # value (e.g. to compute a percentage evolution)
        if diff_col not in index:
            raise ValueError(f'{diff_col} should be in the index list.')
        if (
            new_val not in df[diff_col].values or
            ref_val not in df[diff_col].values
        ):
            if append:
                return(df)
            else:
                return(pd.DataFrame(index=df.index, columns=df.columns))
        col_indexed = (
            df
            .loc[df[diff_col].isin([ref_val, new_val])]
            .set_index(index)
            .unstack(diff_col, fill_value=0)
            .swaplevel(i=0, j=diff_col, axis=1)
            .sort_index(axis=1)
        )
        to_append = (
            col_indexed.loc[:, new_val] - col_indexed.loc[:, ref_val]
        )
        if ratio:
            to_append = to_append.div(col_indexed.loc[:, ref_val], axis=0)
        to_append = pd.concat(
            [to_append],
            axis=1,
            keys=[target_val],
            names=[diff_col]
        ).stack(diff_col).reset_index(index)
        if not append:
            return(to_append)
        else:
            return(pd.concat([df, to_append], axis=0))

    @staticmethod
    def compute_delta(
        df,
        ref_idx='Ignoreurs',
        new_idx='Adopteurs',
        key_idx='Comparaison',
    ):
        # DEPRECATED! SHOULD USE compute_diff instead
        new = pd.concat(
            [df.loc[idx[new_idx], :] - df.loc[idx[ref_idx], :]],
            keys=[key_idx],
        )
        return(pd.concat([df, new]))

    @staticmethod
    def compute_evol(
        df,
        ref_col='P1',
        new_col='P2',
        key_idx='evo',
    ):
        # TO DEPRECATE! Use compute_diff instead
        df_evol_ratio = (
            (df.loc[:, idx[new_col]] - df.loc[:, idx[ref_col]])
            / df.loc[:, idx[ref_col]]
        )
        return(
            pd.concat(
                [
                    df.loc[:, idx[ref_col]],
                    df.loc[:, idx[new_col]],
                    df_evol_ratio
                ],
                axis=1,
                keys=[ref_col, new_col, key_idx],
            )
        )

    @staticmethod
    def reorder_cols(
        df,
        lev1_order=[
            'weight_perbusday',
            'brutrevenue_perbusday',
            'margin_perbusday',
            'PMVK',
            'margin_perkg',
            'margin_percent',
        ],
        lev2_order=['P1', 'P2', 'evo'],
    ):
        # DEPRECATED!
        return(
            df.reindex(
                [('size', '')] + list(product(lev1_order, lev2_order)),
                axis=1,
            )
        )

    dict_rep = {
        '>size<': '>Nb<',
        'weight_perbusday': 'Tonnage (kg/j.o.)',
        'brutrevenue_perbusday': 'CA brut (€/j.o.)',
        'margin_perbusday': 'Marge (€/j.o.)',
        'PMVK': 'PMVK (€/kg)',
        'marginperkg': 'Marge kg (€/kg)',
        'marginpercent': 'Marge % (%)',
        'linecount_perbusday': 'Nb lignes (/j.o.)',
        'lineperorder': 'Nb lignes/com',
    }

    @staticmethod
    def dict_replace(string, dict_replace=dict_rep):
        for k, v in dict_replace.items():
            string = string.replace(k, v)
        return(string)

    def table(
        self,
        dict_rep=dict_rep,
        indicators=None,
    ):
        if not indicators:
            indicators = self.table_indicators
        data = (
            self.aggregations['table']
            .set_index(self.restitutions['table'])
            .unstack('period')
        )
        data = self._merge_with_scaling_factor_df(
            df=data,
            sfactor_df=self.sizes,
            df_keys=self.restitutions['table'],
            sfactor_keys=self.data_structure['population'],
            sfactor_name='size',
        ).set_index('group')
        try:
            data = data.loc[
                ['Adopteurs', 'Ignoreurs', 'Comparaison'],
                indicators,
            ]
        except KeyError:
            return(None)

        html = (
            data
            .to_html(
                header=True,
                index_names=False,
                index=True,
                escape=False,
                formatters={
                    **{('size', ''): lambda x: f'{x:.0f}'},
                    **{
                        (indicator, period): lambda x: f'{x:.2f}' for indicator
                        in data.columns.get_level_values(0)
                        for period in ('P1', 'P2')
                    },
                    **{
                        ('marginpercent', period): lambda x: f'{x:.2%}'
                        for period in ('P1', 'P2')
                    },
                    **{
                        (indicator, 'evo'):
                            ComparativeWebprogress.format_number for indicator
                            in data.columns.get_level_values(0)
                    },
                },
            )
        )

        html += "<style>\n"
        html += "th {text-align: center; horizontal-align: center; "
        html += "padding-left: 3px; padding-right: 3px; font-size: 1.em}\n"
        html += "td {text-align: center;padding-left: 3px; padding-right: 3px;"
        html += " font-size: 1.em;}\n"
        html += "</style>"
        html = (
            html
            .replace(' class="dataframe"', '')
            .replace('      <th></th>\n      <th></th>\n', '')
            .replace('<th></th>', '<th rowspan="2"></th>')
            .replace('<th>size</th>', '<th rowspan="2">size</th>')
            .replace('nan', '-')
            .replace('NaN', '-')
        )
        html = ComparativeWebprogress.dict_replace(html, dict_replace=dict_rep)
        return(html)

    def table_old(
        self,
        period_key=None,
        orgacom=None,
        seg3_l=None,
        groups_of_interest=['Adopteurs', 'Ignoreurs', 'Comparaison'],
        dict_rep=dict_rep,
    ):
        # DEPRECATED !
        if period_key:
            data = (
                self.formatted
                .loc[idx[period_key, orgacom, seg3_l, groups_of_interest], :]
                # .droplevel(list(range(3)))
            )
        else:
            if orgacom:
                data = (
                    self.agg_formatted
                    .loc[idx[orgacom, seg3_l, groups_of_interest], :]
                    # .droplevel(list(range(2)))
                )
            else:
                data = (
                    self.oc_agg_formatted
                    .loc[idx[seg3_l, groups_of_interest], :]
                    # .droplevel(list(range(2)))
                )
        html = (
            data
            .droplevel(data.index.names[:-1])
            .to_html(
                header=True,
                index_names=False,
                index=True,
                escape=False,
                formatters={
                    **{('size', ''): lambda x: f'{x:.0f}'},
                    **{
                        (indicator, period): lambda x: f'{x:.2f}' for indicator
                        in self.formatted.columns.get_level_values(0)
                        for period in ('P1', 'P2')
                    },
                    **{
                        ('margin_percent', period): lambda x: f'{x:.2%}'
                        for period in ('P1', 'P2')
                    },
                    **{
                        (indicator, 'evo'):
                            ComparativeWebprogress.format_number for indicator
                            in self.formatted.columns.get_level_values(0)
                    },
                },
            )
        )

        html += "<style>\n"
        html += "th {text-align: center; horizontal-align: center; "
        html += "padding-left: 3px; padding-right: 3px; font-size: 1.em}\n"
        html += "td {text-align: center;padding-left: 3px; padding-right: 3px;"
        html += " font-size: 1.em;}\n"
        html += "</style>"
        html = (
            html
            .replace(' class="dataframe"', '')
            .replace('      <th></th>\n      <th></th>\n', '')
            .replace('<th></th>', '<th rowspan="2"></th>')
            .replace('<th>size</th>', '<th rowspan="2">size</th>')
            .replace('nan', '-')
            .replace('NaN', '-')
        )
        html = ComparativeWebprogress.dict_replace(html, dict_replace=dict_rep)
        return(html)

    def single_bar(
        self,
        indicator=None,
        group=None,
        hv_group=None,
        hv_label=None,
    ):
        hv_group = hv_group if hv_group else 'Bars'
        hv_label = hv_label if hv_label else ''

        data = self.aggregations['bars']
        data = data.loc[
            (data['group'] == group) &
            (data['period'].isin(['P1', 'P2'])),
            ['period', 'origin2', indicator]
        ]

        bar = hv.Bars(
            data,
            kdims=['period', 'origin2'],
            vdims=hv.Dimension(indicator, **self.dimension_dict[indicator]),
            # dynamic=True,
            group=hv_group,
            label=hv_label,
        ).opts(
            stacked=True,
            cmap=colormaps['origin2'],
            show_legend=False,
            width=120,
            axiswise=True,
            framewise=True,
            bar_width=.5,
            title='',
            xlabel='',
            height=250,
            toolbar=None,
        )
        return(bar)

    def single_bar_old(
        self,
        indicator=None,
        orgacom=None,
        seg3_l=None,
        group=None,
        period_key=None,
        hv_group=None,
        hv_label=None,
    ):
        # DEPRECATED
        indicator_dict = {
            'weight_perbusday': {'label': 'Tonnage', 'unit': 'kg/j.o.'},
            'margin_perbusday': {'label': 'Marge', 'unit': '€/j.o.'},
            'brutrevenue_perbusday': {'label': 'CA brut', 'unit': '€/j.o.'},
        }

        hv_group = hv_group if hv_group else 'Bars'
        hv_label = hv_label if hv_label else ''

        hv_ds = hv.Dataset(
            self.grid_df,
            kdims=[
                'orgacom',
                'seg3_l',
                'group',
                'period_key',
                'period',
                'origin2'
            ],
            vdims=[
                'margin',
                'brutrevenue',
                'weight',
                'linecount',
                'ordercount',
                'margin_perbusday',
                'brutrevenue_perbusday',
                'weight_perbusday',
                'linecount_perbusday',
                'ordercount_perbusday'
            ],
        )
        if not period_key:
            hv_ds = hv_ds.reduce(period_key=np.mean)
        if not orgacom:
            hv_ds = hv_ds.reduce(orgacom=np.mean)
        bar = hv_ds.to(
            hv.Bars,
            kdims=['period', 'origin2'],
            vdims=hv.Dimension(indicator, **indicator_dict[indicator]),
            dynamic=True,
            group=hv_group,
            label=hv_label,
        ).opts(
            stacked=True,
            cmap=colormaps['origin2'],
            show_legend=False,
            width=120,
            axiswise=True,
            framewise=True,
            bar_width=.5,
            title='',
            xlabel='',
            height=250,
            toolbar=None,
        )
        if period_key:
            bar = bar.select(
                orgacom=orgacom,
                group=group,
                seg3_l=seg3_l,
                period_key=period_key,
            )
        else:
            bar = bar.select(
                orgacom=orgacom,
                group=group,
                seg3_l=seg3_l,
            )
        return(bar)

    def pair_barplot(
        self,
        indicator=None,
        orgacom=None,
        seg3_l=None,
        groups=None,
        period_key=None,
    ):
        bar1 = self.single_bar(
            indicator=indicator,
            # orgacom=orgacom,
            # seg3_l=seg3_l,
            # period_key=period_key,
            group=groups[0],
            hv_group=indicator,
            hv_label='Target',
        )
        bar2 = self.single_bar(
            indicator=indicator,
            # orgacom=orgacom,
            # seg3_l=seg3_l,
            # period_key=period_key,
            group=groups[1],
            hv_group=indicator,
            hv_label='Reference',
        ).opts(width=100, ylabel='')
        return((bar1 + bar2).opts(toolbar=None))

    @param.depends('orgacom', 'seg3_l', 'period_key', watch=True)
    def update_from_selection(self):
        self.init_source(individual_input=True)
        self.compute_sizes()
        self.compute_aggregations()
        self.compute_individual()
        self.compute_composite()
        self.compute_evolution()
        self.compute_comparison()

    def init_source(
        self,
        individual_input=False,
    ):
        self.source = self.grid_df.loc[
            self.grid_df['orgacom'].isin(self.orgacom) &
            (self.grid_df['seg3_l'] == self.seg3_l) &
            self.grid_df['period_key'].isin(self.period_key)
        ].copy()
        if individual_input:
            # compute whole group total values if necessary
            self.source[self.accumulation_indicators] = (
                self.source[self.accumulation_indicators].mul(
                    self.source[self.data_structure['pop_size']],
                    axis=0,
                )
            )

    def compute_sizes(
        self,
    ):
        # an ugly patch, pop sizes should always be outside the main dataframe
        data_structure = self.data_structure
        self.sizes = (
            self.source
            .groupby(
                data_structure['population'] + data_structure['observation'],
                observed=True,
                as_index=False,
            )
            [data_structure['pop_size']]
            .sum()
            .drop(data_structure['observation'], axis=1)
            .drop_duplicates()
        )

    def compute_aggregations(self):
        self.aggregations = dict()
        for rest_name, rest_def in self.restitutions.items():
            self.aggregations[rest_name] = ComparativeWebprogress.compute_agg(
                df=self.source,
                groupers=rest_def,
                accumulation_indicators=self.accumulation_indicators,
            )

    @staticmethod
    def compute_agg(
        df=None,
        groupers=None,
        accumulation_indicators=None,
    ):
        df = (
            df
            .groupby(
                groupers,
                observed=True,
                as_index=False)
            [accumulation_indicators]
            .sum()
        )
        return(df)

    @staticmethod
    def _merge_with_scaling_factor_df(
        df=None,
        sfactor_df=None,
        df_keys=None,
        sfactor_keys=None,
        sfactor_name='size',
        how='left',
    ):
        df_keys, sfactor_keys = set(df_keys), set(sfactor_keys)
        intersect_keys = list(sfactor_keys.intersection(df_keys))
        grouped_sizes = (
            sfactor_df
            .groupby(
                intersect_keys,
                observed=True,
                as_index=False,
            )
            [sfactor_name]
            .sum()
        )
        # realign columns level counts in case of multiindex
        df, grouped_sizes = ComparativeWebprogress.align_index_levels(
            df,
            grouped_sizes,
            axis=1,
            fill_bottom=True,
        )
        return(df.merge(grouped_sizes, on=intersect_keys, how=how))

    @staticmethod
    def align_index_levels(
        df,
        other,
        axis=0,
        fill_bottom=True,
    ):
        # returns both df with index levels padded with '' (empty string)
        # so they have the same length
        # use axis=1 to pad columns multiindex
        # when fill_bottom is true the resulting multiindex will be filled
        # on the last levels
        # does not modify dataframes in place, returns copies
        df, other = df.copy(), other.copy()
        df_levs, other_levs = df.axes[axis].nlevels, other.axes[axis].nlevels
        if df_levs > other_levs:
            diff = df_levs - other_levs
            other = ComparativeWebprogress.append_dummy_levels(
                other,
                lev_count=diff,
                axis=axis,
                fill_bottom=fill_bottom,
            )
        elif df_levs < other_levs:
            diff = other_levs - df_levs
            df = ComparativeWebprogress.append_dummy_levels(
                df,
                lev_count=diff,
                axis=axis,
                fill_bottom=fill_bottom,
            )
        return(df, other)

    @staticmethod
    def append_dummy_levels(
        df,
        lev_count=1,
        axis=0,
        fill_bottom=True,
    ):
        # appends dummy (empty) levels to the index of axis axis
        # returns a copy
        df = df.copy()
        init_nlevs = df.axes[axis].nlevels
        df = pd.concat(
            {('', ) * lev_count: df},
            names=[''] * lev_count,
            axis=axis,
        )
        if fill_bottom:
            full_list = list(range(df.axes[axis].nlevels))
            new_order = full_list[-init_nlevs:] + full_list[:-init_nlevs]
            df = df.reorder_levels(
                new_order,
                axis=axis,
            )
        return(df)

    def compute_individual(self):
        # scaling factor for population (sizes)
        # heuristic is: you only keep in scaling factor table key fields that
        # are in data table, groupby/sum, and merge with broadcast on data
        # table
        # not ideal, but replaces aggregations dataframes
        for rest_name, rest_def in self.restitutions.items():
            with_sizes = ComparativeWebprogress._merge_with_scaling_factor_df(
                df=self.aggregations[rest_name],
                sfactor_df=self.sizes,
                df_keys=rest_def,
                sfactor_keys=self.data_structure['population'],
                sfactor_name='size',
            )
            individualed = (
                with_sizes[self.accumulation_indicators]
                .div(with_sizes['size'], axis=0)
            )
            # patch the values in the aggregations dataframe
            self.aggregations[rest_name][self.accumulation_indicators] = (
                individualed[self.accumulation_indicators]
            )

    def compute_composite(self):
        # compute composite indicators from accumulation indicators
        for indicator, definition in self.composite_indicators_dict.items():
            for agg_name, agg_df in self.aggregations.items():
                agg_df[indicator] = (
                    agg_df[definition[0]] / agg_df[definition[1]]
                )

    def compute_evolution(self):
        for agg_name, agg_df in self.aggregations.items():
            self.aggregations[agg_name] = ComparativeWebprogress.compute_diff(
                agg_df,
                index=self.restitutions[agg_name],
                diff_col='period',
                ref_val='P1',
                new_val='P2',
                target_val='evo',
                append=True,
                ratio=True,
            )

    def compute_comparison(self):
        for agg_name, agg_df in self.aggregations.items():
            self.aggregations[agg_name] = ComparativeWebprogress.compute_diff(
                agg_df,
                index=self.restitutions[agg_name],
                diff_col='group',
                ref_val='Ignoreurs',
                new_val='Adopteurs',
                target_val='Comparaison',
                append=True,
                ratio=False,
            )

    # # @param.depends('orgacom', 'seg3_l', 'period_key', watch=True)
    # def update_bar_layout(
    #     self,
    # ):
    #     # DEPRECATED! Is not useful anymore.
    #     layout = hv.Layout([
    #         self.pair_barplot(
    #             indicator=indicator,
    #             orgacom=self.orgacom,
    #             seg3_l=self.seg3_l,
    #             groups=['Adopteurs', 'Ignoreurs'],
    #             period_key=self.period_key,
    #         ) for indicator in [
    #             'weight_perbusday',
    #             'brutrevenue_perbusday',
    #             'margin_perbusday'
    #         ]
    #     ]).opts(toolbar=None)
    #     self.layout = layout

    # # @param.depends('orgacom', 'seg3_l', 'period_key')
    # def get_weight_barplot(self):
    #     # DEPRECATED! Is not useful anymore.
    #     return(self.get_pair_barplot(indicator='weight_perbusday'))

    # def get_pair_barplot(
    #     self,
    #     indicator=None,
    # ):
    #     # DEPRECATED! Is not useful anymore.
    #     indicator = indicator.capitalize()
    #     return(self.layout[indicator])

    @param.depends('orgacom', 'seg3_l', 'period_key')
    def dashboard_title(self):
        title = (
            f'{self.orgacom_descriptor()} - {self.seg3_l}'
            f' - {self.period_descriptor()}'
        )
        return(title)

    def orgacom_descriptor(
        self,
    ):
        if (
            set(self.orgacom) ==
            {oc for oc in suc_libs_inv.values() if oc[0] == '1'}
        ):
            return('Branche PassionFroid')
        if (
            set(self.orgacom) ==
            {oc for oc in suc_libs_inv.values() if oc[0] == '2'}
        ):
            return('Branche EpiSaveurs')
        if len(self.orgacom) == 1:
            return(f'Succursale {self.orgacom[0]}')
        else:
            return('Multiples SV')

    def period_descriptor(
        self,
    ):
        period_l = {
            '6jan_fev': 'Jan & Fev 2019/2020',
            '5nov_dec': 'Nov & Dec 2018/2019',
            '4sep_oct': 'Sep & Oct 2018/2019',
            '3jui_aou': 'Juil & Aou 2018/2019',
            '2mai_jui': 'Mai & Juin 2018/2019',
            '1mar_avr': 'Mar & Avr 2018/2019',
        }
        if len(self.period_key) == len(self.param.period_key.objects):
            return('Toutes périodes')
        elif len(self.period_key) == 1:
            return(period_l[self.period_key[0]])
        else:
            return('Multiples périodes')

    def dashboard_3_plots(
        self,
        indicators=[
            'weight_perbusday',
            'brutrevenue_perbusday',
            'margin_perbusday',
            'linecount_perbusday',
        ],
        orgacom=None,
        seg3_l=None,
        groups=['Adopteurs', 'Ignoreurs'],
        period_key=None,
        title=None,
        bgcolor=None,  # for layout debugging purpose
    ):
        print('dashboard 3 plots called!')
        orgacom = orgacom if orgacom else self.orgacom
        period_key = period_key if period_key else self.period_key
        seg3_l = seg3_l if seg3_l else self.seg3_l

        title = title if title else self.dashboard_title()

        plot_area = [
            pn.layout.HSpacer(background=bgcolor),
        ]
        for indicator in indicators:
            plot_area.extend(
                [
                    self.pair_barplot(
                        period_key=period_key,
                        orgacom=orgacom,
                        seg3_l=seg3_l,
                        groups=groups,
                        indicator=indicator,
                    ),
                    pn.layout.HSpacer(background=bgcolor),
                ]
            )

        # TO BE CONTINUED !
        # plot_area = [
        #     pn.layout.HSpacer(background=bgcolor),
        #     *[
        #         *[
        #             self.pair_barplot(
        #                 period_key=period_key,
        #                 orgacom=orgacom,
        #                 seg3_l=seg3_l,
        #                 groups=groups,
        #                 indicator=indicator,
        #             ),
        #             pn.layout.HSpacer(background=bgcolor),
        #         ] for indicator in indicators
        #     ],            
        # ]

        dashboard = pn.Column(
            pn.Row(
                pn.layout.HSpacer(background=bgcolor),
                pn.pane.HTML(f'<center><h3>{title}</h3></center>', width=600),
                pn.layout.HSpacer(background=bgcolor),
            ),
            pn.layout.VSpacer(background=bgcolor),
            pn.Row(
                *plot_area
                # pn.layout.HSpacer(background=bgcolor),
                # self.pair_barplot(
                #     period_key=period_key,
                #     orgacom=orgacom,
                #     seg3_l=seg3_l,
                #     groups=groups,
                #     indicator=indicators[0],
                # ),
                # pn.layout.HSpacer(background=bgcolor),
                # self.pair_barplot(
                #     period_key=period_key,
                #     orgacom=orgacom,
                #     seg3_l=seg3_l,
                #     groups=groups,
                #     indicator=indicators[1],
                # ),
                # pn.layout.HSpacer(background=bgcolor),
                # self.pair_barplot(
                #     period_key=period_key,
                #     orgacom=orgacom,
                #     seg3_l=seg3_l,
                #     groups=groups,
                #     indicator=indicators[2],
                # ),
                # pn.layout.HSpacer(background=bgcolor),
            ),
            pn.layout.VSpacer(background=bgcolor),
            pn.pane.HTML(self.table(
                # period_key=period_key,
                # orgacom=orgacom,
                # seg3_l=seg3_l,
                # groups_of_interest=groups + ['Comparaison']
            ))
        )
        return(dashboard)
