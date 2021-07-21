import colorcet as cc
import numpy as np
from bokeh.models import ColumnDataSource, Legend, LegendItem
from bokeh.plotting import figure
from numpy import linspace
from scipy.stats.kde import gaussian_kde


def ridge(category, data, scale=10):
    return list(zip([category] * len(data), scale * data))


def plot_violin(df, df_group_column, df_columns, scales=(1, -1), plot_size=(600, 600), x_range=(-5, 100), title=None):
    groups = df[df_group_column].unique()
    assert len(scales) == len(df_columns)
    palette = [cc.glasbey_cool[(i + 1) * (255 // len(df_columns))] for i in range(len(df_columns))]
    x = linspace(0, x_range[1], 250)

    source = ColumnDataSource(data=dict(x=x))
    p = figure(y_range=groups, plot_height=plot_size[1], plot_width=plot_size[0], x_range=x_range,
               title=title)
    mean_renderers = []
    median_renderers = []
    for i, cat in enumerate(reversed(groups)):
        j = 0
        for m, s in zip(df_columns, scales):
            measures = df.loc[df[df_group_column] == cat][m]
            pdf = gaussian_kde(measures)
            data = pdf(x)
            data[0] = 0
            data[-1] = 0
            y = ridge(cat, data, scale=s)
            source.add(y, cat + m)
            p.patch(x='x', y=cat + m, color=palette[j], line_color='gray', alpha=0.5, source=source,
                    muted_alpha=0.1, legend_label=m)
            mean = np.mean(measures)
            r = p.vbar(x=mean,
                       top='top_%s' % cat,
                       bottom='bottom_%s' % cat, color='#00008b',
                       width=0.01,
                       muted_alpha=0,
                       source={'bottom_%s' % cat: [[cat, 0]],
                               'top_%s' % cat: [[cat, pdf(mean)[0] * s]]})
            mean_renderers.append(r)

            median = np.median(measures)
            r = p.vbar(x=median,
                       top='top_%s' % cat,
                       bottom='bottom_%s' % cat,
                       color='#8B0000',
                       muted_alpha=0,
                       width=0.01,
                       source={'bottom_%s' % cat: [[cat, 0]],
                               'top_%s' % cat: [[cat, pdf(median)[0] * s]]})
            median_renderers.append(r)
            j += 1

    li1 = LegendItem(label='mean', renderers=mean_renderers)
    li2 = LegendItem(label='median', renderers=median_renderers)
    legend = Legend(items=[li1, li2], location='bottom_right')
    p.add_layout(legend)
    p.xgrid.ticker = p.xaxis.ticker
    p.axis.axis_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.click_policy = "mute"
    p.legend.background_fill_alpha = 0.1
    return p


_NUMERALS = '0123456789abcdefABCDEF'
_HEXDEC = {v: int(v, 16) for v in (x + y for x in _NUMERALS for y in _NUMERALS)}


def rgb(triplet):
    return np.asarray((_HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]], _HEXDEC[triplet[4:6]]))


def mix_img_and_mask(img, masks):
    img = img * 0.7
    colors = [rgb(cc.glasbey_light[(i + 1) * (255 // len(masks))][1:]) for i in range(len(masks))]
    for m, c in zip(masks, colors):
        img[m > 0] = c * 0.5 + 0.5 * img[m > 0]
    return img.astype(np.uint8)
