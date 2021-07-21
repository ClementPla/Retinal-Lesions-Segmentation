from bokeh.models import ColumnDataSource, FactorRange
from bokeh.palettes import Spectral5
from bokeh.plotting import figure
from bokeh.transform import factor_cmap


def categoral_group_vbar(df, group_col, sub_col, col, title, plot_height=500, plot_width=500):
    groups = df[group_col]
    sub = df[sub_col]
    x = list(zip(groups, sub))
    counts = [df[(df[group_col] == c[0]) & (df[sub_col] == c[1])][col].array[0] for c in x]
    source = ColumnDataSource(data=dict(x=x, counts=counts))

    p = figure(x_range=FactorRange(*x), plot_height=plot_height, plot_width=plot_width,
               title="%s, %s" % (title, col),
               toolbar_location=None, tools="hover", tooltips="%s score @counts" % col)
    cmap = factor_cmap('x', palette=Spectral5, factors=sorted(sub.unique()), start=1, end=len(sub))
    p.vbar(x='x', top='counts', width=0.9, source=source,
           fill_color=cmap, line_color='black', line_alpha=0.25)

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    return p
