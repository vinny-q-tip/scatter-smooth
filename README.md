# scatter-smooth
This repository contains a Python function to visualize scatterplots with various smoothing methods.

Currently, there is support for least squares regression (linear and any polynomial degree), LOWESS (locally weighted scatterplot smoothing), and smoothing splines (with polynomials of a degree of less than 6). The function also exposes varios parameters from matplotlib to allow for great customizability of the plot.

## Example usages

`plot_scatterdata(xs, ys, smoother='linear')`

`plot_scatterdata(xs, ys, smoother='poly', degree=7)`

`plot_scatterdata(xs, ys, smoother='lowess', low_frac=0.33)`

`plot_scatterdata(xs, ys, smoother='splines', spl_smooth=26500, degree=3)`