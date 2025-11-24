import matplotlib.pyplot as plt
import seaborn as sns


def set_notebook_style():
    sns.set_theme(style="whitegrid", palette="colorblind")
    params = {
        "figure.titlesize": 11,
        "figure.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.titlecolor": "#343030",
        "axes.labelcolor": "#343030",
        "yaxis.labellocation": "top",
        "xaxis.labellocation": "left",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.title_fontsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.loc": "upper left",
        "grid.alpha": 0.4,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
    plt.rcParams.update(params)
