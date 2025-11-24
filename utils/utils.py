from matplotlib.ticker import FuncFormatter
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from pandas.api.types import is_integer_dtype, is_float_dtype
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
import os
import shap
import random

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def format_numbers(x: int | float, pos: int, count: bool = False) -> str:
    """
    Formats the number to a more concise form.

    Args:
        x (int | float): The number to be formatted.
        pos (int): The position argument for formatting function. Not used here.
        count (bool): If display number with one or two decimal numbers. Defaults to False.

    Returns:
        str: A formatted string representation of a number.
    """
    abs_x = abs(x)
    if abs_x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    elif abs_x >= 1_000:
        return f"{x / 1_000:.1f}k"
    else:
        if count:
            return f"{x:.1f}"
        else:
            return f"{x:.2f}"


def winsorize_features(
    df: pd.DataFrame, features: list[str], lower: float = None, upper: float = None
) -> pd.DataFrame:
    """
    Clips the feature value between lower and upper percentiles.

    Args:
        df (pd.DataFrame): The input dataframe.
        features (list[str]): List of features.
        lower (float, optional): The lower percentile limit. Defaults to None.
        upper (float, optional): The upper percentile limit. Defaults to None.

    Returns:
        pd.DataFrame: The dataframe with the clipped features.
    """
    for feature in features:
        lower_bound = df[feature].quantile(lower) if lower is not None else None
        upper_bound = df[feature].quantile(upper) if upper is not None else None
        df.loc[:, feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
    return df


def consolidate_credit_type(credit_type: str) -> str:
    """
    Converts the string to another string values for the feature values.

    Args:
        credit_type (str): Feature value.

    Returns:
        str: The new feature value.
    """
    if credit_type in ["Consumer credit", "Microloan", "Cash loan (non-earmarked)"]:
        return "Installment_Unsecured"
    elif credit_type in [
        "Credit card",
        "Mobile operator loan",
        "Loan for purchase of shares (margin lending)",
        "Interbank credit",
    ]:
        return "Revolving"
    elif credit_type in [
        "Mortgage",
        "Car loan",
        "Real estate loan",
        "Loan for the purchase of equipment",
    ]:
        return "Secured_Installment"
    elif credit_type in [
        "Loan for business development",
        "Loan for working capital replenishment",
    ]:
        return "Business_Loan"
    else:
        return "Other_Unknown"


def get_distribution_plots_with_stat(
    df: pd.DataFrame,
    number_cols: list[str],
    right: float,
    top: float,
    width: float = 7,
    name: str = "Bureau",
    log_scale: bool = False,
    descriptive: bool = True,
    missing: bool = True,
) -> pd.DataFrame:
    """
    Plots KDE and boxplots for the selected numerical features and optionally
    returns descriptive statistics.

    Args:
        df (pd.DataFrame): The input dataframe.
        number_cols (list[str]): The list of numerical features.
        right (float): Spacing of the figure suptitle to the right.
        top (float): Spacing of the figure suptitle to the top.
        width (float): Width of the figure. Defaults to 7.
        name (str): Table name. Defaults to "Bureau".
        log_scale (bool): If to use log-scaled axis. Defaults to False.
        descriptive (bool): If to print descriptive statistics. Defaults to True.
        missing (bool): If to print out missing values statistics. Defaults to True.

    Returns:
        pd.DataFrame: Summary statistics of the selected features.
    """
    fig, axes = plt.subplots(
        len(number_cols), 2, figsize=(width, len(number_cols) * 2), squeeze=False
    )

    for i, col in enumerate(number_cols):
        sns.kdeplot(df[col], ax=axes[i, 0], log_scale=log_scale)
        axes[i, 0].set_title(col)
        axes[i, 0].xaxis.set_major_formatter(
            FuncFormatter(partial(format_numbers, count=True))
        )

        sns.boxplot(x=df[col], ax=axes[i, 1], log_scale=log_scale)
        axes[i, 1].set_title(col)
        axes[i, 1].xaxis.set_major_formatter(
            FuncFormatter(partial(format_numbers, count=True))
        )
    plt.tight_layout()
    fig.suptitle(
        f"KDE and Boxplot of Numerical Features in {name} Dataset", x=right, y=top
    )
    plt.show()

    if missing:
        print("Missing Value Summary:")
        for col in number_cols:
            print(f"{col}: {df[col].isnull().sum()} missing")

    if descriptive:
        desc = df[number_cols].describe().T
        desc = desc.map(lambda x: f"{x:,.1f}")
    return desc


def get_boxplots(
    df: pd.DataFrame,
    features: list[str],
    figsize: tuple[float],
    right: float = 0.2,
    top: float = 1.0,
    log_scale: bool = False,
) -> None:
    """
    Plot boxplots of the selected numerical features.

    Args:
        df (pd.DataFrame): The original dataframe.
        features (list[str]): The list of numerical feature names.
        figsize (tuple[float]): Figure size.
        right (float, optional): Spacing of figure suptitle to the right. Defaults to 0.2.
        top (float, optional): Spacing of figure suptitle to the top. Defaults to 1.0.
        log_scale (bool, optional): If log-scaling of the axis. Defaults to False.
    """
    nrows = (len(features) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.boxplot(data=df, x=col, log_scale=log_scale, ax=axes[i])
        axes[i].set_title(f"{col} Distribution")
        axes[i].xaxis.set_major_formatter(
            FuncFormatter(partial(format_numbers, count=True))
        )
    fig.suptitle("Boxplots of Selected Features", x=right, y=top)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.tight_layout()
    plt.show()


def get_countplots(
    df: pd.DataFrame,
    features: list[str],
    figsize: tuple[float],
    right: float = 0.35,
    top: float = 0.97,
) -> None:
    """
    Plotting distribution plots of categorical features.

    Args:
        df (pd.DataFrame): The original dataframe.
        features (list[str]): The list of categorical features.
        figsize (tuple[float]): Figure size.
        right (float, optional): Spacing of the figure suptitle to the right. Defaults to 0.35.
        top (float, optional): Spacing of the figure suptitle to the top. Defaults to 0.97.
    """
    ncols = 2 if len(features) > 1 else 1
    nrows = (len(features) + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.ravel(np.atleast_1d(axes)).flatten()
    for i, col in enumerate(features):
        df_copy = df.copy()
        df_copy[col] = df_copy[col].fillna("Missing")

        order = (
            df_copy[col]
            .value_counts(normalize=True, dropna=False)
            .sort_values(ascending=False)
            .index
        )
        sns.countplot(data=df_copy, y=col, ax=axes[i], stat="percent", order=order)
        for container in axes[i].containers:
            axes[i].bar_label(
                container,
                fmt=lambda x: f"{x:.1f}%",
                label_type="edge",
                padding=3,
                fontsize=8,
            )
        axes[i].set_title(f"{col} Distribution")

    if len(features) > 1:
        fig.suptitle("Distributions of Selected Features", x=right, y=top)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.tight_layout()
    plt.show()


def group_product_groups(product_col: str) -> pd.Series:
    """
    Handles values of the specified column.

    Args:
        product_col (str): Name of the feature.

    Returns:
        pd.Series: Transformed series of the feature.
    """
    product_col = product_col.fillna("Unknown")

    grouped_col = pd.Series("MISSING", index=product_col.index, dtype=object)

    is_cash = product_col.str.contains("Cash", na=False)
    grouped_col[is_cash] = "Cash"

    is_pos = product_col.str.contains("POS", na=False)
    grouped_col[is_pos] = "POS"

    is_card = product_col.str.contains("Card", na=False)
    grouped_col[is_card] = "Card"

    return grouped_col


def get_kde_plots(
    df: pd.DataFrame,
    cols: list[str],
    figsize: tuple[float],
    name: str = "Previous Application",
    ncols: int = 3,
    right: float = 0.35,
    top: float = 1.01,
) -> None:
    """
    Plot kde plots for the selected numerical features.

    Args:
        df (pd.DataFrame): The original dataframe.
        cols (list[str]): The list of feature names.
        figsize (tuple[float]): Figure size.
        name (str, optional): Table name. Defaults to 'Previous Application'.
        ncols (int, optional): Number of columns. Defaults to 3.
        right (float, optional): Spacing of the suptitle to the right. Defaults to 0.35.
        top (float, optional): Spacing of the suptitle to the top. Defaults to 1.01.
    """
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.kdeplot(x=df[col], ax=axes[i])
        axes[i].set_title(col)

    for i in range(len(cols), len(axes)):
        ax = axes[i]
        ax.axis("off")

    fig.suptitle(f"KDE Plots of Numerical Features in {name} Dataset", x=right, y=top)
    plt.tight_layout()
    plt.show()


def get_distribution_plots(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    figsize: tuple[float | int],
    title_right: float,
    title_top: float,
) -> None:
    """
    Makes kde and boxplots for the selected features.

    Args:
        df (pd.DataFrame): The dataframe containing data.
        features (list[str]): Features to plot.
        target (str): Name of the target feature.
        figsize (tuple[float | int]): Size of the subplot.
        title_right (float): Spacing of the suptitle to the right.
        title_top (float): Spacing of the suptitle to the top.
    """

    fig, axes = plt.subplots(len(features), 2, figsize=figsize)
    if len(features) == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(features):
        low, hight = df[feature].quantile([0.01, 0.99])

        sns.kdeplot(
            data=df, x=feature, hue=target, alpha=0.5, log_scale=False, ax=axes[i, 0]
        )
        axes[i, 0].set_title(f"Distribution of {feature}", fontsize=9)
        axes[i, 0].get_legend().remove()

        sns.boxplot(
            data=df,
            x=feature,
            hue=target,
            orient="h",
            fill=False,
            width=0.5,
            gap=0.2,
            log_scale=False,
            ax=axes[i, 1],
        )

        axes[i, 1].axvline(low, color="orange", linestyle="--", label="1st Percentile")
        axes[i, 1].axvline(
            hight, color="orange", linestyle="--", label="99th Percentile"
        )

        axes[i, 1].set_title(f"Boxplot of {feature}", fontsize=9)
        axes[i, 1].legend(
            title="TARGET", bbox_to_anchor=(1.5, 1), loc="upper right", fontsize=8
        )

    plt.tight_layout()
    fig.suptitle(
        f"Distributions of {feature} by Target Class",
        x=title_right,
        y=title_top,
        fontsize=12,
    )
    plt.show()


def plot_target(
    df: pd.DataFrame, target: str, figsize: tuple[float | int], right: float, top: float
) -> None:
    """
    Plot barplots of the feature.

    Args:
        df (pd.DataFrame): Original dataframe.
        target (str): Selected feature.
        figsize (tuple[int]): Plot size.
        right (float): Moving suptitle to left.
        top (float): Moving suptitle top.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    keywords = ["Count", "Proportion"]
    sns.countplot(data=df, x=target, width=0.6, ax=axes[0])
    sns.countplot(data=df, x=target, width=0.6, stat="percent", ax=axes[1])
    for i, ax in enumerate(axes):
        for bar in ax.patches:
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            if i == 1:
                s = f"{height:.1f}"
            else:
                s = f"{height:,.0f}"
            ax.text(
                x,
                (height + 0.02),
                s=s,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.grid(False)
        ax.yaxis.set_major_formatter(FuncFormatter(partial(format_numbers, count=True)))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No", "Yes"])
        ax.set_title(f"{keywords[i]} of {target} feature", pad=20)

    plt.tight_layout()
    fig.suptitle("Target feature distribution", x=right, y=top)
    plt.show()


def plot_distribution_log(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    figsize: tuple[float | int],
    title_right: float = 0.22,
    title_top: float = 1.02,
    padding: float = 15,
) -> None:
    """
    Plot feature distribution plots using log scaling.

    Args:
        df (pd.DataFrame): The original dataframe.
        features (list[str]): Features to plot distributions for.
        target (str): The target feature.
        figsize (tuple[float | int]): Figure size.
        title_right (float, optional): Spacing of the figure suptitle to the left. Defaults to 0.22.
        title_top (float, optional): Spacing of the figure suptitle to the top. Defaults to 1.02.
        padding (float, optional): Padding after subplot title. Defaults to 15.
    """

    fig, axes = plt.subplots(len(features), 3, figsize=figsize)
    if len(features) == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(features):
        low, hight = df[feature].quantile([0.01, 0.99])

        sns.kdeplot(
            data=df, x=feature, hue=target, alpha=0.5, log_scale=False, ax=axes[i, 0]
        )
        axes[i, 0].set_title(f"Distribution of {feature}", fontsize=9.5, pad=padding)
        axes[i, 0].get_legend().remove()

        sns.kdeplot(
            data=df, x=feature, hue=target, alpha=0.5, log_scale=True, ax=axes[i, 1]
        )
        axes[i, 1].set_title(
            f"Log-Scaled Distribution of {feature}", fontsize=9.5, pad=padding
        )
        axes[i, 1].get_legend().remove()

        sns.boxplot(
            data=df,
            x=feature,
            hue=target,
            orient="h",
            fill=False,
            width=0.5,
            gap=0.2,
            log_scale=True,
            ax=axes[i, 2],
        )
        axes[i, 2].axvline(low, color="orange", linestyle="--", label="1st Percentile")
        axes[i, 2].axvline(
            hight, color="orange", linestyle="--", label="99th Percentile"
        )

        axes[i, 2].set_title(
            f"Log-scaled Boxplot of {feature}", fontsize=9.5, pad=padding
        )
        axes[i, 2].legend(
            title="TARGET", bbox_to_anchor=(1.5, 1), loc="upper right", fontsize=8
        )

    plt.tight_layout()
    fig.suptitle(
        "Distributions of Anomalous Features by Target Class",
        x=title_right,
        y=title_top,
        fontsize=12,
    )
    plt.show()


def plot_discrete_distribution(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    figsize: tuple[float | int],
    title_right: float = 0.2,
    title_top: float = 1.02,
) -> None:
    """
    Plotting distribution plots for the discrete features.

    Args:
        data (pd.DataFrame): The original dataframe.
        features (list[str]): The features to plot distribution for.
        target (str): The target variable.
        figsize (tuple[float | int]): The size of the figure.
        title_right (float, optional): The spacing of the suptitle to the left. Defaults to 0.2.
        title_top (float, optional): The spacing of the suptitle to the top. Defaults to 1.02.
    """
    fig, axes = plt.subplots(len(features), 2, figsize=figsize)
    if len(features) == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(features):
        low, hight = data[feature].quantile([0.01, 0.99])

        sns.histplot(
            data=data,
            x=feature,
            hue=target,
            discrete=True,
            ax=axes[i, 0],
            multiple="dodge",
            shrink=0.8,
            stat="density",
        )
        axes[i, 0].set_title(f"Distribution of {feature}", fontsize=9.5)
        axes[i, 0].get_legend().remove()
        axes[i, 0].set_xticks(
            range(int(data[feature].min()), int(data[feature].max()) + 1, 2)
        )

        sns.boxplot(
            data=data,
            x=feature,
            hue=target,
            orient="h",
            fill=False,
            width=0.5,
            gap=0.2,
            ax=axes[i, 1],
        )
        axes[i, 1].set_xticks(
            range(int(data[feature].min()), int(data[feature].max()) + 1, 2)
        )

        axes[i, 1].axvline(low, color="orange", linestyle="--", label="1st Percentile")
        axes[i, 1].axvline(
            hight, color="orange", linestyle="--", label="99th Percentile"
        )

        axes[i, 1].set_title(f"Boxplot of {feature}", fontsize=9.5)
        axes[i, 1].legend(
            title="TARGET",
            bbox_to_anchor=(1.3, 1.1),
            loc="upper right",
            fontsize=8,
            alignment="left",
        )

    plt.tight_layout()
    fig.suptitle(
        "Distributions of Anomalous Features by Target Class",
        x=title_right,
        y=title_top,
    )
    plt.show()


def plot_triangle_heatmap(
    data: pd.DataFrame,
    figsize: tuple[float | int, float | int],
    annot_size: float | int,
    title: str,
    cmap: str,
    vmax: float | int,
    vmin: float | int,
    label_size: float | int = 8,
    title_size: float | int = 12,
    float_precision: str = ".2f",
) -> None:
    """
    Plots lower triangular heatmap from a given correlation matrix.

    Args:
        data (pd.DataFrame): The correlation matrix.
        figsize (tuple[float|int, float|int]): Width and height of the plot.
        annot_size (float | int): Font size for annotations inside heatmap cells.
        title (str): The title of the heatmap.
        cmap (str): Color palette for the heatmap.
        vmax (float | int): Max value to anchor to the heatmap.
        vmin (float | int): Min value to anchor to the heatmap.
        label_size (float | int, optional): Font size for axes labels. Defaults to 9.
        title_size (float | int, optional): Font size for the title. Defaults to 12.
        float_precision (str, optional): String format for displaying numerical values. Defaults to '.1f'.
    """
    mask = np.triu(np.ones_like(data, dtype=bool))
    plt.figure(figsize=figsize)
    sns.heatmap(
        data,
        mask=mask,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        annot=True,
        fmt=float_precision,
        linewidths=0.5,
        annot_kws={"size": annot_size},
    )
    plt.yticks(fontsize=label_size)
    plt.xticks(fontsize=label_size)
    plt.title(title, fontsize=title_size)
    plt.grid(False)
    plt.show()


def plot_numerical_feature_distribution(
    df: pd.DataFrame,
    figsize: tuple[float | int, float | int],
    numerical_features: list,
    target: str,
    title_right: float,
    title_top: float,
    wspace: float,
    hspace: float = None,
    shrink: float | int = 1,
    statistic: str = "percent",
    discrete: bool = False,
    xticks_cust: bool = False,
    step: int = 10,
    count: bool = False,
    bins: int = 10,
    padding: int = 15,
    fontsize: int = 9,
) -> None:
    """
    Plots distribution of the numerical features using target as a hue parameter.

    Args:
        df (pd.DataFrame): The original dataset.
        figsize (tuple[float|int, float|int]): The whole figure size.
        numerical_features (list[str]): The list of numerical features to plot.
        target (str): Target feature.
        title_right (float): The magnitude to move suptitle to the right.
        title_top (float): The magnitude to move suptitle to the top.
        wspace (float): The width of the padding between subplots.
        hspace (float): The height of the padding between subplots.
        shrink (float|int): The higher the coefficient, the wider the bar. Default value is 1.
        statistic (str): Statistic from those available under sns.histplot. Default to percent.
        discrete (bool): If feature is discrete.
        xticks_cust (bool): If to customize x axis ticks.
        step (int): Step used in a histplot axis.
        count (bool): Goes to format_number function. Defaults to False.
        bins (int): Number of bins in the histplot. Defaults to 10.
        padding (int): Padding of the subplot title. Defaults to 15.
        fontsiza (int): Size of the title font.
    """
    nrows = len(numerical_features)
    ncols = 3 if target else 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            feature = numerical_features[row]
            ax.xaxis.set_major_formatter(
                FuncFormatter(partial(format_numbers, count=count))
            )
            if xticks_cust:
                min_feature = int(df[feature].min())
                max_feature = int(df[feature].max())
                ax.set_xticks(
                    np.arange(min_feature, max_feature + 1, step),
                    labels=np.arange(min_feature, max_feature + 1, step),
                )
            if col == 0:
                sns.histplot(
                    ax=ax,
                    data=df,
                    x=feature,
                    shrink=shrink,
                    bins=bins,
                    stat=statistic,
                    discrete=discrete,
                    kde=True,
                )
                ax.set_xlabel(feature)
                ax.set_title(f"{feature}", pad=padding, fontsize=fontsize)
                ax.grid(visible=False, axis="x")
            elif col == 1:
                sns.histplot(
                    ax=ax,
                    data=df,
                    x=feature,
                    hue=target,
                    shrink=shrink,
                    bins=bins,
                    stat=statistic,
                    discrete=discrete,
                    kde=True,
                    multiple="dodge",
                    legend=True,
                )
                ax.set_xlabel(feature)
                ax.set_title(f"{feature} by {target}", pad=padding, fontsize=fontsize)
                ax.grid(visible=False, axis="x")
                sns.move_legend(
                    ax,
                    title=f"{target.capitalize()}",
                    loc="upper left",
                    bbox_to_anchor=(0.98, 1.1),
                    alignment="left",
                )
            elif col == 2:
                sns.boxplot(
                    ax=ax,
                    data=df,
                    x=feature,
                    hue=target,
                    width=0.6,
                    gap=0.3,
                    fill=False,
                )
                ax.set_ylabel(target)
                ax.set_title(
                    f"Boxplots of {feature} by {target}", pad=padding, fontsize=fontsize
                )
                ax.legend(
                    title=f"{target.capitalize()}",
                    loc="upper left",
                    bbox_to_anchor=(0.98, 1.1),
                    alignment="left",
                )

    fig.suptitle(f"Distribution of numerical features", x=title_right, y=title_top)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()


def plot_categorical_features(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    figsize: tuple[int],
    title_top: float,
    title_right: float,
    xticks: list[int | str] = None,
    wspace: float = 0.5,
    hspace: float = 0.6,
    count: bool = True,
    padding: int = 30,
    ordered: bool = False,
    convert_cat: bool = False,
) -> None:
    """
    Plot histplots of the list of categorical features on the subplots.

    Args:
        df (pd.DataFrame): The original dataset.
        target (str): Feature used for hue.
        features (list[str]): The list of features used for calculating distributions.
        figsize (tuple[int]): The size of the subplots.
        title_top (float): The magnitude of moving the suptitle to the top.
        title_right (float): The magnitude of moving the suptitle to the right.
        xticks (list[int], optional): Labels of x axis. Defaults to None.
        wspace (float, optional): The width of the padding between subplots. Defaults to 0.3.
        hspace (float, optional): The height of the padding between subplots. Defaults to 0.4.
        count (bool): Goes to format_numbers function. Defaults to True.
        padding (int): Spacing for annotations on the plot.
        ordered (bool): If bars need to be ordered by value counts.
        convert_cat (bool): If categories need to be converted to integers.
    """
    title_keywords = ["defaulted credits", "categorical feature"]

    num = len(features)
    ylabels = [feature for feature in features for _ in range(2)]
    xlabels = ["Counts", "Percentage share"] * num

    stat = ["count", "percent"] * num
    multiple = ["stack", "fill"] * num
    legends = [False, True] * num
    shrink = [0.8] * len(xlabels)

    fig, axes = plt.subplots(len(features), 2, figsize=figsize)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if ordered == True:
            ord_cat = df[ylabels[i]].value_counts().index.to_list()
            df[ylabels[i]] = pd.Categorical(
                df[ylabels[i]], categories=ord_cat, ordered=True
            )
        else:
            pass
        sns.histplot(
            data=df,
            y=ylabels[i],
            hue=target,
            multiple=multiple[i],
            stat=stat[i],
            ax=ax,
            shrink=shrink[i],
            discrete=True,
            legend=legends[i],
        )
        if i % 2 != 0:
            for container in ax.containers:
                labels = [f"{bar.get_width()*100:.1f}%" for bar in container]
                ax.bar_label(
                    container,
                    labels=labels,
                    label_type="center",
                    padding=padding,
                    fontsize=8.5,
                    color="white",
                    fontweight="bold",
                )

        ax.grid(False, "major", "y")
        if i % 2 == 0:
            ax.set_title(f"Amount of {title_keywords[0]} by {ylabels[i]}", pad=17)
            ax.xaxis.set_major_formatter(
                FuncFormatter(partial(format_numbers, count=count))
            )
        else:
            ax.set_title(f"Proportion of {title_keywords[0]} by {ylabels[i]}", pad=17)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
        ax.set_xlabel(xlabels[i].capitalize())

        if (
            is_integer_dtype(df[ylabels[i]]) and df[ylabels[i]].nunique() == 2
        ) or pd.api.types.is_bool_dtype(df[ylabels[i]]):
            ax.set_yticks([0, 1])
            ax.set_yticklabels(xticks)

        if not ordered:
            if isinstance(df[ylabels[i]].dtype, pd.CategoricalDtype):
                if convert_cat:
                    try:
                        cats_as_ints = df[ylabels[i]].cat.categories.astype(int)
                        min_feature = cats_as_ints.min()
                        max_feature = cats_as_ints.max()
                        ax.set_yticks(
                            np.arange(min_feature, max_feature + 1, 1),
                            labels=np.arange(min_feature, max_feature + 1, 1),
                        )
                        ax.set_ylim(min_feature - 1, max_feature + 1)
                        ax.invert_yaxis()
                    except ValueError:
                        pass
        if legends[i]:
            sns.move_legend(
                ax,
                title=target,
                loc="upper left",
                bbox_to_anchor=(1.05, 1.1),
                alignment="left",
            )
    fig.suptitle(
        f"Amount and proportion of {title_keywords[0]} by {title_keywords[1]}",
        x=title_right,
        y=title_top,
    )
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()


def get_stat_diff_bootstrap_ci(
    df: pd.DataFrame,
    target: str,
    group_0: pd.Series,
    group_1: pd.Series,
    feature: str,
    statistic: str = None,
    target_category: str | int = None,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> tuple[float]:
    """
    Compute bootstrap confidence interval for the difference in statistic.

    Args:
        df (pd. DataFrame): Original dataframe.
        target (str): Target feature.
        group_0 (pd.Series): Series of the feature.
        group_1 (pd.Series): Series of the feature.
        feature (str): Selected feature.
        statistic (str): Statistic to evaluate.
        target_category (str): Selected target category of the feature.
        n_bootstrap (int, optional): The amount of replications to evaluate confidence interval for the difference in medians.
        Defaults to 10_000.
        alpha (float, optional): The significance level. Defaults to 0.05.

    Returns:
        tuple[float]: Values for the confidence interval and the statistic difference itself.
    """

    if callable(statistic):
        stat_name = statistic.__name__
    else:
        stat_name = "proportion"

    df = df.copy().dropna(subset=[feature])

    if (
        is_integer_dtype(df[feature]) or is_float_dtype(df[feature])
    ) and stat_name != "proportion":
        data_1 = df.loc[group_1, feature]
        data_0 = df.loc[group_0, feature]
        true_stat_diff = round(statistic(data_1) - statistic(data_0), 5)
        stat_diffs = np.array(
            [
                statistic(data_1.sample(len(data_1), replace=True))
                - statistic(data_0.sample(len(data_0), replace=True))
                for _ in range(n_bootstrap)
            ]
        )
    else:
        maskA = df[feature] == target_category
        maskB = ~maskA

        insA = df.loc[maskA, target]
        insB = df.loc[maskB, target]
        nA, nB = len(insA), len(insB)
        true_stat_diff = insA.mean() - insB.mean()
        stat_diffs = np.array(
            [
                np.mean(insA.sample(nA, replace=True))
                - np.mean(insB.sample(nB, replace=True))
                for _ in range(n_bootstrap)
            ]
        )

    left = round(np.percentile(stat_diffs, (alpha / 2) * 100), 5)
    right = round(np.percentile(stat_diffs, (1 - alpha / 2) * 100), 5)

    return left, right, true_stat_diff


def interpret_test_results(
    p_value: float,
    alpha: float = 0.05,
) -> None:
    """
    Interprets the results of the performed statistical test.

    Args:
        p_value (float): The p-value after performing the test.
        alpha (float, optional): The significance level. Defaults to 0.05.
    """
    if p_value < alpha:
        print(
            "We reject the null hypothesis that there is no difference in the statistic of two groups."
        )
    else:
        print(
            "We fail to reject the null hypothesis that there is no difference in the statistic of two groups."
        )


def make_chi_test(
    df: pd.DataFrame, feature1: str, feature2: str, alpha: float = 0.05
) -> float:
    """
    Performs Chi square test.

    Args:
        df (pd.DataFrame): The original dataframe.
        feature1 (str): Feature name of the first groupd
        feature2 (str): The second group feature name
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        float: p-value for the statistical significance test.
    """
    contingency_table = pd.crosstab(df[feature1], df[feature2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return round(p, 4)


def make_stat_permutation_test(
    df: pd.DataFrame,
    target: str,
    feature: str,
    group_0: pd.Series,
    group_1: pd.Series,
    statistic: str = None,
    target_category: str | int = None,
    replications: int = 10_000,
) -> float:
    """
    Makes a permutation test fo the dataset. Outputs the p-value for the hypothesis test.

    Args:
        df (pd. DataFrame): Original dataframe.
        target (str): Target feature.
        feature (str): Selected feature.
        group_0 (pd.Series): Series of the feature.
        group_1 (pd.Series): Series of the feature.
        statistic (str): Statistic to evaluate.
        target_category (str): Selected target category of the feature.
        replications (int, optional): The amount of simulation replications. Defaults to 10_000.

    Returns:
        float: The p-value for the significance test.
    """
    if callable(statistic):
        stat_name = statistic.__name__
    else:
        stat_name = "proportion"

    df = df.copy().dropna(subset=[feature])

    if (
        is_integer_dtype(df[feature]) or is_float_dtype(df[feature])
    ) and stat_name != "proportion":
        test_stat0 = statistic(df.loc[group_0, feature])
        test_stat1 = statistic(df.loc[group_1, feature])
        n0 = group_0.sum()
        n1 = group_1.sum()
    else:
        maskA = df[feature] == target_category
        maskB = ~maskA

        insA = df.loc[maskA, target]
        insB = df.loc[maskB, target]
        nA, nB = len(insA), len(insB)
        test_stat1 = insA.mean()
        test_stat0 = insB.mean()

    obs_diff = test_stat1 - test_stat0
    differences = np.array([])
    for _ in np.arange(replications):
        if (
            is_integer_dtype(df[feature]) or is_float_dtype(df[feature])
        ) and stat_name != "proportion":
            shuffled_combine = np.random.permutation(df[feature])
            shuffled_v0 = shuffled_combine[0:n0]
            shuffled_v1 = shuffled_combine[n0 : (n0 + n1)]
            differences = np.append(
                differences, statistic(shuffled_v1) - statistic(shuffled_v0)
            )
        else:
            all_labels = np.concatenate((insA.values, insB.values))
            np.random.shuffle(all_labels)
            dA = all_labels[:nA].mean()
            dB = all_labels[nA:].mean()
            differences = np.append(differences, dA - dB)
    p_value = np.mean(np.abs(differences) >= np.abs(obs_diff))
    return p_value


def get_binary_features(df: pd.DataFrame) -> list[str]:
    """
    Filters out features that have only two distinct values.

    Args:
        df (pd.DataFrame): The original dataframe.

    Returns:
        list[str]: List of binary feature names.
    """
    binary_features = []
    for col in df.columns:
        if df[col].nunique(dropna=True) == 2:
            binary_features.append(col)
    return binary_features


def score_dataset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipe: Pipeline,
    metric: str = "roc_auc",
    n_repeats: int = 4,
) -> tuple[float]:
    """
    Performs cross validation to return average metric score.

    Args:
        X_train (pd.DataFrame): Predictors' dataframe.
        y_train (pd.Series): Series of the target feature.
        pipe (Pipeline): Sklearn pipeline.
        metric (str, optional): Metric for model evaluation. Defaults to 'roc_auc'.
        n_repeats (int): Number or cross validation repetitions. Defaults to 4.

    Returns:
        float: Average score of the metric.
    """
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=n_repeats, random_state=SEED)
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=metric, n_jobs=-1)

    avg_metric = round(scores["test_score"].mean(), 4)
    avg_std = round(scores["test_score"].std(), 4)

    return avg_metric, avg_std


def score_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_list: list[str],
    preprocessor: Pipeline,
    interactions: TransformerMixin = None,
) -> pd.DataFrame:
    """
    Scores different models using cross validation.

    Args:
        X_train (pd.DataFrame): Predictor matrix.
        y_train (pd.Series): Target series.
        model_list (list[str]): Model names.
        preprocessor (Pipeline): Pipeline of columntransformers.
        interactions (TransformerMixin, optional): Transformer with feature interactions. Defaults to None.

    Returns:
        pd.DataFrame: The table of the model scores.
    """
    results_dict = {}
    for model in model_list:
        if interactions:
            pipe = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("interactions", interactions),
                    ("classifier", model),
                ]
            )
        else:
            pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

        results = {}
        model_name = model.__class__.__name__

        results[model_name] = score_dataset(
            X_train, y_train, pipe, metric="roc_auc", n_repeats=4
        )
        results_dict.update(results)

    results_total = pd.DataFrame.from_dict(
        results_dict, orient="index", columns=["Mean ROC AUC", "STD ROC AUC"]
    ).sort_values(by="Mean ROC AUC", ascending=False)

    return results_total


def find_correlating_pairs(
    df_corr: pd.DataFrame, condition: pd.DataFrame
) -> pd.DataFrame:
    """
    Extracts unique pairs of features from correlation matrix that satisfy
    a given condition of correlation coefficient's value.

    Args:
        df_corr (pd.DataFrame): A square correlation matrix with pairwise
                                correlation coefficients between features.
        condition (pd.DataFrame): A DataFrame containing boolean values indicating
                                    whether the condition is satisfied (True) or
                                    not (False).

    Returns:
        pd.DataFrame: A DataFrame of unique feature pairs with their correlation
                            coefficient values.
    """
    np.fill_diagonal(df_corr.values, np.nan)  # To ignore self-correlations
    corr_features = df_corr[condition].stack().reset_index().round(3)
    corr_features.columns = ["feature_1", "feature_2", "correlation"]
    corr_features = corr_features[
        corr_features["feature_1"] < corr_features["feature_2"]
    ].sort_values(by="correlation", ascending=False)
    return corr_features


def plot_errorbar(
    ci_df: pd.DataFrame | list[pd.DataFrame],
    metric: str,
    feature: str,
    fig_size: tuple[int],
    right: float,
    top: float,
    nrows: int = 1,
    ncols: int = 1,
    xlim_left: float | int = None,
    xlim_right: float | int = None,
    zero_line: bool = True,
) -> None:
    """
    Plots the confidence intervals for the features on the errorbar.

    Args:
        ci_df (pd.DataFrame): The dataframe containing confidence interval data.
        metric (str): The name of the metric for which confidence interval is calculated.
        feature (str): The name of the categorical feature (the subgroup of numerical data feature).
        fig_size (tuple[int]): The width and height of the figure.
        right (float): Spacing of the suptitle to the right.
        top (float): Spacing of the suptitle to the top.
        nrows (int): Number of subplot grid rows.
        ncols (int): Number of subplot grid columns.
        xlim_left (float | int, optional): The left limit for the x axis. Defaults to None.
        xlim_right (float | int, optional): The right limit for the x axis. Defaults to None.
        zero_line (boolean, optional): The line at x=0. Defaults to False.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()
    if isinstance(ci_df, list):
        for i, (df, ax) in enumerate(zip(ci_df, axes)):
            x_err = np.array(
                [
                    np.abs(df[metric] - df["CI_lower"]),
                    np.abs(df["CI_upper"] - df[metric]),
                ]
            )
            label = (
                "Difference in medians\n with 95% confidence interval"
                if i == 0
                else None
            )
            ax.errorbar(
                df[metric],
                df[feature],
                xerr=x_err,
                fmt="o",
                capsize=5,
                capthick=2,
                color="steelblue",
                label=label,
            )
            ax.grid(True, linestyle="--", alpha=0.9)
            if zero_line:
                ax.axvline(x=0, color="purple", linestyle="--", linewidth=1)

            ax.set_xlim(left=xlim_left, right=xlim_right)
            ax.set_ylim(-1, len(df))
            ax.set_xlabel("Difference in medians")
            ax.set_ylabel("Feature")

    else:
        ax = axes[0]
        x_err = np.array(
            [
                np.abs(ci_df[metric] - ci_df["CI_lower"]),
                np.abs(ci_df["CI_upper"] - ci_df[metric]),
            ]
        )
        ax.errorbar(
            ci_df[metric],
            ci_df[feature],
            xerr=x_err,
            fmt="o",
            capsize=5,
            capthick=2,
            color="steelblue",
            label="Difference in medians\n with 95% confidence interval",
        )
        ax.grid(True, linestyle="--", alpha=0.9)
        if zero_line:
            ax.axvline(x=0, color="purple", linestyle="--", linewidth=1)

        ax.set_xlim(left=xlim_left, right=xlim_right)
        ax.set_ylim(-1, len(ci_df))
        ax.set_xlabel("Difference in medians")
        ax.set_ylabel("Feature")

    fig.legend(loc="upper left", bbox_to_anchor=(1, 1), ncols=1)
    plt.tight_layout()
    fig.suptitle(
        "Difference in medians of numerical features with their 95% confidence intervals",
        x=right,
        y=top,
        fontsize=11,
    )
    plt.show()


def evaluate_model(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Scores the models using cross validation.

    Args:
        pipe (Pipeline): Pipelines of different ML models.
        X_train (pd.DataFrame): Predictor matrix.
        y_train (pd.Series): Target series.

    Returns:
        dict: Nested dictionary of ML models learning scores.
    """
    cv = StratifiedKFold(n_splits=3)

    if isinstance(pipe, Pipeline):
        _, final_estimator = pipe.steps[-1]
    else:
        final_estimator = pipe
    model_name = final_estimator.__class__.__name__

    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    return model_name, {
        "ROC_AUC": round(scores.mean(), 4),
        "Std": round(scores.std(), 4),
    }


def make_roc_auc_curve(
    pipes: list[Pipeline],
    ncols: int,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    right: float,
    top: float,
    figsize: tuple[float | int],
    legend_coord: tuple[float | int] = (1.05, 0.5),
    wspace: float = 0.3,
    hspace: float = 0.3,
    save: bool = False,
    name: str = None,
) -> tuple[float]:
    """
    Plots ROC AUC curves for the models and returns generalization performance scores.

    Args:
        pipes (Pipeline): Fitted pipelines of transformers, preprocessor and estimator.
        ncols (int): Number of subplot grid columns.
        X_test (pd.DataFrame): Test data of predictors.
        test_preds (np.array): Predictions of the target by the model.
        y_test (pd.Series): Series of the target.
        right (float): Moving the suptitle of the figure to right.
        top (float): Moving the suptitle of the figure to top.
        figsize (tuple[float | int]): Figure size.
        legend_coord (tuple[float|int]): Coordinates of the figure legend. Defaults to (1.05, 0.5).
        wspace (float): Horizontal spacing between the subplots. Defaults to 0.3.
        hspace (float): Vertical spacing between the subplots. Defaults to 0.3.
        save (bool, optional): If saving of the figure is needed. Defaults to False.
        name (str, optional): Name of the file where to save figure. Defaults to None.

    Returns:
        tuple[float]: Tuple of precision and recall scores.
    """
    nrows = int(len(pipes) / ncols)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    axes = axes.ravel()
    results = {}
    for ax, pipe in zip(axes, pipes):

        if isinstance(pipe, Pipeline):
            final_step_name, final_estimator = pipe.steps[-1]
        else:
            final_estimator = pipe

        model_name = final_estimator.__class__.__name__

        y_pred_proba = pipe.predict_proba(X_test)[:, 1]
        score = round(roc_auc_score(y_test, y_pred_proba), 4)
        results[model_name] = score

        RocCurveDisplay.from_estimator(
            pipe,
            X_test,
            y_test,
            name=model_name,
            plot_chance_level=True,
            marker="+",
            chance_level_kw={"color": "orange", "linestyle": "--"},
            ax=ax,
        )
        ax.set_title(f"{model_name}")

    for ax in axes:
        ax.legend(bbox_to_anchor=legend_coord)

    fig.suptitle(f"ROC-AUC curves of the models", x=right, y=top)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if save:
        file_path = name
        if os.path.exists(file_path):
            os.remove(file_path)
        plt.savefig(name)
    plt.show()

    return (
        pd.Series(results)
        .to_frame(name="Test ROC AUC")
        .sort_values(by="Test ROC AUC", ascending=False)
    )


def get_shap_feat_importance(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    figsize: tuple[int | float],
    right: float = 0.6,
    top: float = 0.95,
    rectangle=[0.1, 0.3, 2, 0.97],
) -> None:
    """
    Plots SHAP feature importance on the summary dotplot and barplot.

    Args:
        pipe (Pipeline): Pipeline of transformers, preprocessor and estimator.
        X_train (pd.DataFrame): Predictors' dataframe.
        y_train (pd.Series): Series of the target.
        figsize (tuple[int | float]): Size of the figure.
        right (float): Position of figure title to the right. Defaults to 0.6.
        top (float): Position of figure title to top. Defaults to 0.95.
        rectangle (list[int]): Proportions of the figure. Defaults to [0.1, 0.3, 2, 0.97].

    """
    pipe.fit(X_train, y_train)
    model = pipe.named_steps["classifier"]
    model_name = model.__class__.__name__

    X_train_transf = X_train.copy()
    for name, step in pipe.steps[:-1]:
        if hasattr(step, "transform"):
            X_train_transf = step.transform(X_train_transf)

    if model_name == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_train_transf)
        shap_values = explainer(X_train_transf)
    else:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_train_transf)
        except Exception as e:
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_train_transf, 1_500, random_state=SEED),
            )
            shap_values = explainer.shap_values(
                X_train_transf, nsamples=2_500, random_state=SEED
            )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plt.sca(axes[0])
    shap.summary_plot(shap_values, show=False)
    axes[0].set_title("SHAP Summary (Dot Plot)", fontsize=11)
    axes[0].tick_params(axis="x", labelsize=9)
    axes[0].tick_params(axis="y", labelsize=9)
    axes[0].set_xlabel("SHAP value (impact on model output)", fontsize=9.5)

    plt.sca(axes[1])
    shap.summary_plot(shap_values, plot_type="bar", show=False)
    axes[1].set_title("SHAP Summary (Bar Plot)", fontsize=11)
    axes[1].tick_params(axis="x", labelsize=9)
    axes[1].tick_params(axis="y", labelsize=9)
    axes[1].set_xlabel("Mean Absolute SHAP value", fontsize=9.5)

    fig.suptitle(f"SHAP Values for the Features of {model_name}", x=right, y=top)
    plt.tight_layout(rect=rectangle)
    plt.show()


def make_predictions_save_file(
    pipe: Pipeline,
    data: pd.DataFrame,
    target: pd.Series,
    X_test: pd.DataFrame,
    file_prefix: str,
) -> str:
    """
    Makes predictions for the test data and creates a file.

    Args:
        pipe (Pipeline): Preprocessing pipeline with the estimator.
        data (pd.DataFrame): The predictors' matrix.
        target (pd.Series): The target feature series.
        X_test (pd.DataFrame): Test subset.
        file_prefix (str): String for file name.

    Returns:
        str: Output about the commands executed.
    """
    pipe.fit(data, target)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    submission_df = pd.DataFrame(
        {"SK_ID_CURR": X_test["SK_ID_CURR"], "TARGET": y_proba}
    )

    submission_path = f"../data/submissions/{file_prefix}_predictions.csv"

    if os.path.exists(submission_path):
        os.remove(submission_path)
        print(f"Old submission file removed: {submission_path}.")

    submission_df.to_csv(submission_path, index=False)
    print("Submission file created successfully.")


def plot_final_model_results(
    final_res_df: pd.DataFrame,
    figsize: tuple[float | int],
    title: str,
    limits: tuple[float | int],
    metrics: list[str] = ["ROC_AUC"],
    save: bool = False,
    name: str = None,
) -> None:
    """
    Plots pointplot of metrics for different models.

    Args:
        final_res_df (pd.DataFrame): Dataframe of metrics for models.
        figsize (tuple[float | int]): Figure size.
        title (str): Title of the figure.
        limits (tuple[float | int]): X axis limits.
        metrics (list[str], optional): List of metrics to plot. Defaults to ['ROC_AUC'].
        save (bool, optional): If saving of the figure is needed. Defaults to False.
        name (str, optional): Name of the file where to save the figure. Defaults to None.
    """
    df_melted = final_res_df.melt(
        id_vars="Model",
        value_vars=metrics,
        var_name="Metric",
        value_name="Value",
    )

    plt.figure(figsize=figsize)
    sns.pointplot(data=df_melted, hue="Metric", x="Value", y="Model")
    plt.title(title, pad=15)
    plt.ylabel("Model")
    plt.xlabel("Metric value")
    plt.legend(
        title="Metric", bbox_to_anchor=(1.1, 1), loc="upper left", alignment="left"
    )
    plt.xlim(limits)
    plt.tight_layout()
    if save:
        file_path = name
        if os.path.exists(file_path):
            os.remove(file_path)
        plt.savefig(name)
    plt.show()


def display_model_comparison(
    result_list: list[pd.DataFrame],
    names: list[str],
    figsize: tuple[float],
    title: str,
    limits: tuple[float | int],
    baseline: float,
) -> pd.DataFrame:
    """
    Plots performance metric for the different experiments with the models.

    Args:
        result_list (list[pd.DataFrame]): List of dataframes containing model results.
        names (list[str]): List of model names corresponding to each dataframe.
        title (str): Name of the figure.
        limits (tuple[float | int]): Limits for the x axis.
        baseline (float): Basline ROC AUC score.

    Returns:
        pd.DataFrame: Combined dataframe with model names as index.
    """
    cleaned_results = []
    for i, result in enumerate(result_list):
        if "STD ROC AUC" in result.columns:
            result = result.drop(columns=["STD ROC AUC"])
            result = result.reset_index(drop=False, inplace=False)
            result = result.reset_index().rename(
                columns={"index": "Model", "Mean ROC AUC": "ROC AUC Score"}
            )
            result["Mode"] = names[i]
            result = result[["Model", "Mode", "ROC AUC Score"]]
        cleaned_results.append(result)

    general_df = pd.concat(cleaned_results, ignore_index=True)

    plt.figure(figsize=figsize)
    sns.pointplot(data=general_df, hue="Mode", x="ROC AUC Score", y="Model", dodge=True)
    plt.axvline(
        x=baseline,
        color="purple",
        linestyle="--",
        linewidth=1,
        label="Baseline ROC AUC",
    )
    plt.title(title, pad=15)
    plt.ylabel("Model")
    plt.xlabel("Mode")
    plt.legend(
        title="Modes", bbox_to_anchor=(1.1, 1), loc="upper left", alignment="left"
    )
    plt.xlim(limits)
    plt.tight_layout()
    plt.show()

    return general_df.pivot(index="Mode", columns="Model", values="ROC AUC Score")
