"""
Scientific and academic visualisation utilities for exploratory feature analysis
and post-mortem conversational diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Academic colorblind-safe palette (Okabe-Ito / publication inspired)
# Group 0 (e.g. Unresolved) -> Muted brick/vermilion (#C44E52)
# Group 1 (e.g. Resolved)   -> Deep slate navy (#1F5A8A)
ACADEMIC_PALETTE: list[str] = [
    "#C44E52",  # Muted crimson / brick red
    "#1F5A8A",  # Deep academic navy blue
    "#2B8CBE",  # Cerulean teal
    "#E69F00",  # Amber gold
    "#009E73",  # Bluish green
    "#756BB1",  # Muted purple
    "#636363",  # Charcoal grey
]


def _coerce_inputs(
    x: pd.DataFrame | pd.Series | np.ndarray,
    x_columns: str | Sequence[str] | None,
    groupby: str | pd.Series | np.ndarray | Sequence[Any] | None,
) -> tuple[pd.DataFrame, list[str], str | None]:
    """
    Standardize input data `x`, `x_columns`, and `groupby` into a clean DataFrame,
    list of column names to plot, and optional groupby column name.
    """
    if isinstance(x, pd.DataFrame):
        if x_columns is None:
            raise ValueError(
                "When `x` is a pandas DataFrame, `x_columns` must be provided."
            )
        cols = [x_columns] if isinstance(x_columns, str) else list(x_columns)
        if not cols:
            raise ValueError("`x_columns` must contain at least one column name.")
        missing = [c for c in cols if c not in x.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        df = x[cols].copy()
        group_col: str | None = None

        if groupby is not None:
            if isinstance(groupby, str):
                if groupby not in x.columns:
                    raise ValueError(
                        f"Groupby column '{groupby}' not found in DataFrame."
                    )
                group_col = groupby
                if group_col not in df.columns:
                    df[group_col] = x[group_col].values
            else:
                group_arr = np.asarray(groupby)
                if len(group_arr) != len(df):
                    raise ValueError(
                        f"Length of `groupby` ({len(group_arr)}) does not match "
                        f"length of `x` ({len(df)})."
                    )
                group_col = getattr(groupby, "name", None) or "_group"
                df[group_col] = group_arr

        return df, cols, group_col

    if isinstance(x, pd.Series):
        col_name = (
            x_columns
            if isinstance(x_columns, str)
            else (
                list(x_columns)[0]
                if x_columns is not None and len(x_columns) > 0
                else (str(x.name) if x.name is not None else "feature_0")
            )
        )
        df = pd.DataFrame({col_name: x.values})
        cols = [col_name]
    elif isinstance(x, np.ndarray):
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(
                f"NumPy array `x` must be 1D or 2D, got {arr.ndim}D array."
            )

        n_features = arr.shape[1]
        if x_columns is None:
            cols = [f"feature_{i}" for i in range(n_features)]
        elif isinstance(x_columns, str):
            if n_features != 1:
                raise ValueError(
                    f"Single column name '{x_columns}' provided for array with "
                    f"{n_features} columns."
                )
            cols = [x_columns]
        else:
            cols = list(x_columns)
            if len(cols) != n_features:
                raise ValueError(
                    f"Length of `x_columns` ({len(cols)}) must match number of "
                    f"array columns ({n_features})."
                )
        df = pd.DataFrame(arr, columns=cols)
    else:
        raise TypeError(
            f"`x` must be a pandas DataFrame, Series, or NumPy ndarray; got {type(x).__name__}."
        )

    group_col = None
    if groupby is not None:
        if isinstance(groupby, str):
            raise ValueError(
                "When `x` is a NumPy array or Series, `groupby` must be an array-like "
                "or Series of group labels, not a column name string."
            )
        group_arr = np.asarray(groupby)
        if len(group_arr) != len(df):
            raise ValueError(
                f"Length of `groupby` ({len(group_arr)}) does not match "
                f"length of `x` ({len(df)})."
            )
        group_col = getattr(groupby, "name", None) or "group"
        df[group_col] = group_arr

    return df, cols, group_col


def _plot_single_barchart(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    col: str,
    group_col: str | None,
    groups: list[Any],
    color_map: dict[Any, str],
    density: bool,
    bins: int,
    max_discrete_unique: int = 25,
) -> None:
    """Render a discrete bar chart or binned histogram bar chart on `ax`."""
    valid_series = df[col].dropna()
    if valid_series.empty:
        return

    unique_vals = np.sort(valid_series.unique())
    is_discrete = len(unique_vals) <= max_discrete_unique and np.all(
        np.isclose(unique_vals, np.round(unique_vals))
    )

    if is_discrete:
        x_positions = np.arange(len(unique_vals))
        if group_col is None:
            counts = valid_series.value_counts().reindex(unique_vals, fill_value=0)
            heights = (
                counts.values / counts.values.sum()
                if density and counts.values.sum() > 0
                else counts.values
            )
            ax.bar(
                unique_vals,
                heights,
                width=0.65
                * (np.min(np.diff(unique_vals)) if len(unique_vals) > 1 else 1.0),
                color=ACADEMIC_PALETTE[1],
                edgecolor="#143B5B",
                linewidth=0.8,
                alpha=0.82,
                zorder=3,
            )
        else:
            n_groups = len(groups)
            step = np.min(np.diff(unique_vals)) if len(unique_vals) > 1 else 1.0
            total_width = 0.72 * step
            bar_width = total_width / max(n_groups, 1)

            for idx, g in enumerate(groups):
                sub = df.loc[df[group_col] == g, col].dropna()
                if sub.empty:
                    continue
                mean_val = float(sub.mean())
                counts = sub.value_counts().reindex(unique_vals, fill_value=0)
                heights = (
                    counts.values / counts.values.sum()
                    if density and counts.values.sum() > 0
                    else counts.values
                )
                offset = (idx - (n_groups - 1) / 2.0) * bar_width
                ax.bar(
                    unique_vals + offset,
                    heights,
                    width=bar_width * 0.92,
                    color=color_map[g],
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.85,
                    label=f"{group_col} = {g} ($\\mu$={mean_val:.2f})",
                    zorder=3,
                )
        if len(unique_vals) <= 12:
            ax.set_xticks(unique_vals)
            if len(x_positions) > 0 and np.all(
                np.isclose(unique_vals, np.round(unique_vals))
            ):
                ax.set_xticklabels([f"{int(round(v))}" for v in unique_vals])
    else:
        stat_mode = "density" if density else "count"
        if group_col is None:
            sns.histplot(
                data=df,
                x=col,
                bins=bins,
                stat=stat_mode,
                color=ACADEMIC_PALETTE[1],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.80,
                ax=ax,
                zorder=3,
            )
        else:
            from matplotlib.patches import Patch

            sns.histplot(
                data=df,
                x=col,
                hue=group_col,
                hue_order=groups,
                palette=color_map,
                bins=bins,
                stat=stat_mode,
                common_norm=False,
                multiple="dodge",
                shrink=0.82,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
                legend=False,
                ax=ax,
                zorder=3,
            )
            for g in groups:
                sub = df.loc[df[group_col] == g, col].dropna()
                if sub.empty:
                    continue
                mean_val = float(sub.mean())
                ax.add_patch(
                    Patch(
                        facecolor=color_map[g],
                        edgecolor="white",
                        alpha=0.85,
                        label=f"{group_col} = {g} ($\\mu$={mean_val:.2f})",
                        visible=False,
                    )
                )


def _plot_single_kde(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    col: str,
    group_col: str | None,
    groups: list[Any],
    color_map: dict[Any, str],
    density: bool,
    bw_adjust: float,
) -> None:
    """Render a smooth kernel density area chart on `ax`."""
    from scipy.stats import gaussian_kde

    valid_all = df[col].dropna().to_numpy(dtype=float)
    if valid_all.size == 0:
        return

    x_min, x_max = float(np.min(valid_all)), float(np.max(valid_all))
    if np.isclose(x_min, x_max):
        # Fallback for constant feature
        ax.axvline(x_min, color=ACADEMIC_PALETTE[1], linewidth=2.0)
        return

    grid = np.linspace(x_min, x_max, 300)
    span = x_max - x_min

    if group_col is None:
        if np.std(valid_all) > 1e-12 and valid_all.size > 1:
            kde = gaussian_kde(valid_all, bw_method="scott")
            kde.set_bandwidth(kde.factor * bw_adjust)
            y_vals = kde(grid)
            if not density:
                # Scale density by N * (range / 30) so y-axis represents estimated count scale
                y_vals = y_vals * valid_all.size * (span / 30.0)
            ax.plot(
                grid,
                y_vals,
                color=ACADEMIC_PALETTE[1],
                linewidth=2.0,
                zorder=3,
            )
            ax.fill_between(
                grid,
                0,
                y_vals,
                color=ACADEMIC_PALETTE[1],
                alpha=0.25,
                zorder=2,
            )
    else:
        for g in groups:
            sub = df.loc[df[group_col] == g, col].dropna().to_numpy(dtype=float)
            if sub.size == 0:
                continue
            mean_val = float(np.mean(sub))
            group_label = f"{group_col} = {g} ($\\mu$={mean_val:.2f})"
            if sub.size <= 1 or np.std(sub) <= 1e-12:
                ax.plot([], [], color=color_map[g], linewidth=2.0, label=group_label)
                continue
            kde = gaussian_kde(sub, bw_method="scott")
            kde.set_bandwidth(kde.factor * bw_adjust)
            y_vals = kde(grid)
            if not density:
                y_vals = y_vals * sub.size * (span / 30.0)
            ax.plot(
                grid,
                y_vals,
                color=color_map[g],
                linewidth=2.0,
                label=group_label,
                zorder=3,
            )
            ax.fill_between(
                grid,
                0,
                y_vals,
                color=color_map[g],
                alpha=0.24,
                zorder=2,
            )

    ax.set_ylim(bottom=0)


def plot_distribution(
    x: pd.DataFrame | pd.Series | np.ndarray,
    x_columns: str | Sequence[str] | None = None,
    groupby: str | pd.Series | np.ndarray | Sequence[Any] | None = None,
    *,
    kind: Literal["kde", "area", "bar", "barchart", "hist", "histogram"] = "kde",
    density: bool = True,
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    bins: int = 25,
    bw_adjust: float = 1.0,
    palette: Sequence[str] | dict[Any, str] | None = None,
    suptitle: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """
    Plot publication-grade distribution subplots for one or more features,
    optionally stratified by a grouping variable (e.g., `resolved={0, 1}`).

    Parameters
    ----------
    x : pd.DataFrame | pd.Series | np.ndarray
        Input dataset. Can be a pandas DataFrame, a 1D Series, or a 1D/2D NumPy array.
    x_columns : str | Sequence[str] | None, optional
        Column name(s) to plot. Required when `x` is a pandas DataFrame. When `x` is
        a NumPy array or Series, used as the subplot title(s) if provided.
    groupby : str | pd.Series | np.ndarray | Sequence[Any] | None, optional
        Grouping variable (e.g. `"resolved"` or a 1D array of group labels).
        - If provided: plots stratified distributions and draws vertical lines at each
          group's **mean**.
        - If `None`: plots the overall distribution and draws vertical lines at the
          **mean**, **median**, **p25** (25th percentile), and **p75** (75th percentile).
    kind : {"kde", "area", "bar", "barchart", "hist", "histogram"}, default "kde"
        Visualisation style:
        - `"kde"` or `"area"`: Smooth kernel density area chart.
        - `"bar"`, `"barchart"`, `"hist"`, or `"histogram"`: Bar chart (automatically
          renders exact discrete bars for low-cardinality integer features, or binned
          histogram bars for continuous features).
    density : bool, default True
        Whether to normalize the distribution (`True` for density/proportion; `False`
        for raw sample counts).
    ncols : int | None, optional
        Number of subplot columns. Defaults to `min(n_features, 3)`.
    figsize : tuple[float, float] | None, optional
        Explicit `(width, height)` in inches. Automatically scaled to grid size if omitted.
    dpi : int, default 150
        Figure resolution in dots per inch.
    bins : int, default 25
        Number of bins when rendering continuous histograms (`kind="bar"` / `"hist"`).
    bw_adjust : float, default 1.0
        Bandwidth multiplier for kernel density smoothing (`kind="kde"` / `"area"`).
    palette : Sequence[str] | dict[Any, str] | None, optional
        Custom color palette for groups. Defaults to `ACADEMIC_PALETTE`.
    suptitle : str | None, optional
        Optional overarching figure title.
    save_path : str | Path | None, optional
        If provided, saves the rendered figure as a PNG (or specified extension) at
        this path with `bbox_inches="tight"`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered Matplotlib Figure object.
    axes : np.ndarray of matplotlib.axes.Axes
        2D NumPy array of subplot Axes objects.
    """
    df, cols, group_col = _coerce_inputs(x, x_columns, groupby)

    kind_norm = kind.strip().lower()
    if kind_norm not in {"kde", "area", "kernel", "bar", "barchart", "hist", "histogram"}:
        raise ValueError(
            f"Unsupported `kind='{kind}'`. Choose from 'kde', 'area', 'bar', 'barchart', 'hist'."
        )
    is_kde = kind_norm in {"kde", "area", "kernel"}

    # Determine group levels and color mapping
    groups: list[Any] = []
    color_map: dict[Any, str] = {}
    if group_col is not None:
        raw_groups = df[group_col].dropna().unique()
        try:
            groups = sorted(raw_groups)
        except TypeError:
            groups = list(raw_groups)

        if isinstance(palette, dict):
            color_map = {
                g: palette.get(g, ACADEMIC_PALETTE[i % len(ACADEMIC_PALETTE)])
                for i, g in enumerate(groups)
            }
        else:
            pal_list = list(palette) if palette is not None else ACADEMIC_PALETTE
            color_map = {g: pal_list[i % len(pal_list)] for i, g in enumerate(groups)}

    # Compute subplot grid geometry
    n_plots = len(cols)
    if ncols is None:
        ncols = min(n_plots, 3)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n_plots / ncols))

    if figsize is None:
        subplot_w = 5.2 if ncols == 1 else (4.8 if ncols == 2 else 4.3)
        subplot_h = 3.6
        figsize = (subplot_w * ncols, subplot_h * nrows)

    # Apply clean academic rcParams locally without mutating global user state
    academic_rc = {
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#CBD5E1",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "font.size": 6.8,
        "axes.titlesize": 7.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 6.8,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }

    with plt.rc_context(academic_rc):
        fig, raw_axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
            constrained_layout=True,
        )

        y_label = (
            ("Density" if is_kde else "Proportion / Density")
            if density
            else ("Scaled Count" if is_kde else "Count")
        )

        for idx, col in enumerate(cols):
            r, c = divmod(idx, ncols)
            ax = raw_axes[r, c]

            # 1. Draw main distribution (KDE area or Bar chart)
            if is_kde:
                _plot_single_kde(
                    ax=ax,
                    df=df,
                    col=col,
                    group_col=group_col,
                    groups=groups,
                    color_map=color_map,
                    density=density,
                    bw_adjust=bw_adjust,
                )
            else:
                _plot_single_barchart(
                    ax=ax,
                    df=df,
                    col=col,
                    group_col=group_col,
                    groups=groups,
                    color_map=color_map,
                    density=density,
                    bins=bins,
                )

            # 2. Draw vertical summary statistic lines
            if group_col is not None:
                for g in groups:
                    sub = df.loc[df[group_col] == g, col].dropna()
                    if sub.empty:
                        continue
                    mean_val = float(sub.mean())
                    ax.axvline(
                        mean_val,
                        color=color_map[g],
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.95,
                        zorder=4,
                    )
            else:
                valid = df[col].dropna()
                if not valid.empty:
                    mean_val = float(valid.mean())
                    median_val = float(valid.median())
                    p25_val = float(np.percentile(valid, 25))
                    p75_val = float(np.percentile(valid, 75))

                    ax.axvline(
                        mean_val,
                        color="#C44E52",
                        linestyle="--",
                        linewidth=1.6,
                        label=f"$\\mu$={mean_val:.2f}",
                        zorder=4,
                    )
                    ax.axvline(
                        median_val,
                        color="#009E73",
                        linestyle="-.",
                        linewidth=1.5,
                        label=f"p50={median_val:.2f}",
                        zorder=4,
                    )
                    ax.axvline(
                        p25_val,
                        color="#525252",
                        linestyle=":",
                        linewidth=1.3,
                        label=f"p25={p25_val:.2f}",
                        zorder=4,
                    )
                    ax.axvline(
                        p75_val,
                        color="#525252",
                        linestyle=":",
                        linewidth=1.3,
                        label=f"p75={p75_val:.2f}",
                        zorder=4,
                    )

            # 3. Formatting, title, grid, and legend
            ax.set_title(str(col), pad=8, color="#1E293B")
            ax.set_xlabel("")
            ax.set_ylabel(y_label if c == 0 else "", color="#334155")
            ax.grid(True, axis="both", alpha=0.35, zorder=0)
            sns.despine(ax=ax, top=True, right=True)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles=handles,
                    labels=labels,
                    frameon=True,
                    facecolor="white",
                    edgecolor="#E2E8F0",
                    framealpha=0.9,
                    loc="best",
                )

        # Hide any unused trailing subplot slots
        for idx in range(n_plots, nrows * ncols):
            r, c = divmod(idx, ncols)
            raw_axes[r, c].set_visible(False)

        if suptitle:
            fig.suptitle(
                suptitle,
                fontsize=10.5,
                fontweight="bold",
                color="#0F172A",
            )

        if save_path is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig, raw_axes


def _style_r2_badge_and_axes(
    ax: matplotlib.axes.Axes,
    r_val: float,
    r2_val: float,
    r2_threshold: float,
    show_r: bool = False,
) -> str:
    """
    Annotate subplot `ax` with a compact R^2 badge and highlight the subplot
    when R^2 exceeds `r2_threshold`. Returns the trendline color.
    """
    if np.isnan(r2_val):
        badge_text = "$R^2$=N/A"
    elif show_r:
        badge_text = f"$R^2$={r2_val:.2f} ($r$={r_val:+.2f})"
    else:
        badge_text = f"$R^2$={r2_val:.2f}"

    if not np.isnan(r2_val) and r2_val >= r2_threshold:
        # High collinearity highlight
        box_fc = "#FEE2E2"
        box_ec = "#DC2626"
        text_color = "#991B1B"
        ax.set_facecolor("#FFF5F5")
        for spine in ("bottom", "left", "top", "right"):
            ax.spines[spine].set_color("#DC2626")
            ax.spines[spine].set_linewidth(1.2)
        line_color = "#B91C1C"
    elif not np.isnan(r2_val) and r2_val >= 0.25:
        # Moderate correlation highlight
        box_fc = "#FEF3C7"
        box_ec = "#D97706"
        text_color = "#92400E"
        ax.set_facecolor("#FFFDF5")
        line_color = "#B45309"
    else:
        # Low correlation neutral style
        box_fc = "#F8FAFC"
        box_ec = "#CBD5E1"
        text_color = "#475569"
        ax.set_facecolor("white")
        line_color = "#475569"

    ax.text(
        0.06,
        0.92,
        badge_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        fontweight="bold" if (not np.isnan(r2_val) and r2_val >= r2_threshold) else "normal",
        color=text_color,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": box_fc,
            "edgecolor": box_ec,
            "linewidth": 0.6,
            "alpha": 0.90,
        },
        zorder=6,
    )
    return line_color


def _plot_single_pair_scatter(
    ax: matplotlib.axes.Axes,
    df_full: pd.DataFrame,
    df_plot: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str | None,
    groups: list[Any],
    color_map: dict[Any, str],
    r2_threshold: float,
    fit_reg: bool,
    jitter: float,
    rng: np.random.Generator,
    show_r: bool,
) -> None:
    """Plot a single square pairwise scatter subplot with highlighted R^2."""
    valid_full = df_full[[x_col, y_col]].dropna()
    if len(valid_full) < 2:
        r_val, r2_val = np.nan, np.nan
    else:
        x_full = valid_full[x_col].to_numpy(dtype=float)
        y_full = valid_full[y_col].to_numpy(dtype=float)
        if np.std(x_full) <= 1e-12 or np.std(y_full) <= 1e-12:
            r_val, r2_val = 0.0, 0.0
        else:
            r_val = float(np.corrcoef(x_full, y_full)[0, 1])
            r2_val = float(r_val**2)

    line_color = _style_r2_badge_and_axes(
        ax=ax,
        r_val=r_val,
        r2_val=r2_val,
        r2_threshold=r2_threshold,
        show_r=show_r,
    )

    cols_needed = [x_col, y_col] + ([group_col] if group_col is not None else [])
    valid_plot = df_plot[cols_needed].dropna()
    if valid_plot.empty:
        return

    x_pts = valid_plot[x_col].to_numpy(dtype=float).copy()
    y_pts = valid_plot[y_col].to_numpy(dtype=float).copy()

    if jitter > 0.0:
        x_span = float(np.ptp(x_pts)) or 1.0
        y_span = float(np.ptp(y_pts)) or 1.0
        x_pts += rng.normal(0.0, jitter * x_span, size=x_pts.size)
        y_pts += rng.normal(0.0, jitter * y_span, size=y_pts.size)

    if group_col is None:
        ax.scatter(
            x_pts,
            y_pts,
            s=14,
            color=ACADEMIC_PALETTE[1],
            alpha=0.55,
            edgecolors="white",
            linewidths=0.25,
            rasterized=True,
            zorder=3,
        )
    else:
        g_vals = valid_plot[group_col].to_numpy()
        for g in groups:
            mask = g_vals == g
            if not np.any(mask):
                continue
            ax.scatter(
                x_pts[mask],
                y_pts[mask],
                s=14,
                color=color_map[g],
                alpha=0.60,
                edgecolors="white",
                linewidths=0.25,
                label=f"{group_col} = {g}",
                rasterized=True,
                zorder=3,
            )

    if fit_reg and len(valid_full) >= 2:
        x_full = valid_full[x_col].to_numpy(dtype=float)
        y_full = valid_full[y_col].to_numpy(dtype=float)
        if np.std(x_full) > 1e-12:
            slope, intercept = np.polyfit(x_full, y_full, 1)
            x_line = np.array([np.min(x_full), np.max(x_full)])
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                color=line_color,
                linestyle="--",
                linewidth=1.1,
                alpha=0.85,
                zorder=5,
            )

    # Add clean padding around points so the scatter shape sits comfortably in the square
    x_min, x_max = float(np.min(x_pts)), float(np.max(x_pts))
    y_min, y_max = float(np.min(y_pts)), float(np.max(y_pts))
    x_pad = 0.08 * (x_max - x_min) if x_max > x_min else 0.5
    y_pad = 0.08 * (y_max - y_min) if y_max > y_min else 0.5
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + 2.2 * y_pad)


def plot_correlation_matrix(
    x: pd.DataFrame | np.ndarray,
    x_columns: Sequence[str] | None = None,
    groupby: str | pd.Series | np.ndarray | Sequence[Any] | None = None,
    *,
    r2_threshold: float = 0.50,
    show_r: bool = False,
    fit_reg: bool = True,
    jitter: float = 0.0,
    sample_size: int | None = 1500,
    random_state: int = 42,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    palette: Sequence[str] | dict[Any, str] | None = None,
    suptitle: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """
    Plot a bottom-left triangular matrix of square scatter subplots for all
    ((k^2 - k) / 2) unique feature pairs, with highlighted R^2 badges.

    Every subplot is rendered as an equal-aspect square without axis ticks or tick
    numbers so the geometric shape of each pairwise relationship and its R^2
    stand out cleanly.

    Parameters
    ----------
    x : pd.DataFrame | np.ndarray
        Input dataset (pandas DataFrame or 2D NumPy array) with at least 2 columns.
    x_columns : Sequence[str] | None, optional
        List of column names to include in the pairwise audit (minimum 2). Required
        when `x` is a pandas DataFrame.
    groupby : str | pd.Series | np.ndarray | Sequence[Any] | None, optional
        Optional grouping variable (e.g. `"resolved"`) used to color scatter points.
    r2_threshold : float, default 0.50
        Threshold above which a pair's R^2 badge, border, and background are visually
        highlighted in crimson to flag strong collinearity.
    show_r : bool, default False
        If `True`, appends signed Pearson `r` to the `R^2` badge.
    fit_reg : bool, default True
        If `True`, overlays the linear OLS regression trendline on each subplot.
    jitter : float, default 0.0
        Optional Gaussian jitter scale (as a fraction of feature range, e.g. `0.015`)
        to reveal overlapping discrete integer points.
    sample_size : int | None, default 1500
        Maximum number of scatter points to render per subplot for visual clarity and
        rendering speed. R^2 and OLS trendlines are always computed on 100% of the data.
    random_state : int, default 42
        Random seed used when subsampling scatter points or applying jitter.
    figsize : tuple[float, float] | None, optional
        Explicit `(width, height)` in inches. Defaults to a square `(2.1 * (k-1), 2.1 * (k-1))`.
    dpi : int, default 150
        Figure resolution in dots per inch.
    palette : Sequence[str] | dict[Any, str] | None, optional
        Custom color palette when `groupby` is provided. Defaults to `ACADEMIC_PALETTE`.
    suptitle : str | None, optional
        Optional overarching figure title.
    save_path : str | Path | None, optional
        If provided, saves the rendered figure to this file path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered Matplotlib Figure object.
    axes : np.ndarray of matplotlib.axes.Axes
        2D NumPy array of `(k - 1) x (k - 1)` subplot Axes objects.
    """
    df_full, cols, group_col = _coerce_inputs(x, x_columns, groupby)
    k = len(cols)
    if k < 2:
        raise ValueError(
            f"`plot_correlation_matrix` requires at least 2 columns; got {k}."
        )

    rng = np.random.default_rng(random_state)
    if sample_size is not None and len(df_full) > sample_size:
        sample_idx = rng.choice(len(df_full), size=sample_size, replace=False)
        df_plot = df_full.iloc[sample_idx].copy()
    else:
        df_plot = df_full

    groups: list[Any] = []
    color_map: dict[Any, str] = {}
    if group_col is not None:
        raw_groups = df_full[group_col].dropna().unique()
        try:
            groups = sorted(raw_groups)
        except TypeError:
            groups = list(raw_groups)

        if isinstance(palette, dict):
            color_map = {
                g: palette.get(g, ACADEMIC_PALETTE[i % len(ACADEMIC_PALETTE)])
                for i, g in enumerate(groups)
            }
        else:
            pal_list = list(palette) if palette is not None else ACADEMIC_PALETTE
            color_map = {g: pal_list[i % len(pal_list)] for i, g in enumerate(groups)}

    n_dim = k - 1
    if figsize is None:
        cell_size = 2.2 if n_dim <= 4 else (1.85 if n_dim <= 6 else 1.55)
        figsize = (cell_size * n_dim, cell_size * n_dim)

    academic_rc = {
        "axes.edgecolor": "#475569",
        "axes.linewidth": 0.75,
        "font.size": 6.0,
        "axes.labelsize": 6.0,
        "legend.fontsize": 6.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }

    with plt.rc_context(academic_rc):
        fig, raw_axes = plt.subplots(
            nrows=n_dim,
            ncols=n_dim,
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
        )
        fig.subplots_adjust(wspace=0.06, hspace=0.06)

        for r in range(n_dim):
            y_col = cols[r + 1]
            for c in range(n_dim):
                ax = raw_axes[r, c]
                ax.set_box_aspect(1)

                if c > r:
                    ax.set_visible(False)
                    continue

                x_col = cols[c]
                _plot_single_pair_scatter(
                    ax=ax,
                    df_full=df_full,
                    df_plot=df_plot,
                    x_col=x_col,
                    y_col=y_col,
                    group_col=group_col,
                    groups=groups,
                    color_map=color_map,
                    r2_threshold=r2_threshold,
                    fit_reg=fit_reg,
                    jitter=jitter,
                    rng=rng,
                    show_r=show_r,
                )

                # Remove all ticks and tick numbers; only the scatter shape & R^2 matter
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)

                # Outer row/column feature names so each cell in the triangle is identifiable
                if c == 0:
                    ax.set_ylabel(
                        str(y_col),
                        fontsize=5.8,
                        color="#1E293B",
                        rotation=0 if n_dim >= 5 else 90,
                        ha="right" if n_dim >= 5 else "center",
                        va="center",
                        labelpad=6 if n_dim >= 5 else 4,
                    )
                else:
                    ax.set_ylabel("")

                if r == n_dim - 1:
                    ax.set_xlabel(
                        str(x_col),
                        fontsize=5.8,
                        color="#1E293B",
                        rotation=25 if n_dim >= 5 else 0,
                        ha="right" if n_dim >= 5 else "center",
                        labelpad=4,
                    )
                else:
                    ax.set_xlabel("")

        # Place group legend cleanly in the empty upper-right space of the triangle
        if group_col is not None:
            handles, labels = raw_axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles=handles,
                    labels=labels,
                    loc="upper right",
                    bbox_to_anchor=(0.88, 0.88),
                    frameon=True,
                    facecolor="white",
                    edgecolor="#E2E8F0",
                    framealpha=0.95,
                    markerscale=1.4,
                )

        if suptitle:
            fig.suptitle(
                suptitle,
                fontsize=8.5,
                fontweight="bold",
                color="#0F172A",
                y=0.96,
            )

        if save_path is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig, raw_axes
