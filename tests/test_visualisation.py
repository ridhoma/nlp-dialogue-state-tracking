import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from data_science.visualisations import plot_correlation_matrix, plot_distribution

matplotlib.use("Agg")


def test_plot_distribution_dataframe_with_groupby(tmp_path):
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "num_turns": rng.poisson(lam=12, size=100),
            "sop_edit_distance_norm": rng.uniform(0.0, 1.0, size=100),
            "resolved": rng.integers(0, 2, size=100),
        }
    )

    out_png = tmp_path / "dist_grouped.png"
    fig, axes = plot_distribution(
        df,
        x_columns=["num_turns", "sop_edit_distance_norm"],
        groupby="resolved",
        kind="kde",
        density=True,
        save_path=out_png,
    )

    assert axes.shape == (1, 2)
    assert axes[0, 0].get_title() == "num_turns"
    assert axes[0, 1].get_title() == "sop_edit_distance_norm"
    # Verify single legend entry per group with (\mu=...) extension
    _, labels = axes[0, 0].get_legend_handles_labels()
    assert len(labels) == 2
    assert any(lbl.startswith("resolved = 0 ($\\mu$=") for lbl in labels)
    assert any(lbl.startswith("resolved = 1 ($\\mu$=") for lbl in labels)
    assert out_png.exists()
    plt.close(fig)


def test_plot_distribution_without_groupby_quartiles():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"slot_filling_latency": rng.exponential(scale=5.0, size=80)})

    fig, axes = plot_distribution(
        df,
        x_columns="slot_filling_latency",
        groupby=None,
        kind="bar",
        density=False,
    )

    assert axes.shape == (1, 1)
    assert axes[0, 0].get_title() == "slot_filling_latency"
    _, labels = axes[0, 0].get_legend_handles_labels()
    for stat_lbl in ("$\\mu$=", "p50=", "p25=", "p75="):
        assert any(stat_lbl in lbl for lbl in labels)
    plt.close(fig)


def test_plot_distribution_numpy_2d_and_barchart():
    rng = np.random.default_rng(99)
    arr = rng.integers(0, 5, size=(120, 4))
    groups = rng.integers(0, 2, size=120)

    fig, axes = plot_distribution(
        arr,
        x_columns=["feat_a", "feat_b", "feat_c", "feat_d"],
        groupby=groups,
        kind="barchart",
        density=True,
        ncols=2,
    )

    assert axes.shape == (2, 2)
    assert axes[0, 0].get_title() == "feat_a"
    assert axes[1, 1].get_title() == "feat_d"
    plt.close(fig)


def test_plot_distribution_validation_errors():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="x_columns"):
        plot_distribution(df)

    with pytest.raises(ValueError, match="not found"):
        plot_distribution(df, x_columns=["non_existent"])


def test_plot_correlation_matrix_triangle_and_r2_highlight(tmp_path):
    rng = np.random.default_rng(123)
    x1 = rng.normal(0, 1, size=150)
    x2 = 2.0 * x1 + rng.normal(0, 0.2, size=150)  # High R^2 > 0.9
    x3 = rng.normal(0, 1, size=150)  # Low R^2
    x4 = -0.7 * x1 + rng.normal(0, 0.5, size=150)
    df = pd.DataFrame(
        {
            "f1": x1,
            "f2": x2,
            "f3": x3,
            "f4": x4,
            "resolved": rng.integers(0, 2, size=150),
        }
    )

    out_png = tmp_path / "corr_triangle.png"
    fig, axes = plot_correlation_matrix(
        df,
        x_columns=["f1", "f2", "f3", "f4"],
        groupby="resolved",
        r2_threshold=0.50,
        save_path=out_png,
    )

    # For k=4 features, lower-triangle grid is (4-1) x (4-1) = 3x3 with (16-4)/2 = 6 visible subplots
    assert axes.shape == (3, 3)
    visible_axes = [ax for ax in axes.flat if ax.get_visible()]
    assert len(visible_axes) == 6
    # Verify every visible subplot is square (box_aspect == 1) and has no ticks
    for ax in visible_axes:
        assert ax.get_box_aspect() == 1
        assert len(ax.get_xticks()) == 0
        assert len(ax.get_yticks()) == 0
    # Check that f2 vs f1 (axes[0, 0]) contains an R^2 badge and is highlighted
    texts = [t.get_text() for t in axes[0, 0].texts]
    assert any("$R^2$" in t for t in texts)
    assert out_png.exists()
    plt.close(fig)


def test_plot_correlation_matrix_numpy_array():
    rng = np.random.default_rng(42)
    arr = rng.normal(size=(100, 3))

    fig, axes = plot_correlation_matrix(
        arr,
        x_columns=["col_a", "col_b", "col_c"],
        jitter=0.01,
    )

    # 3 features -> (3-1) x (3-1) = 2x2 grid with 3 visible lower-triangle subplots
    assert axes.shape == (2, 2)
    visible_axes = [ax for ax in axes.flat if ax.get_visible()]
    assert len(visible_axes) == 3
    plt.close(fig)
