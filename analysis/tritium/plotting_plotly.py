import plotly.graph_objects as go
import numpy as np
from typing import List
from libra_toolbox.tritium import ureg
from libra_toolbox.tritium.lsc_measurements import LIBRASample, GasStream
from libra_toolbox.tritium.model import Model, quantity_to_activity
from libra_toolbox.tritium.plotting import (
    replace_water,
    COLLECTION_VOLUME,
    LSC_SAMPLE_VOLUME,
)


def plot_bars_plotly(
    measurements: List[LIBRASample] | GasStream | dict,
    index=None,
    bar_width=0.35,
):
    """
    Plot bar charts using Plotly.

    Parameters:
    - measurements: List of LIBRASample, GasStream, or dict containing measurement data.
    - index: Custom x-axis positions for the bars.
    - bar_width: Width of the bars.
    - stacked: Whether to stack the bars or group them.

    Returns:
    - A Plotly Figure object.
    """
    if isinstance(measurements, dict):
        raise NotImplementedError("plot_bars_old is not converted to Plotly yet.")

    if isinstance(measurements, GasStream):
        measurements = measurements.samples

    vial_1_vals = ureg.Quantity.from_list(
        [sample.samples[0].activity for sample in measurements]
    ).magnitude
    vial_2_vals = ureg.Quantity.from_list(
        [sample.samples[1].activity for sample in measurements]
    ).magnitude
    vial_3_vals = ureg.Quantity.from_list(
        [sample.samples[2].activity for sample in measurements]
    ).magnitude
    vial_4_vals = ureg.Quantity.from_list(
        [sample.samples[3].activity for sample in measurements]
    ).magnitude

    if index is None:
        index = np.arange(len(measurements))

    traces = []
    # Add stacked bars
    traces.append(
        go.Bar(
            x=index,
            y=vial_3_vals,
            name="Vial 3",
            marker_color="#FB8500",
        )
    )
    traces.append(
        go.Bar(
            x=index,
            y=vial_4_vals,
            name="Vial 4",
            marker_color="#FFB703",
        )
    )
    traces.append(
        go.Bar(
            x=index,
            y=vial_1_vals,
            name="Vial 1",
            marker_color="#219EBC",
        )
    )
    traces.append(
        go.Bar(
            x=index,
            y=vial_2_vals,
            name="Vial 2",
            marker_color="#8ECAE6",
        )
    )

    return traces


def plot_irradiation(model, ymax=3.5):
    # Plot irradiation
    x_irr = []
    y_irr = []
    for irr in model.irradiations:
        x_irr += [
            irr[0].to(ureg.day).magnitude,
            irr[0].to(ureg.day).magnitude,
            irr[1].to(ureg.day).magnitude,
            irr[1].to(ureg.day).magnitude,
        ]
        y_irr += [0, ymax, ymax, 0]
    # Create a filled rectangle for each irradiation period
    scatter = go.Scatter(
        x=x_irr,
        y=y_irr,
        fill="toself",
        fillcolor="rgba(239, 91, 91, 0.5)",
        line=dict(color="rgba(255,255,255,0)"),  # No border line
        name="Irradiation",
        showlegend=True,
    )
    return scatter


def plot_sample_activity_top(
    model,
    replacement_times,
    collection_vol=COLLECTION_VOLUME,
    lsc_sample_vol=LSC_SAMPLE_VOLUME,
):
    integrated_top = quantity_to_activity(model.integrated_release_top()).to(ureg.Bq)
    sample_activity_top = integrated_top / collection_vol * lsc_sample_vol
    sample_activity_top, times = replace_water(
        sample_activity_top, model.times, replacement_times
    )
    return go.Scatter(
        x=times.to(ureg.day),
        y=sample_activity_top,
        mode="lines",
        line=dict(color="#023047"),
        name="Top",
    )


def plot_sample_activity_walls(
    model,
    replacement_times,
    collection_vol=COLLECTION_VOLUME,
    lsc_sample_vol=LSC_SAMPLE_VOLUME,
):
    integrated_walls = quantity_to_activity(model.integrated_release_wall()).to(ureg.Bq)
    sample_activity_wall = integrated_walls / collection_vol * lsc_sample_vol
    sample_activity_wall, times = replace_water(
        sample_activity_wall, model.times, replacement_times
    )
    return go.Scatter(
        x=times.to(ureg.day),
        y=sample_activity_wall,
        mode="lines",
        line=dict(color="green"),
        name="Wall",
    )
