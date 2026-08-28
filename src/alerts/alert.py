"""
Malaria alert system.

Provides:
    calculate_alert_thresholds()
    classify_alerts()
    plot_alert_forecast()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "Cas Confirmés (par TDR) palu consultations externes"

SEASONAL_WINDOW = 2
CONFIDENCE_Z = 1.645


# ============================================================
# CALCULATE HISTORICAL THRESHOLDS
# ============================================================

def calculate_alert_thresholds(
    history,
    target=TARGET,
    date_col="PERIODE",
):
    """
    Calculate historical malaria alert thresholds.

    Parameters
    ----------
    history : pd.DataFrame
        Historical malaria data.

    target : str
        Target malaria variable.

    date_col : str
        Date column.

    Returns
    -------
    dict
        Contains:

        seasonal_threshold
        average_threshold
        alert_threshold
        monthly_stats
    """

    data = history.copy()

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required = [
        date_col,
        target,
    ]

    missing = [
        col
        for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns for alert calculation: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Prepare dates
    # --------------------------------------------------------

    data[date_col] = pd.to_datetime(
        data[date_col]
    )

    data = data.dropna(
        subset=[target]
    )

    if data.empty:
        raise ValueError(
            "No historical malaria observations available."
        )

    # --------------------------------------------------------
    # Monthly seasonality
    # --------------------------------------------------------

    data["month"] = data[date_col].dt.month

    # --------------------------------------------------------
    # Historical monthly statistics
    # --------------------------------------------------------

    monthly_stats = (
        data
        .groupby("month")[target]
        .agg(
            historical_mean="mean",
            historical_sd="std",
            historical_n="count",
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Replace undefined SD
    #
    # This can happen when only one historical observation
    # exists for a month.
    # --------------------------------------------------------

    monthly_stats["historical_sd"] = (
        monthly_stats["historical_sd"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Upper confidence / alert curve
    # --------------------------------------------------------

    monthly_stats["alert_curve"] = (
        monthly_stats["historical_mean"]
        + CONFIDENCE_Z
        * monthly_stats["historical_sd"]
    )

    # --------------------------------------------------------
    # Average threshold
    #
    # Maximum of the historical monthly mean curve.
    # --------------------------------------------------------

    average_threshold = (
        monthly_stats["historical_mean"]
        .max()
    )

    # --------------------------------------------------------
    # Alert threshold
    #
    # Maximum upper confidence boundary.
    # --------------------------------------------------------

    alert_threshold = (
        monthly_stats["alert_curve"]
        .max()
    )

    # --------------------------------------------------------
    # Seasonal threshold
    #
    # 95th percentile of historical observations.
    # --------------------------------------------------------

    seasonal_threshold = np.percentile(
        data[target],
        95
    )

    return {
        "seasonal_threshold": float(
            seasonal_threshold
        ),

        "average_threshold": float(
            average_threshold
        ),

        "alert_threshold": float(
            alert_threshold
        ),

        "monthly_stats": monthly_stats,
    }


# ============================================================
# CLASSIFY FORECAST
# ============================================================

def classify_alerts(
    forecast,
    thresholds,
    prediction_col="prediction",
    date_col="PERIODE",
):
    """
    Classify each forecast month against its historical
    calendar-month distribution.
    """

    result = forecast.copy()

    result[date_col] = pd.to_datetime(
        result[date_col]
    )

    result["month"] = (
        result[date_col].dt.month
    )

    monthly_stats = thresholds[
        "monthly_stats"
    ].copy()

    result = result.merge(
        monthly_stats[
            [
                "month",
                "historical_mean",
                "historical_sd",
                "alert_curve",
            ]
        ],
        on="month",
        how="left",
    )

    def classify(row):

        value = row[prediction_col]

        if value < row["historical_mean"]:
            return "Below seasonal"

        elif value < row["alert_curve"]:
            return "Seasonal / elevated"

        else:
            return "ALERT"

    result["status"] = result.apply(
        classify,
        axis=1,
    )

    result["alert_threshold"] = (
        result["alert_curve"]
    )

    return result


# ============================================================
# PLOT FORECAST + ALERTS
# ============================================================

def plot_alert_forecast(
    forecast,
    thresholds,
    prediction_col="prediction",
    date_col="PERIODE",
    title="Regional malaria forecast and alert thresholds",
):
    """
    Plot malaria forecast together with alert thresholds.

    Returns
    -------
    matplotlib.figure.Figure
    """

    data = forecast.copy()

    data[date_col] = pd.to_datetime(
        data[date_col]
    )

    seasonal_threshold = thresholds[
        "seasonal_threshold"
    ]

    average_threshold = thresholds[
        "average_threshold"
    ]

    alert_threshold = thresholds[
        "alert_threshold"
    ]

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    ax.plot(
        data[date_col],
        data[prediction_col],
        marker="o",
        linewidth=2,
        label="Predicted malaria cases",
    )

    # --------------------------------------------------------
    # Seasonal threshold
    # --------------------------------------------------------

    ax.axhline(
        seasonal_threshold,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Seasonal threshold ({seasonal_threshold:.1f})",
    )

    # --------------------------------------------------------
    # Average threshold
    # --------------------------------------------------------

    ax.axhline(
        average_threshold,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Average threshold ({average_threshold:.1f})",
    )

    # --------------------------------------------------------
    # Alert threshold
    # --------------------------------------------------------

    ax.axhline(
        alert_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Alert threshold ({alert_threshold:.1f})",
    )

    # --------------------------------------------------------
    # Highlight ALERT periods
    # --------------------------------------------------------

    alerts = data[
        data["status"] == "ALERT"
    ]

    if not alerts.empty:

        ax.scatter(
            alerts[date_col],
            alerts[prediction_col],
            color="red",
            s=100,
            zorder=5,
            label="Predicted ALERT",
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_title(title)

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Predicted malaria cases"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.autofmt_xdate()

    plt.tight_layout()
    
    plt.show()

    return fig
