"""
Dynamic Chart Generation for Financial Data

This module provides intelligent chart generation based on financial query results.
It analyzes data shape, selects appropriate chart types, and generates Vega-Lite specs.
"""

from typing import Any, Dict, List, Literal, Optional

from app.logging_config import get_logger
from app.models import FinancialDataRecord

logger = get_logger(__name__)


# =============================================================================
# Data Profiling
# =============================================================================


class DataProfile:
    """Analyzes financial data to understand its structure for charting."""

    def __init__(self, records: List[FinancialDataRecord]):
        self.records = records
        self.row_count = len(records)

        # Extract unique values (using set comprehensions per linter)
        self.companies = list({r.company for r in records})
        self.symbols = list({r.symbol for r in records})
        self.years = sorted({r.year for r in records})
        self.metrics = list({r.standard_item for r in records})

        # Cardinality
        self.num_companies = len(self.companies)
        self.num_years = len(self.years)
        self.num_metrics = len(self.metrics)

        # Data quality
        self.has_numeric = any(r.item is not None for r in records)
        self.null_count = sum(1 for r in records if r.item is None)

    def summary(self) -> str:
        """Human-readable summary for logging/debugging."""
        return (
            f"DataProfile: {self.row_count} rows, "
            f"{self.num_companies} companies, {self.num_years} years, "
            f"{self.num_metrics} metrics, has_numeric={self.has_numeric}"
        )


# =============================================================================
# Chartability Detection
# =============================================================================


def is_chartable(records: List[FinancialDataRecord]) -> bool:
    """
    Determine if the data is suitable for charting.

    Returns True if:
    - More than 1 record
    - Has numeric data
    - Has variation (multiple companies OR multiple years)
    """
    if len(records) < 2:
        return False

    # Check if we have numeric data
    has_numeric = any(r.item is not None for r in records)
    if not has_numeric:
        return False

    # Check for variety (not all same company/year)
    unique_companies = len({r.company for r in records})
    unique_years = len({r.year for r in records})

    return unique_companies > 1 or unique_years > 1


# =============================================================================
# Chart Type Selection
# =============================================================================


ChartType = Literal["line", "bar", "grouped_bar", "horizontal_bar", "table"]


def select_chart_type(profile: DataProfile) -> ChartType:
    """
    Select the best chart type based on data shape.

    Heuristics:
    - 1 company, multiple years → line (time series)
    - Multiple companies, 1 year → bar (comparison)
    - Multiple companies, multiple years → grouped_bar
    - 1 company, 1 year, multiple metrics → horizontal_bar
    """
    if profile.num_companies == 1 and profile.num_years > 1:
        return "line"
    elif profile.num_companies > 1 and profile.num_years == 1:
        return "bar"
    elif profile.num_companies > 1 and profile.num_years > 1:
        # Multi-company, multi-year: use grouped bar or multi-line
        if profile.num_years <= 5:
            return "grouped_bar"
        else:
            return "line"  # Too many years for grouped bar
    elif profile.num_metrics > 1:
        return "horizontal_bar"
    else:
        return "bar"  # Default fallback


# =============================================================================
# Vega-Lite Spec Generation
# =============================================================================


def generate_vega_lite_spec(
    records: List[FinancialDataRecord],
    chart_type: ChartType,
    profile: DataProfile,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a Vega-Lite specification for the given data and chart type.
    """
    # Prepare data for Vega-Lite
    data_values = []
    for r in records:
        if r.item is not None:
            # Calculate actual value with multiplier
            actual_value = r.item * (r.unit_multiplier or 1)
            data_values.append(
                {
                    "company": r.company,
                    "symbol": r.symbol,
                    "year": r.year,
                    "metric": r.standard_item,
                    "value": actual_value,
                    "formatted": r.formatted_value,
                }
            )

    if not data_values:
        return {}

    # Auto-generate title if not provided
    if not title:
        if profile.num_companies == 1:
            company = profile.symbols[0] if profile.symbols else profile.companies[0]
            metric = profile.metrics[0] if profile.metrics else "Data"
            title = f"{company} {metric.replace('_', ' ').title()}"
            if profile.num_years > 1:
                title += f" ({profile.years[0]}-{profile.years[-1]})"
        else:
            metric = profile.metrics[0] if profile.metrics else "Data"
            title = f"{metric.replace('_', ' ').title()} Comparison"

    # Base spec with interactivity
    spec: Dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "data": {"values": data_values},
        "width": 500,
        "height": 300,
        "params": [],  # Will hold selection parameters
    }

    # Determine the color field for interactivity
    color_field = "symbol"  # default
    if chart_type == "grouped_bar":
        color_field = "year"
    elif chart_type == "horizontal_bar":
        color_field = "metric"

    # Add hover highlight selection
    spec["params"].append(
        {
            "name": "hover",
            "select": {
                "type": "point",
                "on": "pointerover",
                "clear": "pointerout",
            },
        }
    )

    # Add legend selection for multi-series charts
    if profile.num_companies > 1 or chart_type == "grouped_bar":
        spec["params"].append(
            {
                "name": "legend_select",
                "select": {"type": "point", "fields": [color_field]},
                "bind": "legend",
            }
        )

    # Chart-type specific encoding
    if chart_type == "line":
        spec["mark"] = {"type": "line", "point": {"size": 80}}
        spec["encoding"] = {
            "x": {
                "field": "year",
                "type": "ordinal",
                "title": "Year",
                "sort": "ascending",
            },
            "y": {
                "field": "value",
                "type": "quantitative",
                "title": _format_metric_title(profile.metrics[0] if profile.metrics else "Value"),
            },
        }
        # Add color encoding if multiple companies
        if profile.num_companies > 1:
            spec["encoding"]["color"] = {
                "field": "symbol",
                "type": "nominal",
                "title": "Company",
            }
            # Add opacity for highlight effect
            spec["encoding"]["opacity"] = {
                "condition": [
                    {"param": "hover", "empty": False, "value": 1},
                    {"param": "legend_select", "value": 1},
                ],
                "value": 0.3,
            }
            spec["encoding"]["strokeWidth"] = {
                "condition": {"param": "hover", "empty": False, "value": 3},
                "value": 2,
            }

    elif chart_type == "bar":
        spec["mark"] = {"type": "bar", "cursor": "pointer"}
        spec["encoding"] = {
            "x": {
                "field": "symbol",
                "type": "nominal",
                "title": "Company",
                "sort": "-y",
            },
            "y": {
                "field": "value",
                "type": "quantitative",
                "title": _format_metric_title(profile.metrics[0] if profile.metrics else "Value"),
            },
            "color": {
                "field": "symbol",
                "type": "nominal",
                "legend": None,
            },
            "opacity": {
                "condition": {"param": "hover", "empty": False, "value": 1},
                "value": 0.7,
            },
        }

    elif chart_type == "grouped_bar":
        spec["mark"] = {"type": "bar", "cursor": "pointer"}
        spec["encoding"] = {
            "x": {
                "field": "symbol",
                "type": "nominal",
                "title": "Company",
            },
            "xOffset": {
                "field": "year",
                "type": "nominal",
            },
            "y": {
                "field": "value",
                "type": "quantitative",
                "title": _format_metric_title(profile.metrics[0] if profile.metrics else "Value"),
            },
            "color": {
                "field": "year",
                "type": "nominal",
                "title": "Year",
            },
            "opacity": {
                "condition": [
                    {"param": "hover", "empty": False, "value": 1},
                    {"param": "legend_select", "value": 1},
                ],
                "value": 0.4,
            },
        }

    elif chart_type == "horizontal_bar":
        spec["mark"] = {"type": "bar", "cursor": "pointer"}
        spec["encoding"] = {
            "y": {
                "field": "metric",
                "type": "nominal",
                "title": "Metric",
                "sort": "-x",
            },
            "x": {
                "field": "value",
                "type": "quantitative",
                "title": "Value",
            },
            "color": {
                "field": "metric",
                "type": "nominal",
                "legend": None,
            },
            "opacity": {
                "condition": {"param": "hover", "empty": False, "value": 1},
                "value": 0.7,
            },
        }

    # Add tooltip
    spec["encoding"]["tooltip"] = [
        {"field": "symbol", "type": "nominal", "title": "Symbol"},
        {"field": "company", "type": "nominal", "title": "Company"},
        {"field": "year", "type": "ordinal", "title": "Year"},
        {"field": "formatted", "type": "nominal", "title": "Value"},
    ]

    return spec


def _format_metric_title(metric: str) -> str:
    """Format metric name for display."""
    return metric.replace("_", " ").title()


# =============================================================================
# Main Chart Generation Function
# =============================================================================


def generate_chart(
    records: List[FinancialDataRecord],
    query: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Generate a chart specification for the given financial data.

    Returns None if data is not chartable.

    Returns:
        dict with keys: chart_type, title, description, vega_lite
    """
    if not is_chartable(records):
        logger.debug(f"Data not chartable: {len(records)} records")
        return None

    # Profile the data
    profile = DataProfile(records)
    logger.info(f"Chart generation: {profile.summary()}")

    # Select chart type
    chart_type = select_chart_type(profile)
    logger.info(f"Selected chart type: {chart_type}")

    # Generate Vega-Lite spec
    vega_spec = generate_vega_lite_spec(records, chart_type, profile)

    if not vega_spec:
        logger.warning("Failed to generate Vega-Lite spec")
        return None

    # Build description
    description = _build_chart_description(chart_type, profile)

    return {
        "chart_type": chart_type,
        "title": vega_spec.get("title", "Financial Data"),
        "description": description,
        "vega_lite": vega_spec,
    }


def _build_chart_description(chart_type: ChartType, profile: DataProfile) -> str:
    """Build a human-readable description of the chart."""
    if chart_type == "line":
        return f"Time series showing {profile.metrics[0] if profile.metrics else 'data'} over {profile.num_years} years"
    elif chart_type == "bar":
        return f"Comparison of {profile.num_companies} companies"
    elif chart_type == "grouped_bar":
        return f"Comparison of {profile.num_companies} companies across {profile.num_years} years"
    elif chart_type == "horizontal_bar":
        return f"Comparison of {profile.num_metrics} metrics"
    return "Financial data visualization"
