import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from pydantic import BaseModel, Field, field_validator
from matplotlib.ticker import FuncFormatter

from src.formulae import (
    compute_dimensionless_pressure,
    compute_dimension_pressure_derivative,
    compute_dimensionless_pressure_gradient,
)


# Configure x-axis formatter for better small number display
def format_func(x, pos):
    if x >= 1:
        return f'{x:.1f}'
    elif x >= 0.01:
        return f'{x:.3f}'
    else:
        return f'{x:.1e}'


class ReservoirParameters(BaseModel):
    """Pydantic model for reservoir parameter validation."""

    alpha: float = Field(
        ..., gt=0, description="Alpha parameter (dimensionless)", title="Alpha (α)"
    )
    exponential_integral: float = Field(
        ..., description="Exponential integral value", title="Exponential Integral (Ei)"
    )
    dimensionless_length: float = Field(
        ...,
        gt=0,
        description="Dimensionless length parameter",
        title="Dimensionless Length (LD)",
    )
    dimensionless_wellbore_radius: float = Field(
        ...,
        gt=0,
        description="Dimensionless wellbore radius",
        title="Dimensionless Wellbore Radius (r_wD)",
    )
    wellbore_storage_constant: float = Field(
        ...,
        gt=0,
        description="Wellbore storage constant",
        title="Wellbore Storage Constant (CD)",
    )
    distance_to_boundary: float = Field(
        ..., gt=0, description="Distance to boundary", title="Distance to Boundary (d)"
    )
    skin_factor: float = Field(
        default=0.0, description="Skin factor (default: 0.0)", title="Skin Factor (S)"
    )

    @field_validator("exponential_integral")
    def validate_exponential_integral(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("Exponential integral must be a number")
        return v


class TimeParameters(BaseModel):
    """Pydantic model for time parameter validation."""

    min_time: float = Field(
        ..., gt=0, description="Minimum dimensionless time", title="Min Time (tD)"
    )
    max_time: float = Field(
        ..., gt=0, description="Maximum dimensionless time", title="Max Time (tD)"
    )
    step_size: float = Field(
        ..., gt=0, description="Step size for time range", title="Step Size"
    )

    @field_validator("max_time")
    @classmethod
    def validate_max_time(cls, v, info):
        if info.data.get("min_time") and v <= info.data.get("min_time"):
            raise ValueError("Max time must be greater than min time")
        return v

    @field_validator("step_size")
    @classmethod
    def validate_step_size(cls, v, info):
        if info.data.get("min_time") and info.data.get("max_time"):
            if v >= (info.data.get("max_time") - info.data.get("min_time")):
                raise ValueError("Step size must be smaller than the time range")
        return v


class VaryingParameter(BaseModel):
    """Pydantic model for varying parameter validation."""

    parameter_name: str = Field(
        ..., description="Name of the parameter to vary"
    )
    min_value: float = Field(
        ..., description="Minimum value of the parameter"
    )
    max_value: float = Field(
        ..., description="Maximum value of the parameter"
    )
    step_size: float = Field(
        ..., gt=0, description="Step size for parameter range"
    )

    @field_validator("max_value")
    @classmethod
    def validate_max_value(cls, v, info):
        if info.data.get("min_value") and v <= info.data.get("min_value"):
            raise ValueError("Max value must be greater than min value")
        return v

    @field_validator("step_size")
    @classmethod
    def validate_step_size(cls, v, info):
        if info.data.get("min_value") and info.data.get("max_value"):
            if v >= (info.data.get("max_value") - info.data.get("min_value")):
                raise ValueError("Step size must be smaller than the parameter range")
        return v


def create_time_array(time_params: TimeParameters) -> np.ndarray:
    """Create linearly spaced time array based on min/max/step."""
    return np.arange(time_params.min_time, time_params.max_time + time_params.step_size, time_params.step_size)


def create_parameter_array(varying_param: VaryingParameter) -> np.ndarray:
    """Create linearly spaced parameter array based on min/max/step."""
    return np.arange(varying_param.min_value, varying_param.max_value + varying_param.step_size, varying_param.step_size)


def compute_all_functions_single(
    params: ReservoirParameters, 
    time_array: np.ndarray, 
    include_gradient: bool = True
) -> pd.DataFrame:
    """Compute all functions for the given time array with single parameter values."""
    results = []

    for td in time_array:
        try:
            # Compute dimensionless pressure
            pd_value = compute_dimensionless_pressure(
                alpha=params.alpha,
                exponential_integral=params.exponential_integral,
                dimensionless_length=params.dimensionless_length,
                dimensionless_wellbore_radius=params.dimensionless_wellbore_radius,
                dimensionless_time=td,
                wellbore_storage_constant=params.wellbore_storage_constant,
                distance_to_boundary=params.distance_to_boundary,
                skin_factor=params.skin_factor,
            )

            # Compute pressure derivative
            pd_derivative = compute_dimension_pressure_derivative(
                alpha=params.alpha,
                dimensionless_length=params.dimensionless_length,
                dimensionless_wellbore_radius=params.dimensionless_wellbore_radius,
                dimensionless_time=td,
                wellbore_storage_constant=params.wellbore_storage_constant,
                distance_to_boundary=params.distance_to_boundary,
            )

            result_row = {
                "Dimensionless Time (tD)": td,
                "Dimensionless Pressure (pD)": pd_value,
                "Pressure Derivative (p'D)": pd_derivative,
            }

            # Optionally compute pressure gradient
            if include_gradient:
                pd_gradient = compute_dimensionless_pressure_gradient(
                    alpha=params.alpha,
                    dimensionless_length=params.dimensionless_length,
                    dimensionless_wellbore_radius=params.dimensionless_wellbore_radius,
                    dimensionless_time=td,
                    wellbore_storage_constant=params.wellbore_storage_constant,
                    distance_to_boundary=params.distance_to_boundary,
                )
                result_row["Pressure Gradient (∂pD/∂tD)"] = pd_gradient

            results.append(result_row)

        except Exception as e:
            st.error(f"Error computing values at tD = {td}: {str(e)}")
            continue

    return pd.DataFrame(results)


def compute_all_functions_varying(
    base_params: ReservoirParameters, 
    time_array: np.ndarray, 
    varying_param: VaryingParameter,
    param_array: np.ndarray,
    include_gradient: bool = True
) -> pd.DataFrame:
    """Compute all functions for varying parameter values across time array."""
    results = []

    for td in time_array:
        result_row = {"Dimensionless Time (tD)": td}
        
        for param_value in param_array:
            # Create a copy of base parameters and update the varying parameter
            current_params = base_params.model_copy()
            
            # Update the specific parameter
            if varying_param.parameter_name == "dimensionless_length":
                current_params.dimensionless_length = param_value
            elif varying_param.parameter_name == "dimensionless_wellbore_radius":
                current_params.dimensionless_wellbore_radius = param_value
            elif varying_param.parameter_name == "wellbore_storage_constant":
                current_params.wellbore_storage_constant = param_value
            elif varying_param.parameter_name == "distance_to_boundary":
                current_params.distance_to_boundary = param_value
            elif varying_param.parameter_name == "skin_factor":
                current_params.skin_factor = param_value

            try:
                # Compute dimensionless pressure
                pd_value = compute_dimensionless_pressure(
                    alpha=current_params.alpha,
                    exponential_integral=current_params.exponential_integral,
                    dimensionless_length=current_params.dimensionless_length,
                    dimensionless_wellbore_radius=current_params.dimensionless_wellbore_radius,
                    dimensionless_time=td,
                    wellbore_storage_constant=current_params.wellbore_storage_constant,
                    distance_to_boundary=current_params.distance_to_boundary,
                    skin_factor=current_params.skin_factor,
                )

                # Compute pressure derivative
                pd_derivative = compute_dimension_pressure_derivative(
                    alpha=current_params.alpha,
                    dimensionless_length=current_params.dimensionless_length,
                    dimensionless_wellbore_radius=current_params.dimensionless_wellbore_radius,
                    dimensionless_time=td,
                    wellbore_storage_constant=current_params.wellbore_storage_constant,
                    distance_to_boundary=current_params.distance_to_boundary,
                )

                # Create column names with parameter values
                param_suffix = f"({varying_param.parameter_name}={param_value:.3f})"
                result_row[f"pD {param_suffix}"] = pd_value
                result_row[f"p'D {param_suffix}"] = pd_derivative

                # Optionally compute pressure gradient
                if include_gradient:
                    pd_gradient = compute_dimensionless_pressure_gradient(
                        alpha=current_params.alpha,
                        dimensionless_length=current_params.dimensionless_length,
                        dimensionless_wellbore_radius=current_params.dimensionless_wellbore_radius,
                        dimensionless_time=td,
                        wellbore_storage_constant=current_params.wellbore_storage_constant,
                        distance_to_boundary=current_params.distance_to_boundary,
                    )
                    result_row[f"∂pD/∂tD {param_suffix}"] = pd_gradient

            except Exception as e:
                st.error(f"Error computing values at tD = {td}, {varying_param.parameter_name} = {param_value}: {str(e)}")
                continue

        results.append(result_row)

    return pd.DataFrame(results)


def create_individual_plots_single(df: pd.DataFrame, show_markers: bool = True, show_grid: bool = True, line_width: float = 2.5, include_gradient: bool = True) -> tuple:
    """Create individual plots for each output variable with single parameter values."""
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Determine number of subplots based on available data
    num_plots = 3 if include_gradient and 'Pressure Gradient (∂pD/∂tD)' in df.columns else 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    if num_plots == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    title_suffix = f"Semilog Plots (Log tD axis) - {num_plots} Parameters"
    fig.suptitle(f'Water Breakthrough Analysis - {title_suffix}', fontsize=16, fontweight='bold')
    
    time_data = df['Dimensionless Time (tD)']
    marker_style = 'o' if show_markers else None
    marker_size = 4 if show_markers else 0
    
    # Configure matplotlib to handle small numbers better
    plt.rcParams['axes.formatter.limits'] = (-3, 4)
    plt.rcParams['axes.formatter.use_mathtext'] = False
    
    # Plot 1: Dimensionless Pressure
    axes[0].semilogx(time_data, df['Dimensionless Pressure (pD)'], 'b-', linewidth=line_width, marker=marker_style, markersize=marker_size)
    axes[0].set_xlabel('Dimensionless Time (tD) - Log Scale', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Dimensionless Pressure (pD)', fontsize=12)
    axes[0].set_title('Dimensionless Pressure vs Log(tD)', fontsize=13, fontweight='bold')
    axes[0].grid(show_grid, alpha=0.3, which='both')  # Show both major and minor grid lines
    axes[0].set_xlim(time_data.min(), time_data.max())
    axes[0].xaxis.set_major_formatter(FuncFormatter(format_func))
    
    # Plot 2: Pressure Derivative
    marker_style2 = 's' if show_markers else None
    axes[1].semilogx(time_data, df["Pressure Derivative (p'D)"], 'r-', linewidth=line_width, marker=marker_style2, markersize=marker_size)
    axes[1].set_xlabel('Dimensionless Time (tD) - Log Scale', fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Pressure Derivative (p'D)", fontsize=12)
    axes[1].set_title("Pressure Derivative vs Log(tD)", fontsize=13, fontweight='bold')
    axes[1].grid(show_grid, alpha=0.3, which='both')  # Show both major and minor grid lines
    axes[1].set_xlim(time_data.min(), time_data.max())
    axes[1].xaxis.set_major_formatter(FuncFormatter(format_func))
    
    # Plot 3: Pressure Gradient (optional)
    if num_plots == 3 and 'Pressure Gradient (∂pD/∂tD)' in df.columns:
        marker_style3 = '^' if show_markers else None
        axes[2].semilogx(time_data, df['Pressure Gradient (∂pD/∂tD)'], 'g-', linewidth=line_width, marker=marker_style3, markersize=marker_size)
        axes[2].set_xlabel('Dimensionless Time (tD) - Log Scale', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Pressure Gradient (∂pD/∂tD)', fontsize=12)
        axes[2].set_title('Pressure Gradient vs Log(tD)', fontsize=13, fontweight='bold')
        axes[2].grid(show_grid, alpha=0.3, which='both')  # Show both major and minor grid lines
        axes[2].set_xlim(time_data.min(), time_data.max())
        axes[2].xaxis.set_major_formatter(FuncFormatter(format_func))
    
    plt.tight_layout()
    return fig, axes


def create_individual_plots_varying(df: pd.DataFrame, varying_param: VaryingParameter, show_markers: bool = True, show_grid: bool = True, line_width: float = 2.5, include_gradient: bool = True) -> tuple:
    """Create individual plots for each output variable with varying parameter lines."""
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Get parameter values from column names
    param_values = []
    for col in df.columns:
        if col.startswith("pD ("):
            param_val = float(col.split("=")[1].split(")")[0])
            param_values.append(param_val)
    
    # Determine number of subplots based on available data
    has_gradient = include_gradient and any(col.startswith("∂pD/∂tD") for col in df.columns)
    num_plots = 3 if has_gradient else 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    if num_plots == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    title_suffix = f"Varying {varying_param.parameter_name.replace('_', ' ').title()}"
    fig.suptitle(f'Water Breakthrough Analysis - {title_suffix}', fontsize=16, fontweight='bold')
    
    time_data = df['Dimensionless Time (tD)']
    
    # Configure matplotlib to handle small numbers better
    plt.rcParams['axes.formatter.limits'] = (-3, 4)
    plt.rcParams['axes.formatter.use_mathtext'] = False
    
    # Get color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(param_values)))
    
    # Plot 1: Dimensionless Pressure
    for i, param_val in enumerate(param_values):
        col_name = f"pD ({varying_param.parameter_name}={param_val:.3f})"
        marker_style = 'o' if show_markers else None
        marker_size = 4 if show_markers else 0
        
        axes[0].semilogx(time_data, df[col_name], color=colors[i], linewidth=line_width, 
                        marker=marker_style, markersize=marker_size, label=f'{param_val:.3f}')
    
    axes[0].set_xlabel('Dimensionless Time (tD) - Log Scale', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Dimensionless Pressure (pD)', fontsize=12)
    axes[0].set_title('Dimensionless Pressure vs Log(tD)', fontsize=13, fontweight='bold')
    axes[0].grid(show_grid, alpha=0.3, which='both')
    axes[0].set_xlim(time_data.min(), time_data.max())
    axes[0].xaxis.set_major_formatter(FuncFormatter(format_func))
    axes[0].legend(title=varying_param.parameter_name.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Pressure Derivative
    for i, param_val in enumerate(param_values):
        col_name = f"p'D ({varying_param.parameter_name}={param_val:.3f})"
        marker_style = 's' if show_markers else None
        marker_size = 4 if show_markers else 0
        
        axes[1].semilogx(time_data, df[col_name], color=colors[i], linewidth=line_width, 
                        marker=marker_style, markersize=marker_size, label=f'{param_val:.3f}')
    
    axes[1].set_xlabel('Dimensionless Time (tD) - Log Scale', fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Pressure Derivative (p'D)", fontsize=12)
    axes[1].set_title("Pressure Derivative vs Log(tD)", fontsize=13, fontweight='bold')
    axes[1].grid(show_grid, alpha=0.3, which='both')
    axes[1].set_xlim(time_data.min(), time_data.max())
    axes[1].xaxis.set_major_formatter(FuncFormatter(format_func))
    axes[1].legend(title=varying_param.parameter_name.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Pressure Gradient (optional)
    if has_gradient:
        for i, param_val in enumerate(param_values):
            col_name = f"∂pD/∂tD ({varying_param.parameter_name}={param_val:.3f})"
            marker_style = '^' if show_markers else None
            marker_size = 4 if show_markers else 0
            
            axes[2].semilogx(time_data, df[col_name], color=colors[i], linewidth=line_width, 
                            marker=marker_style, markersize=marker_size, label=f'{param_val:.3f}')
        
        axes[2].set_xlabel('Dimensionless Time (tD) - Log Scale', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Pressure Gradient (∂pD/∂tD)', fontsize=12)
        axes[2].set_title('Pressure Gradient vs Log(tD)', fontsize=13, fontweight='bold')
        axes[2].grid(show_grid, alpha=0.3, which='both')
        axes[2].set_xlim(time_data.min(), time_data.max())
        axes[2].xaxis.set_major_formatter(FuncFormatter(format_func))
        axes[2].legend(title=varying_param.parameter_name.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig, axes


def create_combined_plot(df: pd.DataFrame, plot_type: str = "separate_y", show_markers: bool = True, show_grid: bool = True, line_width: float = 2.5, include_gradient: bool = True) -> plt.Figure:
    """Create combined plots with different scaling options."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    time_data = df['Dimensionless Time (tD)']
    marker_size = 4 if show_markers else 0
    has_gradient = include_gradient and 'Pressure Gradient (∂pD/∂tD)' in df.columns
    
    # Configure matplotlib to handle small numbers better
    plt.rcParams['axes.formatter.limits'] = (-3, 4)
    plt.rcParams['axes.formatter.use_mathtext'] = False
    
    # Configure x-axis formatter for better small number display
    from matplotlib.ticker import FuncFormatter
    def format_func(x, pos):
        if x >= 1:
            return f'{x:.1f}'
        elif x >= 0.01:
            return f'{x:.3f}'
        else:
            return f'{x:.1e}'
    
    if plot_type == "separate_y":
        # Separate y-axes for different scales
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Plot dimensionless pressure on primary y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Dimensionless Time (tD) - Log Scale', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Dimensionless Pressure (pD)', color=color1, fontsize=12)
        marker1 = 'o' if show_markers else None
        line1 = ax1.semilogx(time_data, df['Dimensionless Pressure (pD)'], 
                            color=color1, linewidth=line_width, marker=marker1, markersize=marker_size, label='pD')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(show_grid, alpha=0.3, which='both')  # Show both major and minor grid lines
        ax1.xaxis.set_major_formatter(FuncFormatter(format_func))
        
        # Create second y-axis for derivative
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel("Pressure Derivative (p'D)", color=color2, fontsize=12)
        marker2 = 's' if show_markers else None
        line2 = ax2.semilogx(time_data, df["Pressure Derivative (p'D)"], 
                            color=color2, linewidth=line_width, marker=marker2, markersize=marker_size, label="p'D")
        ax2.tick_params(axis='y', labelcolor=color2)
        
        lines = line1 + line2
        
        # Create third y-axis for gradient (optional)
        if has_gradient:
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            color3 = 'tab:green'
            ax3.set_ylabel('Pressure Gradient (∂pD/∂tD)', color=color3, fontsize=12)
            marker3 = '^' if show_markers else None
            line3 = ax3.semilogx(time_data, df['Pressure Gradient (∂pD/∂tD)'], 
                                color=color3, linewidth=line_width, marker=marker3, markersize=marker_size, label='∂pD/∂tD')
            ax3.tick_params(axis='y', labelcolor=color3)
            lines += line3
        
        # Add legend
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        params_count = "3 Parameters" if has_gradient else "2 Parameters"
        plt.title(f'Combined Water Breakthrough Analysis - Semilog Plot ({params_count})', 
                 fontsize=14, fontweight='bold', pad=20)
        
    else:  # normalized plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalize all values to 0-1 range for comparison
        pressure_norm = (df['Dimensionless Pressure (pD)'] - df['Dimensionless Pressure (pD)'].min()) / \
                       (df['Dimensionless Pressure (pD)'].max() - df['Dimensionless Pressure (pD)'].min())
        derivative_norm = (df["Pressure Derivative (p'D)"] - df["Pressure Derivative (p'D)"].min()) / \
                         (df["Pressure Derivative (p'D)"].max() - df["Pressure Derivative (p'D)"].min())
        
        marker1 = 'o' if show_markers else None
        marker2 = 's' if show_markers else None
        
        ax.semilogx(time_data, pressure_norm, 'b-', linewidth=line_width, marker=marker1, markersize=marker_size, label='pD (normalized)')
        ax.semilogx(time_data, derivative_norm, 'r-', linewidth=line_width, marker=marker2, markersize=marker_size, label="p'D (normalized)")
        
        # Add gradient if included
        if has_gradient:
            gradient_norm = (df['Pressure Gradient (∂pD/∂tD)'] - df['Pressure Gradient (∂pD/∂tD)'].min()) / \
                           (df['Pressure Gradient (∂pD/∂tD)'].max() - df['Pressure Gradient (∂pD/∂tD)'].min())
            marker3 = '^' if show_markers else None
            ax.semilogx(time_data, gradient_norm, 'g-', linewidth=line_width, marker=marker3, markersize=marker_size, label='∂pD/∂tD (normalized)')
        
        ax.set_xlabel('Dimensionless Time (tD) - Log Scale', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Values (0-1)', fontsize=12)
        params_count = "3 Parameters" if has_gradient else "2 Parameters"
        ax.set_title(f'Combined Water Breakthrough Analysis - Semilog Plot (Normalized, {params_count})', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(show_grid, alpha=0.3, which='both')  # Show both major and minor grid lines
        ax.set_xlim(time_data.min(), time_data.max())
        ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    
    plt.tight_layout()
    return fig


def create_separate_tables(df: pd.DataFrame, varying_param: VaryingParameter, include_gradient: bool = True):
    """Create separate tables for each pressure type when using parameter variation."""
    
    # Get parameter values from column names
    param_values = []
    for col in df.columns:
        if col.startswith("pD ("):
            param_val = float(col.split("=")[1].split(")")[0])
            param_values.append(param_val)
    
    # Create separate DataFrames for each pressure type
    tables = {}
    
    # Dimensionless Pressure Table
    pD_data = {"Dimensionless Time (tD)": df["Dimensionless Time (tD)"]}
    for param_val in param_values:
        col_name = f"pD ({varying_param.parameter_name}={param_val:.3f})"
        if col_name in df.columns:
            pD_data[f"pD ({param_val:.3f})"] = df[col_name]
    tables["Dimensionless Pressure (pD)"] = pd.DataFrame(pD_data)
    
    # Pressure Derivative Table
    pD_deriv_data = {"Dimensionless Time (tD)": df["Dimensionless Time (tD)"]}
    for param_val in param_values:
        col_name = f"p'D ({varying_param.parameter_name}={param_val:.3f})"
        if col_name in df.columns:
            pD_deriv_data[f"p'D ({param_val:.3f})"] = df[col_name]
    tables["Pressure Derivative (p'D)"] = pd.DataFrame(pD_deriv_data)
    
    # Pressure Gradient Table (optional)
    if include_gradient:
        pD_grad_data = {"Dimensionless Time (tD)": df["Dimensionless Time (tD)"]}
        for param_val in param_values:
            col_name = f"∂pD/∂tD ({varying_param.parameter_name}={param_val:.3f})"
            if col_name in df.columns:
                pD_grad_data[f"∂pD/∂tD ({param_val:.3f})"] = df[col_name]
        if len(pD_grad_data) > 1:  # More than just time column
            tables["Pressure Gradient (∂pD/∂tD)"] = pd.DataFrame(pD_grad_data)
    
    return tables


def save_plot_as_image(fig: plt.Figure, filename: str, dpi: int = 300) -> BytesIO:
    """Save plot as image in memory."""
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    img_buffer.seek(0)
    return img_buffer


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Water Breakthrough Analysis", page_icon="�", layout="wide"
    )

    st.title("� Water Breakthrough Analysis Calculator")
    st.markdown("""
    This application analyzes factors affecting water breakthrough time for horizontal wells near constant pressure boundaries.
    It computes dimensionless pressure, pressure derivative, and pressure gradient to understand water breakthrough behavior.
    
    **📋 Instructions:**
    1. Enter well and reservoir parameters (Alpha and Exponential Integral remain constant)
    2. Configure time range using min/max/step format for breakthrough analysis
    3. Select ONE parameter to vary and set its range (min/max/step)
    4. Click "Compute Results" to generate calculations for all parameter combinations
    5. View results in tabular format with multiple columns for each parameter value
    6. Analyze plots showing multiple lines for different parameter values
    """)

    st.divider()

    st.subheader("🏗️ Well & Reservoir Parameters")
    st.markdown("*Configure the horizontal well and reservoir properties (these remain constant)*")

    # Input fields for constant reservoir parameters
    alpha = st.number_input(
        "Alpha (α) - Constant",
        min_value=None,
        value=2.0,
        help="**Alpha parameter** - A dimensionless parameter that characterizes the reservoir system. Must be positive. Typical range: 0.1 - 10.0. This parameter remains constant.",
    )

    exponential_integral = st.number_input(
        "Exponential Integral (Ei) - Constant",
        value=-0.5,
        help="**Exponential integral value** - Mathematical function used in water breakthrough pressure calculations. Can be positive or negative. Typical range: -5.0 to 5.0. This parameter remains constant.",
    )

    # Parameter variation section - moved before parameter inputs
    st.divider()
    st.subheader("🔄 Parameter Variation")
    st.markdown("*Choose whether to use single values or vary one parameter for comprehensive analysis*")

    # Parameter variation toggle
    use_parameter_variation = st.checkbox(
        "Enable Parameter Variation",
        value=False,
        help="Check this to vary one parameter across a range. Uncheck to use single parameter values for all calculations."
    )

    selected_param = None
    if use_parameter_variation:
        st.info("🔄 **Parameter Variation Mode**: One parameter will be varied while others remain constant.")
        
        # Parameter selection
        param_options = {
            "Dimensionless Length (LD)": "dimensionless_length",
            "Dimensionless Wellbore Radius (r_wD)": "dimensionless_wellbore_radius", 
            "Wellbore Storage Constant (CD)": "wellbore_storage_constant",
            "Distance to Water Boundary (d)": "distance_to_boundary",
            "Skin Factor (S)": "skin_factor"
        }

        selected_param = st.selectbox(
            "Choose parameter to vary:",
            list(param_options.keys()),
            help="Select which parameter to vary while keeping others constant. This will create multiple curves in the plots."
        )

        # Parameter range inputs
        param_col1, param_col2, param_col3 = st.columns(3)

        with param_col1:
            param_min = st.number_input(
                f"Min {selected_param}",
                value=0.1 if "Skin Factor" not in selected_param else -2.0,
                format="%.3f",
                help=f"**Minimum value for {selected_param}**"
            )

        with param_col2:
            param_max = st.number_input(
                f"Max {selected_param}",
                value=2.0,
                format="%.3f", 
                help=f"**Maximum value for {selected_param}**"
            )

        with param_col3:
            param_step = st.number_input(
                f"Step Size",
                min_value=0.001,
                value=0.5,
                format="%.3f",
                help=f"**Step size for {selected_param}** increments"
            )
    else:
        st.info("📊 **Single Value Mode**: All parameters will use their base values for calculations.")
        param_min = param_max = param_step = None

    # Now define parameter inputs with conditional display
    st.divider()
    st.subheader("⚙️ Parameter Configuration")
    st.markdown("*Configure remaining parameters (those not being varied)*")
    
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        st.markdown("**🏗️ Well & Reservoir Parameters**")
        
        # Get the selected parameter key for variation
        selected_param_key = None
        if use_parameter_variation and selected_param:
            param_options = {
                "Dimensionless Length (LD)": "dimensionless_length",
                "Dimensionless Wellbore Radius (r_wD)": "dimensionless_wellbore_radius", 
                "Wellbore Storage Constant (CD)": "wellbore_storage_constant",
                "Distance to Water Boundary (d)": "distance_to_boundary",
                "Skin Factor (S)": "skin_factor"
            }
            selected_param_key = param_options.get(selected_param, None)

        # Dimensionless Length input
        if not (use_parameter_variation and selected_param_key == "dimensionless_length"):
            dimensionless_length = st.number_input(
                "Dimensionless Length (LD) - Base Value",
                min_value=None,
                value=1.0,
                help="**Dimensionless length parameter** - Base value when not varying this parameter. Characterizes the horizontal well length relative to reservoir extent. Must be positive. Typical range: 100 - 10,000",
            )
        else:
            dimensionless_length = 1.0  # Default value when varying
            st.info("🔄 **Dimensionless Length (LD)** is set to vary - configured above in Parameter Variation section")

        # Dimensionless Wellbore Radius input
        if not (use_parameter_variation and selected_param_key == "dimensionless_wellbore_radius"):
            dimensionless_wellbore_radius = st.number_input(
                "Dimensionless Wellbore Radius (r_wD) - Base Value",
                min_value=None,
                value=0.1,
                help="**Dimensionless wellbore radius** - Base value when not varying this parameter. Ratio of horizontal well radius to characteristic length. Must be positive. Typical range: 0.01 - 1.0",
            )
        else:
            dimensionless_wellbore_radius = 0.1  # Default value when varying
            st.info("🔄 **Dimensionless Wellbore Radius (r_wD)** is set to vary - configured above in Parameter Variation section")

    with param_col2:
        st.markdown("**🌊 Boundary & Storage Parameters**")
        
        # Wellbore Storage Constant input
        if not (use_parameter_variation and selected_param_key == "wellbore_storage_constant"):
            wellbore_storage_constant = st.number_input(
                "Wellbore Storage Constant (CD) - Base Value",
                min_value=0.001,
                value=1.0,
                help="**Wellbore storage constant** - Base value when not varying this parameter. Dimensionless parameter representing horizontal wellbore storage effects on water breakthrough. Must be positive. Typical range: 0.01 - 1000",
            )
        else:
            wellbore_storage_constant = 1.0  # Default value when varying
            st.info("🔄 **Wellbore Storage Constant (CD)** is set to vary - configured above in Parameter Variation section")

        # Distance to Boundary input
        if not (use_parameter_variation and selected_param_key == "distance_to_boundary"):
            distance_to_boundary = st.number_input(
                "Distance to Water Boundary (d) - Base Value",
                min_value=0.001,
                value=500.0,
                help="**Distance to water boundary** - Base value when not varying this parameter. Distance from horizontal well to the constant pressure water boundary. Critical for breakthrough timing. Typical range: 100 - 10,000 ft",
            )
        else:
            distance_to_boundary = 500.0  # Default value when varying
            st.info("🔄 **Distance to Water Boundary (d)** is set to vary - configured above in Parameter Variation section")

        # Skin Factor input
        if not (use_parameter_variation and selected_param_key == "skin_factor"):
            skin_factor = st.number_input(
                "Skin Factor (S) - Base Value",
                value=0.0,
                help="**Skin factor** - Base value when not varying this parameter. Dimensionless parameter representing near-wellbore condition affecting water breakthrough. Positive = damage, Negative = stimulation. Typical range: -5.0 to +10.0",
            )
        else:
            skin_factor = 0.0  # Default value when varying
            st.info("🔄 **Skin Factor (S)** is set to vary - configured above in Parameter Variation section")

    # Time parameters section
    st.divider()
    st.subheader("⏱️ Breakthrough Time Analysis")
    st.markdown("*Configure the time range for water breakthrough analysis*")

    st.info(
        "💡 **Tip:** Use appropriate step size for time range. Smaller steps provide more detailed analysis but take longer to compute."
    )

    time_col1, time_col2, time_col3 = st.columns(3)

    with time_col1:
        min_time = st.number_input(
            "Min Time (tD)",
            min_value=0.0001,
            value=0.001,
            format="%.4f",
            help="**Minimum dimensionless time** - Beginning of breakthrough analysis period. Must be positive.",
        )

    with time_col2:
        max_time = st.number_input(
            "Max Time (tD)",
            min_value=0.0001,
            value=1.0,
            format="%.4f",
            help="**Maximum dimensionless time** - End of breakthrough analysis period. Must be greater than min time.",
        )

    with time_col3:
        time_step = st.number_input(
            "Time Step Size",
            min_value=0.0001,
            value=0.01,
            format="%.4f",
            help="**Time step size** - Step size for time increments. Smaller steps = more data points.",
        )

    # Validate inputs and compute results
    st.divider()
    
    # Add option for pressure gradient computation
    st.markdown("**🔧 Computation Options:**")
    compute_gradient = st.checkbox(
        "Include Pressure Gradient (∂pD/∂tD) Analysis",
        value=True,
        help="Check to include dimensionless pressure gradient computation and visualization. This provides additional insight into pressure sensitivity to time changes during water breakthrough."
    )

    # Add validation warnings
    if max_time <= min_time:
        st.warning("⚠️ Max time must be greater than min time")
    
    if use_parameter_variation and (param_max <= param_min):
        st.warning("⚠️ Max parameter value must be greater than min parameter value")

    if st.button(
        "🧮 Compute Water Breakthrough Analysis",
        type="primary",
        help="Click to compute dimensionless pressure values for water breakthrough analysis across the specified ranges",
    ):
        try:
            # Validate reservoir parameters
            reservoir_params = ReservoirParameters(
                alpha=alpha,
                exponential_integral=exponential_integral,
                dimensionless_length=dimensionless_length,
                dimensionless_wellbore_radius=dimensionless_wellbore_radius,
                wellbore_storage_constant=wellbore_storage_constant,
                distance_to_boundary=distance_to_boundary,
                skin_factor=skin_factor,
            )

            # Validate time parameters
            time_params = TimeParameters(
                min_time=min_time, 
                max_time=max_time, 
                step_size=time_step
            )

            # Create time array
            time_array = create_time_array(time_params=time_params)

            if use_parameter_variation:
                # Validate varying parameter
                varying_param = VaryingParameter(
                    parameter_name=param_options[selected_param],
                    min_value=param_min,
                    max_value=param_max,
                    step_size=param_step
                )

                # Create parameter array
                param_array = create_parameter_array(varying_param=varying_param)

                # Compute results with varying parameters
                with st.spinner("Computing water breakthrough analysis with parameter variation..."):
                    results_df = compute_all_functions_varying(
                        base_params=reservoir_params, 
                        time_array=time_array, 
                        varying_param=varying_param, 
                        param_array=param_array, 
                        include_gradient=compute_gradient
                    )

                success_message = f"✅ Successfully computed {len(results_df)} time points across {len(param_array)} parameter values!"
                
                # Store results in session state
                st.session_state['results_df'] = results_df
                st.session_state['computation_done'] = True
                st.session_state['include_gradient'] = compute_gradient
                st.session_state['varying_param'] = varying_param
                st.session_state['use_parameter_variation'] = True
            else:
                # Compute results with single parameter values
                with st.spinner("Computing water breakthrough analysis with single parameter values..."):
                    results_df = compute_all_functions_single(
                        params=reservoir_params, 
                        time_array=time_array, 
                        include_gradient=compute_gradient
                    )

                success_message = f"✅ Successfully computed {len(results_df)} data points for breakthrough analysis!"
                
                # Store results in session state
                st.session_state['results_df'] = results_df
                st.session_state['computation_done'] = True
                st.session_state['include_gradient'] = compute_gradient
                st.session_state['use_parameter_variation'] = False

            if not results_df.empty:
                st.success(success_message)

        except Exception as e:
            st.error(f"❌ Validation Error: {str(e)}")
            st.info(
                """
                **💡 Common Issues:**
                - Ensure all positive parameters are greater than 0.001
                - End time must be greater than start time
                - Check that exponential integral value is reasonable (-10 to 10)
                - Verify that dimensionless parameters are within typical ranges
                """
            )

    # Display results if computation was done
    if st.session_state.get('computation_done', False) and 'results_df' in st.session_state:
        results_df = st.session_state['results_df']
        include_gradient = st.session_state.get('include_gradient', True)
        use_parameter_variation = st.session_state.get('use_parameter_variation', False)
        varying_param = st.session_state.get('varying_param', None)
        
        # Display results as table
        st.divider()
        st.subheader("📋 Water Breakthrough Analysis Results")
        
        if use_parameter_variation and varying_param:
            # Parameter variation mode - show separate tables
            varying_param_name = varying_param.parameter_name.replace('_', ' ').title()
            st.markdown(f"*Computed values with varying {varying_param_name} - Separate tables for each pressure type*")
            
            # Create separate tables
            tables = create_separate_tables(df=results_df, varying_param=varying_param, include_gradient=include_gradient)
            
            # Display each table in a separate tab
            table_names = list(tables.keys())
            if len(table_names) > 1:
                tabs = st.tabs(table_names)
                for i, (table_name, table_df) in enumerate(tables.items()):
                    with tabs[i]:
                        st.markdown(f"**{table_name} Table**")
                        # Round values for better display
                        formatted_table = table_df.copy()
                        for col in formatted_table.columns:
                            if col != 'Dimensionless Time (tD)':
                                formatted_table[col] = formatted_table[col].round(6)
                        st.dataframe(formatted_table, width="stretch", height=400)
            else:
                # Single table
                table_name, table_df = list(tables.items())[0]
                st.markdown(f"**{table_name} Table**")
                formatted_table = table_df.copy()
                for col in formatted_table.columns:
                    if col != 'Dimensionless Time (tD)':
                        formatted_table[col] = formatted_table[col].round(6)
                st.dataframe(formatted_table, width="stretch", height=400)
            
            # Combined download button for all tables
            all_tables_csv = ""
            for table_name, table_df in tables.items():
                all_tables_csv += f"\n{table_name}\n"
                all_tables_csv += table_df.to_csv(index=False)
                all_tables_csv += "\n"
            
            st.download_button(
                label="📥 Download All Breakthrough Results as CSV",
                data=all_tables_csv,
                file_name=f"water_breakthrough_analysis_varying_{varying_param.parameter_name}.csv",
                mime="text/csv",
                help="Download all pressure tables as a combined CSV file for further analysis",
            )
        else:
            # Single parameter mode - show single table
            pressure_types = ["Dimensionless Pressure (pD)", "Pressure Derivative (p'D)"]
            if include_gradient:
                pressure_types.append("Pressure Gradient (∂pD/∂tD)")
            
            st.markdown(f"*Computed values for {', '.join(pressure_types)} using single parameter values*")

            # Format the dataframe for better display
            formatted_df = results_df.copy()
            # Round all numeric columns except time
            for col in formatted_df.columns:
                if col != 'Dimensionless Time (tD)':
                    formatted_df[col] = formatted_df[col].round(6)

            st.dataframe(formatted_df, width="stretch", height=400)

            # Download button for results
            csv = formatted_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Water Breakthrough Results as CSV",
                data=csv,
                file_name="water_breakthrough_analysis_results.csv",
                mime="text/csv",
                help="Download the complete water breakthrough analysis table as a CSV file for further analysis",
            )

        # Plotting section
        st.divider()
        st.subheader("📊 Water Breakthrough Visualization")
        st.markdown("*Interactive plots for breakthrough timing analysis and interpretation*")
        
        # Interactive plot options
        st.markdown("**🎛️ Plot Customization:**")
        customize_col1, customize_col2 = st.columns(2)
        
        with customize_col1:
            show_markers = st.checkbox("Show data point markers", value=True, 
                                     help="Toggle visibility of data point markers on lines")
            show_grid = st.checkbox("Show grid lines", value=True,
                                  help="Toggle grid lines for easier reading")
        
        with customize_col2:
            line_width = st.slider("Line width", min_value=1.0, max_value=4.0, value=2.5, step=0.5,
                                 help="Adjust the thickness of plot lines")
            plot_dpi = st.selectbox("Image quality (DPI)", [150, 200, 300, 400], index=2,
                                  help="Higher DPI = better quality but larger file size")
        
        # Plotting options
        plot_col1, plot_col2 = st.columns(2)
        
        with plot_col1:
            st.markdown("**🎯 Breakthrough Plot Type:**")
            plot_option = st.selectbox(
                "Choose visualization type:",
                ["Individual Plots", "Combined (Separate Y-Axes)", "Combined (Normalized)"],
                help="Select how you want to visualize the water breakthrough data"
            )
        
        with plot_col2:
            st.markdown("**💾 Download Options:**")
            if st.button("🖼️ Generate & Download Plots", help="Create high-quality plots for download"):
                with st.spinner("Generating plots..."):
                    if plot_option == "Individual Plots":
                        if use_parameter_variation and varying_param:
                            fig, _ = create_individual_plots_varying(
                                df=results_df, 
                                varying_param=varying_param, 
                                show_markers=show_markers, 
                                show_grid=show_grid, 
                                line_width=line_width, 
                                include_gradient=include_gradient
                            )
                            img_buffer = save_plot_as_image(fig=fig, filename="individual_plots_varying.png", dpi=plot_dpi)
                            st.download_button(
                                label="📥 Download Individual Plots",
                                data=img_buffer.getvalue(),
                                file_name=f"water_breakthrough_individual_plots_varying_{varying_param.parameter_name}.png",
                                mime="image/png"
                            )
                        else:
                            fig, _ = create_individual_plots_single(
                                df=results_df, 
                                show_markers=show_markers, 
                                show_grid=show_grid, 
                                line_width=line_width, 
                                include_gradient=include_gradient
                            )
                            img_buffer = save_plot_as_image(fig=fig, filename="individual_plots_single.png", dpi=plot_dpi)
                            st.download_button(
                                label="📥 Download Individual Plots",
                                data=img_buffer.getvalue(),
                                file_name="water_breakthrough_individual_plots.png",
                                mime="image/png"
                            )
                    elif plot_option == "Combined (Separate Y-Axes)":
                        if not use_parameter_variation:
                            fig = create_combined_plot(
                                df=results_df, 
                                plot_type="separate_y", 
                                show_markers=show_markers, 
                                show_grid=show_grid, 
                                line_width=line_width, 
                                include_gradient=include_gradient
                            )
                            img_buffer = save_plot_as_image(fig=fig, filename="combined_separate_y.png", dpi=plot_dpi)
                            st.download_button(
                                label="📥 Download Combined Plot",
                                data=img_buffer.getvalue(),
                                file_name="water_breakthrough_combined_plot.png",
                                mime="image/png"
                            )
                        else:
                            st.info("Combined plots are only available for single parameter mode.")
                    else:  # Normalized
                        if not use_parameter_variation:
                            fig = create_combined_plot(
                                df=results_df, 
                                plot_type="normalized", 
                                show_markers=show_markers, 
                                show_grid=show_grid, 
                                line_width=line_width, 
                                include_gradient=include_gradient
                            )
                            img_buffer = save_plot_as_image(fig=fig, filename="combined_normalized.png", dpi=plot_dpi)
                            st.download_button(
                                label="📥 Download Normalized Plot",
                                data=img_buffer.getvalue(),
                                file_name="water_breakthrough_normalized_plot.png",
                                mime="image/png"
                            )
                        else:
                            st.info("Combined plots are only available for single parameter mode.")
                    plt.close(fig)  # Free memory
        
        # Display the selected plot
        try:
            # Validate data before plotting
            if results_df['Dimensionless Time (tD)'].min() <= 0:
                st.warning("⚠️ Warning: Some time values are zero or negative. This may cause plotting issues.")
                # Filter out problematic values
                results_df = results_df[results_df['Dimensionless Time (tD)'] > 0]
            
            if plot_option == "Individual Plots":
                param_count = "3 parameters" if include_gradient else "2 parameters"
                
                if use_parameter_variation and varying_param:
                    varying_param_name = varying_param.parameter_name.replace('_', ' ').title()
                    st.markdown(f"**📈 Individual Water Breakthrough Parameter Plots (Semilog - {param_count}):**")
                    st.info(f"Each parameter plotted on semilog scale with multiple lines for varying {varying_param_name}. Logarithmic time axis for detailed analysis of water breakthrough behavior across time decades.")
                    fig, _ = create_individual_plots_varying(
                        df=results_df, 
                        varying_param=varying_param, 
                        show_markers=show_markers, 
                        show_grid=show_grid, 
                        line_width=line_width, 
                        include_gradient=include_gradient
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.markdown(f"**📈 Individual Water Breakthrough Parameter Plots (Semilog - {param_count}):**")
                    st.info(f"Each parameter plotted on semilog scale with tD axis logarithmic for detailed analysis of water breakthrough behavior across time decades. Currently showing {param_count}.")
                    fig, _ = create_individual_plots_single(
                        df=results_df, 
                        show_markers=show_markers, 
                        show_grid=show_grid, 
                        line_width=line_width, 
                        include_gradient=include_gradient
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                    
            elif plot_option == "Combined (Separate Y-Axes)":
                if use_parameter_variation:
                    st.info("Combined plots are only available for single parameter mode. Please use Individual Plots for parameter variation analysis.")
                else:
                    param_count = "3 parameters" if include_gradient else "2 parameters"
                    st.markdown(f"**📊 Combined Water Breakthrough Semilog Plot with Separate Y-Axes ({param_count}):**")
                    st.info(f"All {param_count} on one semilog plot with separate y-axes to analyze water breakthrough patterns across different scales with logarithmic time axis.")
                    fig = create_combined_plot(
                        df=results_df, 
                        plot_type="separate_y", 
                        show_markers=show_markers, 
                        show_grid=show_grid, 
                        line_width=line_width, 
                        include_gradient=include_gradient
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                
            else:  # Normalized
                if use_parameter_variation:
                    st.info("Combined plots are only available for single parameter mode. Please use Individual Plots for parameter variation analysis.")
                else:
                    param_count = "3 parameters" if include_gradient else "2 parameters"
                    st.markdown(f"**🔄 Combined Normalized Water Breakthrough Semilog Plot ({param_count}):**")
                    st.info(f"All {param_count} normalized to 0-1 range on semilog scale for direct comparison of water breakthrough trends with logarithmic time axis.")
                    fig = create_combined_plot(
                        df=results_df, 
                        plot_type="normalized", 
                        show_markers=show_markers, 
                        show_grid=show_grid, 
                        line_width=line_width, 
                        include_gradient=include_gradient
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                
        except Exception as plot_error:
            st.error(f"Error generating plot: {str(plot_error)}")
            st.info("Please try a different plot type or check your data.")

        # Plot interpretation guide
        with st.expander("📖 Water Breakthrough Plot Interpretation Guide", expanded=False):
            st.markdown("""
            ### How to Interpret the Water Breakthrough Semilog Plots:
            
            **📊 Semilog Plot Features:**
            - **X-axis (tD):** Logarithmic scale for time - allows viewing behavior across multiple time decades
            - **Y-axis:** Linear scale for pressure values - shows actual magnitudes
            - **Log Scale Benefits:** Better visualization of early-time and late-time behavior simultaneously
            
            **📈 Individual Semilog Plots:**
            - **Dimensionless Pressure (pD):** Shows pressure buildup as water approaches the horizontal well
            - **Pressure Derivative (p'D):** Indicates rate of pressure change - critical for identifying breakthrough timing
            - **Pressure Gradient (∂pD/∂tD):** Shows pressure sensitivity to time - helps predict breakthrough acceleration
            
            **📊 Combined Semilog Plots:**
            - **Separate Y-Axes:** Compare actual values with different scales to understand breakthrough mechanisms
            - **Normalized:** Compare trends and patterns directly to identify breakthrough timing relationships
            
            **🌊 Water Breakthrough Key Observations on Semilog Scale:**
            - **Early Time (Low tD, left side):** Wellbore storage effects dominate, no water breakthrough yet
            - **Intermediate Time (middle):** Pressure response indicates water movement toward the well
            - **Late Time (High tD, right side):** Water boundary effects dominate, breakthrough occurs
            
            **📋 Semilog Analysis Tips:**
            - Look for straight-line segments indicating specific flow regimes
            - Slope changes on semilog plots reveal transitions between flow periods
            - Early-time behavior visible on left, late-time on right of log scale
            - Use log scale to observe breakthrough behavior across multiple time decades
            - Monitor pressure derivative inflection points to identify characteristic breakthrough patterns
            """)
        
        # Summary statistics
        st.divider()
        st.subheader("📈 Water Breakthrough Analysis Summary")
        
        if use_parameter_variation and varying_param:
            st.markdown("*Key metrics from the computed breakthrough analysis across parameter variations*")
            
            # Get parameter values for summary
            param_values = []
            for col in results_df.columns:
                if col.startswith("pD ("):
                    param_val = float(col.split("=")[1].split(")")[0])
                    param_values.append(param_val)
            
            # Create summary for each parameter value
            if len(param_values) > 0:
                varying_param_name = varying_param.parameter_name.replace('_', ' ').title()
                st.markdown(f"**Summary across {len(param_values)} {varying_param_name} values:**")
                
                # Create columns for each parameter value (limit to 4 for display)
                display_params = param_values[:4] if len(param_values) > 4 else param_values
                summary_cols = st.columns(len(display_params))
                
                for i, param_val in enumerate(display_params):
                    with summary_cols[i]:
                        st.markdown(f"**{varying_param_name} = {param_val:.3f}**")
                        
                        # Max pD for this parameter value
                        pd_col = f"pD ({varying_param.parameter_name}={param_val:.3f})"
                        if pd_col in results_df.columns:
                            max_pd = results_df[pd_col].max()
                            max_pd_time = results_df.loc[results_df[pd_col].idxmax(), 'Dimensionless Time (tD)']
                            st.metric(
                                "Max pD",
                                f"{max_pd:.4f}",
                                f"at tD = {max_pd_time:.4f}",
                            )
                        
                        # Max p'D for this parameter value
                        pd_deriv_col = f"p'D ({varying_param.parameter_name}={param_val:.3f})"
                        if pd_deriv_col in results_df.columns:
                            max_pd_deriv = results_df[pd_deriv_col].max()
                            max_pd_deriv_time = results_df.loc[results_df[pd_deriv_col].idxmax(), 'Dimensionless Time (tD)']
                            st.metric(
                                "Max p'D",
                                f"{max_pd_deriv:.4f}",
                                f"at tD = {max_pd_deriv_time:.4f}",
                            )
                
                if len(param_values) > 4:
                    st.info(f"Showing summary for first 4 parameter values. Total computed: {len(param_values)} values.")
        else:
            st.markdown("*Key metrics from the computed breakthrough analysis*")
            
            # Dynamic column layout based on available data
            if 'Pressure Gradient (∂pD/∂tD)' in results_df.columns:
                summary_col1, summary_col2, summary_col3 = st.columns(3)
            else:
                summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.metric(
                    "Max pD",
                    f"{results_df['Dimensionless Pressure (pD)'].max():.4f}",
                    f"at tD = {results_df.loc[results_df['Dimensionless Pressure (pD)'].idxmax(), 'Dimensionless Time (tD)']:.4f}",
                )

            with summary_col2:
                derivative_col = "Pressure Derivative (p'D)"
                max_derivative = results_df[derivative_col].max()
                max_derivative_time = results_df.loc[
                    results_df[derivative_col].idxmax(), "Dimensionless Time (tD)"
                ]
                st.metric(
                    "Max p'D",
                    f"{max_derivative:.4f}",
                    f"at tD = {max_derivative_time:.4f}",
                )

            # Only show gradient metric if it was computed
            if 'Pressure Gradient (∂pD/∂tD)' in results_df.columns:
                with summary_col3:
                    gradient_col = "Pressure Gradient (∂pD/∂tD)"
                    max_gradient = results_df[gradient_col].max()
                    max_gradient_time = results_df.loc[
                        results_df[gradient_col].idxmax(), "Dimensionless Time (tD)"
                    ]
                    st.metric(
                        "Max ∂pD/∂tD",
                        f"{max_gradient:.4f}",
                        f"at tD = {max_gradient_time:.4f}",
                    )
        
        # Add a button to clear results
        if st.button("🔄 Clear Results & Start New Analysis", help="Clear current results to start a new analysis"):
            st.session_state['computation_done'] = False
            for key in ['results_df', 'varying_param']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    else:
        st.info("👆 Enter parameters above and click 'Compute Water Breakthrough Analysis' to see results and visualizations.")

    # Information section
    st.divider()
    with st.expander("ℹ️ Water Breakthrough Analysis Information & Parameter Guide", expanded=False):
        tab1, tab2, tab3 = st.tabs(
            ["📐 Mathematical Models", "📋 Parameter Guide", "🌊 Water Breakthrough Interpretation"]
        )

        with tab1:
            st.markdown("""
            ### Mathematical Models for Water Breakthrough Analysis:
            
            **1. Dimensionless Pressure:**
            ```
            pD = (-α / (4 * LD)) * Ei(-r_wD² / (4 * tD / CD)) + S + (α / (4 * LD)) * Ei(-d² / tD)
            ```
            
            **2. Pressure Derivative:**
            ```
            p'D = (α / (4 * LD)) * exp(-r_wD² / (4 * tD / CD)) - (α / (4 * LD)) * exp(-d² / tD)
            ```
            
            **3. Pressure Gradient:**
            ```
            ∂pD/∂tD = (α / (4 * LD)) * [exp(-r_wD² / (4 * tD / CD)) + exp(-d² / tD)]
            ```
            
            These equations model the pressure behavior of a horizontal well near a constant pressure water boundary.
            """)

        with tab2:
            st.markdown("""
            ### Parameter Descriptions & Typical Ranges for Water Breakthrough:
            
            | Parameter | Symbol | Description | Typical Range | Units |
            |-----------|--------|-------------|---------------|-------|
            | **Alpha** | α | System parameter for water breakthrough | 0.1 - 10.0 | - |
            | **Dimensionless Length** | LD | Horizontal well length parameter | 100 - 10,000 | - |
            | **Wellbore Radius** | r_wD | Dimensionless horizontal well radius | 0.01 - 1.0 | - |
            | **Storage Constant** | CD | Horizontal wellbore storage effects | 0.01 - 1000 | - |
            | **Distance to Water Boundary** | d | Critical distance affecting breakthrough timing | 100 - 10,000 | ft |
            | **Skin Factor** | S | Near-wellbore condition | -5.0 to +10.0 | - |
            | **Exponential Integral** | Ei | Mathematical function | -10.0 to +10.0 | - |
            | **Dimensionless Time** | tD | Time parameter for breakthrough analysis | 0.01 - 10,000 | - |
            """)

        with tab3:
            st.markdown("""
            ### Water Breakthrough Analysis Interpretation:
            
            **💧 Dimensionless Pressure (pD):**
            - Represents pressure change as water approaches the horizontal well
            - Higher values indicate stronger water drive and earlier breakthrough
            - Critical for determining breakthrough timing and well performance
            
            **📈 Pressure Derivative (p'D):**
            - Rate of pressure change - key indicator for breakthrough prediction
            - Inflection points signal onset of water breakthrough
            - Used to identify flow regime transitions affecting breakthrough
            
            **⚡ Pressure Gradient (∂pD/∂tD):**
            - Sensitivity of pressure to time changes during water breakthrough
            - Indicates how quickly water breakthrough will occur
            - Higher gradients suggest faster breakthrough progression
            
            **🌊 Water Breakthrough Flow Regimes:**
            - **Early time:** Wellbore storage dominates, minimal water movement
            - **Intermediate time:** Pressure response indicates water movement toward well
            - **Late time:** Water boundary effects dominate, breakthrough occurs
            
            **📊 Factors Affecting Breakthrough Time:**
            - **Distance to water boundary:** Closer boundaries → faster breakthrough
            - **Horizontal well length:** Longer wells → different breakthrough patterns
            - **Wellbore storage:** Higher storage → delayed breakthrough response
            - **Skin factor:** Damage → delayed breakthrough, stimulation → faster breakthrough
            """)
        
if __name__ == "__main__":
    main()
