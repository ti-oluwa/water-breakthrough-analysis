import math
import typing
from scipy.special import expi


def exponential_integral(x: float) -> float:
    """
    Adaptive evaluation of Ei(-x) (or equivalently -E1(x)) used in
    reservoir engineering dimensionless pressure calculations.

    For small x, the logarithmic approximation is used:
        Ei(-x) ≈ -γ - ln(x), valid when x << 1
    Otherwise, the exact computation via scipy.special.expi is used.

    :param x: Argument to the exponential integral function (must be positive).
    :return: Value of Ei(-x)
    :raises ValueError: If x is not positive.
    """
    if x <= 0:
        raise ValueError("x must be positive.")

    # Logarithmic approximation region (log approximately holds for x < 0.01 with error < 0.01)
    if x <= 0.01:  # threshold can be tuned depending on desired accuracy
        return math.log(1.781 * x)

    # Use scipy's expi for stable and accurate evaluation
    return expi(-x)


Ei = exponential_integral  # Alias for convenience


def compute_dimensionless_pressure(
    alpha: float,
    dimensionless_length: float,
    dimensionless_wellbore_radius: float,
    dimensionless_time: float,
    wellbore_storage_constant: float,
    distance_to_boundary: float,
    exponential_integral_func: typing.Callable[[float], float] = Ei,
    skin_factor: float = 0.0,
) -> float:
    """
    Computes the dimensionless pressure in a reservoir using the provided parameters.

    The formula used is:

        pD = (-α / (4 * LD)) * Ei(-r_wD^2 / (4 * tD / CD)) + S + (α / (4 * LD)) * Ei(-d^2 / tD)

        where:
        - pD: Dimensionless pressure
        - α: Alpha parameter
        - LD: Dimensionless length
        - Ei(-x): Exponential integral
        - x: Argument of the exponential integral
            x = r_wD^2 / (4 * tD / CD) for the first term
            x = d^2 / tD for the third term
        - r_wD: Dimensionless wellbore radius
        - tD: Dimensionless time
        - CD: Wellbore storage constant
        - S: Skin factor
        - d: Distance to constant boundary

    :param alpha: Alpha parameter.
    :param dimensionless_length: Dimensionless length parameter.
    :param dimensionless_wellbore_radius: Dimensionless wellbore radius.
    :param dimensionless_time: Dimensionless time parameter.
    :param wellbore_storage_constant: Wellbore storage constant.
    :param distance_to_boundary: Distance to constant boundary.
    :param exponential_integral_func: Function to compute the exponential integral (default is `exponential_integral`).
    :param skin_factor: Skin factor (default is 0.0).
    """
    if dimensionless_time <= 0:
        raise ValueError("Dimensionless time must be greater than zero.")

    alpha_term = alpha / (4 * dimensionless_length)
    x_first_term = (dimensionless_wellbore_radius**2) / (
        4 * dimensionless_time / wellbore_storage_constant
    )
    first_term = -alpha_term * exponential_integral_func(x_first_term)

    x_third_term = distance_to_boundary**2 / dimensionless_time
    third_term = alpha_term * exponential_integral_func(x_third_term)
    return first_term + skin_factor + third_term


def compute_dimension_pressure_derivative(
    alpha: float,
    dimensionless_length: float,
    dimensionless_wellbore_radius: float,
    dimensionless_time: float,
    wellbore_storage_constant: float,
    distance_to_boundary: float,
) -> float:
    """
    Computes the derivative of the dimensionless pressure in a reservoir.

    The formula used is:

        p'D = (α / (4 * LD)) * exp(-r_wD^2 / (4 * tD / CD)) - (α / (4 * LD)) * exp(-d^2 / tD)

        where:
        - p'D: Dimensionless pressure derivative
        - α: Alpha parameter
        - LD: Dimensionless length
        - r_wD: Dimensionless wellbore radius
        - tD: Dimensionless time
        - CD: Wellbore storage constant
        - d: Distance to boundary

    :param alpha: Alpha parameter.
    :param dimensionless_length: Dimensionless length parameter.
    :param dimensionless_wellbore_radius: Dimensionless wellbore radius.
    :param dimensionless_time: Dimensionless time parameter.
    :param wellbore_storage_constant: Wellbore storage constant.
    :param distance_to_boundary: Distance to boundary.
    """
    if dimensionless_time <= 0:
        raise ValueError("Dimensionless time must be greater than zero.")

    alpha_term = alpha / (4 * dimensionless_length)
    first_term = alpha_term * math.exp(
        -(dimensionless_wellbore_radius**2)
        / (4 * dimensionless_time / wellbore_storage_constant)
    )
    second_term = -alpha_term * math.exp(
        -(distance_to_boundary**2) / dimensionless_time
    )
    return first_term + second_term


def compute_dimensionless_pressure_gradient(
    alpha: float,
    dimensionless_length: float,
    dimensionless_wellbore_radius: float,
    dimensionless_time: float,
    wellbore_storage_constant: float,
    distance_to_boundary: float,
) -> float:
    """
    Computes the gradient of the dimensionless pressure in a reservoir.

    The formula used is:

        ∂pD/∂tD = (α / (4 * LD)) * [exp(-r_wD^2 / (4 * tD / CD)) + exp(-d^2 / tD)]

        where:
        - ∂pD/∂tD: Dimensionless pressure gradient with respect to time
        - α: Alpha parameter
        - LD: Dimensionless length
        - r_wD: Dimensionless wellbore radius
        - tD: Dimensionless time
        - CD: Wellbore storage constant
        - d: Distance to boundary

    :param alpha: Alpha parameter.
    :param dimensionless_length: Dimensionless length parameter.
    :param dimensionless_wellbore_radius: Dimensionless wellbore radius.
    :param dimensionless_time: Dimensionless time parameter.
    :param wellbore_storage_constant: Wellbore storage constant.
    :param distance_to_boundary: Distance to boundary.
    """
    if dimensionless_time <= 0:
        raise ValueError("Dimensionless time must be greater than zero.")

    alpha_term = alpha / (4 * dimensionless_length)
    first_term = math.exp(
        -(dimensionless_wellbore_radius**2)
        / (4 * dimensionless_time / wellbore_storage_constant)
    )
    second_term = math.exp(-(distance_to_boundary**2) / dimensionless_time)
    return alpha_term * (first_term + second_term)
