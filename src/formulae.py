import math


def compute_dimensionless_pressure(
    alpha: float,
    exponential_integral: float,
    dimensionless_length: float,
    dimensionless_wellbore_radius: float,
    dimensionless_time: float,
    wellbore_storage_constant: float,
    distance_to_boundary: float,
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
        - Ei: Exponential integral
        - r_wD: Dimensionless wellbore radius
        - tD: Dimensionless time
        - CD: Wellbore storage constant
        - S: Skin factor
        - d: Distance to constant boundary

    :param alpha: Alpha parameter.
    :param exponential_integral: Exponential integral value.
    :param dimensionless_length: Dimensionless length parameter.
    :param dimensionless_wellbore_radius: Dimensionless wellbore radius.
    :param dimensionless_time: Dimensionless time parameter.
    :param wellbore_storage_constant: Wellbore storage constant.
    :param distance_to_boundary: Distance to constant boundary.
    :param skin_factor: Skin factor (default is 0.0).
    """
    if dimensionless_time <= 0:
        raise ValueError("Dimensionless time must be greater than zero.")

    alpha_term = alpha / (4 * dimensionless_length)
    first_term = (
        -alpha_term
        * exponential_integral
        * (
            (-(dimensionless_wellbore_radius**2))
            / (4 * dimensionless_time / wellbore_storage_constant)
        )
    )
    third_term = (
        alpha_term
        * exponential_integral
        * (-(distance_to_boundary**2) / dimensionless_time)
    )
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
