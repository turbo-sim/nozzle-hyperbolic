"""Host-side fitting and plotting of tabulated nozzle wall coordinates."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def fit_nozzle_shape(file_path, sheet_name, degree):
    """Fit polynomial and cubic-spline wall profiles from an Excel worksheet.

    Parameters
    ----------
    file_path : str or path-like
        Workbook containing numeric ``x`` and ``y`` wall-coordinate columns.
    sheet_name : str or int
        Worksheet name or index passed to ``pandas.read_excel``.
    degree : int
        Degree of the least-squares polynomial fit.

    Returns
    -------
    coefficients : numpy.ndarray
        Polynomial coefficients ordered from highest to lowest power.
    polynomial : numpy.poly1d
        Callable polynomial fit.
    spline : scipy.interpolate.CubicSpline
        Interpolating cubic spline through the same coordinate data.

    Raises
    ------
    FileNotFoundError
        If the workbook does not exist.
    ValueError
        If the sheet cannot be read or the coordinates cannot define a spline.

    Notes
    -----
    This host-side preprocessing helper requires pandas and an Excel engine.
    Coordinates must use consistent units and strictly increasing x values.
    """
    # Read the Excel file
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError as e:
        raise ValueError(f"Error reading the sheet: {sheet_name}. {e}")

    # Extract data points
    x = data['x'].values  # Ensure the column names match your Excel file
    y = data['y'].values  # Ensure the column names match your Excel file

    # Fit the data to a high-degree polynomial
    coefficients = np.polyfit(x, y, degree)

    # Create the polynomial function
    polynomial = np.poly1d(coefficients)

    spline = CubicSpline(x, y)


    return coefficients, polynomial, spline

def plot_nozzle_shape(x, y, polynomial, degree):
    """Plot tabulated wall coordinates and their polynomial fit.

    Parameters
    ----------
    x, y : array_like
        Original wall coordinates in consistent physical units.
    polynomial : callable
        Fitted wall-coordinate function evaluated on a uniform plotting grid.
    degree : int
        Polynomial degree displayed in the legend.

    Notes
    -----
    Draws on the current Matplotlib axes and calls ``matplotlib.pyplot.show``.
    """
    plt.scatter(x, y, color='red', label='Data Points')
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = polynomial(x_fit)
    plt.plot(x_fit, y_fit, color='blue', label=f'Fitted Polynomial Degree {degree}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def nozzle_shape(x, polynomial):
    """Evaluate a fitted wall-coordinate polynomial.

    Parameters
    ----------
    x : array_like or float
        Coordinates at which to evaluate the wall profile.
    polynomial : callable
        Fitted polynomial accepting the supplied coordinates.

    Returns
    -------
    array_like or float
        Wall coordinates returned by the polynomial, in the fit's original units.
    """
    return polynomial(x)

def define_radius_function(lengths, radii):
    """Build a scalar radius profile from three straight segments.

    Parameters
    ----------
    lengths : sequence of float
        Four strictly increasing axial coordinates defining the segment endpoints.
    radii : sequence of float
        Four corresponding wall radii in consistent physical units.

    Returns
    -------
    radius_function : callable
        Function accepting a scalar coordinate in the closed interval from the
        first to the last axial coordinate. Evaluation outside it raises ValueError.

    Notes
    -----
    This helper is host-side and accepts scalar evaluation points. It is not a
    JAX geometry callback and does not validate the supplied endpoint sequences.
    """
    # Calculate slopes and intercepts for each segment
    slopes = [(radii[i+1] - radii[i]) / (lengths[i+1] - lengths[i]) for i in range(3)]
    intercepts = [radii[i] - slopes[i] * lengths[i] for i in range(3)]

    # Define the piecewise radius function
    def radius_function(x):
        if lengths[0] <= x < lengths[1]:
            return slopes[0] * x + intercepts[0]
        elif lengths[1] <= x < lengths[2]:
            return slopes[1] * x + intercepts[1]
        elif lengths[2] <= x <= lengths[3]:
            return slopes[2] * x + intercepts[2]
        else:
            raise ValueError("x is outside the defined length range.")

    return radius_function

# To test the new interpolation functions
if __name__ == "__main__":

    file_path = 'Spinelli_nozzles.xlsx'
    sheet_name = 'nozzle Ma1.5'  # Adjust as necessary
    degree = 8

    # Fit the nozzle shape
    coefficients, polynomial, spline = fit_nozzle_shape(file_path, sheet_name, degree)

    # Extract data for plotting
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    x = np.linspace(0, 15, 100)
    x_spline = np.linspace(25, 35, 800)
    h_spline = lambda x: spline(x)
    h = polynomial(x)

    m1 = -(5-0.12)/27.35
    m2 = (0.195-0.12)/56.15

    # # Piecewise nozzle shape
    # def nozzle_shape(x):
    #     if x <= 25:
    #         return 5 + m1*x
    #     elif x > 35:
    #         return 0.12+m2*(x-27.35)
    #     else:
    #         return h_spline(x)

    # h = np.array([nozzle_shape(x_elem) for x_elem in x])


    # lenghts = [0, 1, 2, 3]
    # radius = [0.5, 0.2, 0.2, 0.3]
    # radius_function = define_radius_function(lenghts, radius)
    # y = np.zeros_like(x)

    # for j in range(len(y)):
    #     y[j] = radius_function(x[j])

    # ax1 = plt.gca()
    # ax1.set_ylim([0, 0.75])

    plt.plot(x,h)
    plt.plot(data.x, data.y, "o", markersize=2)
    plt.show()
