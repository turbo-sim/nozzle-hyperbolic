# # Fit the nozzle shape from Spinelli´s paper and save the coefficients
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Read the Excel file
# file_path = 'nozzles points.xlsx'  # Ensure this file is in the same folder as the script
# data = pd.read_excel(file_path, sheet_name='nozzle Ma1.5')  # Replace with the actual sheet name if different

# # Extract data points
# x = data['x0'].values  # Replace 'x' with the actual name of the x-column in your Excel file
# y = data['y0'].values  # Replace 'y' with the actual name of the y-column in your Excel file

# # Fit the data to a high-degree polynomial
# degree = 12  # Use the desired degree
# coefficients = np.polyfit(x, y, degree)

# # Create the polynomial function
# polynomial = np.poly1d(coefficients)

# # Plot the original points and the fitted polynomial
# plt.scatter(x, y, color='red', label='Data Points')
# x_fit = np.linspace(min(x), max(x), 100)
# y_fit = polynomial(x_fit)
# plt.plot(x_fit, y_fit, color='blue', label=f'Fitted Polynomial Degree {degree}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()


# def nozzle_shape(x):
#     return polynomial(x)


# common/nozzle_shape.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def fit_nozzle_shape(file_path, sheet_name, degree):
    """
    Fit the nozzle shape from an Excel file and return the polynomial coefficients.
    
    Parameters:
    - file_path: str, path to the Excel file containing the nozzle points.
    - sheet_name: str, name of the sheet to read the data from.
    - degree: int, degree of the polynomial to fit.
    
    Returns:
    - coefficients: np.ndarray, coefficients of the fitted polynomial.
    - polynomial: np.poly1d, polynomial function for the nozzle shape.
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
    """
    Plot the original points and the fitted polynomial.
    
    Parameters:
    - x: np.ndarray, x-coordinates of the original data points.
    - y: np.ndarray, y-coordinates of the original data points.
    - polynomial: np.poly1d, polynomial function for the nozzle shape.
    - degree: int, degree of the fitted polynomial.
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
    """
    Calculate the nozzle shape for given x values using the fitted polynomial.
    
    Parameters:
    - x: np.ndarray or float, x-coordinate(s) for which to compute the nozzle shape.
    - polynomial: np.poly1d, polynomial function for the nozzle shape.
    
    Returns:
    - y: np.ndarray or float, calculated y-coordinate(s) for the nozzle shape.
    """
    return polynomial(x)

def define_radius_function(lengths, radii):
    """
    Creates a function that returns the radius at a given length x
    based on three straight-line segments defined by lengths and radii.

    Parameters:
    - lengths: List of length values (e.g., [L1, L2, L3, L4])
    - radii: List of corresponding radius values (e.g., [R1, R2, R3, R4])

    Returns:
    - radius_function: A function that takes x and returns the radius at x
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
