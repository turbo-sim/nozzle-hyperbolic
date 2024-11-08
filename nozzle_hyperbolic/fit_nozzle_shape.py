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

    return coefficients, polynomial

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

if __name__ == "__main__":
    # This block will only run when this script is executed directly
    file_path = 'nozzles points.xlsx'
    sheet_name = 'nozzle Ma1.5'  # Adjust as necessary
    degree = 12

    # Fit the nozzle shape
    coefficients, polynomial = fit_nozzle_shape(file_path, sheet_name, degree)

    # Extract data for plotting
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    x = data['x'].values
    y = data['y'].values

    # Plot the nozzle shape
    plot_nozzle_shape(x, y, polynomial, degree)
