import os
import re
import time
import yaml
import numbers

import numpy as np

from numbers import Number
from collections.abc import Iterable


def is_float(element: any) -> bool:
    """
    Check if the given element can be converted to a float.

    Parameters
    ----------
    element : any
        The element to be checked.

    Returns
    -------
    bool
        True if the element can be converted to a float, False otherwise.
    """
    
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False    

def is_numeric(value):
    """
    Check if a value is a numeric type, including both Python and NumPy numeric types.

    This function checks if the given value is a numeric type (int, float, complex) 
    in the Python standard library or NumPy, while explicitly excluding boolean types.

    Parameters
    ----------
    value : any type
        The value to be checked for being a numeric type.

    Returns
    -------
    bool
        Returns True if the value is a numeric type (excluding booleans), 
        otherwise False.
    """
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        return True
    if isinstance(value, (np.int_, np.float_, np.complex_)):
        return True
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.number) and not np.issubdtype(value.dtype, np.bool_)
    return False

def wait_for_file(file_path, timeout=None, poll_interval=0.1):
    """
    Wait until the specified file is created.

    This function is used to wait until a Fluent transcript file is created.

    Parameters
    ----------
    file_path : str
        Path to the file to wait for.
    timeout : float, optional
        Maximum time to wait in seconds. If None, waits indefinitely.
    poll_interval : int, optional
        Time interval between checks in seconds.

    Returns
    -------
    bool
        True if the file was found, False otherwise (only if timeout is set).
    """
    start_time = time.time()
    while True:
        if os.path.exists(file_path):
            return True
        if timeout is not None and time.time() - start_time > timeout:
            raise FileNotFoundError(f"Timeout waiting for file: {file_path}")
        time.sleep(poll_interval)


def savefig_in_formats(fig, path_without_extension, formats=['.png', '.svg', ".eps"]):
    """
    Save a given matplotlib figure in multiple file formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to be saved.
    path_without_extension : str
        The full path to save the figure excluding the file extension.
    formats : list of str, optional
        A list of string file extensions to specify which formats the figure should be saved in. 
        Default is ['.png', '.svg', '.pdf'].

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 1], [0, 1])
    >>> save_fig_in_formats(fig, "/path/to/figure/filename")

    This will save the figure as "filename.png", "filename.svg", and "filename.pdf" in the "/path/to/figure/" directory.
    """
    for ext in formats:
        fig.savefig(f"{path_without_extension}{ext}")
        # fig.savefig(f"{path_without_extension}{ext}", bbox_inches="tight")


def compare_contents_or_files(file_or_content_1, file_or_content_2):
    """
    Compare the content of two inputs, which can be either file paths or strings.

    This function accepts two arguments. Each argument can be:
    1. A file path pointing to a file containing text content.
    2. A string containing text content directly.

    If the argument is a file path that exists, the function reads its content.
    If the argument is a string, it's directly used for comparison.

    Parameters
    ----------
    file_or_content1 : str
        First input which can be a file path or string content.
    file_or_content2 : str
        Second input which can be a file path or string content.

    Returns
    -------
    bool
        True if the contents of the two inputs are identical, False otherwise.

    Examples
    --------
    >>> content_same("path/to/file1.txt", "path/to/file2.txt")
    True
    >>> content_same("Hello, world!", "path/to/file_with_hello_world_content.txt")
    True
    >>> content_same("Hello, world!", "Goodbye, world!")
    False
    """
    # If the first argument is a filepath and it exists, read its content
    if os.path.exists(file_or_content_1):
        with open(file_or_content_1, "r") as f1:
            file_or_content_1 = f1.read()

    # If the second argument is a filepath and it exists, read its content
    if os.path.exists(file_or_content_2):
        with open(file_or_content_2, "r") as f2:
            file_or_content_2 = f2.read()

    return file_or_content_1 == file_or_content_2



def validate_keys(checked_dict, required_keys, allowed_keys=None):
    """
    Validate the presence of required keys and check for any unexpected keys in a dictionary.

    Give required keys and allowed keys to have complete control
    Give required keys twice to check that the list of keys is necessary and sufficient
    Give only required keys to allow all extra additional key

    Parameters
    ----------
    checked_dict : dict
        The dictionary to be checked.
    required_keys : set
        A set of keys that are required in the dictionary.
    allowed_keys : set
        A set of keys that are allowed in the dictionary.

    Raises
    ------
    ConfigurationError
        If either required keys are missing or unexpected keys are found.
    """

    # Convert input lists to sets for set operations
    checked_keys = set(checked_dict.keys())
    required_keys = set(required_keys)

    # Set allowed_keys to all present keys if not provided
    if allowed_keys is None:
        allowed_keys = checked_keys
    else:
        allowed_keys = set(allowed_keys)

    # Check for extra and missing keys
    missing_keys = required_keys - checked_keys
    extra_keys = checked_keys - allowed_keys

    # Prepare error messages
    error_messages = []
    if missing_keys:
        error_messages.append(f"Missing required keys: {missing_keys}")
    if extra_keys:
        error_messages.append(f"Found unexpected keys: {extra_keys}")

    # Raise combined error if there are any issues
    if error_messages:
        raise DictionaryValidationError("; ".join(error_messages))


class DictionaryValidationError(Exception):
    """Exception raised for errors in the expected options."""

    def __init__(self, message, key=None, value=None):
        self.message = message
        self.key = key
        self.value = value
        super().__init__(self._format_message())

    def _format_message(self):
        if self.key is not None and self.value is not None:
            return f"{self.message} Key: '{self.key}', Value: {self.value}"
        return self.message


def read_configuration_file(filename):
    """Reads and validates a YAML configuration file (return multiple outputs)"""

    # Read configuration file
    try:
        with open(filename, "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        raise Exception(f"Error parsing configuration file: '{filename}'. Original error: {e}")

    # Convert options to Numpy when possible
    config = convert_configuration_options(config)

    # # Validate the configuration dictionary
    # config, info, error = validate_configuration_options(config, cfg.CONFIGURATION_OPTIONS)

    return config


def convert_configuration_options(config):
    """
    Processes configuration data by evaluating string expressions as numerical values and converting lists to numpy arrays.

    This function iteratively goes through the configuration dictionary and converts string representations of numbers
    (e.g., "1+2", "2*np.pi") into actual numerical values using Python's `eval` function. It also ensures that all numerical
    values are represented as Numpy types for consistency across the application.

    Parameters
    ----------
    config : dict
        The configuration data loaded from a YAML file, typically containing a mix of strings, numbers, and lists.

    Returns
    -------
    dict
        The postprocessed configuration data where string expressions are evaluated as numbers, and all numerical values
        are cast to corresponding NumPy types.

    Raises
    ------
    ConfigurationError
        If a list contains elements of different types after conversion, indicating an inconsistency in the expected data types.
    """

    def convert_strings_to_numbers(data):
        """
        Recursively converts string expressions within the configuration data to numerical values.

        This function handles each element of the configuration: dictionaries are traversed recursively, lists are processed
        element-wise, and strings are evaluated as numerical expressions. Non-string and valid numerical expressions are
        returned as is. The conversion supports basic mathematical operations and is capable of recognizing Numpy functions
        and constants when used in the strings.

        Parameters
        ----------
        data : dict, list, str, or number
            A piece of the configuration data that may contain strings representing numerical expressions.

        Returns
        -------
        The converted data, with all string expressions evaluated as numbers.
        """
        if isinstance(data, dict):
            return {
                key: convert_strings_to_numbers(value) for key, value in data.items()
            }
        elif isinstance(data, list):
            return [convert_strings_to_numbers(item) for item in data]
        elif isinstance(data, bool):
            return data
        elif isinstance(data, str):
            # Evaluate strings as numbers if possible
            try:
                data = eval(data)
                return convert_numbers_to_numpy(data)
            except (NameError, SyntaxError, TypeError):
                return data
        elif isinstance(data, Number):
            # Convert Python number types to corresponding NumPy number types
            return convert_numbers_to_numpy(data)
        else:
            return data

    def convert_numbers_to_numpy(data):
        """
        Casts Python native number types (int, float) to corresponding Numpy number types.

        This function ensures that all numeric values in the configuration are represented using Numpy types.
        It converts integers to `np.int64` and floats to `np.float64`.

        Parameters
        ----------
        data : int, float
            A numerical value that needs to be cast to a Numpy number type.

        Returns
        -------
        The same numerical value cast to the corresponding Numpy number type.

        """
        if isinstance(data, int):
            return np.int64(data)
        elif isinstance(data, float):
            return np.float64(data)
        else:
            return data

    def convert_to_arrays(data, parent_key=""):
        """
        Convert lists within the input data to Numpy arrays.

        Iterates through the input data recursively. If a list is encountered, the function checks if all elements are of the same type.
        If they are, the list is converted to a Numpy array. If the elements are of different types, a :obj:`ConfigurationError` is raised.

        Parameters
        ----------
        data : dict or list or any
            The input data which may contain lists that need to be converted to NumPy arrays. The data can be a dictionary (with recursive processing for each value), a list, or any other data type.
        parent_key : str, optional
            The key in the parent dictionary corresponding to `data`, used for error messaging. The default is an empty string.

        Returns
        -------
        dict or list or any
            The input data with lists converted to NumPy arrays. The type of return matches the type of `data`. Dictionaries and other types are returned unmodified.

        Raises
        ------
        ValueError
            If a list within `data` contains elements of different types. The error message includes the problematic list and the types of its elements.
        
        """
        if isinstance(data, dict):
            return {k: convert_to_arrays(v, parent_key=k) for k, v in data.items()}
        elif isinstance(data, list):
            if not data:  # Empty list
                return data
            first_type = type(data[0])
            if not all(isinstance(item, first_type) for item in data):
                element_types = [type(item) for item in data]
                raise ValueError(
                    f"Option '{parent_key}' contains elements of different types: {data}, "
                    f"types: {element_types}"
                )
            return np.array(data)
        else:
            return data

    # Convert the configuration options to Numpy arrays when relevant
    config = convert_strings_to_numbers(config)
    config = convert_to_arrays(config)

    return config



def ensure_iterable(obj):
    """
    Ensure that an object is iterable. If the object is already an iterable
    (except for strings, which are not treated as iterables in this context),
    it will be returned as is. If the object is not an iterable, or if it is
    a string, it will be placed into a list to make it iterable.

    Parameters
    ----------
    obj : any type
        The object to be checked and possibly converted into an iterable.

    Returns
    -------
    Iterable
        The original object if it is an iterable (and not a string), or a new
        list containing the object if it was not iterable or was a string.

    Examples
    --------
    >>> ensure_iterable([1, 2, 3])
    [1, 2, 3]
    >>> ensure_iterable('abc')
    ['abc']
    >>> ensure_iterable(42)
    [42]
    >>> ensure_iterable(np.array([1, 2, 3]))
    array([1, 2, 3])
    """
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        return obj
    else:
        return [obj]
    

def render_and_evaluate(expression, data):
    """
    Render variables prefixed with '$' in an expression and evaluate the resulting expression.

    This function processes an input string `expr`, identifying all occurrences of variables
    indicated by a leading '$' symbol. Each such variable is resolved to its value from the
    provided `context` dictionary. The expression with all variables resolved is then evaluated
    and the result is returned.

    This function is useful to render strings defined in a YAML configuration file to values 
    that are calculated within the code and stored in a dicitonary.

    Parameters
    ----------
    expr : str
        The expression string containing variables to be rendered. Variables in the
        expression are expected to be prefixed with a '$' symbol.
    data : dict
        A dictionary containing variables and their corresponding values. These variables
        are used to render values in the expression.

    Returns
    -------
    The result of evaluating the rendered expression. The type of the result depends on the
    expression.

    Notes
    -----
    - `pattern`: A regular expression pattern used to identify variables within the expression.
      Variables are expected to be in the format `$variableName`, potentially with dot-separated
      sub-properties (e.g., `$variable.property`).

    - `replace_with_value`: An inner function that takes a regex match object and returns
      the value of the variable from `context`. `match.group(1)` returns the first captured
      group from the matched text, which in this case is the variable name excluding the
      leading '$' symbol. For example, in `$variableName`, `match.group(1)` would return
      `variableName`.

    - The function uses Python's `eval` for evaluation, which should be used cautiously as
      it can execute arbitrary code. Ensure that the context and expressions are from a trusted
      source.
    """
    # Pattern to find $variable expressions
    pattern = re.compile(r"\$(\w+(\.\w+)*)")

   # Function to replace each match with its resolved value
    def replace_with_value(match):
        nested_key = match.group(1)
        try:
            return str(render_nested_value(nested_key, data))
        except KeyError:
            raise KeyError(f"Variable '{nested_key}' not found in the provided data context.")
        
    try:
        # Replace all $variable with their actual values
        resolved_expr = pattern.sub(replace_with_value, expression)

        # Check if any unresolved variables remain
        if '$' in resolved_expr:
            raise ValueError(f"Unresolved variable in expression: '{resolved_expr}'")

        # Now evaluate the expression
        return eval(resolved_expr, data)
    except SyntaxError:
        raise SyntaxError(f"Syntax error in expression: '{expression}'")
    except Exception as e:
        raise TypeError(f"Error evaluating expression '{expression}': {e}")



def render_nested_value(nested_key, data):
    """
    Retrieves a value from a nested structure (dictionaries or objects with attributes) using a dot-separated key.

    This function is designed to navigate through a combination of dictionaries and objects. For an object to be
    compatible with this function, it must implement a `keys()` method that returns its attribute names.

    This function is intended as a subroutine of the more genera ``render_expression``

    Parameters
    ----------
    nested_key : str
        A dot-separated key string that specifies the path in the structure.
        For example, 'level1.level2.key' will retrieve data['level1']['level2']['key'] if data is a dictionary,
        or data.level1.level2.key if data is an object or a combination of dictionaries and objects.

    data : dict or object
        The starting dictionary or object from which to retrieve the value. This can be a nested structure
        of dictionaries and objects.

    Returns
    -------
    value
        The value retrieved from the nested structure using the specified key. 
        The type of the value depends on what is stored at the specified key in the structure.

    Raises
    ------
    KeyError
        If the specified nested key is not found in the data structure. The error message includes the part
        of the path that was successfully traversed and the available keys or attributes at the last valid level.
    """
    keys = nested_key.split('.')
    value = data
    traversed_path = []

    for key in keys:
        if isinstance(value, dict):
            # Handle dictionary-like objects
            if key in value:
                traversed_path.append(key)
                value = value[key]
            else:
                valid_keys = ', '.join(value.keys())
                traversed_path_str = '.'.join(traversed_path) if traversed_path else 'root'
                raise KeyError(f"Nested key '{key}' not found at '{traversed_path_str}'. Available keys: {valid_keys}")
        elif hasattr(value, key):
            # Handle objects with attributes
            traversed_path.append(key)
            value = getattr(value, key)
        else:
            traversed_path_str = '.'.join(traversed_path)
            available_keys = ', '.join(value.keys())
            raise KeyError(f"Key '{key}' not found in object at '{traversed_path_str}'. Available keys: {available_keys}")

    if not is_numeric(value):
        raise ValueError(f"The key '{nested_key}' is not numeric. Key value is: {value}")

    return value


def evaluate_constraints(data, constraints):
    """
    Evaluates the constraints based on the provided data and constraint definitions.

    Parameters
    ----------
    data : dict
        A dictionary containing performance data against which the constraints will be evaluated.
    constraints : list of dicts
        A list of constraint definitions, where each constraint is defined as a dictionary.
        Each dictionary must have 'variable' (str), 'type' (str, one of '=', '>', '<'), and 'value' (numeric).

    Returns
    -------
    tuple of numpy.ndarray
        Returns two numpy arrays: the first is an array of equality constraints, and the second is an array of 
        inequality constraints. These arrays are flattened and concatenated from the evaluated constraint values.

    Raises
    ------
    ValueError
        If an unknown constraint type is specified in the constraints list.
    """
    # Initialize constraint lists
    c_eq = []    # Equality constraints
    c_ineq = []  # Inequality constraints

    # Loop over all constraint from configuration file
    for constraint in constraints:
        name = constraint['variable']
        type = constraint['type']
        target = constraint['value']
        normalize = constraint.get("normalize", False)

        # Get the performance value for the given variable name
        current = render_nested_value(name, data)

        # Evaluate constraint
        mismatch = current - target

        # Normalize constraint according to specifications
        normalize_factor = normalize if is_numeric(normalize) else target
        if normalize is not False:
            if normalize_factor == 0:
                raise ValueError(f"Cannot normalize constraint '{name} {type} {target}' because the normalization factor is '{normalize_factor}' (division by zero).")
            mismatch /= normalize_factor

        # Add constraints to lists
        if type == '=':
            c_eq.append(mismatch)
        elif type == '>':
            c_ineq.append(mismatch)
        elif type == '<':
            # Change sign because optimizer handles c_ineq > 0
            c_ineq.append(-mismatch)
        else:
            raise ValueError(f"Unknown constraint type: {type}")

    # Flatten and concatenate constraints
    c_eq = np.hstack([np.atleast_1d(item) for item in c_eq]) if c_eq else np.array([])
    c_ineq = np.hstack([np.atleast_1d(item) for item in c_ineq]) if c_ineq else np.array([])

    return c_eq, c_ineq


def evaluate_objective_function(data, objective_function):
    """
    Evaluates the objective function based on the provided data and configuration.

    Parameters
    ----------
    data : dict
        A dictionary containing performance data against which the objective function will be evaluated.
    objective_function : dict
        A dictionary defining the objective function. It must have 'variable' (str) 
        and 'type' (str, either 'minimize' or 'maximize').

    Returns
    -------
    float
        The value of the objective function, adjusted for optimization. Positive for minimization and 
        negative for maximization.

    Raises
    ------
    ValueError
        If an unknown objective function type is specified in the configuration.
    """

    # Get the performance value for the given variable name
    name = objective_function['variable']
    type = objective_function['type']
    value = render_nested_value(name, data)

    if not np.isscalar(value):
        raise ValueError(f"The objective function '{name}' must be an scalar, but the value is: {value}")

    if type == 'minimize':
        return value
    elif type == 'maximize':
        return -value
    else:
        raise ValueError(f"Unknown objective function type: {type}")


def convert_numpy_to_python(data, precision=10):
    """
    Recursively converts numpy arrays, scalars, and other numpy types to their Python counterparts
    and rounds numerical values to the specified precision.

    Parameters:
    - data: The numpy data to convert.
    - precision: The decimal precision to which float values should be rounded.

    Returns:
    - The converted data with all numpy types replaced by native Python types and float values rounded.
    """
    if data is None:
        return None

    if isinstance(data, dict):
        return {k: convert_numpy_to_python(v, precision) for k, v in data.items()}

    elif isinstance(data, list):
        return [convert_numpy_to_python(item, precision) for item in data]

    elif isinstance(data, np.ndarray):
        # If the numpy array has more than one element, it is iterable.
        if data.ndim > 0:
            return [convert_numpy_to_python(item, precision) for item in data.tolist()]
        else:
            # This handles the case of a numpy array with a single scalar value.
            return convert_numpy_to_python(data.item(), precision)

    elif isinstance(
        data,
        (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64),
    ):
        return int(data.item())

    elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
        return round(float(data.item()), precision)

    elif isinstance(data, np.bool_):
        return bool(data.item())

    elif isinstance(data, (np.str_, np.unicode_)):
        return str(data.item())

    # This will handle Python built-in types and other types that are not numpy.
    elif isinstance(data, (float, int, str, bool)):
        if isinstance(data, float):
            return round(data, precision)
        return data

    else:
        raise TypeError(f"Unsupported data type: {type(data)}")



def print_dict(data, indent=0):
    """
    Recursively prints nested dictionaries with indentation.

    Parameters
    ----------
    data : dict
        The dictionary to print. It can contain nested dictionaries as values.
    indent : int, optional
        The initial level of indentation for the keys of the dictionary, by default 0.
        It controls the number of spaces before each key.

    Returns
    -------
    None

    Examples
    --------
    >>> data = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    >>> print_dict(data)
    a: 1
    b:
        c: 2
        d:
            e: 3
    """

    for key, value in data.items():
        print("    " * indent + str(key) + ":", end=" ")
        if isinstance(value, dict):
            print("")
            print_dict(value, indent + 1)
        else:
            print(value)


def print_object(obj):
    """
    Prints all attributes and methods of an object, sorted alphabetically.

    - Methods are identified as callable and printed with the prefix 'Method: '.
    - Attributes are identified as non-callable and printed with the prefix 'Attribute: '.

    Parameters
    ----------
    obj : object
        The object whose attributes and methods are to be printed.

    Returns
    -------
    None
        This function does not return any value. It prints the attributes and methods of the given object.

    """
    # Retrieve all attributes and methods
    attributes = dir(obj)

    # Sort them alphabetically, case-insensitive
    sorted_attributes = sorted(attributes, key=lambda x: x.lower())

    # Iterate over sorted attributes
    for attr in sorted_attributes:
        # Retrieve the attribute or method from the object
        attribute_or_method = getattr(obj, attr)

        # Check if it is callable (method) or not (attribute)
        if callable(attribute_or_method):
            print(f"Method: {attr}")
        else:
            print(f"Attribute: {attr}")



def postprocess_ode(t, y, ode_handle):
    """
    Post-processes the output of an ordinary differential equation (ODE) solver.

    This function takes the time points and corresponding ODE solution matrix,
    and for each time point, it calls a user-defined ODE handling function to
    process the state of the ODE system. It collects the results into a
    dictionary where each key corresponds to a state variable and the values
    are numpy arrays of that state variable at each integration step

    Parameters
    ----------
    t : array_like
        Integration points at which the ODE was solved, as a 1D numpy array.
    y : array_like
        The solution of the ODE system, as a 2D numpy array with shape (n,m) where
        n is the number of points and m is the number of state variables.
    ode_handle : callable
        A function that takes in a integration point and state vector and returns a tuple,
        where the first element is ignored (can be None) and the second element
        is a dictionary representing the processed state of the system.

    Returns
    -------
    ode_out : dict
        A dictionary where each key corresponds to a state variable and each value
        is a numpy array containing the values of that state variable at each integration step.
    """
    # Initialize ode_out as a dictionary
    ode_out = {}
    for t_i, y_i in zip(t, y.T):
        _, out = ode_handle(t_i, y_i)

        for key, value in out.items():
            # Initialize with an empty list
            if key not in ode_out:
                ode_out[key] = []
            # Append the value to list of current key
            ode_out[key].append(value)

    # Convert lists to numpy arrays
    for key in ode_out:
        ode_out[key] = np.array(ode_out[key])

    return ode_out