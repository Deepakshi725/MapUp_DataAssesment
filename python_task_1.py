import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values,
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here

    # Create a pivot table using id_1 as index, id_2 as columns, and car as values
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')

    # Fill NaN values with 0
    car_matrix = car_matrix.fillna(0)

    # Set diagonal values to 0
    for index in car_matrix.index:
        car_matrix.at[index, index] = 0

    return car_matrix


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here

    # Add a new categorical column 'car_type' based on values of the 'car' column
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_count = df['car_type'].value_counts().to_dict()

    return dict(sorted(type_count.items()))


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here

    # Calculate the mean value of the 'bus' column
    mean_bus_value = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here

    # Group by 'route' and calculate the average of the 'truck' column for each group
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' values is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes in ascending order
    selected_routes.sort()

    return selected_routes


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here

    # Create a deep copy of the input DataFrame to avoid modifying the original
    modified_matrix = matrix.copy()

    # Apply the specified logic to each value in the DataFrame
    for column in modified_matrix.columns:
        for index in modified_matrix.index:
            value = modified_matrix.at[index, column]
            if value > 20:
                modified_matrix.at[index, column] = round(value * 0.75, 1)
            else:
                modified_matrix.at[index, column] = round(value * 1.25, 1)

    return modified_matrix

def check_completeness(group):
    # Check if the timestamps cover a full 24-hour period
    full_24_hours = group['time_diff'].between(0, 24 * 60 * 60).all()

    # Check if the timestamps span all 7 days of the week
    span_all_days = group['start_timestamp'].dt.dayofweek.nunique() == 7

    # Return True if both conditions are satisfied, indicating completeness
    return full_24_hours and span_all_days

def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    # Combine 'startDay' and 'startTime' to create a datetime column for start timestamp
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%A %H:%M:%S')

    # Combine 'endDay' and 'endTime' to create a datetime column for end timestamp
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%A %H:%M:%S')

    # Calculate the time difference between start and end timestamps
    df['time_diff'] = (df['end_timestamp'] - df['start_timestamp']).dt.total_seconds()

    # Group by (id, id_2) and check completeness for each group
    completeness_check = df.groupby(['id', 'id_2']).apply(check_completeness).reset_index(drop=True)

    return completeness_check

#Question 1: Car Matrix Generation
df = pd.read_csv("C:/Users/Aniket/MapUp-Data-Assessment-F/datasets/dataset-1.csv")
result = generate_car_matrix(df)
print(result)

#Question 2: Car Type Count Calculation
result = get_type_count(df)
print(result)

#Question 3: Bus Count Index Retrieval
result = get_bus_indexes(df)
print(result)

#Question 4: Route Filtering
result = filter_routes(df)
print(result)

#Question 5: Matrix Value Modification
result = multiply_matrix(result)
print(result)

#Question 6: Time Check
df = pd.read_csv("C:/Users/Aniket/MapUp-Data-Assessment-F/datasets/dataset-2.csv")
result = time_check(df)
print(result)