import pandas as pd
import numpy as np
from datetime import datetime, time , timedelta
import networkx as nx



def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    # Create a directed graph to represent toll locations and distances
    G = nx.DiGraph()

    # Add edges to the graph with distances
    for index, row in df.iterrows():
        G.add_edge(row['id_start'], row['id_end'], distance=row['distance'])

    # Create a DataFrame with toll locations as both index and columns
    toll_locations = sorted(set(df['id_start'].unique()).union(set(df['id_end'].unique())))
    distance_matrix = pd.DataFrame(index=toll_locations, columns=toll_locations)

    # Populate the DataFrame with cumulative distances
    for source in toll_locations:
        for target in toll_locations:
            if source == target:
                distance_matrix.at[source, target] = 0
            else:
                try:
                    # Use the shortest_path_length function to get cumulative distances
                    distance = nx.shortest_path_length(G, source=source, target=target, weight='distance')
                    distance_matrix.at[source, target] = distance
                except nx.NetworkXNoPath:
                    # If no path exists, set the distance to NaN
                    distance_matrix.at[source, target] = float('nan')

    # Ensure the matrix is symmetric
    distance_matrix = distance_matrix.combine_first(distance_matrix.T)

    return distance_matrix


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

    # Create an empty list to store unrolled distance information
    unrolled_data = []

    # Iterate over the rows of the distance matrix
    for i in range(len(distance_matrix.index)):
        id_start = distance_matrix.index[i]

        # Iterate over the columns of the distance matrix
        for j in range(len(distance_matrix.columns)):
            id_end = distance_matrix.columns[j]

            # Skip diagonal entries (same id_start and id_end)
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]

                # Append the information to the unrolled list
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a DataFrame from the unrolled list
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    # Filter rows with the specified reference value
    reference_df = df[df['id_start'] == reference_id]

    # Calculate the average distance for the reference value
    average_distance = reference_df['distance'].mean()

    # Calculate the threshold range (within 10% of the average distance)
    threshold_low = average_distance - (0.1 * average_distance)
    threshold_high = average_distance + (0.1 * average_distance)

    # Filter rows within the threshold range and get unique 'id_start' values
    filtered_ids = df[(df['distance'] >= threshold_low) & (df['distance'] <= threshold_high)]['id_start'].unique()

    # Sort the list of values and return
    sorted_filtered_ids = sorted(filtered_ids)
    return sorted_filtered_ids


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Create new columns for each vehicle type with their respective rate coefficients
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient


    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    df['startDay'] = ""
    df['endDay'] = ""
    df['startTime'] = time(0, 0, 0)
    df['endTime'] = time(0, 0, 0)

    print(df.columns)

    # Define time ranges and discount factors
    time_ranges_weekday = [(time(0, 0, 0), time(10, 0, 0)),
                           (time(10, 0, 0), time(18, 0, 0)),
                           (time(18, 0, 0), time(23, 59, 59))]

    time_ranges_weekend = [(time(0, 0, 0), time(23, 59, 59))]

    discount_factors_weekday = [0.8, 1.2, 0.8]
    discount_factor_weekend = 0.7


    # Apply discount factors based on time ranges and day of the week
    for i, (start_time, end_time) in enumerate(time_ranges_weekday):
        mask = ((df['startTime'] >= start_time) & (df['endTime'] <= end_time) & (
            df['startDay'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])))
        df.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= discount_factors_weekday[i]

    for start_time, end_time in time_ranges_weekend:
        mask = ((df['startTime'] >= start_time) & (df['endTime'] <= end_time) & (
            df['startDay'].isin(['Saturday', 'Sunday'])))
        df.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= discount_factor_weekend

    return df

#Question 1: Distance Matrix Calculation
df = pd.read_csv("C:/Users/Aniket/MapUp-Data-Assessment-F/datasets/dataset-3.csv")
distance_matrix = calculate_distance_matrix(df)
print(distance_matrix)

#Question 2: Unroll Distance Matrix
unrolled_result = unroll_distance_matrix(distance_matrix)
print(unrolled_result)

#Question 3: Finding IDs within Percentage Threshold
reference_id = unrolled_result['id_start'].sample().iloc[0]
result_ids = find_ids_within_ten_percentage_threshold(unrolled_result, reference_id)
print(result_ids)

#Question 4: Calculate Toll Rate
result_with_toll_rates = calculate_toll_rate(unrolled_result)
print(result_with_toll_rates)

#Question 5: Calculate Time-Based Toll Rates
result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_ids)
print(result_with_time_based_toll_rates)