# The primary goal of this function is to analyze the movement of multiple body parts within a specified Region of Interest (ROI) and provide insights into their behavior. Specifically, it calculates:
# Entry Count: How many times each body part enters the ROI.
# Time in ROI: The total number of frames each body part spends inside the ROI.
# Average Speed per Entry Bout: The average speed of each body part during each entry into the ROI.

import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def is_in_roi(x, y, roi_coords):
    """Checks if a point (x, y) is inside a given ROI defined by a list of coordinates"""
    # Create points from the roi coordinates, create the polygon from the points
    # ray casting algorythm
    n = len(roi_coords)
    inside = False
    p1x, p1y = roi_coords[0]
    for i in range(n + 1):
        p2x, p2y = roi_coords[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def analyze_multi_bodypart_entries(body_part_data, roi_definitions, df, body_parts, roi_name):
    """
    Analyzes multiple body parts for entry counts, time spent in ROI, and average speed per entry bout.

    Args:
        body_part_data (dict): Dictionary containing x and y coordinates for each body part.
        roi_definitions (dict): Dictionary defining ROIs with name and coordinates.
        df (DataFrame): DataFrame containing all the data.
        body_parts (list): A list of body part names to analyze.
        roi_name (str): The name of the ROI to analyze.

    Returns:
        dict: A dictionary containing entry counts, time spent in ROI, and average speed per entry bout for each body part.
    """
    results = {}
    try:
        roi_coords = roi_definitions[roi_name]
    except KeyError:
        logging.error(f"ROI '{roi_name}' not found in ROI definitions.")
        return results

    for part in body_parts:
        if part not in body_part_data:
            logging.warning(f"Body part '{part}' not found in data. Skipping.")
            continue
        results[part] = {
            "entry_count": 0,
            "time_in_roi": 0,
            "entry_speeds": [],
        }

    in_roi = {part: False for part in body_parts}  # Track if each body part is currently in the ROI
    entry_start_times = {part: None for part in body_parts}  # Track the start time of each entry

    for index in range(len(df)):
        for part in body_parts:
            if part not in body_part_data:
                continue
            x = body_part_data[part]["x"][index]
            y = body_part_data[part]["y"][index]
            is_currently_in_roi = is_in_roi(x, y, roi_coords)

            if is_currently_in_roi:
                results[part]["time_in_roi"] += 1
                if not in_roi[part]:
                    results[part]["entry_count"] += 1
                    in_roi[part] = True
                    entry_start_times[part] = index
            elif not is_currently_in_roi and in_roi[part]:
                in_roi[part] = False
                if entry_start_times[part] is not None:
                    entry_start_index = entry_start_times[part]
                    entry_end_index = index
                    if entry_end_index > entry_start_index:
                        speeds = []
                        for i in range(entry_start_index, entry_end_index - 1):
                            dist = calculate_distance([body_part_data[part]["x"][i], body_part_data[part]["y"][i]],
                                                     [body_part_data[part]["x"][i + 1], body_part_data[part]["y"][i + 1]])
                            speeds.append(dist)
                        if speeds:
                            avg_speed = np.mean(speeds)
                            results[part]["entry_speeds"].append(avg_speed)
                    entry_start_times[part] = None
    logging.info(f"Multi-body part analysis complete for body parts: {body_parts} in ROI: {roi_name}")
    return results

if __name__ == '__main__':
    # Example usage:
    import pandas as pd
    import numpy as np
    import json

    csv_file = "test.csv"
    roi_definitions = {'center': [[20, 10], [40, 10], [40, 30], [20, 30]]}

    # Load the DeepLabCut CSV file, using the second and third rows as the header
    df = pd.read_csv(csv_file, header=[1, 2])

    # Create a dictionary to map body part names to coordinate columns
    body_part_data = {}
    # Get unique body parts from the first level of the MultiIndex (excluding 'bodyparts')
    unique_body_parts = df.columns.get_level_values(0).unique().tolist()
    if 'bodyparts' in unique_body_parts:
        unique_body_parts.remove('bodyparts')

    for part in unique_body_parts:
        # Use the MultiIndex to access the data
        body_part_data[part] = {
            "x": df.loc[:, (part, 'x')].to_numpy(),
            "y": df.loc[:, (part, 'y')].to_numpy(),
            "likelihood": df.loc[:, (part, 'likelihood')].to_numpy()
        }

    body_parts_to_analyze = ["Nose", "Thorax"]
    roi_to_analyze = "center"
    analysis_results = analyze_multi_bodypart_entries(body_part_data, roi_definitions, df, body_parts_to_analyze, roi_to_analyze)
    print(f"Multi-body part analysis results: {analysis_results}")