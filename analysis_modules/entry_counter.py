# Entry and Exit Counting Module
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def count_entries(body_part_data, roi_definitions, df, body_part, roi_name, frame_threshold=5):
    """
    Counts the number of entries a body part makes into a specific ROI, considering a frame threshold.

    Args:
        body_part_data (dict): Dictionary containing x and y coordinates for each body part.
        roi_definitions (dict): Dictionary defining ROIs with name and coordinates.
        df (DataFrame): DataFrame containing all the data.
        body_part (str): The name of the body part to analyze.
        roi_name (str): The name of the ROI to analyze.
        frame_threshold (int): The number of previous frames to consider for entry detection.

    Returns:
        int: The number of entries into the ROI.
    """
    try:
        x_coords = body_part_data[body_part]["x"]
        y_coords = body_part_data[body_part]["y"]
    except KeyError:
        logging.error(f"Body part '{body_part}' not found in data.")
        return None

    try:
        roi_coords = roi_definitions[roi_name]
    except KeyError:
        logging.error(f"ROI '{roi_name}' not found in ROI definitions.")
        return None

    entry_count = 0
    in_roi_history = [False] * frame_threshold  # Keep track of the last 'frame_threshold' frames
    previous_in_roi = False

    for index in range(len(df)):
        x = x_coords[index]
        y = y_coords[index]
        is_currently_in_roi = is_in_roi(x, y, roi_coords)

        # Update the history
        in_roi_history.pop(0)
        in_roi_history.append(is_currently_in_roi)

        # Check for entry
        if is_currently_in_roi and not previous_in_roi and not all(in_roi_history):
            entry_count += 1
        previous_in_roi = is_currently_in_roi

    return entry_count

def count_exits(body_part_data, roi_definitions, df, body_part, roi_name, frame_threshold=5):
    """
    Counts the number of exits a body part makes from a specific ROI, considering a frame threshold.

    Args:
        body_part_data (dict): Dictionary containing x and y coordinates for each body part.
        roi_definitions (dict): Dictionary defining ROIs with name and coordinates.
        df (DataFrame): DataFrame containing all the data.
        body_part (str): The name of the body part to analyze.
        roi_name (str): The name of the ROI to analyze.
        frame_threshold (int): The number of previous frames to consider for exit detection.

    Returns:
        int: The number of exits from the ROI.
    """
    try:
        x_coords = body_part_data[body_part]["x"]
        y_coords = body_part_data[body_part]["y"]
    except KeyError:
        logging.error(f"Body part '{body_part}' not found in data.")
        return None

    try:
        roi_coords = roi_definitions[roi_name]
    except KeyError:
        logging.error(f"ROI '{roi_name}' not found in ROI definitions.")
        return None

    exit_count = 0
    in_roi_history = [False] * frame_threshold  # Keep track of the last 'frame_threshold' frames
    previous_in_roi = False

    for index in range(len(df)):
        x = x_coords[index]
        y = y_coords[index]
        is_currently_in_roi = is_in_roi(x, y, roi_coords)

        # Update the history
        in_roi_history.pop(0)
        in_roi_history.append(is_currently_in_roi)

        # Check for exit
        if not is_currently_in_roi and previous_in_roi and not all(in_roi_history):
            exit_count += 1
        previous_in_roi = is_currently_in_roi

    return exit_count

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

    body_part_to_analyze = "Nose"
    roi_to_analyze = "center"
    frame_threshold = 5
    entry_count = count_entries(body_part_data, roi_definitions, df, body_part_to_analyze, roi_to_analyze, frame_threshold)
    exit_count = count_exits(body_part_data, roi_definitions, df, body_part_to_analyze, roi_to_analyze, frame_threshold)
    print(f"Number of entries of {body_part_to_analyze} into {roi_to_analyze}: {entry_count}")
    print(f"Number of exits of {body_part_to_analyze} from {roi_to_analyze}: {exit_count}")