# event_duration module is focused on Freezing Behaivors Only (for now)
# The function's logic is built around the idea that:
# 1. Movement is Calculated: It calculates the sum of distances between consecutive frames over a specified frame_threshold.
# 2. Freezing is Defined: If this movement_sum is below a freezing_threshold, the animal is considered to be freezing.
# 3. Duration is Tracked: The function then tracks the start and end frames of these freezing events to calculate the total duration.

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

def analyze_freezing(body_part_data, roi_definitions, df, body_part, frame_threshold, freezing_threshold=1, roi_name=None):
    """
    Analyzes freezing behavior based on the change in position of a specified body part, optionally within an ROI.

    Args:
        body_part_data (dict): Dictionary containing body part coordinates.
        roi_definitions (dict): Dictionary of ROI definitions.
        df (pd.DataFrame): DataFrame containing the DeepLabCut data.
        body_part (str): The body part to track for freezing analysis.
        frame_threshold (int): Number of frames over which to assess movement for freezing.
        freezing_threshold (float): The threshold for movement to be considered freezing.
        roi_name (str, optional): The name of the ROI to analyze. If None, analyzes overall freezing.

    Returns:
        dict: Dictionary with freezing duration (in frames) for the specified body part.
    """
    freezing_durations = {}

    if not isinstance(frame_threshold, int) or frame_threshold <= 0:
        logging.error("Invalid frame threshold. Please provide a positive integer.")
        return freezing_durations

    if not isinstance(freezing_threshold, (int, float)) or freezing_threshold < 0:
        logging.error("Invalid freezing threshold. Please provide a non-negative number.")
        return freezing_durations

    if body_part not in body_part_data:
        logging.warning(f"Body part '{body_part}' not found in data. Skipping.")
        return freezing_durations

    x_coords = body_part_data[body_part]["x"]
    y_coords = body_part_data[body_part]["y"]

    if len(x_coords) < frame_threshold:
        logging.warning(f"Not enough frames for analysis of '{body_part}'. Skipping.")
        return freezing_durations

    if roi_name:
        try:
            roi_coords = roi_definitions[roi_name]
        except KeyError:
            logging.error(f"ROI '{roi_name}' not found in ROI definitions.")
            return freezing_durations

    freezing = False
    freeze_start_frame = None
    total_freeze_duration = 0

    for i in range(len(x_coords) - frame_threshold):
        x = x_coords[i]
        y = y_coords[i]
        if roi_name:
            if not is_in_roi(x, y, roi_coords):
                continue  # Skip if not in ROI

        movement_sum = 0
        for j in range(1, frame_threshold):
            dist = calculate_distance((x_coords[i], y_coords[i]), (x_coords[i + j], y_coords[i + j]))
            movement_sum += dist

        if movement_sum < freezing_threshold:  # If animal has not moved much, start freezing event
            if not freezing:
                freezing = True
                freeze_start_frame = i
        else:
            if freezing:  # if animal has now moved, end freezing event
                freezing = False
                total_freeze_duration += i - freeze_start_frame

    if freezing:  # checks for the case that animal has frozen until the end of the video
        total_freeze_duration += (len(x_coords) - 1) - freeze_start_frame

    freezing_durations[body_part] = total_freeze_duration
    if total_freeze_duration == 0:
        logging.info(f"No freezing detected for '{body_part}' with the given parameters.")
    else:
        logging.info(f"Freezing analysis complete for '{body_part}'. Total freezing duration: {total_freeze_duration} frames.")
    return freezing_durations

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

    frame_threshold = 5
    freezing_threshold = 1
    body_part_to_analyze = "Nose"
    # Example 1: Overall freezing analysis
    freezing_results_overall = analyze_freezing(body_part_data, roi_definitions, df, body_part_to_analyze, frame_threshold, freezing_threshold)
    print(f"Freezing durations (overall): {freezing_results_overall}")

    # Example 2: Freezing analysis within an ROI
    roi_to_analyze = "center"
    freezing_results_roi = analyze_freezing(body_part_data, roi_definitions, df, body_part_to_analyze, frame_threshold, freezing_threshold, roi_to_analyze)
    print(f"Freezing durations in '{roi_to_analyze}': {freezing_results_roi}")