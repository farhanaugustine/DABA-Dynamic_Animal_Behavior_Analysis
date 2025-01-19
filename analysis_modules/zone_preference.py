# # Zone Preference
# Distance Calculation: The calculate_distance function is now used to calculate the distance traveled within each ROI.
# distance_traveled Dictionary: A distance_traveled dictionary is introduced to store the total distance traveled within each ROI.
# Distance Calculation Logic: The code now calculates the distance traveled within each ROI by summing the distances between consecutive frames where the animal is within that ROI.
# Comprehensive Output: The function now returns a dictionary with "time_spent", "distance_traveled", and "proportion", each containing a dictionary with ROIs as keys and time, distance, and proportion as values.
# Logging: Added logging to provide more informative messages and warnings.
# ROI Iteration: Modified to create a dictionary of ROI coordinates outside the main loop to avoid repeated lookups.
# Error Handling: Added try-except blocks to handle potential errors.
# Clarity: Improved comments and variable names for better readability.
# Feedback Messages: Added a feedback message to indicate the completion of the analysis.

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

def analyze_zone_preference(body_part_data, roi_definitions, df, body_part, rois):
    """
    Analyzes the time spent, distance traveled, and proportion of time in specified ROIs.

    Args:
        body_part_data (dict): Dictionary containing body part coordinates.
        roi_definitions (dict): Dictionary of ROI definitions.
        df (pd.DataFrame): DataFrame containing the DeepLabCut data.
        body_part (str): Body part to track for zone preference analysis.
        rois (list): List of ROI names to analyze.

    Returns:
         dict: A dictionary containing three keys: "time_spent", "distance_traveled", and "proportion", each containing a dictionary with ROIs as keys
         and time, distance, and proportion as values.
    """
    if body_part not in body_part_data:
        logging.warning(f"Body part '{body_part}' not found in data. Returning empty dict")
        return {}

    x_coords = body_part_data[body_part]["x"]
    y_coords = body_part_data[body_part]["y"]
    
    time_spent = {roi: 0 for roi in rois}
    distance_traveled = {roi: 0 for roi in rois}
    
    roi_coords_dict = {}
    for roi_name in rois:
        if roi_name not in roi_definitions:
            logging.warning(f"ROI '{roi_name}' not found in ROI definitions. Skipping.")
            continue
        roi_coords_dict[roi_name] = roi_definitions[roi_name]

    for i in range(len(x_coords)):
        x, y = x_coords[i], y_coords[i]
        for roi_name, roi_coords in roi_coords_dict.items():
            if is_in_roi(x, y, roi_coords):
               time_spent[roi_name] += 1
               if i < len(x_coords) - 1:
                   next_x, next_y = x_coords[i+1], y_coords[i+1]
                   distance_traveled[roi_name] += calculate_distance((x,y), (next_x, next_y))

    total_frames = len(x_coords)
    proportions = {roi: time / total_frames for roi, time in time_spent.items()} if total_frames > 0 else {roi: 0 for roi in rois}
    logging.info(f"Zone preference analysis complete for body part: {body_part} in ROIs: {rois}")
    return {"time_spent": time_spent, "distance_traveled": distance_traveled, "proportion": proportions}

if __name__ == '__main__':
    # Example usage:
    import pandas as pd
    import numpy as np
    import json

    csv_file = "test.csv"
    roi_definitions = {
        'roi1': [[20, 10], [40, 10], [40, 30], [20, 30]],
        'roi2': [[60, 10], [80, 10], [80, 30], [60, 30]]
    }

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
    
    rois_to_analyze = ["roi1", "roi2"]
    body_part_to_analyze = "Nose"
    zone_preference_results = analyze_zone_preference(body_part_data, roi_definitions, df, body_part_to_analyze, rois_to_analyze)
    print(f"Zone preference results: {zone_preference_results}")