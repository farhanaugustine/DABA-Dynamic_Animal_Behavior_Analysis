# Distance Calculation Module
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

def analyze_distance_in_roi(body_part_data, roi_definitions, df, body_part, roi_name):
    """
    Calculates the total distance a specified body part traveled within a specified ROI.

    Args:
        body_part_data (dict): Dictionary containing x and y coordinates for each body part.
        roi_definitions (dict): Dictionary defining ROIs with name and coordinates.
        df (DataFrame): DataFrame containing all the data.
        body_part (str): The name of the body part to analyze.
        roi_name (str): The name of the ROI to analyze.

    Returns:
        float: The total distance the specified body part traveled within the ROI
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

    total_distance_in_roi = 0
    for i in range(len(x_coords) - 1):
        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_coords[i+1], y_coords[i+1]

        if is_in_roi(x1, y1, roi_coords) or is_in_roi(x2, y2, roi_coords):
            total_distance_in_roi += calculate_distance([x1, y1], [x2, y2])

    return total_distance_in_roi

if __name__ == '__main__':
    # Example usage:
    # Import the data loading script and load variables df, body_part_data, and roi_definitions
    # to mimic the setup of running this script in the intended way
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
    
    total_distance = analyze_distance_in_roi(body_part_data, roi_definitions, df, "Nose", "center")
    print(f"Total distance traveled by the nose in the center ROI: {total_distance}")