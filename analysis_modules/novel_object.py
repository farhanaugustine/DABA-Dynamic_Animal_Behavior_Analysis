# Novel Object Interaction Analysis Module
# Flexible Object Coordinates: The object_positions parameter can now accept:
# A list of coordinates (tuples) to define a polygon-shaped object.
# A single coordinate (tuple) to define a point-like object.
# Interaction Bout Counting: The code now tracks and counts the number of interaction bouts with each object. An interaction bout is defined as a continuous period of interaction with an object.

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

def analyze_novel_object_interaction(body_part_data, roi_definitions, df, body_part, object_positions, interaction_threshold, exploration_threshold, frame_threshold):
    """
    Analyzes novel object interaction and general exploration.

    Args:
        body_part_data (dict): Dictionary containing body part coordinates.
        roi_definitions (dict): Dictionary of ROI definitions.
        df (pd.DataFrame): DataFrame containing the DeepLabCut data.
        body_part (str): Body part to track.
        object_positions (dict): Dictionary with object names as keys and coordinates as values (list of coordinates for each object).
        interaction_threshold (float): Distance threshold for defining an interaction with an object.
        exploration_threshold (float): Distance threshold for defining exploration (movement threshold between frames).
        frame_threshold (int): Number of frames over which to assess movement for exploration.

    Returns:
         dict: A dictionary with "object_interaction", "object_interaction_bouts", and "exploration_rate" as keys.
         -   object_interaction: Returns a dictionary with the time (number of frames) the animal spent within a certain distance from the novel objects.
         -   object_interaction_bouts: Returns a dictionary with the number of interaction bouts with each object.
         -   exploration_rate: Returns a number representing the proportion of 'exploratory' frames.
    """
    if not isinstance(interaction_threshold, (int, float)) or interaction_threshold < 0:
        logging.error("Invalid interaction threshold. Please provide a non-negative number.")
        return {}

    if not isinstance(exploration_threshold, (int, float)) or exploration_threshold < 0:
        logging.error("Invalid exploration threshold. Please provide a non-negative number.")
        return {}

    if not isinstance(frame_threshold, int) or frame_threshold <= 0:
        logging.error("Invalid frame threshold. Please provide a positive integer.")
        return {}

    if body_part not in body_part_data:
        logging.warning(f"Body part '{body_part}' not found in data. Returning empty dictionary.")
        return {}

    x_coords = body_part_data[body_part]["x"]
    y_coords = body_part_data[body_part]["y"]

    interaction_times = {obj: 0 for obj in object_positions}
    interaction_bouts = {obj: 0 for obj in object_positions}
    interacting = {obj: False for obj in object_positions} # Track if the animal is currently interacting with an object

    for i in range(len(x_coords)):
        x, y = x_coords[i], y_coords[i]
        for obj, obj_coords in object_positions.items():
            is_interacting = False
            if isinstance(obj_coords, list) and all(isinstance(coord, tuple) and len(coord) == 2 for coord in obj_coords):
                # If object coordinates are a list of tuples, use is_in_roi
                is_interacting = is_in_roi(x, y, obj_coords)
            elif isinstance(obj_coords, tuple) and len(obj_coords) == 2:
                # If object coordinates are a single tuple, use distance
                distance = calculate_distance((x, y), obj_coords)
                if distance <= interaction_threshold:
                    is_interacting = True
            else:
                logging.warning(f"Invalid object coordinates for '{obj}'. Skipping.")
                continue

            if is_interacting:
                interaction_times[obj] += 1
                if not interacting[obj]:
                    interaction_bouts[obj] += 1
                    interacting[obj] = True
            else:
                interacting[obj] = False

    if len(x_coords) < frame_threshold:
        logging.warning(f"Not enough frames for exploration analysis. Returning 0 for exploration rate.")
        exploration_rate = 0
    else:
        exploration_count = 0
        for i in range(len(x_coords) - frame_threshold):
            movement_sum = 0
            for j in range(1, frame_threshold):
                dist = calculate_distance((x_coords[i], y_coords[i]), (x_coords[i + j], y_coords[i + j]))
                movement_sum += dist
            if movement_sum >= exploration_threshold:
                exploration_count += 1
        exploration_rate = exploration_count / (len(x_coords) - frame_threshold)

    logging.info(f"Novel object analysis complete for body part: {body_part}.")
    return {"object_interaction": interaction_times, "object_interaction_bouts": interaction_bouts, "exploration_rate": exploration_rate}

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

    object_positions = {
        "object1": [(30, 20), (32, 22), (30, 24)],  # Example object with multiple coordinates (polygon)
        "object2": (10, 10)  # Example object with a single coordinate (center)
    }
    interaction_threshold = 10
    exploration_threshold = 5
    frame_threshold = 5
    novel_object_results = analyze_novel_object_interaction(body_part_data, roi_definitions, df, "Nose", object_positions, interaction_threshold, exploration_threshold, frame_threshold)
    print(f"Novel object interaction results: {novel_object_results}")