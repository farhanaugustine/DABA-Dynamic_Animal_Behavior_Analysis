#Average Velocity Module
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

def calculate_average_velocity(body_part_data, roi_definitions, df, body_part, roi_name=None, frame_rate=None, timestamps=None):
    """
    Calculates the average velocity of a body part, either within specific ROIs or overall.

    Args:
        body_part_data (dict): Dictionary containing x and y coordinates for each body part.
        roi_definitions (dict): Dictionary defining ROIs with name and coordinates.
        df (DataFrame): DataFrame containing all the data.
        body_part (str): The name of the body part to analyze.
        roi_name (str, optional): The name of the ROI to analyze. If None, calculates overall velocity.
        frame_rate (float, optional): The frame rate of the video. If provided, velocity is in real-world units.
        timestamps (list, optional): List of timestamps for each frame.

    Returns:
        dict or float: A dictionary where keys are ROI names and values are the average velocities, or a float for overall velocity.
    """
    try:
        x_coords = body_part_data[body_part]["x"]
        y_coords = body_part_data[body_part]["y"]
    except KeyError:
        logging.error(f"Body part '{body_part}' not found in data.")
        return None

    if timestamps is not None:
        if len(timestamps) != len(df):
            logging.error("Length of timestamps does not match the number of frames.")
            return None
        time_intervals = np.diff(timestamps)
    elif frame_rate:
        time_intervals = np.ones(len(df) - 1) / frame_rate
    else:
        logging.warning("Frame rate not provided. Assuming each frame is one unit of time.")
        time_intervals = np.ones(len(df) - 1)

    if roi_name is None:
        distances = np.array([calculate_distance([x_coords[i], y_coords[i]], [x_coords[i+1], y_coords[i+1]]) for i in range(len(x_coords) - 1)])
        if not distances.size:
            logging.warning("No movement detected for the body part.")
            return 0
        total_distance = np.sum(distances)
        total_time = np.sum(time_intervals)
        average_velocity = total_distance / total_time
        return average_velocity
    else:
        results = {}
        for roi_name_def, roi in roi_definitions.items():
            distances = []
            for i in range(len(x_coords) - 1):
                dist = calculate_distance([x_coords[i], y_coords[i]], [x_coords[i+1], y_coords[i+1]])
                if is_in_roi(x_coords[i], y_coords[i], roi) or is_in_roi(x_coords[i+1], y_coords[i+1], roi):
                    distances.append(dist)
            if not distances:
                logging.warning(f"No movement detected within ROI '{roi_name_def}'.")
                results[roi_name_def] = 0
            else:
                total_distance = np.sum(distances)
                total_time = np.sum(time_intervals[:len(distances)])
                average_velocity = total_distance / total_time
                results[roi_name_def] = average_velocity
        return results

if __name__ == '__main__':
    # Example usage:
    import pandas as pd
    import numpy as np
    import json

    csv_file = "test.csv"
    roi_definitions = {'center': [[20, 10], [40, 10], [40, 30], [20, 30]]}
    frame_rate = 30  # Example frame rate
    timestamps = np.arange(0, 10, 1/frame_rate) # Example timestamps

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

    avg_velocity_overall = calculate_average_velocity(body_part_data, roi_definitions, df, "Nose", frame_rate=frame_rate, timestamps=timestamps)
    avg_velocity_in_roi = calculate_average_velocity(body_part_data, roi_definitions, df, "Nose", "center", frame_rate=frame_rate, timestamps=timestamps)
    print(f"Average velocity (overall): {avg_velocity_overall}")
    print(f"Average velocity in 'center' ROI: {avg_velocity_in_roi}")