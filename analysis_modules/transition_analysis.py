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

def analyze_roi_transitions(body_part_data, roi_definitions, df, body_part, rois, frame_threshold=3):
    """
    Analyzes transitions between specified ROIs based on entries and exits,
    allowing for transitions even if ROIs are not directly adjacent.

    Args:
        body_part_data (dict): Dictionary containing body part coordinates.
        roi_definitions (dict): Dictionary of ROI definitions.
        df (pd.DataFrame): DataFrame containing the DeepLabCut data.
        body_part (str): Body part to track for transitions.
        rois (list): List of ROI names to analyze transitions between.
        frame_threshold (int): Number of consecutive frames required to confirm an entry or exit.

    Returns:
        dict: Dictionary with transition counts between ROIs (e.g., {('roi1', 'roi2'): 5}).
    """
    if not isinstance(frame_threshold, int) or frame_threshold <= 0:
        logging.error("Invalid frame threshold. Please provide a positive integer.")
        return {}

    if body_part not in body_part_data:
        logging.warning(f"Body part '{body_part}' not found in data. Returning empty dict")
        return {}

    x_coords = body_part_data[body_part]["x"]
    y_coords = body_part_data[body_part]["y"]

    transition_counts = {}  # Store counts of transitions
    in_roi = {roi: False for roi in rois}  # Track if the body part is currently in each ROI
    entry_start_frames = {roi: None for roi in rois} # Track the start frame of each entry
    exit_start_frames = {roi: None for roi in rois} # Track the start frame of each exit
    last_roi = None # Track the last ROI the animal was in

    for i in range(len(x_coords)):
        x, y = x_coords[i], y_coords[i]
        current_roi = None
        for roi_name in rois:
            if roi_name not in roi_definitions:
                logging.warning(f"ROI '{roi_name}' not found in ROI definitions. Skipping.")
                continue
            roi_coords = roi_definitions[roi_name]
            if is_in_roi(x, y, roi_coords):
                current_roi = roi_name
                break

        if current_roi and not in_roi[current_roi]:
            # Entry detected
            if entry_start_frames[current_roi] is None:
                entry_start_frames[current_roi] = i
            elif i - entry_start_frames[current_roi] >= frame_threshold:
                in_roi[current_roi] = True
                entry_start_frames[current_roi] = None
                if last_roi and last_roi != current_roi:
                    transition = (last_roi, current_roi)
                    if transition in transition_counts:
                        transition_counts[transition] += 1
                    else:
                        transition_counts[transition] = 1
                last_roi = current_roi
        elif not current_roi:
            # Exit detected
            for roi_name in rois:
                if in_roi[roi_name]:
                    if exit_start_frames[roi_name] is None:
                        exit_start_frames[roi_name] = i
                    elif i - exit_start_frames[roi_name] >= frame_threshold:
                        in_roi[roi_name] = False
                        exit_start_frames[roi_name] = None
                        last_roi = roi_name
    logging.info(f"Transition analysis complete for body part: {body_part} between ROIs: {rois}")
    return transition_counts

def test_is_in_roi():
    """Tests the is_in_roi function with sample data."""
    roi1_coords = [[20, 10], [40, 10], [40, 30], [20, 30]]
    roi2_coords = [[60, 10], [80, 10], [80, 30], [60, 30]]

    print("Testing is_in_roi function:")
    print(f"Point (30, 20) in roi1: {is_in_roi(30, 20, roi1_coords)}")  # Should be True
    print(f"Point (70, 20) in roi2: {is_in_roi(70, 20, roi2_coords)}")  # Should be True
    print(f"Point (10, 10) in roi1: {is_in_roi(10, 10, roi1_coords)}")  # Should be False
    print(f"Point (50, 10) in roi2: {is_in_roi(50, 10, roi2_coords)}")  # Should be False
    print(f"Point (30, 40) in roi1: {is_in_roi(30, 40, roi1_coords)}")  # Should be False
    print(f"Point (70, 40) in roi2: {is_in_roi(70, 40, roi2_coords)}")  # Should be False

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
    transition_results = analyze_roi_transitions(body_part_data, roi_definitions, df, body_part_to_analyze, rois_to_analyze)
    print(f"Transition counts: {transition_results}")
    test_is_in_roi()