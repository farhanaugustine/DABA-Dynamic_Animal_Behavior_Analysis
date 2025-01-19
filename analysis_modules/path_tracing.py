import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

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

def trace_path(body_part_data, roi_definitions, df, body_part):
    """
    Traces the path of a given body part and identifies time spent in ROIs.
    Args:
        body_part_data (dict): Dictionary containing body part coordinates.
        roi_definitions (dict): Dictionary of ROI definitions.
        df (pd.DataFrame): DataFrame containing the DeepLabCut data.
        body_part (str): Body part to track the path for.
    Returns:
         tuple (list, dict): A list of (x, y) coordinates for the body part path and a dictionary for time spent in ROIs.
    """
    if body_part not in body_part_data:
        print(f"Warning: Body part '{body_part}' not found in data. Returning empty path and empty dictionary.")
        return [], {}

    x_coords = body_part_data[body_part]["x"]
    y_coords = body_part_data[body_part]["y"]
    path = list(zip(x_coords, y_coords))
    time_in_rois = {}  # Store time spent in each ROI

    current_roi = None
    for i, (x, y) in enumerate(path):
        new_roi = None
        for roi_name in roi_definitions:
            if is_in_roi(x, y, roi_definitions[roi_name]):
                new_roi = roi_name
                break
        if new_roi and current_roi and new_roi == current_roi:
            time_in_rois[new_roi] += 1
        elif new_roi and current_roi and new_roi != current_roi:
            current_roi = new_roi
            if new_roi in time_in_rois:
                time_in_rois[new_roi] += 1
            else:
                time_in_rois[new_roi] = 1
        elif new_roi and not current_roi:
            current_roi = new_roi
            if new_roi in time_in_rois:
                time_in_rois[new_roi] += 1
            else:
                time_in_rois[new_roi] = 1
    return path, time_in_rois


def graph_path(path, video_path, roi_definitions, body_part, time_in_rois, output_path="path_plot.png"):
    """Graphs the traced path on a frame from the video, including ROIs and time spent labels.
    Args:
      path (list): List of (x,y) coordinates of the path.
      video_path (str): Path to the video file.
      roi_definitions (dict): Dictionary of ROI definitions.
      body_part (str): Body part being tracked.
      time_in_rois (dict): Dictionary with time spent in each ROI.
      output_path (str): Path to save the resulting image file.
    """
    if not path:
        print("Warning: no path to display, skipping graph")
        return
    # Get first frame from the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")
    try:
        ret, frame = cap.read()
        if not ret:
            raise IOError("Error reading video frame")
    finally:
        cap.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)
    # Plot the path as a line
    x, y = zip(*path)
    plt.plot(x, y, marker='.', markersize=1, linestyle='-', linewidth=0.5, label=f'{body_part} Path')

    # Draw ROIs and labels on the frame
    for roi_name, roi_points in roi_definitions.items():
        roi_points = np.array(roi_points)
        polygon = plt.Polygon(roi_points, closed=True, fill=False, edgecolor='green', linewidth=2)
        plt.gca().add_patch(polygon)
        text_x = roi_points[0][0]
        text_y = roi_points[0][1] - 10 if roi_points[0][1] > 20 else roi_points[0][1] + 20
        plt.text(text_x, text_y, f"{roi_name}: {time_in_rois.get(roi_name, 0)} frames", color='green', fontsize=8)

    plt.title(f"Path of {body_part} with Time Spent in ROIs")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.savefig(output_path)
    plt.close()
    print(f"Path graph saved to {output_path}")


def draw_path_on_video(path, video_path, roi_definitions, body_part, output_video_path="path_video.mp4"):
    """Draws the path on the video frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    try:
      for i, (x, y) in enumerate(path):
        ret, frame = cap.read()
        if not ret:
            print("Warning: Reached end of the video. Some of path might not be drawn.")
            break
        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)
        # Draw the path on the frame
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        for roi_name, roi_points in roi_definitions.items():
            roi_points = np.array(roi_points, dtype=np.int32)
            cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=1)
            text_x = roi_points[0][0]
            text_y = roi_points[0][1] - 10 if roi_points[0][1] > 20 else roi_points[0][1] + 20
            cv2.putText(frame, roi_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)


        # Write the modified frame to the output video
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    except Exception as e:
        print(f"Error during video drawing: {e}")
    finally:
        cap.release()
        out.release()
        print (f"Path tracing video saved to {output_video_path}")


def process_path_tracing(body_part_data, roi_definitions, df, body_part, video_path):
    """
    Orchestrates the path tracing process, including plotting and video drawing based on user choice.
    Args:
        body_part_data (dict): Dictionary containing body part coordinates.
        roi_definitions (dict): Dictionary of ROI definitions.
        df (pd.DataFrame): DataFrame containing the DeepLabCut data.
        body_part (str): Body part to track the path for.
        video_path (str): Path to the video file.
    """
    path, time_in_rois = trace_path(body_part_data, roi_definitions, df, body_part)

    if not path:
        return

    # Ask the user if they want video drawing
    user_choice = input("Do you want to draw the path on the video frame by frame? (yes/no): ").strip().lower()

    if user_choice == 'yes':
      draw_path_on_video(path, video_path, roi_definitions, body_part)
    else:
      graph_path(path, video_path, roi_definitions, body_part, time_in_rois)


if __name__ == '__main__':
    # Example usage:
    import pandas as pd
    import json
    import os
    # Create dummy csv file for testing
    csv_file = "test.csv"
    roi_definitions = {
        'roi1': [[20, 10], [40, 10], [40, 30], [20, 30]],
        'roi2': [[60, 10], [80, 10], [80, 30], [60, 30]]
    }
    video_path = "test.mp4"
    #Create a dummy video file for testing
    try:
       fourcc = cv2.VideoWriter_fourcc(*'mp4v')
       frame_rate = 30
       width = 100
       height = 100
       out = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
       for _ in range(10):
           frame = np.zeros((height, width, 3), dtype=np.uint8)
           out.write(frame)
       out.release()
    except IOError as e:
        print(f"Error creating a dummy video for testing: {e}")
    # Create a dummy csv file
    data = {
    ('bodyparts', 'Nose', 'x'): [25, 30, 35, 65, 70, 75, 70, 65, 35, 30],
    ('bodyparts', 'Nose', 'y'): [15, 20, 25, 15, 20, 25, 20, 15, 25, 20],
    ('bodyparts', 'Nose', 'likelihood'): [0.9, 0.8, 0.7, 0.9, 0.8, 0.7, 0.9, 0.8, 0.7, 0.9]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file)
    # Load the DeepLabCut CSV file, using the second and third rows as the header
    df = pd.read_csv(csv_file, header=[0, 1])

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
    process_path_tracing(body_part_data, roi_definitions, df, body_part_to_analyze, video_path)
    os.remove(csv_file)
    os.remove(video_path)