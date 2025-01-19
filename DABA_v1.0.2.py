import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import os
import pandas as pd
import numpy as np
import json
import cv2
from tabulate import tabulate
import re
import math
import yaml
import logging
import multiprocessing
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import threading
import asyncio

# --- Helper Functions ---
def calculate_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[0])**2)

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

def calculate_total_time_in_roi(roi_time_spent):
    """Calculates the total time spent in ROIs."""
    total_time = 0
    for key in roi_time_spent:
        total_time += roi_time_spent[key]
    return total_time

def extract_frame_from_video(video_path, frame_number=0):
    """Extracts a single frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            raise IOError("Error reading frame from video")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def draw_rois(video_path, roi_definitions):
    """Draws ROIs on a single frame from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")
    try:
        ret, frame = cap.read()
        if not ret:
            raise IOError("Error reading video frame")

        frame_with_rois = frame.copy()

        for roi_name, roi_points in roi_definitions.items():
            roi_points = np.array(roi_points, dtype=np.int32)
            cv2.polylines(frame_with_rois, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)
            text_x = roi_points[0][0]
            text_y = roi_points[0][1] - 10 if roi_points[0][1] > 20 else roi_points[0][1] + 20
            cv2.putText(frame_with_rois, roi_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame_with_rois
    finally:
        cap.release()

def get_roi_points(video_path):
    """Allows the user to draw ROIs interactively on a single frame from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")
    try:
        ret, frame = cap.read()
        if not ret:
            raise IOError("Error reading video frame")

        roi_points = {}
        temp_roi = []
        roi_name = None
        drawing = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal temp_roi, roi_name, drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                temp_roi.append((x, y))
                drawing = True
            elif event == cv2.EVENT_RBUTTONDOWN:
                if temp_roi:
                    cv2.destroyWindow("Select ROIs")
                    roi_name = input("Enter ROI name: ")
                    roi_points[roi_name] = temp_roi
                    temp_roi = []
                    drawing = False
                    cv2.namedWindow("Select ROIs")
                    cv2.setMouseCallback("Select ROIs", mouse_callback)
            elif event == cv2.EVENT_MBUTTONDOWN:
                if temp_roi:
                    temp_roi.pop()

        cv2.namedWindow("Select ROIs")
        cv2.setMouseCallback("Select ROIs", mouse_callback)

        while True:
            frame_copy = frame.copy()
            if temp_roi:
                temp_roi_np = np.array(temp_roi, dtype=np.int32)
                cv2.polylines(frame_copy, [temp_roi_np], isClosed=False, color=(0, 255, 0), thickness=2)
            for name, points in roi_points.items():
                points_np = np.array(points, dtype=np.int32)
                cv2.polylines(frame_copy, [points_np], isClosed=True, color=(0, 255, 0), thickness=2)
                text_x = points[0][0]
                text_y = points[0][1] - 10 if points[0][1] > 20 else points[0][1] + 20
                cv2.putText(frame_copy, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Select ROIs", frame_copy)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter key
                if temp_roi:
                    cv2.destroyWindow("Select ROIs")  # Close the OpenCV window
                    roi_name = input("Enter ROI name: ")
                    roi_points[roi_name] = temp_roi
                    temp_roi = []
                    drawing = False
                    cv2.namedWindow("Select ROIs")  # Recreate the window
                    cv2.setMouseCallback("Select ROIs", mouse_callback)
            elif key == ord('c'):  # 'c' key to clear current ROI
                temp_roi = []
                drawing = False
            elif key == ord('q'):
                break

        return roi_points
    finally:
        cv2.destroyAllWindows()
        cap.release()

def save_rois(roi_definitions, json_path):
    """Saves the roi_definitions dictionary to a JSON file."""
    try:
        with open(json_path, 'w') as f:
            json.dump(roi_definitions, f, indent=4)
    except IOError as e:
        logging.error(f"Error saving ROI definitions to JSON: {e}")

def load_rois(json_path):
    """Loads ROI definitions from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: ROI definitions file not found at {json_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing ROI definitions file: {e}")
        return None

# Modified analysis_map to use the generalized modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis_modules'))
from analysis_modules import distance_in_roi, time_in_roi, average_velocity, multi_bodypart_analysis, entry_counter, speed_per_entry, event_duration, novel_object, transition_analysis, zone_preference, path_tracing
# Import the code generator module
from analysis_modules import code_generator

analysis_map = {
    "time_in_roi": time_in_roi.analyze_time_in_roi,
    "distance_in_roi": distance_in_roi.analyze_distance_in_roi,
    "average_velocity": average_velocity.calculate_average_velocity,
    "multi_bodypart_analysis": multi_bodypart_analysis.analyze_multi_bodypart_entries,
    "entry_counter": entry_counter.count_entries,
    "speed_per_entry": speed_per_entry.calculate_speed_per_entry,
    "exit_counter": entry_counter.count_exits,
    "event_duration": event_duration.analyze_freezing,
    "novel_object": novel_object.analyze_novel_object_interaction,
    "transition_analysis": transition_analysis.analyze_roi_transitions,
    "zone_preference": zone_preference.analyze_zone_preference,
    "path_tracing": path_tracing.process_path_tracing # Updated reference to the path tracing orchestrator
}

import yaml
import json

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Error: Configuration file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        return {}

def get_user_input(prompt, options=None, default=None):
    """Gets user input with optional options and default."""
    if options:
        print(prompt)
        for i, option in enumerate(options):
            print(f"  {i+1}. {option}")
        while True:
            choice = input(f"Enter your choice (or press Enter for default '{default}'): ")
            if not choice and default:
                return default
            try:
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(options):
                    return options[choice_index]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number or press Enter for default.")
    else:
        return input(f"{prompt} (or press Enter for default '{default}'): ") or default

class DABA_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DABA - Dynamic Animal Behavior Analysis")

        # --- Load Configuration ---
        self.config = load_config()

        # Access settings
        self.roi_file_path = self.config.get("roi_file_path", "rois.json")
        log_level = self.config.get("log_level", "INFO")
        log_file = self.config.get("log_file", "behavior_analysis.log")
        self.ollama_url = self.config.get("ollama_url", "http://localhost:11434")

        # Configure logging
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file)

        # --- Variables ---
        self.csv_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.roi_definitions = None
        self.df = None
        self.body_part_data = None
        self.frame = None
        self.analysis_type = tk.StringVar(value="standard")
        self.selected_analysis = tk.StringVar()
        self.frame_rate = tk.DoubleVar()
        self.frame_threshold = tk.IntVar()
        self.object_positions = tk.StringVar()
        self.interaction_threshold = tk.DoubleVar()
        self.exploration_threshold = tk.DoubleVar()
        self.rois_to_analyze = tk.StringVar()
        self.body_part_to_analyze = tk.StringVar()
        self.code_generation_prompt = tk.StringVar()
        self.code_generation_output_folder = tk.StringVar()

        # --- GUI Layout ---
        self.create_widgets()

    def create_widgets(self):
        # --- File Input Frame ---
        file_frame = ttk.LabelFrame(self.root, text="File Input")
        file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(file_frame, text="DeepLabCut CSV File:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(file_frame, textvariable=self.csv_path, width=50).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_csv).grid(row=0, column=2, sticky="w", padx=5, pady=5)

        ttk.Label(file_frame, text="Behavior Video File:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(file_frame, textvariable=self.video_path, width=50).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_video).grid(row=1, column=2, sticky="w", padx=5, pady=5)

        # --- ROI Frame ---
        roi_frame = ttk.LabelFrame(self.root, text="ROI Definition")
        roi_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        ttk.Button(roi_frame, text="Define ROIs", command=self.define_rois).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(roi_frame, text="Load ROIs", command=self.load_rois_from_file).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(roi_frame, text="Save ROIs", command=self.save_rois_to_file).grid(row=0, column=2, padx=5, pady=5)

        # --- Analysis Selection Frame ---
        analysis_frame = ttk.LabelFrame(self.root, text="Analysis Selection")
        analysis_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(analysis_frame, text="Analysis Type:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        analysis_options = ["standard", "code"]
        analysis_dropdown = ttk.Combobox(analysis_frame, textvariable=self.analysis_type, values=analysis_options, state="readonly")
        analysis_dropdown.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        analysis_dropdown.bind("<<ComboboxSelected>>", self.update_analysis_options)

        # --- Standard Analysis Options Frame ---
        self.standard_analysis_options_frame = ttk.LabelFrame(analysis_frame, text="Standard Analysis Options")
        self.standard_analysis_options_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.standard_analysis_options_frame.grid_remove()

        ttk.Label(self.standard_analysis_options_frame, text="Analysis:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.analysis_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.selected_analysis, values=list(analysis_map.keys()), state="readonly")
        self.analysis_dropdown.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.analysis_dropdown.bind("<<ComboboxSelected>>", self.update_standard_analysis_options)

        self.standard_analysis_options_widgets = {}

        # --- Code Generation Options Frame ---
        self.code_generation_options_frame = ttk.LabelFrame(analysis_frame, text="Code Generation Options")
        self.code_generation_options_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.code_generation_options_frame.grid_remove()

        ttk.Label(self.code_generation_options_frame, text="Analysis Description:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.code_generation_prompt_entry = ttk.Entry(self.code_generation_options_frame, textvariable=self.code_generation_prompt, width=50)
        self.code_generation_prompt_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(self.code_generation_options_frame, text="Output Folder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.code_generation_output_folder_entry = ttk.Entry(self.code_generation_options_frame, textvariable=self.code_generation_output_folder, width=50)
        self.code_generation_output_folder_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # --- Output Frame ---
        output_frame = ttk.LabelFrame(self.root, text="Output")
        output_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

        self.output_text_area = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=80, height=15)
        self.output_text_area.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)

        # --- Analyze Button ---
        ttk.Button(self.root, text="Analyze", command=self.analyze).grid(row=5, column=0, pady=10)

        # --- Configure Grid Weights ---
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(4, weight=1)

    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.csv_path.set(file_path)
        self.load_data()  # Load data after CSV is selected

    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        self.video_path.set(file_path)

    def define_rois(self):
        try:
            if not self.video_path.get():
                self.output_text_area.insert(tk.END, "Error: Please select a video file first.\n")
                return
            self.roi_definitions = get_roi_points(self.video_path.get())
            if self.roi_definitions:
                table_data = []
                for roi_name, coords in self.roi_definitions.items():
                    table_data.append([roi_name, coords])
                table_str = tabulate(table_data, headers=["ROI Name", "Coordinates"])
                self.output_text_area.insert(tk.END, f"\nROI Definitions:\n{table_str}\n")
            else:
                self.output_text_area.insert(tk.END, "\nNo ROIs were defined.\n")
        except Exception as e:
            self.output_text_area.insert(tk.END, f"Error defining ROIs: {e}\n")

    def load_rois_from_file(self):
        self.roi_definitions = load_rois(self.roi_file_path)
        if self.roi_definitions:
            table_data = []
            for roi_name, coords in self.roi_definitions.items():
                table_data.append([roi_name, coords])
            table_str = tabulate(table_data, headers=["ROI Name", "Coordinates"])
            self.output_text_area.insert(tk.END, f"ROIs loaded from file:\n{table_str}\n")
        else:
            self.output_text_area.insert(tk.END, "Error: Could not load ROIs from file.\n")

    def save_rois_to_file(self):
        if self.roi_definitions:
            save_rois(self.roi_definitions, self.roi_file_path)
            self.output_text_area.insert(tk.END, "ROIs saved to file.\n")
        else:
            self.output_text_area.insert(tk.END, "Error: No ROIs defined to save.\n")

    def update_analysis_options(self, event=None):
        if self.analysis_type.get() == "standard":
            self.standard_analysis_options_frame.grid()
            self.code_generation_options_frame.grid_remove()
            # Trigger update of body part dropdowns when switching to standard analysis
            self.update_standard_analysis_options()
        elif self.analysis_type.get() == "code":
            self.standard_analysis_options_frame.grid_remove()
            self.code_generation_options_frame.grid()

    def update_standard_analysis_options(self, event=None):
        selected_analysis = self.selected_analysis.get()
        for widget in self.standard_analysis_options_widgets.values():
            widget.destroy()
        self.standard_analysis_options_widgets = {}

        if selected_analysis == "average_velocity":
            ttk.Label(self.standard_analysis_options_frame, text="Body Part:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            body_part_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.body_part_to_analyze, values=self.get_body_parts(), state="readonly")
            body_part_dropdown.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["body_part"] = body_part_dropdown

            ttk.Label(self.standard_analysis_options_frame, text="Frame Rate:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            frame_rate_entry = ttk.Entry(self.standard_analysis_options_frame, textvariable=self.frame_rate, width=10)
            frame_rate_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
            self.standard_analysis_options_widgets["frame_rate"] = frame_rate_entry

            ttk.Label(self.standard_analysis_options_frame, text="ROI:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
            roi_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.rois_to_analyze, values=list(self.roi_definitions.keys()) if self.roi_definitions else [], state="readonly")
            roi_dropdown.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["roi"] = roi_dropdown

        elif selected_analysis in ["time_in_roi", "distance_in_roi"]:
            ttk.Label(self.standard_analysis_options_frame, text="Body Part:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            body_part_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.body_part_to_analyze, values=self.get_body_parts(), state="readonly")
            body_part_dropdown.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["body_part"] = body_part_dropdown

            ttk.Label(self.standard_analysis_options_frame, text="ROI:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            roi_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.rois_to_analyze, values=list(self.roi_definitions.keys()) if self.roi_definitions else [], state="readonly")
            roi_dropdown.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["roi"] = roi_dropdown

        elif selected_analysis in ["entry_counter", "exit_counter"]:
            ttk.Label(self.standard_analysis_options_frame, text="Body Part:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            body_part_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.body_part_to_analyze, values=self.get_body_parts(), state="readonly")
            body_part_dropdown.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["body_part"] = body_part_dropdown

            ttk.Label(self.standard_analysis_options_frame, text="ROI:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            roi_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.rois_to_analyze, values=list(self.roi_definitions.keys()) if self.roi_definitions else [], state="readonly")
            roi_dropdown.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["roi"] = roi_dropdown

            ttk.Label(self.standard_analysis_options_frame, text="Frame Threshold:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
            frame_threshold_entry = ttk.Entry(self.standard_analysis_options_frame, textvariable=self.frame_threshold, width=10)
            frame_threshold_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
            self.standard_analysis_options_widgets["frame_threshold"] = frame_threshold_entry

        elif selected_analysis == "speed_per_entry":
            ttk.Label(self.standard_analysis_options_frame, text="Body Part:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            body_part_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.body_part_to_analyze, values=self.get_body_parts(), state="readonly")
            body_part_dropdown.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["body_part"] = body_part_dropdown

            ttk.Label(self.standard_analysis_options_frame, text="ROI:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            roi_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.rois_to_analyze, values=list(self.roi_definitions.keys()) if self.roi_definitions else [], state="readonly")
            roi_dropdown.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["roi"] = roi_dropdown

        elif selected_analysis == "event_duration":
            ttk.Label(self.standard_analysis_options_frame, text="Body Part:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            body_part_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.body_part_to_analyze, values=self.get_body_parts(), state="readonly")
            body_part_dropdown.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["body_part"] = body_part_dropdown

            ttk.Label(self.standard_analysis_options_frame, text="Frame Threshold:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            frame_threshold_entry = ttk.Entry(self.standard_analysis_options_frame, textvariable=self.frame_threshold, width=10)
            frame_threshold_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
            self.standard_analysis_options_widgets["frame_threshold"] = frame_threshold_entry

            ttk.Label(self.standard_analysis_options_frame, text="ROI:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
            roi_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.rois_to_analyze, values=list(self.roi_definitions.keys()) if self.roi_definitions else [], state="readonly")
            roi_dropdown.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["roi"] = roi_dropdown

        elif selected_analysis == "novel_object":
            ttk.Label(self.standard_analysis_options_frame, text="Body Part:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            body_part_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.body_part_to_analyze, values=self.get_body_parts(), state="readonly")
            body_part_dropdown.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["body_part"] = body_part_dropdown

            ttk.Label(self.standard_analysis_options_frame, text="Object Positions:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            object_positions_entry = ttk.Entry(self.standard_analysis_options_frame, textvariable=self.object_positions, width=50)
            object_positions_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["object_positions"] = object_positions_entry

            ttk.Label(self.standard_analysis_options_frame, text="Interaction Threshold:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
            interaction_threshold_entry = ttk.Entry(self.standard_analysis_options_frame, textvariable=self.interaction_threshold, width=10)
            interaction_threshold_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
            self.standard_analysis_options_widgets["interaction_threshold"] = interaction_threshold_entry

            ttk.Label(self.standard_analysis_options_frame, text="Exploration Threshold:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
            exploration_threshold_entry = ttk.Entry(self.standard_analysis_options_frame, textvariable=self.exploration_threshold, width=10)
            exploration_threshold_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
            self.standard_analysis_options_widgets["exploration_threshold"] = exploration_threshold_entry

            ttk.Label(self.standard_analysis_options_frame, text="Frame Threshold:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
            frame_threshold_entry = ttk.Entry(self.standard_analysis_options_frame, textvariable=self.frame_threshold, width=10)
            frame_threshold_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)
            self.standard_analysis_options_widgets["frame_threshold"] = frame_threshold_entry

        elif selected_analysis == "transition_analysis":
            ttk.Label(self.standard_analysis_options_frame, text="Body Part:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            body_part_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.body_part_to_analyze, values=self.get_body_parts(), state="readonly")
            body_part_dropdown.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["body_part"] = body_part_dropdown

            ttk.Label(self.standard_analysis_options_frame, text="ROIs (comma-separated):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            rois_entry = ttk.Entry(self.standard_analysis_options_frame, textvariable=self.rois_to_analyze, width=50)
            rois_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["rois"] = rois_entry

            ttk.Label(self.standard_analysis_options_frame, text="Frame Threshold:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
            frame_threshold_entry = ttk.Entry(self.standard_analysis_options_frame, textvariable=self.frame_threshold, width=10)
            frame_threshold_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
            self.standard_analysis_options_widgets["frame_threshold"] = frame_threshold_entry

        elif selected_analysis == "zone_preference":
            ttk.Label(self.standard_analysis_options_frame, text="Body Part:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            body_part_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.body_part_to_analyze, values=self.get_body_parts(), state="readonly")
            body_part_dropdown.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["body_part"] = body_part_dropdown

            ttk.Label(self.standard_analysis_options_frame, text="ROIs (comma-separated):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            rois_entry = ttk.Entry(self.standard_analysis_options_frame, textvariable=self.rois_to_analyze, width=50)
            rois_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["rois"] = rois_entry

        elif selected_analysis == "path_tracing":
            ttk.Label(self.standard_analysis_options_frame, text="Body Part:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            body_part_dropdown = ttk.Combobox(self.standard_analysis_options_frame, textvariable=self.body_part_to_analyze, values=self.get_body_parts(), state="readonly")
            body_part_dropdown.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["body_part"] = body_part_dropdown
        elif selected_analysis == "multi_bodypart_analysis":
            ttk.Label(self.standard_analysis_options_frame, text="Body Parts (comma-separated):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            body_parts_entry = ttk.Entry(self.standard_analysis_options_frame, textvariable=self.body_part_to_analyze, width=50)
            body_parts_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["body_parts"] = body_parts_entry

            ttk.Label(self.standard_analysis_options_frame, text="ROI:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            roi_entry = ttk.Entry(self.standard_analysis_options_frame, textvariable=self.rois_to_analyze, width=50)
            roi_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
            self.standard_analysis_options_widgets["roi"] = roi_entry

    def get_body_parts(self):
        if self.df is not None:
            unique_body_parts = self.df.columns.get_level_values(0).unique().tolist()
            if 'bodyparts' in unique_body_parts:
                unique_body_parts.remove('bodyparts')
            return unique_body_parts
        return []

    def load_data(self):
        if not self.csv_path.get():
            return
        try:
            self.df = pd.read_csv(self.csv_path.get(), header=[1, 2])
            self.frame = extract_frame_from_video(self.video_path.get()) if self.video_path.get() else None
            self.body_part_data = {}
            unique_body_parts = self.df.columns.get_level_values(0).unique().tolist()
            if 'bodyparts' in unique_body_parts:
                unique_body_parts.remove('bodyparts')

            for part in unique_body_parts:
                self.body_part_data[part] = {
                    "x": self.df.loc[:, (part, 'x')].to_numpy(),
                    "y": self.df.loc[:, (part, 'y')].to_numpy(),
                    "likelihood": self.df.loc[:, (part, 'likelihood')].to_numpy()
                }

            # Update body part dropdowns after loading CSV
            for widget_name, widget in self.standard_analysis_options_widgets.items():
                if widget_name == "body_part" and isinstance(widget, ttk.Combobox):
                    widget['values'] = self.get_body_parts()

        except Exception as e:
            self.output_text_area.insert(tk.END, f"Error loading data: {e}\n")
            return

    def analyze(self):
        self.output_text_area.delete(1.0, tk.END)
        if not self.csv_path.get() or not self.video_path.get():
            self.output_text_area.insert(tk.END, "Error: Please select both CSV and video files.\n")
            return

        if self.df is None or self.body_part_data is None:
            self.load_data()
            if self.df is None or self.body_part_data is None:
                self.output_text_area.insert(tk.END, "Error: Could not load data. Please check CSV file.\n")
                return

        if self.roi_definitions is None:
            self.roi_definitions = load_rois(self.roi_file_path)
            if self.roi_definitions:
                table_data = []
                for roi_name, coords in self.roi_definitions.items():
                    table_data.append([roi_name, coords])
                table_str = tabulate(table_data, headers=["ROI Name", "Coordinates"])
                self.output_text_area.insert(tk.END, f"ROIs loaded from file:\n{table_str}\n")
            else:
                self.output_text_area.insert(tk.END, "Error: No ROIs defined or loaded.\n")
                return

        if self.analysis_type.get() == "standard":
            self.run_standard_analysis()
        elif self.analysis_type.get() == "code":
            self.run_code_generation()

    def run_standard_analysis(self):
        selected_analysis = self.selected_analysis.get()
        if not selected_analysis:
            self.output_text_area.insert(tk.END, "Error: Please select an analysis type.\n")
            return

        analysis_function = analysis_map.get(selected_analysis)
        if not analysis_function:
            self.output_text_area.insert(tk.END, f"Error: Invalid analysis type: {selected_analysis}\n")
            return

        try:
            if selected_analysis == "average_velocity":
                body_part_to_analyze = self.body_part_to_analyze.get()
                frame_rate = self.frame_rate.get() if self.frame_rate.get() else None
                roi_to_analyze = self.rois_to_analyze.get() if self.rois_to_analyze.get() else None
                if not body_part_to_analyze:
                    self.output_text_area.insert(tk.END, "Error: Please select a body part.\n")
                    return
                if roi_to_analyze:
                    result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_part_to_analyze, roi_to_analyze, frame_rate=frame_rate)
                else:
                    result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_part_to_analyze, frame_rate=frame_rate)
                if isinstance(result, dict):
                    for roi, avg_vel in result.items():
                        self.output_text_area.insert(tk.END, f"Average velocity in {roi}: {avg_vel:.2f}\n")
                else:
                    self.output_text_area.insert(tk.END, f"Result of {selected_analysis}: {result}\n")
            elif selected_analysis in ["time_in_roi", "distance_in_roi"]:
                body_part_to_analyze = self.body_part_to_analyze.get()
                roi_to_analyze = self.rois_to_analyze.get() if self.rois_to_analyze.get() else None
                if not body_part_to_analyze:
                    self.output_text_area.insert(tk.END, "Error: Please select a body part.\n")
                    return
                if roi_to_analyze:
                    result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_part_to_analyze, roi_to_analyze)
                else:
                    result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_part_to_analyze)
                if isinstance(result, dict):
                    for roi, value in result.items():
                        self.output_text_area.insert(tk.END, f"Result of {selected_analysis} in {roi}: {value}\n")
                else:
                    self.output_text_area.insert(tk.END, f"Result of {selected_analysis}: {result}\n")
            elif selected_analysis in ["entry_counter", "exit_counter"]:
                body_part_to_analyze = self.body_part_to_analyze.get()
                roi_to_analyze = self.rois_to_analyze.get()
                frame_threshold = self.frame_threshold.get()
                if not body_part_to_analyze or not roi_to_analyze:
                    self.output_text_area.insert(tk.END, "Error: Please select a body part and ROI.\n")
                    return
                result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_part_to_analyze, roi_to_analyze, frame_threshold)
                self.output_text_area.insert(tk.END, f"Result of {selected_analysis}: {result}\n")
            elif selected_analysis == "speed_per_entry":
                body_part_to_analyze = self.body_part_to_analyze.get()
                roi_to_analyze = self.rois_to_analyze.get()
                if not body_part_to_analyze or not roi_to_analyze:
                    self.output_text_area.insert(tk.END, "Error: Please select a body part and ROI.\n")
                    return
                result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_part_to_analyze, roi_to_analyze)
                if isinstance(result, dict):
                    for roi, speeds in result.items():
                        formatted_speeds = ", ".join([f"{speed:.2f}" for speed in speeds])
                        self.output_text_area.insert(tk.END, f"Result of {selected_analysis} in {roi}: [{formatted_speeds}]\n")
                else:
                    self.output_text_area.insert(tk.END, f"Result of {selected_analysis}: {result}\n")
            elif selected_analysis == "event_duration":
                body_part_to_analyze = self.body_part_to_analyze.get()
                frame_threshold = self.frame_threshold.get()
                roi_to_analyze = self.rois_to_analyze.get() if self.rois_to_analyze.get() else None
                if not body_part_to_analyze:
                    self.output_text_area.insert(tk.END, "Error: Please select a body part.\n")
                    return
                if roi_to_analyze:
                    result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_part_to_analyze, frame_threshold, roi_name=roi_to_analyze)
                else:
                    result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_part_to_analyze, frame_threshold)
                if isinstance(result, dict):
                    for body_part, duration in result.items():
                        self.output_text_area.insert(tk.END, f"Freezing duration for {body_part}: {duration}\n")
                else:
                    self.output_text_area.insert(tk.END, f"Result of {selected_analysis}: {result}\n")
            elif selected_analysis == "novel_object":
                body_part_to_analyze = self.body_part_to_analyze.get()
                object_positions_str = self.object_positions.get()
                interaction_threshold = self.interaction_threshold.get()
                exploration_threshold = self.exploration_threshold.get()
                frame_threshold = self.frame_threshold.get()
                if not body_part_to_analyze or not object_positions_str:
                    self.output_text_area.insert(tk.END, "Error: Please select a body part and object positions.\n")
                    return
                try:
                    object_positions = json.loads(object_positions_str)
                except json.JSONDecodeError:
                    self.output_text_area.insert(tk.END, "Error: Invalid object positions format. Please use a valid dictionary string.\n")
                    return
                result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_part_to_analyze, object_positions, interaction_threshold, exploration_threshold, frame_threshold)
                if isinstance(result, dict):
                    self.output_text_area.insert(tk.END, f"Object interaction times: {result['object_interaction']}\n")
                    self.output_text_area.insert(tk.END, f"Object interaction bouts: {result['object_interaction_bouts']}\n")
                    self.output_text_area.insert(tk.END, f"Exploration rate: {result['exploration_rate']:.2f}\n")
                else:
                    self.output_text_area.insert(tk.END, f"Result of {selected_analysis}: {result}\n")
            elif selected_analysis == "transition_analysis":
                body_part_to_analyze = self.body_part_to_analyze.get()
                rois_to_analyze_str = self.rois_to_analyze.get()
                frame_threshold = self.frame_threshold.get()
                if not body_part_to_analyze or not rois_to_analyze_str:
                    self.output_text_area.insert(tk.END, "Error: Please select a body part and ROIs.\n")
                    return
                rois_to_analyze = [roi.strip() for roi in rois_to_analyze_str.split(',')]
                result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_part_to_analyze, rois_to_analyze, frame_threshold)
                if isinstance(result, dict):
                    if result:
                        self.output_text_area.insert(tk.END, f"Transition counts: {result}\n")
                    else:
                        self.output_text_area.insert(tk.END, "No transitions detected between the specified ROIs.\n")
                else:
                    self.output_text_area.insert(tk.END, f"Result of {selected_analysis}: {result}\n")
            elif selected_analysis == "zone_preference":
                body_part_to_analyze = self.body_part_to_analyze.get()
                rois_to_analyze_str = self.rois_to_analyze.get()
                if not body_part_to_analyze or not rois_to_analyze_str:
                    self.output_text_area.insert(tk.END, "Error: Please select a body part and ROIs.\n")
                    return
                rois_to_analyze = [roi.strip() for roi in rois_to_analyze_str.split(',')]
                result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_part_to_analyze, rois_to_analyze)
                if isinstance(result, dict):
                    self.output_text_area.insert(tk.END, f"Time spent in each zone: {result['time_spent']}\n")
                    self.output_text_area.insert(tk.END, f"Distance traveled in each zone: {result['distance_traveled']}\n")
                    self.output_text_area.insert(tk.END, f"Proportion of time in each zone: {result['proportion']}\n")
                else:
                    self.output_text_area.insert(tk.END, f"Result of {selected_analysis}: {result}\n")
            elif selected_analysis == "path_tracing":
                body_part_to_analyze = self.body_part_to_analyze.get()
                if not body_part_to_analyze:
                    self.output_text_area.insert(tk.END, "Error: Please select a body part.\n")
                    return
                result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_part_to_analyze, self.video_path.get())
                self.output_text_area.insert(tk.END, f"Result of {selected_analysis}: {result}\n")
            elif selected_analysis == "multi_bodypart_analysis":
                body_parts_to_analyze = [part.strip() for part in self.body_part_to_analyze.get().split(",")]
                roi_to_analyze = self.rois_to_analyze.get()
                if not body_parts_to_analyze or not roi_to_analyze:
                    self.output_text_area.insert(tk.END, "Error: Please select body parts and ROI.\n")
                    return
                result = analysis_function(self.body_part_data, self.roi_definitions, self.df, body_parts_to_analyze, roi_to_analyze)
                if isinstance(result, dict):
                    for part, data in result.items():
                        self.output_text_area.insert(tk.END, f"Results for {part}:\n")
                        self.output_text_area.insert(tk.END, f"  Entry Count: {data['entry_count']}\n")
                        self.output_text_area.insert(tk.END, f"  Time in ROI: {data['time_in_roi']}\n")
                        # Format the average entry speeds
                        formatted_speeds = ", ".join([f"{speed:.2f}" for speed in data['entry_speeds']])
                        self.output_text_area.insert(tk.END, f"  Average Entry Speeds: [{formatted_speeds}]\n")
        except Exception as e:
            self.output_text_area.insert(tk.END, f"Error during standard analysis: {e}\n")

    def run_code_generation(self):
        user_prompt = self.code_generation_prompt.get()
        output_folder = self.code_generation_output_folder.get()
        ollama_url = self.ollama_url
        roi_file_path = self.roi_file_path

        if not user_prompt or not output_folder:
            self.output_text_area.insert(tk.END, "Error: Please enter both analysis description and output folder.\n")
            return

        def run_code_gen():
            self.output_text_area.delete(1.0, tk.END)  # Clear previous output
            async def stream_code():
                async for chunk in code_generator.generate_analysis_code(user_prompt, ollama_url, roi_file_path):
                    if chunk == "No code generated":
                        self.output_text_area.insert(tk.END, f"Error: {chunk}\n")
                        return
                    elif chunk.startswith("Error:"):
                        self.output_text_area.insert(tk.END, f"Error: {chunk}\n")
                        return
                    else:
                        self.output_text_area.insert(tk.END, chunk)
                        self.output_text_area.see(tk.END)  # Scroll to the end

                try:
                    os.makedirs(output_folder, exist_ok=True)
                    file_path = os.path.join(output_folder, "generated_analysis.py")
                    with open(file_path, "w") as f:
                        f.write(self.output_text_area.get(1.0, tk.END))
                    self.output_text_area.insert(tk.END, f"\nGenerated code saved to: {file_path}\n")
                except Exception as e:
                    self.output_text_area.insert(tk.END, f"Error saving generated code: {e}\n")

            asyncio.run(stream_code())

        threading.Thread(target=run_code_gen).start()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = DABA_GUI(root)
    root.mainloop()