# Dynamic Animal Behavior Analysis (DABA)


```
   ____    _    ____    _    
  |  _ \  / \  | __ )  / \   
  | | | |/ _ \ |  _ \ / _ \  
  | |_| / ___ \| |_) / ___ \ 
  |____/_/   \_\____/_/   \_\
Dynamic Animal Behavior Analysis
```

$\textcolor{#76b900}{\\#\ ***\ Updates\ ***}$

#### LLMs created in Ollama for this project's code generation module are available for download on [https://Ollama.com](https://ollama.com/FarhanAugustine)
###### Evaluate the sizes and relative accuracies of models based on your own hardware. The semi-quantatitative LLM scores (below) are provided as a baseline obtained during testing of DABA Code generation functionality. The following scoring criteria were used: *1. Code functionality (0-30 pts), 2. Code completeness & Prompt adherence (0-10 pts), 3. Output format (0-2 pts), 4. Code Comments (0-2 pts), 5. Code security (0-2 pts), and 6. Code Efficiency (0-2 pts)*. Note: Each LLM was prompted twice with the same prompt and scores were averaged. Individual LLM performance may vary, use with caution!

![Figure_1Github](https://github.com/user-attachments/assets/9f276477-dce6-46d8-9d23-f252373d7fd4)
|Prompt # | Prompts Used for LLM Testing |
|----|----|
| 1. |"Calculate the distance traveled by the Animal's Head in a user defined ROI, which the user will input as a list of tuples [(x,y),(x,y),(x,y),(x,y)]. Ensure that the code is standalone and ready to execute using only the DeepLabCut output CSV." 
| 2. | "Calculate the time spend by the Animal's Head in a user defined ROI only. The user will input the ROI as a list of tuples [(x,y),(x,y),(x,y),(x,y)]. Ensure that the code is standalone and ready to execute using only the DeepLabCut output CSV."|
|3. | "Calculate the time to first entry by the Animal's Head in a user defined ROI only. The user will input the ROI as a list of tuples [(x,y),(x,y),(x,y),(x,y)]. The criteria for entry is that the Animal's Head must be in the ROI for at least 30 Consecutive Frames before the Entry is to be counted. Ensure that the code is standalone and ready to execute using only the DeepLabCut output CSV."|

#### Hardware used for Testing
| Hardware| |
|---|---|
|Operating System |Windows|
|CPU|i9-14900F |
|GPU| RTX 4080 Super (16GB vRAM)|
RAM | 64 GB|

$\textcolor{#76b900}{\\#\ ***\ End\ of\ Updates\ ***}$

## This project provides a set of tools for analyzing animal behavior following DeepLabCut (DLC) pose-estimation.

It includes modules for:
- Standard analysis
- Code generation ⚠️ **(Use with Caution! Local LLMs are not very robust and can generate useless code)**

## Community Contributions are Welcomed! 🎁

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Features](#2-features)
3.  [Installation](#3-installation)
4.  [Configuration](#4-configuration)
5.  [Usage](#5-usage)
    *   [Standard Analysis](#standard-analysis)
    *   [Code Generation](#code-generation)
6.  [Analysis Modules](#6-analysis-modules)
7.  [Code Generation Module](#7-code-generation-module)
8.  [Troubleshooting](#8-troubleshooting)
9.  [Contributing](#9-contributing)
10. [License](#10-license)

## 1. Introduction

This project aims to simplify the analysis of animal behavior data obtained from DeepLabCut (DLC) tracking. It provides a user-friendly interface for performing common analyses, leveraging the power of Large Language Models (LLMs) for more complex tasks, and generating custom analysis scripts.

## 2. Features

*   **Standard Analysis:** Provides a set of pre-defined analysis modules for common behavioral metrics, such as time in ROI, distance traveled, average velocity, and more.
*   **Code Generation:** Enables users to generate standalone Python scripts for custom analysis tasks, which can be further modified and extended.
*   **ROI Definition:** Supports defining Regions of Interest (ROIs) using a simple, interactive interface.
*   **Logging:** Includes comprehensive logging for debugging and tracking the analysis process.
*   **Configurable:** Allows users to configure various settings, such as the Ollama server URL, ROI file path, and model names.

## 3. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/farhanaugustine/Dynamic-Animal-Behavior-Analysis-DABA-.git
    cd Dynamic-Animal-Behavior-Analysis-DABA-
    ```

2.  **Create and activate the Anaconda environment (recommended):**

    ```bash
    conda env create -f environment.yml
    conda activate DABA
    ```

3.  **Install Ollama:**

    *   Download and install Ollama from [https://ollama.ai/](https://ollama.ai/).
    *   Run Ollama and ensure that the models you intend to use are downloaded.
    *   Find Ollama LLMs here [Find Ollama models here: https://ollama.com/search]

## 4. Configuration

The project uses a `config.yaml` file to store configuration settings. Create a `config.yaml` file in the root directory of the project with the following structure:

```yaml
ollama_url: "http://localhost:11434"  # URL of your Ollama server
roi_file_path: "rois.json"          # Path to your ROI definitions file
code_gen_model: "codegemma:latest"   # Model for code generation
```

## 5. Usage

To run the project, execute the main script:

```bash
python DABA_v1.0.2.py
```

The script will prompt you to choose an analysis type:

- `standard`: For pre-defined analysis modules.
- `code`: For code generation.
- `exit`: To quit the program.

### Standard Analysis

1. Choose `standard` when prompted.
2. Choose the analysis types you want to run.
3. Enter any additional parameters needed for each analysis type.
4. The results of the analysis will be printed in the console or Output window on the GUI.
5. For all analyses in the `standard analysis`: ***Time is measured in frames, and distance is measured in pixels unless otherwise noted.***

### Code Generation

1. Choose `code` when prompted.
2. Describe the analysis you want to generate code for in natural language.
3. Enter the folder where you want to save the generated code.
4. The generated code will be printed to the console and saved to a file named `generated_analysis.py` in the specified folder.

## 6. Analysis Modules

The `analysis_modules` directory contains the following pre-defined analysis modules:

- `time_in_roi.py`: Calculates the time spent by a body part within a specified ROI.
- `distance_in_roi.py`: Calculates the distance traveled by a body part within a specified ROI.
- `average_velocity.py`: Calculates the average velocity of a body part.
- `multi_bodypart_analysis.py`: Performs analysis on multiple body parts.
- `entry_counter.py`: Counts the number of times a body part enters a specified ROI.
- `speed_per_entry.py`: Calculates the average speed of a specified body part during each continuous entry into a given Region of Interest (ROI). **In essence, it's the average speed of the body part while it is actively inside the ROI for each separate visit.**
    The speed for each entry is calculated as follows:
    1. Identify periods where the body part is continuously inside the ROI.
    2. For each such entry, calculate the distance traveled between consecutive frames *while the body part is inside the ROI*.
    3. Average these frame-by-frame distances over the duration of the entry to get the average speed for that entry.
    This metric focuses on the movement speed *only during the time the animal is within the ROI*.
- `exit_counter.py`: Counts the number of times a body part exits a specified ROI.
- `event_duration.py`: Calculates the duration of events (⚠️ freezing) based on a frame threshold.
- `novel_object.py`: Analyzes interactions with novel objects based on user-defined `interaction_threshold` and `exploration_threshold`. 
  * ***`interaction_threshold`**: **defines** how close the tracked `body_part` needs to be to an `object_position` to be considered an "interaction." It's essentially the maximum distance within which the animal is deemed to be engaging with the object.*
     *  ***Impact of Value***:
     *  ***Lower value (e.g., 2)**: Requires the animal to be very close to the object to be considered interacting. This might be appropriate if you're looking for direct contact or very near proximity. It reduces the chance of false positives (detecting interaction when the animal is just nearby).*
      *   ***Higher value (e.g., 15)**: Allows for a larger "buffer zone" around the object. The animal can be further away and still be considered interacting. This might be useful if you're interested in behaviors like sniffing or orienting towards the object from a slight distance. However, it increases the risk of false positives.*
        *  ***Why it's important**: This threshold allows you to define what "interaction" means in the context of your experiment. It depends on the behavior you're interested in.*

  * ***`exploration_threshold`**: This threshold, in conjunction with frame_threshold, determines when the animal is considered to be "exploring." It's a measure of the total distance the tracked `body_part` moves over a specified number of frames.*
     *  ***Impact of Value***:
     *  ***Lower value (e.g., 3)**: A small amount of movement within the frame_threshold will be enough to classify as exploration. This is more sensitive to any kind of movement.*
      *   ***Higher value (e.g., 15)**: Requires more significant movement within the frame_threshold to be considered exploration. This filters out smaller, less intentional movements and focuses on more deliberate exploratory actions.*
        *  ***Why it's important**: It helps differentiate between general activity and more focused exploratory behavior. You might want to distinguish between simply moving around the enclosure and actively investigating the environment.*
          
          * ***`frame_threshold`**: This threshold defines the time window (in terms of the number of video frames) over which the movement is assessed for the `exploration_threshold`.*
     *  ***Impact of Value***:
     *  ***Lower value (e.g., 3)**: Looks at movement over a very short period. It's sensitive to rapid, short bursts of movement.*
      *   ***Higher value (e.g., 15)**: Examines movement over a longer period. It captures more sustained movement patterns and is less influenced by brief twitches or minor adjustments.*
        *  ***Why it's important**: It determines the timescale of the exploration analysis. A smaller frame_threshold focuses on immediate movement, while a larger one looks at movement trends over a slightly longer duration.*
- `transition_analysis.py`: Analyzes transitions between ROIs.
            * ***`frame_threshold`**: This parameter determines the minimum number of consecutive video frames that the tracked `body_part` must remain inside an ROI (for an entry) or outside of an ROI (for an exit) to be considered a valid transition. It acts as a confirmation mechanism to prevent brief, accidental crossings or noisy tracking data from being counted as legitimate transitions.*
     *  ***Impact of Value***:
     *  ***Lower value (e.g., 3)**: Requires the body part to be consistently inside or outside an ROI for a very short duration to register a transition. This makes the analysis more sensitive to even brief entries and exits. It can be useful if you're interested in very quick interactions or if your tracking data is highly accurate and stable. However, it also increases the chance of counting brief, spurious crossings as real transitions due to tracking noise or minor hesitations at the ROI boundary.*
      *   ***Higher value (e.g., 15)**: Requires the body part to remain consistently within or outside an ROI for a longer period. This makes the analysis more robust against noise and brief, accidental crossings. It's useful when you're interested in more sustained occupancy within a region and want to avoid counting brief or uncertain movements as transitions. However, it might miss very quick but intentional transitions.*
        *  ***Why it's important**: Reduces Noise: Tracking data, even from good algorithms, can sometimes have minor fluctuations where a point might briefly appear inside or outside an ROI due to noise or temporary occlusion. The frame_threshold helps filter out these false positives. `Defines "Intentional" Transitions`: By requiring a certain duration of stay, the threshold helps ensure that the detected transitions represent a more deliberate movement into or out of a region, rather than a momentary passing through. `Adaptability to Tracking Quality`: If your tracking data is less stable or prone to jitter, a higher frame_threshold can make the transition analysis more reliable. `Behavioral Relevance`: The appropriate value depends on the type of behavior you're studying. For example, if you're analyzing rapid switching between locations, a lower threshold might be appropriate. If you're interested in longer periods of occupancy, a higher threshold would be better.*
- `zone_preference.py`: Analyzes the time spent and distance traveled in different zones.
          * ***`is_in_roi`**: While not a direct numerical threshold, the logic within the is_in_roi function acts as a binary threshold. A point is either inside the ROI or outside. There's no gradual measure of "how close" a point is to the boundary affecting its classification.*
     *  ***Impact**: The precision of your ROI definitions directly affects this. A slightly different set of coordinates for an ROI could include or exclude points near the boundary, influencing the time spent and distance traveled within that ROI.*
     *  ***Impact of Temporal Resolution (Frame Rate)**: Implicit "Time Step": The analysis calculates time spent by counting the number of frames the body part is within an ROI. The duration each frame represents (determined by your video's frame rate) is an implicit "time step."*
      *   ***Spatial Resolution (Tracking Accuracy)**: The accuracy of your body part tracking influences the results. If the tracking is noisy or imprecise, the calculated position of the body part might fluctuate around the ROI boundaries, leading to potentially inaccurate assessments of whether the animal is inside or outside.*
       
- `path_tracing.py`: Generates a video of the animal's path. The `User` will be prompted to select the body part to be used as a dot (tracker) and to define the path to save the generated video. This module uses CV2 and can be time-consuming for long videos. 

## 7. Code Generation Module

The `code_generator.py` module is responsible for generating custom analysis scripts using an LLM. It uses the following:

- **Configuration Loading**: It loads settings like the Ollama API URL and the desired LLM model from a `config.yaml` file.
- **Prompt Construction**: It dynamically creates prompts for the LLM using f-strings, incorporating the user's request and the path to an ROI (Region of Interest) definition file.
- **LLM Interaction (Ollama API)**: The `_query_ollama_stream` function asynchronously sends the constructed prompt to the Ollama API. It handles streaming responses, retries with exponential backoff in case of errors (timeout, HTTP errors, etc.), and decodes the JSON response chunks.
- **Code Streaming**: The generate_analysis_code function orchestrates the process, sending the prompt to Ollama and yielding the generated code in chunks as it's received from the LLM. This allows for processing of potentially large code outputs without waiting for the entire generation.
- **Error Handling**: The module includes error handling for file operations (config file loading), API requests, and JSON decoding.
- ⚠️***No Explicit Post-Processing***: *The provided code does not include functions for explicit code extraction, syntax checking, or formatting. It directly streams the output from the LLM. Therefore, whichever LLM that you choose to run, it must be tuned to understand CSV data structure.* ⚠️

To download Ollama Model, ensure that you have Ollama downloaded and running on your computer before using the following commands:
```bash
ollama run 'name-of-the-model-you-want-to-download' (e.g., ollama run llama2, ollama run gemma2,...)
```
To download my models from Ollama: 
```bash
For q4_K_M quantized (43GB) Model: `ollama run FarhanAugustine/nemotron_DLC`
For q4_K_M quantized (43GB) Model: `ollama run FarhanAugustine/llama3.3_DLC`
For q3_K_S quantized (31GB) Model: `ollama run FarhanAugustine/llama3.3_DLC_q3_K_S`
```

## 8. Troubleshooting

- **ModuleNotFoundError**: Ensure you have installed all required packages using.
- **Ollama Connection Issues**: Make sure Ollama is running and that the `ollama_url` in `config.yaml` is correct.
- **Code Saving Errors**: Ensure the output folder path is valid and delete the previously generated response. The script does not automatically overwrite the previous response.
- **AttributeError: 'NoneType' object has no attribute 'startswith'**: Ensure the `ollama_url` is correct and that the LLM is running.

## 9. Contributing

Contributions are welcome! 
If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## 10. License

This project is licensed under the MIT License. See the LICENSE info on this Repo.

** *Disclaimer:* ** This code and these scripts are provided "as is" and without any warranties, express or implied. The user assumes all responsibility for verifying the accuracy of results, validating the methodology, and ensuring adherence to scientific rigor. No guarantee is made regarding the code's functionality or suitability for any specific purpose.
