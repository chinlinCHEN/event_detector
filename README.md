# event_detector
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://badge.fury.io/gh/tterb%2FHyde.svg)](https://badge.fury.io/gh/tterb%2FHyde)

This repository includes the event detection function as a part of the analysis pipeline of https://github.com/NeLy-EPFL/Ascending_neuron_screen_analysis_pipeline/ for the manuscripts [**Ascending neurons convey behavioral state to integrative sensory and action selection centers in the brain**] in bioRxiv (https://www.biorxiv.org/content/10.1101/2022.02.09.479566v1). It is a package allow user to semi-automatically detect the deflection on the trace by labelling the initiating and ending timing as an event. It is flexible to let the user to determine the detection criteria to isolate the targeted event. 

## Content


## Installation
To install the AN environment for Python scripts, please refer the installation at https://github.com/NeLy-EPFL/Ascending_neuron_screen_analysis_pipeline/


## Event detection cirteria
### Parameters:
1. kinx_factor: ```-k```; determine the starting of an event and avoiding from the steepest point. It is more physiological and inspired from action potential kinx. 
2. shortest_evt_dur (sec): ```-sd```; the criteria to filter out too short event
3. longest_evt_dur (sec): ```-ld```; the range of the event
4. raw_thrsld: ```-r```; the criteria on amplitude of the normalized trace to filter out false-detective fluctuation.
5. diff_thrsld: ```-d```; the criteria on differentiated amplitude of the normalized trace to filter out false-detective fluctuation.
6. diff_window (sec): ```-dw```; the interval of differentiation, which will affect the amplitude of differentiated trace.

<p align="left">
  <img align="center" width="780" src="/images/event_detection_diagram.png">
</p>

todo: insert an diagram to explain the parameters here.


## Usage
### Input data structure
A input ```.csv``` file must contain:
1. Datapoints at each time tag
2. Total time (sec).

todo: next version in command line style



### Output files
#### A plot of detected event (```evt.png```)
1. The raw trace with detected event epochs (gray box) and starting (blue arrow head) and ending (red arrow head) timepoint.
2. The differentiated trace with detected event epochs (gray box) and starting point as peak*kink_factor (blue arrow head) and the differentiated peak (green arrow head).
3. The binary trace indicated the event period.
<p align="left">
  <img align="center" width="780" src="/output_events/evt.png">
</p>

<p align="left">
  <img align="center" width="780" src="/output_events/event_overlay.png">
</p>


#### The detection resutls in dictionary format (```detected_events.pkl```)
1. A binary trace
2. Event start index: corresponding to Event end index
3. Event end index: corresponding to Event start index
4. Sampling frequency













