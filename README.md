# event_detector
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://badge.fury.io/gh/tterb%2FHyde.svg)](https://badge.fury.io/gh/tterb%2FHyde)

This repository includes the event detection function as a part of the analysis pipeline of https://github.com/NeLy-EPFL/Ascending_neuron_screen_analysis_pipeline/ for the manuscripts [**Ascending neurons convey behavioral state to integrative sensory and action selection centers in the brain**] in bioRxiv (https://www.biorxiv.org/content/10.1101/2022.02.09.479566v1). It is a package allow user to semi-automatically detect the deflection on the trace by labelling the initiating and ending timing as an event. It is flexible to let the user to determine the detection criteria to isolate the targeted event. 

## Content


## Installation
To install the AN environment for Python scripts, please refer the installation at https://github.com/NeLy-EPFL/Ascending_neuron_screen_analysis_pipeline/


## Event detection cirteria
### Parameters:
1. ```kink_factor```: ```-k```; determine the starting of an event and avoiding from the steepest point. It is more physiological and inspired from action potential . 
2. ```raw_thrsld```: ```-r```; the criteria on amplitude of the normalized trace to filter out false-detective fluctuation.
3. ```diff_thrsld```: ```-d```; the criteria on differentiated amplitude of the normalized trace to filter out false-detective fluctuation.
4. ```shortest_evt_dur``` (sec): ```-sd```; the criteria to filter out too short event
5. ```longest_evt_dur``` (sec): ```-ld```; the range of the event. Not a criteria
6. ```diff_window``` (sec): ```-i```; the interval of differentiation, which will affect the amplitude of differentiated trace.

<p align="left">
  <img align="center" width="360" src="/images/event_detection_criteriaDiagram.png">
</p>

todo: insert an diagram to explain the parameters here.


## Usage
### Input data structure
An input ```.csv``` file must contain two columns:
1. ```values```: Datapoints at each time tag
2. ```time```: Time tags (sec).


### Set criteria
Criteria can be set in command-line interface:
```
python main.py -k 0.4 -i 0.3 -r 0.55 -d 0.2 -sd 0.5 -ld 2 -input ./data/trial001_0.csv -output ./output_events/ -plot_overlay True
```

Check the explanation of the usage:
```
python main.py -h
```

```
usage: main.py [-h] [-k kink factor] [-i interval for differentiation]
               [-r threshold on raw value]
               [-d threshold on differentiated value]
               [-sd shortest event duration] [-ld longest event duration]
               [-input input file] [-output output directory]
               [-plot_overlay plot overlaid events]

event detector

optional arguments:
  -h, --help            show this help message and exit
  -k kink factor, --kink_factor kink factor
                        It is coefficient (usually between 0 and 1) to decides
                        the event initiating point (local maximum change is
                        not a physiologically starting timing). Default = 0.4.
  -i interval for differentiation, --diff_window interval for differentiation
                        The time interval (s) for differentiating the trace.
                        It would affect the decision of diff_thrsld. Default =
                        0.3.
  -r threshold on raw value, --raw_thrsld threshold on raw value
                        An event detection criteria. The normalized amplitude
                        of an event should higher than this. Default = 0.55.
  -d threshold on differentiated value, --diff_thrsld threshold on differentiated value
                        An event detection criteria. The differentiated
                        normalized amplitude of an event should higher than
                        this. The choice of number can be affected by
                        diff_window. Default = 0.2.
  -sd shortest event duration, --shortest_evt_dur shortest event duration
                        An event detection criteria. The duration of an event
                        should longer than this (s). Default = 0.5.
  -ld longest event duration, --longest_evt_dur longest event duration
                        Not an event detection criteria. The duration of an
                        event (s). Default = 2.
  -input input file     input directory + filename.csv
  -output output directory
                        output directory
  -plot_overlay plot overlaid events
                        an optional plot of overlaid detected events. Default
                        = True.
```


### Output files
#### The detection resutls in dictionary format (```detected_events.pkl```)
1. A binary trace
2. Event start index: corresponding to Event end index
3. Event end index: corresponding to Event start index
4. Sampling frequency

#### A plot of detected event (```evt.png``` and ```event_overlay.png```)
1. The raw trace with detected event epochs (gray box) and starting (blue arrow head) and ending (red arrow head) timepoint.
2. The differentiated trace with detected event epochs (gray box) and starting point as peak*kink_factor (blue arrow head) and the differentiated peak (green arrow head).
3. The binary trace indicated the event period.
<p align="left">
  <img align="middle" width="1000" src="/output_events/evt.png">
</p>

#### [Optional] Extended application: Plot overlaid events (```event_overlay.png```)
The detected event can be extended with baseline period to align the event to the initiation time.
<p align="left">
  <img align="middle" width="300" src="/output_events/event_overlay.png">
</p>


















