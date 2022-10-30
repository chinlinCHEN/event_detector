import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')
import sys
import pandas as pd 
import os
import pickle
from multiprocessing import Pool
from itertools import repeat


import event_detector.EventDetector as EventDetector
import utils.general_utils as general_utils
import utils.plot_utils as plot_utils
import utils.cli as cli_utils




def main():

	args=cli_utils.parse_cli_args()

	kink_factor=args.kink_factor
	raw_thrsld=args.raw_thrsld
	diff_thrsld=args.diff_thrsld
	shortest_evt_dur=args.shortest_evt_dur
	longest_evt_dur=args.longest_evt_dur
	diff_window=args.diff_window

	input_file_dir=args.input
	output_dir=args.output


	if not os.path.exists(output_dir):
		os.makedirs(output_dir)


	df = pd.read_csv(input_file_dir)
	timeSec=df['time'].tolist()
	trace=df['values'].tolist()
	samplingFreq=int(len(timeSec)/timeSec[-1])

	print('samplingFreq', samplingFreq)



	smth_trace=EventDetector.filtered_traces(trace, filtermode='running_window', frame_window=int(0.6*samplingFreq)) #0.1s
	norm_range_smth_GC_trace ,_ ,_ = EventDetector.normalize_trace(smth_trace, frame_window=int(10*samplingFreq), mode='btwn_0and1')


	evt_bin_trace, evt_startIdx_list, evt_endIdx_list = EventDetector.detect_event(norm_range_smth_GC_trace, output_dir, 'evt.png', fps=samplingFreq, \
	kink_factor=kink_factor, \
	evt_shortest_dur=shortest_evt_dur, \
	evt_longest_dur=longest_evt_dur, \
	raw_thrsld=raw_thrsld, \
	diff_thrsld=diff_thrsld, \
	diff_window=diff_window\
	)



	events_2d_list, evt_mean_trace=prep_for_evtOverlay_plot(trace, evt_startIdx_list, evt_endIdx_list, samp_freq=samplingFreq, bsl_dur=1)

	plot_evt_overlay=True
	if plot_evt_overlay==True:
		plot_utils.plot_overlaid_events(events_2d_list, evt_mean_trace, filepath=output_dir, filename='event_overlay.png', bsl_s=1, samp_freq=samplingFreq)



	save_GCevt_dic(output_dir, 'detected_events.pkl', evt_bin_trace, evt_startIdx_list, evt_endIdx_list, samplingFreq)



	return



def save_GCevt_dic(save_dir, filename, evt_bin_trace, evt_startIdx_list, evt_endIdx_list, samplingFreq):

	print('Saving GCevent-based.dic')

	GCevt_dic={}
	GCevt_dic.update({'evt_bin_trace':evt_bin_trace})
	GCevt_dic.update({'evt_startIdx_list':evt_startIdx_list})
	GCevt_dic.update({'evt_endIdx_list':evt_endIdx_list})

	GCevt_dic.update({'samplingFreq':samplingFreq})
	# GCevt_dic.update({'bsl_s':bsl_s})

	# GCevt_dic.update({'GC_evt_set_list':GC_evt_set_list})

	pickle.dump( GCevt_dic, open( save_dir + filename, "wb" ) ) 

	return




def prep_for_evtOverlay_plot(trace, evt_start_idx_list, evt_end_idx_list, samp_freq=1, bsl_dur=0):

	corrected_evt_start_idx=[]
	corrected_evt_end_idx=[]
	events_2d_list=[]
	for i, idx in enumerate(evt_start_idx_list):
		if idx-int(bsl_dur*samp_freq)>=0:
			corrected_evt_start_idx.append(idx-int(bsl_dur*samp_freq))
			corrected_evt_end_idx.append(evt_end_idx_list[i])

			evt=trace[idx-int(bsl_dur*samp_freq):evt_end_idx_list[i]]

			events_2d_list.append(evt)


	## make all events have same length by adding nan tail
	events_2d_list=sorted(events_2d_list, key=len)
	longest_evt=len(events_2d_list[-1])


	events_2d_list_nantail=[]
	for i, evt in enumerate(events_2d_list):
		len_nantail=longest_evt-len(evt)
		new_evt=evt+[np.nan]*len_nantail
		events_2d_list_nantail.append(new_evt)

	### Make a mean trace representing the trend of all events
	evt_mean_trace=np.mean(events_2d_list_nantail, axis=0)



	return events_2d_list, evt_mean_trace





if __name__ == "__main__":
    main()


