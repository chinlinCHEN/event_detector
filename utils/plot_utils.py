import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import os




def Plot_traces(series_set=None, savepath=None):

	if series_set==None:
		print('No data series to plot ...')
		pass

	else:
		print('Plotting '+savepath)

		keys_series_set=list(series_set.keys())
		values_series_set=list(series_set.values())

		fig=plt.figure(facecolor='black', figsize=(25, 10), dpi=200)
		for i in range(0, len(series_set)):
			plt.subplot(int(str(len(series_set))+'1'+str(i+1)))
			plt.plot(values_series_set[i], linewidth=1)
			plt.title(keys_series_set[i])
		plt.tight_layout()
		plt.savefig(savepath)
		plt.clf()
		plt.close(fig)


	return



def plot_overlaid_events(events_2d_list, trend_trace, filepath, filename, bsl_s=1, samp_freq=1):

	events_2d_list=sorted(events_2d_list, key=len)

	x_len=len(events_2d_list[-1])
	x_start=0-bsl_s
	x_end=x_start+x_len/samp_freq
	
	trace_dur=np.linspace(x_start, x_end, x_len)
	
	print('x_start', x_start, 'x_end', x_end)
	

	fig = plt.figure(facecolor='white', figsize=(5,5), dpi=170)
	axGC= fig.add_subplot(1,1,1)
	for i, trace in enumerate(events_2d_list):
		axGC.plot(trace_dur[:len(trace)], trace, color='grey', label='ROI#'+str(i), linewidth=0.5, alpha = 0.5)
	axGC.plot(trace_dur[:len(trend_trace)],trend_trace, color='k', label='ROI#'+str(i), linewidth=1.5, alpha = 1)
	axGC.spines['top'].set_visible(False)
	axGC.spines['right'].set_visible(False)
	axGC.axhline(0, linestyle='dashed',color='gray',linewidth=0.5)
	axGC.axvline(0, linestyle='dashed',color='gray',linewidth=0.5)
	axGC.set_xlabel('time (s)',size=14, color='k')
	axGC.set_ylabel(r'Raw values',size=14,color='k')

	plt.savefig(filepath + filename , facecolor=fig.get_facecolor(), edgecolor='none', transparent=True) #bbox_inches='tight',      
	plt.savefig(filepath + filename , facecolor=fig.get_facecolor(), edgecolor='none', transparent=True) #bbox_inches='tight', 

	plt.clf
	plt.close(fig)

	return


























