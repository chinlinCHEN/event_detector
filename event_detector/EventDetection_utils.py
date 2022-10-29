import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')
import sys
import pandas as pd 
import os
import pickle
import cv2
import more_itertools as mit
from skimage import io
# import statsmodels.api as sm # need to downgrade to scipy==1.2




def detect_event(trace, save_dir, filename, fps=30, evt_shortest_dur=0.27, evt_longest_dur=2, diff_window=0.3, raw_thrsld=0.2, diff_thrsld=0.03, kink_factor=0.2, raw_change_thrsld=0.1, find_peak_distance=0.8, kinx_search_range=0.4, select_specific_dur_evt=False):

	print('Detecting events...')

	evt_shortest_distance=int(evt_shortest_dur*fps)
	evt_longest_distance=int(evt_longest_dur*fps)
	find_peak_distance=int(find_peak_distance*fps)



	#grad_trace = np.gradient(trace,0.25)
	grad_trace = diff_trace(trace, samplerate=fps, diff_window_s=diff_window)
	grad_trace = filtered_traces(grad_trace, filtermode='running_window')



	peaks_idx_rawTrace, _ = scipy.signal.find_peaks(trace, height = raw_thrsld, distance=find_peak_distance)
	peaks_idx_gradTrace, _ = scipy.signal.find_peaks(grad_trace, height = diff_thrsld, distance=find_peak_distance)
	#peaks_idx_gradTrace_cwt = scipy.signal.find_peaks_cwt(grad_trace, np.arange(1,20), max_distances=np.arange(1, 30)*2)
	#print('peaks_idx', peaks_idx)
	print('peaks_idx_gradTrace', peaks_idx_gradTrace)
	

	peaks_of_rawTrace_on_rawTrace = np.array(trace)[peaks_idx_rawTrace]

	peaks_of_gradTrace_on_rawTrace = np.array(trace)[peaks_idx_gradTrace]
	peaks_of_gradTrace_on_gradTrace = np.array(grad_trace)[peaks_idx_gradTrace]

	# peaks_idx_gradTrace_cwt_on_rawTrace = np.array(trace)[peaks_idx_gradTrace_cwt]
	# peaks_idx_gradTrace_cwt_on_gradTrace = np.array(grad_trace)[peaks_idx_gradTrace_cwt]

	## Find start kinx of event
	kinx_idx_rawTrace=detect_kinx(grad_trace, peaks_idx_gradTrace, mode='forward', fps=fps, kink_factor=kink_factor, srch_range=kinx_search_range)
	print('kinx_idx_rawTrace', kinx_idx_rawTrace)
	#print('trace[kinx_idx_rawTrace[-1]]', trace[kinx_idx_rawTrace[-1]])

	## Backward find nearest point of kinx as for the end of the event
	end_idx_rawTrace=detect_kinx(trace, kinx_idx_rawTrace, mode='backward', fps=fps, kink_factor=kink_factor, srch_range=evt_longest_dur, no_after_these_idx=kinx_idx_rawTrace, height_cond=peaks_of_gradTrace_on_rawTrace)
	print('end_idx_rawTrace', end_idx_rawTrace)

	# print('kinx_idx_rawTrace', kinx_idx_rawTrace)
	# print('len kinx_idx_rawTrace', len(kinx_idx_rawTrace))

	# print('end_idx_rawTrace', end_idx_rawTrace)
	# print('len end_idx_rawTrace', len(end_idx_rawTrace))	

	startIdx_rawTrace, endIdx_rawTrace=clean_FalsePositive_detection(kinx_idx_rawTrace, end_idx_rawTrace, trace, mode='remove_short_period', threshold=evt_shortest_distance)
	startIdx_rawTrace, endIdx_rawTrace=clean_FalsePositive_detection(startIdx_rawTrace, endIdx_rawTrace, trace, mode='remove_small_value',threshold=raw_thrsld)
	startIdx_rawTrace, endIdx_rawTrace=clean_FalsePositive_detection(startIdx_rawTrace, endIdx_rawTrace, trace, mode='remove_small_change',threshold=raw_change_thrsld)
	if select_specific_dur_evt==True:
		print('select_specific_dur_evt')
		startIdx_rawTrace, endIdx_rawTrace=clean_FalsePositive_detection(startIdx_rawTrace, endIdx_rawTrace, trace, mode='select_specific_dur_evt',threshold= [evt_shortest_distance, evt_longest_distance/2])


	start_idx_rawTrace_on_rawTrace = np.array(trace)[startIdx_rawTrace]
	start_idx_rawTrace_on_gradTrace = np.array(grad_trace)[startIdx_rawTrace]	
	end_idx_rawTrace_on_rawTrace = np.array(trace)[endIdx_rawTrace]

	if len(startIdx_rawTrace)!= len(endIdx_rawTrace):
		print('startIdx_rawTrace numbers is not equal to endIdx_rawTrace')
		sys.exit(0)

	else:
		print('startIdx_rawTrace numbers is equal to endIdx_rawTrace')





	evt_bin_trace=[0]*len(trace)
	for i, evt_startIdx in enumerate(startIdx_rawTrace):
		evt_endIdx=endIdx_rawTrace[i]
		## If current start index == previous end index, move current start index one index ffurther to seperate from the previous event.  
		if evt_startIdx-1 == endIdx_rawTrace[i-1]:
			print('evt_startIdx-1 == endIdx_rawTrace[i-1], move curretn startIdx one poitn further.')
			print('evt_startIdx', evt_startIdx)
			print('endIdx_rawTrace[i-1]', endIdx_rawTrace[i-1])

			startIdx_rawTrace[i]=evt_startIdx+1

			for j in range(evt_startIdx+1, evt_endIdx+1):
					evt_bin_trace[j]=1
		else:
			for j in range(evt_startIdx, evt_endIdx+1):
					evt_bin_trace[j]=1

		# for j in range(evt_startIdx, evt_endIdx+1):
		# 		evt_bin_trace[j]=1


	Plot=True
	if Plot==True:

		print('==Plotting preview of event detection==')
		fig=plt.figure(facecolor='black', figsize=(12, 5), dpi=200)

		
		plt.subplot(311)
		plt.title('Normalized trace (fold relative to baseline)')
		plt.plot(trace, color='k', linewidth=1)
		#plt.plot(trace_med, color='r', linewidth=1)
		#plt.plot(peaks_idx_rawTrace, peaks_of_rawTrace_on_rawTrace, marker='x', color='r',linestyle = 'None')
		#plt.plot(peaks_idx_gradTrace, peaks_of_gradTrace_on_rawTrace, marker='o', color='g',linestyle = 'None')
		plt.plot(startIdx_rawTrace, start_idx_rawTrace_on_rawTrace, marker='^', color='b',linestyle = 'None')
		plt.plot(endIdx_rawTrace, end_idx_rawTrace_on_rawTrace, marker='v', color='r',linestyle = 'None')
		for i, evt_startIdx in enumerate(startIdx_rawTrace):
			evt_endIdx=endIdx_rawTrace[i]
			plt.axvspan(evt_startIdx, evt_endIdx, color='k', alpha=0.25, linewidth=0)
		
		plt.subplot(312)
		plt.title('Differentiated normalized trace')
		plt.plot(grad_trace, color='k',linewidth=1)
		plt.plot(peaks_idx_gradTrace, peaks_of_gradTrace_on_gradTrace, marker='o', color='g',linestyle = 'None')
		plt.plot(startIdx_rawTrace, start_idx_rawTrace_on_gradTrace, marker='^', color='b',linestyle = 'None')
		for i, evt_startIdx in enumerate(startIdx_rawTrace):
			evt_endIdx=endIdx_rawTrace[i]
			plt.axvspan(evt_startIdx, evt_endIdx, color='k', alpha=0.25, linewidth=0)
		

		plt.subplot(313)
		plt.title('Binary event trace')
		plt.plot(evt_bin_trace, color='k',linewidth=1)

		
		plt.tight_layout()
		plt.savefig(save_dir+filename)
		plt.clf()
		plt.close(fig)




	return evt_bin_trace, startIdx_rawTrace, endIdx_rawTrace







def diff_trace(trace, samplerate=1, diff_window_s=0.2 ):

	intvl_dif=int(samplerate*diff_window_s)
	ddata_series=[]
	for i in range(0,len(trace)-intvl_dif):
		ddatapoint = trace[i+intvl_dif]-trace[i]
		ddata_series.append(ddatapoint) 

    
	# put 0 to the place where there is no derivative
	for i in range(0,len(trace)-len(ddata_series)):
		ddata_series.append(0)   

	return ddata_series





def normalize_trace(trace, frame_window=100, mode=None, max_val_manual=None):


	if mode == 'btwn_0and1':

		if max_val_manual==None:
			smth_trace=smooth_trace(trace,frame_window)
			max_val=max(smth_trace)
		else:
			max_val=max_val_manual

		# print('max_val', max_val)

		
		#print('smth_trace', smth_trace)
		# print('min smth_trace', min(smth_trace))
		# print('max smth_trace', max(smth_trace))


		#baseline = np.nanmin(smth_trace[int((1/7)*len(trace)):int((6/7)*len(trace))])


		temp_trace=[1]*len(trace)
		mean_trace = np.nanmean(smth_trace) 
		for i, val in enumerate(trace):
			if val>1.3*mean_trace:
				temp_trace[i]=np.nan
		
		baseline=np.nanmean(np.multiply(trace, temp_trace))



		print('baseline', baseline)

		range_trace=max_val-baseline
		print('range_trace', range_trace)
		
		# plt.plot(smth_trace)
		# plt.show()
		# plt.pause(3)
		# plt.clf()

		#print('min(np.asarray(trace)-baseline)' , min(np.asarray(trace)-baseline))
		norm_0and1_trace=(np.asarray(trace)-baseline)/range_trace

		# norm_0and1_trace=[]
		# for val in trace:
		# 	norm_val=(val-baseline)/range_trace
		# 	norm_0and1_trace.append(norm_val)

		return norm_0and1_trace, range_trace, baseline


	elif mode == 'fold_of_baseline':

		smth_trace=smooth_trace(trace,frame_window)
		if len(trace)>5000:
			baseline = min(smth_trace[int((1/7)*len(trace)):int((6/7)*len(trace))])
		else:
			baseline = min(smth_trace) 


		norm_trace = (np.asarray(trace)-baseline)/baseline



		return norm_trace






def detect_kinx(trace, peaks_idx, mode='forward', srch_range=0.4, kink_factor=0.2, fps=30, no_after_these_idx=None, height_cond=None):

	print('\n'+mode+' detecting ...\n')

	data_samplerate=fps
	print('peaks_idx', peaks_idx)
	print('data_samplerate*srch_range', data_samplerate*srch_range)

	evt_kinx_idx_series=[]




	for i, peak_idx in enumerate(peaks_idx):

		ajst_thrsld = trace[peak_idx]*kink_factor
		print('ajst_thrsld', ajst_thrsld)

		if mode=='forward':
			
			if int(peak_idx-data_samplerate*srch_range)>0:
				nearest_value=find_nearest(trace[int(peak_idx-data_samplerate*srch_range):int(peak_idx)], ajst_thrsld)
				nearest_value_idx = np.where(trace[int(peak_idx-data_samplerate*srch_range):int(peak_idx)] == nearest_value)[0]
				if len(nearest_value_idx)>1:					
					min_trace=min(trace[int(peak_idx-data_samplerate*srch_range):int(peak_idx)])					
					idxs_min=np.where(trace[int(peak_idx-data_samplerate*srch_range):int(peak_idx)]==min_trace)[0]					
					maxIdx=max(idxs_min)					
					nearest_idx=maxIdx+int(peak_idx-data_samplerate*srch_range)
				else:
					nearest_idx=nearest_value_idx[0]+int(peak_idx-data_samplerate*srch_range)

			## Touch the initial index
			elif int(peak_idx-data_samplerate*srch_range)<=0:
				nearest_value=find_nearest(trace[0:int(peak_idx)], ajst_thrsld)
				nearest_value_idx = np.where(trace[0:int(peak_idx)] == nearest_value)[0]
				if len(nearest_value_idx)>1:					
					min_trace=min(trace[0:int(peak_idx)])					
					idxs_min=np.where(trace[0:int(peak_idx)]==min_trace)[0]				
					maxIdx=max(idxs_min)					
					nearest_idx=maxIdx+0
				else:
					nearest_idx=nearest_value_idx[0]+0		


		elif mode=='backward':

			height_cond_val=height_cond[i]*(0.7)

			print('trace[peak_idx]', trace[peak_idx])

			# Not touch trace end
			if int(peak_idx+data_samplerate*srch_range)<len(trace)-1:
				print('Evt# '+str(i)+' at frame# '+str(peak_idx)+': '+'backward not touch trace end')
				
				# Not last start idx
				if i+1<len(no_after_these_idx):
					# print('Not last start idx')
					# Not touch next start idx
					if int(peak_idx+data_samplerate*srch_range)<no_after_these_idx[i+1]:
						print('Evt# '+str(i)+' at index# '+str(peak_idx)+': '+'Not touch next start idx')

						nearest_value=find_nearest(trace[int(peak_idx):int(peak_idx+data_samplerate*srch_range)], trace[peak_idx], condition='over_max', height_cond=height_cond_val)
						nearest_value_idx = np.where(trace[int(peak_idx):int(peak_idx+data_samplerate*srch_range)] == nearest_value)[0]
						print('nearest_value_idx', nearest_value_idx)
						
						if len(nearest_value_idx)>1:
							max_trace=max(trace[int(peak_idx):int(peak_idx+data_samplerate*srch_range)])
							idxs_max=np.where(trace[int(peak_idx):int(peak_idx+data_samplerate*srch_range)]==max_trace)[0]
							maxIdx=max(idxs_max)
							nearest_idx=find_nearest(nearest_value_idx, maxIdx)+peak_idx+1
						else:
							nearest_idx=nearest_value_idx[0]+peak_idx

					#Touch next start idx
					else:
						# print('touch next start idx')
						# print('i',i)
						# print('peak_idx', peak_idx)
						# print('no_after_these_idx', no_after_these_idx)
						
						nearest_value=find_nearest(trace[int(peak_idx):no_after_these_idx[i+1]], trace[peak_idx], condition='over_max', height_cond=height_cond_val)
						nearest_value_idx = np.where(trace[int(peak_idx):no_after_these_idx[i+1]] == nearest_value)[0]
						if len(nearest_value_idx)>1:
							max_trace=max(trace[int(peak_idx):no_after_these_idx[i+1]])
							idxs_max=np.where(trace[int(peak_idx):no_after_these_idx[i+1]]==max_trace)[0]
							maxIdx=max(idxs_max)
							nearest_idx=find_nearest(nearest_value_idx, maxIdx)+peak_idx+1							
						else:
							nearest_idx=nearest_value_idx[0]+int(peak_idx)

				# Last start idx
				else:		
					print('Evt# '+str(i)+' at frame# '+str(peak_idx)+': '+'Last start idx')		
					nearest_value=find_nearest(trace[int(peak_idx):int(peak_idx+data_samplerate*srch_range)], trace[peak_idx], condition='over_max', height_cond=height_cond_val)
					nearest_value_idx = np.where(trace[int(peak_idx):int(peak_idx+data_samplerate*srch_range)] == nearest_value)[0]
					print('nearest_value_idx', nearest_value_idx)
					if len(nearest_value_idx)>1:
						max_trace=max(trace[int(peak_idx):int(peak_idx+data_samplerate*srch_range)])
						idxs_max=np.where(trace[int(peak_idx):int(peak_idx+data_samplerate*srch_range)]==max_trace)[0]
						maxIdx=max(idxs_max)
						nearest_idx=find_nearest(nearest_value_idx, maxIdx)+peak_idx+1
					else:
						nearest_idx=nearest_value_idx[0]+peak_idx


			# Touch trace end
			else:
				print('Evt# '+str(i)+' at frame# '+str(peak_idx)+': '+'backward touch trace end')
				if i+1<len(no_after_these_idx):
					print('Evt# '+str(i)+' at frame# '+str(peak_idx)+': '+'Not last start idx')
					# Not touch next start idx
					if int(peak_idx+data_samplerate*srch_range)<no_after_these_idx[i+1]:
						print('Evt# '+str(i)+' at frame# '+str(peak_idx)+': '+'Not touch next start idx')
						nearest_value=find_nearest(trace[int(peak_idx):-1], trace[peak_idx], condition='over_max', height_cond=height_cond_val)
						nearest_value_idx=np.where(trace[int(peak_idx):-1]==nearest_value)
						if len(nearest_value_idx)>1:
							max_trace=max(trace[int(peak_idx):int(peak_idx+data_samplerate*srch_range)])
							idxs_max=np.where(trace[int(peak_idx):int(peak_idx+data_samplerate*srch_range)]==max_trace)[0]
							maxIdx=max(idxs_max)
							nearest_idx=find_nearest(nearest_value_idx, maxIdx)+peak_idx+1							
						else:
							nearest_idx=nearest_value_idx[0]+peak_idx


					#Touch next start idx
					else:
						print('Evt# '+str(i)+' at frame# '+str(peak_idx)+': '+'touch next start idx')
						nearest_value=find_nearest(trace[int(peak_idx):no_after_these_idx[i+1]], trace[peak_idx], condition='over_max', height_cond=height_cond_val)
						nearest_value_idx = np.where(trace[int(peak_idx):no_after_these_idx[i+1]] == nearest_value)[0]

						if len(nearest_value_idx)>1:
							max_trace=max(trace[int(peak_idx):no_after_these_idx[i+1]])
							idxs_max=np.where(trace[int(peak_idx):no_after_these_idx[i+1]]==max_trace)[0]
							maxIdx=max(idxs_max)
							nearest_idx=find_nearest(nearest_value_idx, maxIdx)+peak_idx+1							
						else:
							nearest_idx=nearest_value_idx[0]+int(peak_idx)

				# Last start idx
				else:		
					print('Evt# '+str(i)+' at frame# '+str(peak_idx)+': '+'Last start idx')		
					nearest_value=find_nearest(trace[int(peak_idx):-1], trace[peak_idx], condition='over_max', height_cond=height_cond_val)
					nearest_value_idx=np.where(trace[int(peak_idx):-1] == nearest_value)[0]
					if len(nearest_value_idx)>1:
						max_trace=max(trace[int(peak_idx):-1])
						idxs_max=np.where(trace[int(peak_idx):-1]==max_trace)[0]
						maxIdx=max(idxs_max)
						nearest_idx=find_nearest(nearest_value_idx, maxIdx)+peak_idx
					else:
						nearest_idx=nearest_value_idx[0]+peak_idx			

		# print('ddata_series searching range:', int(peak_idx-data_samplerate*srch_range), int(peak_idx))
		#print('nearest_value_idx', nearest_value_idx)
		print('nearest_idx', nearest_idx)
		evt_kinx_idx_series.append(nearest_idx)
		evt_kinx_idx_series.sort()

	print('evt_kinx_idx_series')


	return evt_kinx_idx_series



def find_nearest(ori_array, ori_value, condition=None, height_cond=None):

	# print('len ori_array', len(ori_array))
    

	array = np.asarray(ori_array)
	# print('array', array)
	# print('len array', len(array))

	if condition==None:
		idx = (np.abs(ori_array - ori_value)).argmin()

		return ori_array[idx]

	elif condition=='over_max':

		array, range_trace, baseline=normalize_trace(ori_array, frame_window=1, mode='btwn_0and1')
		value=(ori_value-baseline)/range_trace

		if len(array)<10:
			
			print('Skip detecting end point of this event! It is too short!')
			return ori_array[1]

		else:

			peak_idx_array, _ = scipy.signal.find_peaks(array, height=0.5)
			if len(peak_idx_array)==0:
				max_idx=array.argmax()
			else:
				max_idx=peak_idx_array[0]

			# print('max_idx', max_idx)
			# print('array', array)
			# print('len array', len(array))
			#print('array[max_idx:-1]', array[max_idx:-1])

			# if the array is too short, then skip



			Similarity_with_value = 1 - np.abs(array[max_idx:-1] - value) 
			similarity_to_startVal=0.8
			local_max_idx, _ = scipy.signal.find_peaks(Similarity_with_value, height=similarity_to_startVal)

			similarity_grid=sorted(np.arange(0.5, 0.9, step=0.05), reverse = True)
			#print('similarity_grid',similarity_grid)

			for i, similarity in enumerate(similarity_grid):
				local_max_idx, _ = scipy.signal.find_peaks(Similarity_with_value, height=similarity)
				# print('similarity',similarity,'local_max_idx',local_max_idx)
				if len(local_max_idx)==0:
					if i==len(similarity_grid)-1:
						print('	no local maximum found ... ')
						print('	Instead looking for closet value  ... ')
						# print('similarity', similarity)
						# print('Similarity_with_value', Similarity_with_value)
						Similarity_with_value_thres=np.asarray(list(Similarity_with_value).copy())
						Similarity_with_value_thres[(np.asarray(Similarity_with_value)<similarity)]=0

						if np.sum(Similarity_with_value_thres)==0:
							if max_idx==len(array)-1:
								print('	max_idx')
								idx=max_idx
							else:
								print('	all values < threshold...')
								print('	take last value as the end of the event...')
								idx_temp=len(Similarity_with_value_thres)-1
								idx=max_idx+idx_temp
								# print('idx_temp',idx_temp)

						else:

							idx_temp=Similarity_with_value_thres.argmax()
							idx=max_idx+idx_temp 
							# print('idx_temp',idx_temp)

							continue
							plt.title('similarity ='+str(similarity)+' frame# ='+str(idx))
							plt.plot(Similarity_with_value_thres)
							plt.plot(idx_temp, Similarity_with_value_thres[idx_temp], 'ro')
							plt.savefig(outputPERdir+'local_evt_no_local_max.png')
							plt.clf()


					else:
						continue

				else:
					

					idx=max_idx+local_max_idx[0]

					continue
					plt.title('similarity ='+str(similarity)+' frame# ='+str(idx))
					plt.plot(0-max_idx,value,'x')
					plt.plot(array[max_idx:-1],'r') 
					plt.plot(np.abs(array[max_idx:-1] - value),'g')
					plt.plot(Similarity_with_value,'b')
					plt.plot(local_max_idx, Similarity_with_value[local_max_idx], "x")
					plt.plot(idx-max_idx, array[idx], "v")

					#plt.savefig(outputPERdir+'local_evt.png')
					plt.clf()

					break			

			##Normalize y-value with x-value for detecting distance between start point and end point
			# scaling_factor_for_fair_dist_cal=int(len(array[max_idx:-1])/max(array[max_idx:-1]))
			# slope_list=[]
			# dist_list=[]
			# for i, val in enumerate(array[max_idx:-1]):
			# 	dist=calc_length(0, value*scaling_factor_for_fair_dist_cal, max_idx+i, array[max_idx+i]*scaling_factor_for_fair_dist_cal)
			# 	dist_list.append(dist)
			# 	slope=np.abs((array[max_idx+i]-value)/(max_idx+i-0))
			# 	slope_list.append(slope)
			
			# print('len slope_list', len(slope_list))
			# print('len dist_list', len(dist_list))

			# min_slope_idx=np.asarray(slope_list).argmin()
			# min_dist_idx=np.asarray(dist_list).argmin()
			# print('min_slope_idx', min_slope_idx)
			# print('min_dist_idx', min_dist_idx)
			# idx=min_dist_idx


			return ori_array[idx]






def clean_FalsePositive_detection(startIdx_series, stopIdx_series, ref_trace, mode='remove_small', threshold=0.5):



	if mode=='remove_small_change':

		new_startIdx_series=[]
		new_stopIdx_series=[]
		for i, evt_startIdx in enumerate(startIdx_series):
			evt_endIdx=stopIdx_series[i]

			startVal=ref_trace[evt_startIdx]
			pealVal=max(ref_trace[evt_startIdx:evt_endIdx])

			if (pealVal-startVal)>threshold:
				new_startIdx_series.append(evt_startIdx)
				new_stopIdx_series.append(evt_endIdx)


	elif mode=='remove_small_value':

		new_startIdx_series=[]
		new_stopIdx_series=[]
		for i, evt_startIdx in enumerate(startIdx_series):
			evt_endIdx=stopIdx_series[i]

			pealVal=max(ref_trace[evt_startIdx:evt_endIdx])
			if pealVal>threshold:
				new_startIdx_series.append(evt_startIdx)
				new_stopIdx_series.append(evt_endIdx)		


	elif mode=='remove_short_period':
		new_startIdx_series=[]
		new_stopIdx_series=[]		
		for i, evt_startIdx in enumerate(startIdx_series):
			evt_endIdx=stopIdx_series[i]

			if (evt_endIdx-evt_startIdx)>threshold:
				new_startIdx_series.append(evt_startIdx)
				new_stopIdx_series.append(evt_endIdx)

	elif mode=='select_specific_dur_evt':
		new_startIdx_series=[]
		new_stopIdx_series=[]		
		for i, evt_startIdx in enumerate(startIdx_series):
			evt_endIdx=stopIdx_series[i]

			if (evt_endIdx-evt_startIdx)>threshold[0] and (evt_endIdx-evt_startIdx)<threshold[1]-1:
				new_startIdx_series.append(evt_startIdx)
				new_stopIdx_series.append(evt_endIdx)


	return new_startIdx_series, new_stopIdx_series




def smooth_trace(trace, frame_window=9):

	window = np.ones(frame_window)/frame_window
	trace_smooth = np.convolve(trace, window, mode='same')
	trace_smooth[0] = trace[0]
	trace_smooth[-1] = trace[-1]

	return trace_smooth


def median_filter(trace, frame_window=9):

	return scipy.signal.medfilt(trace,kernel_size=frame_window)


def savgol_filter(trace, frame_window=9, polyorder=3):

	trace_hat = scipy.signal.savgol_filter(trace, frame_window, polyorder)

	return trace_hat


def FitSARIMAXModel(x,pcutoff=10,alpha=0.1,ARdegree=3,MAdegree=1,nforecast = 0, disp=False):
    # Seasonal Autoregressive Integrated Moving-Average with eXogenous regressors (SARIMAX)
    # see http://www.statsmodels.org/stable/statespace.html#seasonal-autoregressive-integrated-moving-average-with-exogenous-regressors-sarimax
    Y=x.copy()
    Y=np.asarray(Y)
    #Y[p<pcutoff]=np.nan # Set uncertain estimates to nan (modeled as missing data)
    if np.sum(np.isfinite(Y))>10:

        # SARIMAX implemetnation has better prediction models than simple ARIMAX (however we do not use the seasonal etc. parameters!)
        mod = sm.tsa.statespace.SARIMAX(Y.flatten(), order=(ARdegree,0,MAdegree),seasonal_order=(0, 0, 0, 0),simple_differencing=True)
        #Autoregressive Moving Average ARMA(p,q) Model
        #mod = sm.tsa.ARIMA(Y, order=(ARdegree,0,MAdegree)) #order=(ARdegree,0,MAdegree)
        try:
            res = mod.fit(disp=disp)
        except ValueError: #https://groups.google.com/forum/#!topic/pystatsmodels/S_Fo53F25Rk (let's update to statsmodels 0.10.0 soon...)
            startvalues=np.array([convertparms2start(pn) for pn in mod.param_names])
            res= mod.fit(start_params=startvalues,disp=disp)
        except np.linalg.LinAlgError:
            # The process is not stationary, but the default SARIMAX model tries to solve for such a distribution...
            # Relaxing those constraints should do the job.
            mod = sm.tsa.statespace.SARIMAX(Y.flatten(), order=(ARdegree, 0, MAdegree),
                                            seasonal_order=(0, 0, 0, 0), simple_differencing=True,
                                            enforce_stationarity=False, enforce_invertibility=False,
                                            use_exact_diffuse=False)
            res = mod.fit(disp=disp)

        predict = res.get_prediction(end=mod.nobs + nforecast-1)
        predict.predicted_mean[0]=Y[0]

        return predict.predicted_mean,predict.conf_int(alpha=alpha)

    else:
        return np.nan*np.zeros(len(Y)),np.nan*np.zeros((len(Y),2))



def butter_lowpass_filter(data, cutOff, fs, order=3):

    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = scipy.signal.butter(order, normalCutoff, btype='low', analog = True)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def fft_filter(trace, lf=2.5,hf=1.5, filename='fft_space.png'):

	fps=30
	start_time=0
	end_time=int(len(trace)/fps)
	time=np.linspace(start_time,end_time,len(trace))
	trace_copy=trace.copy()
	signal=np.asarray(trace_copy)

	

	W = fftfreq(signal.size, d=time[1]-time[0])
	f_signal = rfft(signal)
	f_signal[(W<0.1)] = 0

	# print('W',W)
	# print('len W', len(W))

	# If our original signal time was in seconds, this is now in Hz    
	band_f_signal = f_signal.copy()
	band_f_signal[(W<hf)] = 0
	band_f_signal[(W>lf)] = 0
	band_signal = irfft(band_f_signal)

	high_f_signal = f_signal.copy()
	high_f_signal[(W<7)] = 0
	high_f_signal[(W>15)] = 0
	high_signal = irfft(high_f_signal)

	low_f_signal = f_signal.copy()
	low_f_signal[(W<0)] = 0
	low_f_signal[(W>0.6)] = 0
	low_signal = irfft(low_f_signal)

	# Substract raw signal from low and high signals
	#Sub_raw_signal=signal-high_signal-low_signal
	Sub_raw_signal=signal-high_signal

	# #Plot fft
	# print('Plotting f-space of fft filter')
	# fig=plt.figure(facecolor='black', figsize=(25, 17), dpi=200)
	
	# plt.subplot(911)
	# plt.title('raw_signal')
	# plt.plot(time,signal)
	
	# plt.subplot(912)
	# plt.title('raw_f_signal')
	# plt.plot(W,f_signal)

	# plt.subplot(913)
	# plt.title('band_signal, '+str(lf)+'hz > W > '+str(hf)+' hz')
	# plt.plot(time, band_signal)

	# plt.subplot(914)
	# plt.title('band_f_signal, '+str(lf)+'hz > W > '+str(hf)+' hz')
	# plt.plot(W, band_f_signal)

	# plt.subplot(915)
	# plt.title('high_signal, 15 hz > W > 7 hz')
	# plt.plot(time, high_signal)

	# plt.subplot(916)
	# plt.title('high_f_signal, 15 hz > W > 7 hz')
	# plt.plot(W, high_f_signal)	

	# plt.subplot(917)
	# plt.title('low_signal, 0.6 hz > W > 0 hz')
	# plt.plot(time, low_signal)

	# plt.subplot(918)
	# plt.title('low_f_signal, 0.6 hz > W > 0 hz')
	# plt.plot(W, low_f_signal)

	# plt.subplot(919)
	# plt.title('raw_signal-high_signal')
	# plt.plot(time, Sub_raw_signal)



	# plt.tight_layout()
	# plt.savefig(outputPERdir+filename)
	# plt.clf()
	# plt.close(fig)



	return Sub_raw_signal



def filtered_traces(trace, filtermode='median', frame_window=3):

	trace_copy=trace.copy()



	if filtermode == 'running_window':
		trace_filt=smooth_trace(trace_copy, frame_window=frame_window)

	elif filtermode == 'median':
		trace_filt=median_filter(trace_copy, frame_window=frame_window)

	elif filtermode=='savgol_filter':
		trace_filt=savgol_filter(trace_copy, frame_window=frame_window, polyorder=2)

	elif filtermode=='sarimax':
		trace_filt, CI=FitSARIMAXModel(trace_copy)

	elif filtermode=='butter_lowpass':
		trace_filt = butter_lowpass_filter(trace_copy, cutOff=0.7, fs=frame_window, order=2) # cutoff frequency in rad/s; sampling frequency in rad/s; order of filter
	
	elif filtermode=='fft_filter':
		trace_filt=fft_filter(trace_copy, lf=7,hf=-0.01, filename='fft_space.png')

	return trace_filt
	








# def find_dominant_event(evt_startIdx_list, evt_endIdx_list, trace_set, save_dir, filename, ref_ROI=0, difference_thrsld=0.5, fps=30, mode='Norm_trace_to_0_1'):

# 	print('Finding dominant events ...')

# 	ref_trace_raw=trace_set[ref_ROI]

# 	if mode=='Norm_trace_to_0_1':
# 		ref_trace, _ ,_=normalize_trace(ref_trace_raw, frame_window=int(10*fps), mode='btwn_0and1')


# 	max_ref_trace=max(ref_trace)

# 	ROI_num_all=[i for i in range(len(trace_set))]
# 	print('ROI_num_all', ROI_num_all)

# 	ROI_num_all_copy=ROI_num_all.copy()
# 	ROI_num_all_copy.remove(ref_ROI)
# 	ROI_num_the_other=ROI_num_all_copy
# 	print('ROI_num_the_other', ROI_num_the_other)

# 	the_other_trace_set=[]
# 	the_other_ROI_ID_set=[]
# 	for i in ROI_num_the_other:
# 		trace=trace_set[i]
# 		if mode=='Norm_trace_to_0_1':
# 			trace, _ ,_ = normalize_trace(trace, frame_window=int(10*fps), mode='btwn_0and1')
# 		the_other_trace_set.append(trace)
# 		the_other_ROI_ID_set.append(i)

	
# 	dominant_evt_startIdx_list=[]
# 	dominant_evt_endIdx_list=[]
# 	for i, evt_startIdx in enumerate(evt_startIdx_list):

# 		evt_endIdx=evt_endIdx_list[i]

# 		mean_refROI_evt=np.mean(ref_trace[evt_startIdx:evt_endIdx])

# 		# Calcuale event mean of other ROIs
# 		mean_otherROI_evt=[]
# 		for j, trace in enumerate(the_other_trace_set):
# 			mean_curROI_evt=np.mean(trace[evt_startIdx:evt_endIdx])
# 			mean_otherROI_evt.append(mean_curROI_evt)

# 		## Count how many other ROIs' mean were lower than the reference ROI's mean
# 		Count_true=0
# 		for value in mean_otherROI_evt:
# 			if mean_refROI_evt-mean_otherROI_evt>difference_thrsld*max_ref_trace:
# 				Count_true+=1
# 		## if reference ROI's mean is larger than all the other, then it is dominant event. 
# 		if Count_true==len(the_other_ROI_ID_set):
# 			dominant_evt_startIdx_list.append(evt_startIdx)
# 			dominant_evt_endIdx_list.append(evt_endIdx)


# 	print(len(dominant_evt_startIdx_list), 'domiant events are found !')



# 	print('==Plotting preview of dominant event detection==')

# 	dominant_evt_startIdx_on_rawTrace=[]
# 	dominant_evt_endIdx_on_rawTrace=[]
# 	for i, evt_startIdx in enumerate(dominant_evt_startIdx_list):
# 		evt_endIdx=dominant_evt_endIdx_list[i]
# 		raw_startValue=ref_trace_raw[evt_startIdx]
# 		raw_endValue=ref_trace_raw[evt_endIdx]
# 		dominant_evt_startIdx_on_rawTrace.append(raw_startValue)
# 		dominant_evt_endIdx_on_rawTrace.append(raw_endValue)



# 	fig=plt.figure(facecolor='black', figsize=(25, 10), dpi=200)

	
# 	for idx, trace in enumerate(trace_set):
		
# 		total_row_num=len(trace_set)

# 		plt.subplot(total_row_num,1,idx+1)
# 		if len(evt_startIdx_list)==0:
# 			plt.title('ROI#'+str(idx)+' No dominant event found ')
# 		else:
# 			plt.title('ROI#'+str(idx))
# 		plt.plot(trace, color='k', linewidth=1)
# 		#plt.plot(trace_med, color='r', linewidth=1)
# 		#plt.plot(peaks_idx_rawTrace, peaks_of_rawTrace_on_rawTrace, marker='x', color='r',linestyle = 'None')
# 		#plt.plot(peaks_idx_gradTrace, peaks_of_gradTrace_on_rawTrace, marker='o', color='g',linestyle = 'None')
# 		if idx == ref_ROI:
# 			plt.plot(dominant_evt_startIdx_list, dominant_evt_startIdx_on_rawTrace, marker='^', color='b',linestyle = 'None')
# 			plt.plot(dominant_evt_endIdx_list, dominant_evt_endIdx_on_rawTrace, marker='v', color='r',linestyle = 'None')
		

# 		for i, evt_startIdx in enumerate(dominant_evt_startIdx_list):
# 			evt_endIdx=dominant_evt_endIdx_list[i]
# 			plt.axvspan(evt_startIdx, evt_endIdx, color='k', alpha=0.25, linewidth=0)

		
# 	# plt.subplot(212)
# 	# plt.title('Binary event trace')
# 	# plt.plot(evt_bin_trace, color='k',linewidth=1)

	
# 	plt.tight_layout()
# 	plt.savefig(save_dir+filename)
# 	plt.clf()
# 	plt.close(fig)



# 	return dominant_evt_startIdx_list, dominant_evt_endIdx_list













