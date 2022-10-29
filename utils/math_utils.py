
# import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import warnings
import sys
import math
import scipy.stats
from multiprocessing import Pool
from itertools import repeat
import more_itertools as mit
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# from skimage.filters import threshold_otsu
# from skimage.filters import threshold_minimum
# from skimage.filters import threshold_mean

import skimage.filters




def smooth_data(data, windowlen=10):

    if windowlen==0:
        smooth_data=data
    else:
        window = np.ones(int(windowlen))/float(windowlen)
        smooth_data=np.convolve(data,window,'same')

    return smooth_data



def median_filter(trace, frame_window=9):

	return scipy.signal.medfilt(trace,kernel_size=frame_window)


def grouping_consecutivePoints_into_evt(idx_beh):

    
    idx_beh_evt=[]
    for evt in mit.consecutive_groups(idx_beh):
        #type(evt) is map, once it is converted into list, it become an empty array
        #print(list(evt))
        idx_beh_evt.append(list(evt))    

    return idx_beh_evt


def linear_regress(arr1, arr2):

	slope, intercept, Corr_coeff, p_value, std_err = scipy.stats.linregress(arr1, arr2)

	return slope, intercept, Corr_coeff, p_value, std_err


def compute_CI_and_mean(data, confidence=0.95):


	a = 1*np.array(data)
	n = len(a)
	m = np.mean(a)
	se = scipy.stats.sem(a)
	intvl = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)


	return (m, m-intvl, m+intvl)



def compute_mean_CI_sem(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a, axis=0), scipy.stats.sem(a, axis=0, nan_policy='propagate')
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, m-se, m+se



def mp_worker_for_CI_mean_trace(idx, dataset, confidence, least_realNum_amount):

	dataset=np.asarray(dataset)
	# print('idx', idx)
	# print('shape dataset', np.shape(dataset))
	# print('least_realNum_amount', least_realNum_amount)
	# print('len dataset', len(dataset))
	data=np.asarray(dataset[:, idx])
	# print('type data', type(data))
	# print('data', data)
	

	non_nan_counts=data.size - np.count_nonzero(np.isnan(data))

	if non_nan_counts > least_realNum_amount:

		nonnan_data_list=data[np.logical_not(np.isnan(data))] 

		(mean, down_CI, up_CI) = compute_CI_and_mean(nonnan_data_list, confidence=confidence)

		return mean, down_CI, up_CI

	else:

		return np.nan, np.nan, np.nan




def compute_CI_and_mean_trace(dataset, confidence=0.95, least_realNum_amount=4):

	

	idx_range=[i for i in range(0,np.shape(dataset)[1])]
	# print('colums_range', colums_range)

	p=Pool()
	mean_downCI_upCI = p.starmap(mp_worker_for_CI_mean_trace, zip(idx_range, repeat(dataset), repeat(confidence), repeat(least_realNum_amount)))
	#mean_trace, down_err_trace, up_err_trace = p.starmap(worker_for_CI_mean_trace, args_for_pool)

	p.close()
	p.join()
	del p	

	mean_trace=np.asarray(mean_downCI_upCI)[:,0]
	down_err_trace=np.asarray(mean_downCI_upCI)[:,1]
	up_err_trace=np.asarray(mean_downCI_upCI)[:,2]


    
	return mean_trace, down_err_trace, up_err_trace



def add_NaN_tail(raw_trace, expected_len):

    if not isinstance(raw_trace, list): 

        raw_trace=raw_trace.tolist()

    if expected_len>len(raw_trace):

        NaN_len=expected_len-len(raw_trace)

        NaN_tail=[np.nan]*NaN_len

        # print('len NaN_tail', len(NaN_tail))

        raw_trace_NaNtail=raw_trace+NaN_tail

        #  print('len raw_trace_NaNtail', len(raw_trace_NaNtail))

    elif expected_len==len(raw_trace):
        raw_trace_NaNtail=raw_trace

    else:
        print('raw_trace length < expexted length. Please check if something wrong ...')

    return raw_trace_NaNtail


def add_NaNtail_to_each_Evt(Evt_list):

	Evt_len_list=[]
	for evt in Evt_list:
		# print('np.shape(evt', np.shape(evt))
		Evt_len_list.append(np.shape(evt)[0])

	max_EvtLen=max(Evt_len_list)

	# print('max_EvtLen', max_EvtLen)

	EvtNaNtail_list=[]
	for evt in Evt_list:
		evt_NaNtail=add_NaN_tail(evt, max_EvtLen)
		#print('len evt_NaNtail', len(evt_NaNtail))
		EvtNaNtail_list.append(evt_NaNtail)

	return EvtNaNtail_list







def lowpass_binary_trace(trace, fps=30, corner_dur_s=0.5):



	low_pass_frames=int(fps*corner_dur_s)
	#print('low_pass_frames', low_pass_frames)

	evt_idxs=np.where(np.asarray(trace) == 1)[0]
	# print(idx_behevt_consecut)
	idx_evts=grouping_consecutivePoints_into_evt(evt_idxs)

	print('shape idx_evts', np.shape(idx_evts))

	print('len trace', len(trace))


	trace_cp=np.asarray(trace.copy())
	print('len trace_cp', len(trace_cp))
	for evt_idx in idx_evts:
		# print('evt_idx', evt_idx)
		# print('len evt_idx', len(evt_idx))
		# print(beh_trace_cp[evt_idx])
		if len(evt_idx)<low_pass_frames:
			trace_cp[evt_idx]=0
		# print(beh_trace_cp[evt_idx])
	trace_LP=list(trace_cp)

	print('len trace_LP', len(trace_LP))

	# list_trace=[
	# trace,
	# trace_LP
	# ]

	# subtitle=['raw', 'LP']

	# plot_utils.Plot_traces(list_trace, glm_GC_Beh_summary_dir, 'Beh_traces_LP.png', subtitle_list=subtitle, plot_mode='row_by_row')

	# print('Plotting LP beh traces')



	return trace_LP




def norm_to_max(data, percentile_th_to_norm=100):

    data = np.asarray(data)

    data_cp=[x for x in data if x==x]


    data_cp=sorted(data_cp)
    max_data = data_cp[int((percentile_th_to_norm/100)*len(data_cp))-1]

    # max_data=np.nanmax(data)

    # print('max_data', max_data)



    norm_data = data/max_data

    #print('max norm_data', np.nanmax(norm_data))
    # print('data', data)
    # print('norm_data', norm_data)

    

    return norm_data


def norm_to_val(data, val=1):

	data = np.asarray(data)
	norm_data = data/val

	return norm_data



def compute_dFF(gcamp, len_baseline):
    """
    Compute and return the fluorescence traces computed as dF/F and dR/R.
    
    Parameters
    ----------
    tdtom : ndarray
        Time series of average tdTomato intensity for each axon. If an axon is 
        not present on a frame, its intensity is set to np.nan for that frame.
        Shape is n_axons x n_images.
    gcamp : ndarray
        Same, but for the GCaMP fluorohpore.
    len_baseline : int
        Number of consecutive frames to consider for the baseline computation.
        eg., len_baseline = int(BIN_S * RATE_HZ + 0.5), BIN_S=10,  RATE_HZ=2.41
    
    Returns
    -------
    dFF : ndarray
        Fluorescence traces for each axon computed as dF/F, in percent.
        Shape is n_axons x n_images.
    dRR : ndarray
        Same, but computed as dR/R.
    """
    dFF = np.zeros_like(gcamp)
    
    # Loop on axons
    for n in range(dFF.shape[0]):
        F_t = gcamp[n].copy()
        
        # Compute baselines by looping on frames
        F_0 = []
        for i in range(max(0, dFF.shape[1] - len_baseline)):
            # Ignore empty slice warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                F_0.append(np.nanmean(F_t[i: i + len_baseline]))
        F_0 = np.nanmin(F_0)
        
        dFF[n] = (F_t - F_0) / F_0 * 100
    
    return dFF


def hysteresis_filter(seq, n=5, n_false=None):
    """
    This function implements a hysteresis filter for boolean sequences.
    The state in the sequence only changes if n consecutive element are in a different state.
    Parameters
    ----------
    seq : 1D np.array of type boolean
        Sequence to be filtered.
    n : int, default=5
        Length of hysteresis memory.
    n_false : int, optional, default=None
        Length of hystresis memory applied for the false state.
        This means the state is going to change to false when it encounters
        n_false consecutive entries with value false.
        If None, the same value is used for true and false.
    Returns
    -------
    seq : 1D np.array of type boolean
        Filtered sequence.
    """
    if n_false is None:
        n_false = n
    seq = np.asarray(seq).astype(np.bool)
    state = seq[0]
    start_of_state = 0
    memory = 0

    current_n = n
    if state:
        current_n = n_false

    for i in range(len(seq)):
        if state != seq[i]:
            memory += 1
        elif memory < current_n:
            memory = 0
            continue
        if memory == current_n:
            seq[start_of_state : i - current_n + 1] = state
            start_of_state = i - current_n + 1
            state = not state
            if state:
                current_n = n_false
            else:
                current_n = n
            memory = 0
    seq[start_of_state:] = state


    

    return seq




def compute_mean_with_diffrerent_row_length(list_w_diffrent_rowsize, samplerate=1500, cutting_head_s=0):

	mean_list=[]
	for row in list_w_diffrent_rowsize:
		#print('row', row)
		if cutting_head_s*samplerate<len(row):
			row_mean=np.nanmean(row[int(cutting_head_s*samplerate):-1])
			mean_list.append(row_mean)

	clean_mean_list = [x for x in mean_list if str(x) != 'nan']

	clean_mean_list=np.asarray(clean_mean_list)

	print('len clean_mean_list', len(clean_mean_list))

	return clean_mean_list





def grouping_consecutivePoints_into_evt(idx_beh):

   
    idx_beh_evt=[]
    for evt in mit.consecutive_groups(idx_beh):
        #type(evt) is map, once it is converted into list, it become an empty array
        #print(list(evt))
        idx_beh_evt.append(list(evt))    

    return idx_beh_evt




def find_corresponding_evt_from_groupIdxs(idx_evt, corresponding_trace, baseline=0, fps=30):

    # baseline is for including the previous trace as baseline for the corresponding epoch

    
	bsl_len=int(baseline*fps)

	if len(idx_evt)>0:

		corr_evt=[]
		for j in range(0,len(idx_evt)): 
			if idx_evt[j][0]-bsl_len>0:          
				corr_evt.append(corresponding_trace[idx_evt[j][0]-bsl_len:idx_evt[j][-1]+1])


	else:
		corr_evt=[]
		for j in range(0,len(idx_evt)):           
			corr_evt.append(np.nan)


	return corr_evt 






def calc_residual_sum_of_squares(arr1, arr2):

	if np.shape(arr1)==np.shape(arr2):
		rss=0
		for i in range(0, len(arr1)):
			ele_arr1=arr1[i]
			ele_arr2=arr2[i]

			ss=(ele_arr1-ele_arr2)**2
			rss+=ss

	else:
		print('The two array must have the same length to compute...')
		rss=np.nan


	return rss


def cal_diff_between_dmean_and_3STD(old_arr, new_arr):

	old_std=np.std(old_arr)
	old_mean=np.nanmean(old_arr)
	
	new_mean=np.nanmean(new_arr)

	print('old_mean', old_mean)
	print('new_mean', new_mean)

	d_mean = abs(new_mean-old_mean)


	return d_mean, old_std



def align_to_bsl_for_evtlist(List2d, bsl_len=0, data_freq=1):

	aligned_evtlist=[]

	for i, evt in enumerate(List2d):

		bsl_mean=np.nanmean(evt[:int(bsl_len*data_freq)])

		algined_evt=np.asarray(evt)-bsl_mean
		aligned_evtlist.append(algined_evt)


	return aligned_evtlist





def find_baseline_of_trace_by_thresholding_histogram(trace, filepath, filename, samplerate=1500, descent=True):

	print('len trace', len(trace))

	trace_pre= np.asarray(trace)
	iter_count=0
	iteration=False
	while iteration==True:

		thresh_otsu = skimage.filters.threshold_otsu(trace_pre)
		thresh_mean = skimage.filters.threshold_mean(trace_pre)
		thresh_yen = skimage.filters.threshold_yen(trace_pre)
		thresh_li = skimage.filters.threshold_li(trace_pre)
		thresh_iso = skimage.filters.threshold_isodata(trace_pre)
		

		trace_bsl_otsu_bool = trace_pre < thresh_otsu
		trace_bsl_mean_bool = trace_pre < thresh_mean
		trace_bsl_yen_bool = trace_pre < thresh_yen
		trace_bsl_li_bool = trace_pre < thresh_li


		trace_bsl_otsu = trace_bsl_otsu_bool*trace_pre
		trace_bsl_mean = trace_bsl_mean_bool*trace_pre
		trace_bsl_yen = trace_bsl_yen_bool*trace_pre
		trace_bsl_li = trace_bsl_li_bool*trace_pre


		print('trace_bsl_otsu', trace_bsl_otsu)
		print('trace_bsl_otsu_bool', trace_bsl_otsu_bool)
		print('len trace_bsl_otsu', len(trace_bsl_otsu))	
		print('type trace_bsl_otsu', type(trace_bsl_otsu))

		# min_trace_pre=min(min_trace_pre)
		# max_trace_pre=min(max_trace_pre)
		# range_trace_pre=max_trace_pre-min_trace_pre
		# intvl_hist=10

		hist, bin_edges = np.histogram(trace_pre, bins=10, density=True)
		print('hist', hist)
		print('bin_edges', bin_edges)

		idx_max = list(hist).index(max(hist))
		print('idx_max', idx_max)
		thresh_manual_idx = idx_max+1
		thresh_manual = bin_edges[thresh_manual_idx]
		trace_bsl_manual_bool = trace_pre < thresh_manual
		trace_bsl_manual = trace_bsl_manual_bool*trace_pre

		if descent==False:
			iteration=False

		mean_diff, std_pre=cal_diff_between_dmean_and_3STD(trace_pre, trace_bsl_otsu)
		print('mean_diff', mean_diff)
		print('std_pre', std_pre)

		diff_mean_std = mean_diff-4*std_pre

		if iter_count==0:
			diff_mean_std=1
		print('diff_mean_std', diff_mean_std)
		if diff_mean_std<0:
			iteration=False

		trace_pre = trace_bsl_otsu
		iter_count+=1
		print('iter_count', iter_count)



	fig = plt.figure(facecolor='white', figsize=(20,4*5), dpi=170)

	plt.subplot(6,1,1)
	plt.hist(trace, len(bin_edges)-1, density=True, facecolor='g', alpha=0.75)
	plt.axvline(x=thresh_manual,color='r')
	plt.axvline(x=thresh_mean,color='c')
	plt.axvline(x=thresh_yen,color='y')
	plt.axvline(x=thresh_li,color='m')
	plt.axvline(x=thresh_otsu,color='g')


	plt.subplot(6,1,2)
	plt.title('manual')
	plt.plot(trace, color='k') 
	plt.plot(trace_bsl_manual, color='r') 


	plt.subplot(6,1,3)
	plt.title('mean')
	plt.plot(trace, color='k') 
	plt.plot(trace_bsl_mean, color='c') 


	plt.subplot(6,1,4)
	plt.title('yen')
	plt.plot(trace, color='k') 
	plt.plot(trace_bsl_yen, color='y') 


	plt.subplot(6,1,5)
	plt.title('li')
	plt.plot(trace, color='k') 
	plt.plot(trace_bsl_li,  color='m') 


	plt.subplot(6,1,6)
	plt.title('otsu')
	plt.plot(trace, color='k') 
	plt.plot(trace_bsl_otsu,  color='g') 


	plt.savefig(filepath + filename+'.png', edgecolor='none', transparent=True) #bbox_inches='tight', 
	plt.clf()
	plt.close()  


	nonzeroidx_bsl = np.where(trace_bsl_manual_bool)[0]
	# nonzeroidx_bsl = np.where(trace_bsl_mean_bool)[0]
	# nonzeroidx_bsl = np.where(trace_bsl_otsu_bool)[0]

	idx_bsl_epoch=grouping_consecutivePoints_into_evt(nonzeroidx_bsl)
	bsl_epochs=find_corresponding_evt_from_groupIdxs(idx_bsl_epoch, trace, baseline=0, fps=samplerate)


	filtered_bsl_epcoh=[]
	for i, v in enumerate(bsl_epochs):
		if len(v)>1*samplerate:
			filtered_bsl_epcoh.append(v)

	#print('idx_bsl_epoch', idx_bsl_epoch)
	print('shape idx_bsl_epoch', np.shape(idx_bsl_epoch))
	print('shape bsl_epochs', np.shape(bsl_epochs))


	trace_concat=trace_bsl_manual
	# trace_concat=trace_bsl_mean
	bsl_datapoint_nonzero_concat = trace_concat[trace_concat!=0]



	return bsl_datapoint_nonzero_concat, bsl_epochs




def find_baseline_of_trace_by_runningWindow(trace_set, filepath, filename, tune_range_fold=0.1, window_len_s=10, samplerate=1500):
	print('find_baseline_of_trace_by_runningWindow....')
	print('window_len_s', window_len_s)
	print('tune_range_fold', tune_range_fold)

	print('shape trace_set', np.shape(trace_set))

	gapfree_trace=[]
	gapfree_bsl=[]
	bsl_epochs_all=[]

	up_bounds=[]

	for trace in trace_set:

		trace=np.asarray(trace)
		# trace=smooth_data(trace, windowlen=1*samplerate)
		trace=smooth_data(trace, windowlen=0.3*samplerate)

		gapfree_trace.extend(trace)

		window_size=int(window_len_s*samplerate)

		running_mean_list=[]
		windowTrace_list=[]

		looping_position_list=range(0,len(trace), int(samplerate/4.3))

		for i, v in enumerate(looping_position_list):
			if v+window_size<len(trace):
				trace_in_window = trace[v:v+window_size]
				running_mean=np.nanmean(trace_in_window)
				windowTrace_list.append(trace_in_window)
				running_mean_list.append(running_mean)

		
		idx_min_run_mean = running_mean_list.index(np.nanmin(running_mean_list))
		windowTrace_w_minMean=list(windowTrace_list[idx_min_run_mean])
		print('len windowTrace_w_minMean', len(windowTrace_w_minMean))
		print('type windowTrace_w_minMean', type(windowTrace_w_minMean))

		windowTrace_w_minMean.sort()
		# print('windowTrace_w_minMean', windowTrace_w_minMean)
		print('len windowTrace_w_minMean', len(windowTrace_w_minMean))
		print('type windowTrace_w_minMean', type(windowTrace_w_minMean))
		print('int(0.9*len(windowTrace_w_minMean))', int(0.9*len(windowTrace_w_minMean)))

		max_windowTrace_w_minMean=windowTrace_w_minMean[int(0.9*len(windowTrace_w_minMean))]
		min_windowTrace_w_minMean=np.nanmin(trace)
		print('max_windowTrace_w_minMean', max_windowTrace_w_minMean)
		print('min_windowTrace_w_minMean', min_windowTrace_w_minMean)


		up_bound = max_windowTrace_w_minMean+max_windowTrace_w_minMean*tune_range_fold 
		low_bound = min_windowTrace_w_minMean+min_windowTrace_w_minMean*tune_range_fold
		print('up_bound', up_bound)
		print('low_bound', low_bound)

		up_bounds.append(up_bound)
		


		trace_thresh_bool=[]
		for i, v in enumerate(trace):
			if v > low_bound and v < up_bound:
				trace_thresh_bool.append(True)
			else:
				trace_thresh_bool.append(False)

		nonzeroidx_bsl = np.where(trace_thresh_bool)[0]
		idx_bsl_epoch=grouping_consecutivePoints_into_evt(nonzeroidx_bsl)
		bsl_epochs=find_corresponding_evt_from_groupIdxs(idx_bsl_epoch, trace, baseline=0, fps=samplerate)	


		trace_bsl = trace_thresh_bool*trace
		trace_bsl = np.where(trace_bsl==0, np.nan, trace_bsl)
		bsl_datapoint_nonzero = trace_bsl[trace_bsl!=np.nan]

		bsl_epochs_all.extend(bsl_epochs)
		gapfree_bsl.extend(bsl_datapoint_nonzero)

	print('shape gapfree_trace', np.shape(gapfree_trace))
	print('shape bsl_epochs_all', np.shape(bsl_epochs_all))


	hist, bin_edges = np.histogram(gapfree_trace, bins=15, density=True)

	fig = plt.figure(facecolor='white', figsize=(20,4*5), dpi=170)
	plt.subplot(6,1,1)
	plt.hist(gapfree_trace, len(bin_edges)-1, density=True, facecolor='g', alpha=0.75)
	for i, thres in enumerate(up_bounds):
		plt.axvline(x=thres,color='r')
		plt.text(thres,max(hist),str(thres), color='r', size=20)
 
	plt.subplot(6,1,2)
	plt.title('manual')
	plt.plot(gapfree_trace, color='k') 
	plt.plot(gapfree_bsl, color='r') 


	plt.savefig(filepath + filename+'.png', edgecolor='none', transparent=True) #bbox_inches='tight', 
	plt.clf()
	plt.close()  


	return gapfree_bsl, bsl_epochs_all




def find_baseline_of_trace01(trace01_set, trace_set, filepath, filename, tune_range_fold=0.1, window_len_s=10, samplerate=1500):

	print('Computing baseline in trace01 ...')


	gapfree_trace_raw=[]
	gapfree_trace_01=[]
	gapfree_bsl=[]
	bsl_epochs_all=[]

	up_bounds=[]

	for i, trace in enumerate(trace_set):
		# print('shape trace', np.shape(trace))
		gapfree_trace_raw.extend(list(trace))
	# print('shape gapfree_trace_raw', np.shape(gapfree_trace_raw))

	fold=20
	votes=[]
	for i in range(0,fold):
		gapfree_trace_sampled=np.random.choice(gapfree_trace_raw, size=150)
		# _, p_normality=scipy.stats.normaltest(gapfree_trace_sampled)
		# _, p_normality=scipy.stats.anderson(gapfree_trace_sampled)
		_, p_normality=scipy.stats.shapiro(gapfree_trace_sampled)
		# print('p_normality', p_normality)
		votes.append(p_normality)

	count_p_lessthan_005=sum(map(lambda x : x<0.05, votes))
	print(count_p_lessthan_005, 'out of', fold, 'normality test are <0.05.')

	# count_p_lessthan_005=len(votes)



	if count_p_lessthan_005<=len(votes)*0.6:
		print('-->The trace are baseline noise')
		for i, trace01 in enumerate(trace01_set):

			thresh_trace=np.nanmax(trace01)

			trace_thresh_bool = trace01 <= thresh_trace
			trace_bsl=trace01

			trace_bsl = np.where(trace_bsl==0, np.nan, trace_bsl)
			bsl_datapoint_nonzero = trace_bsl[trace_bsl!=np.nan]

			# print('trace_bsl', trace_bsl)
			# print('bsl_datapoint_nonzero', bsl_datapoint_nonzero)

			bsl_concat_evt=[x for x in trace_bsl if x==x]
			# bsl_concat_evt=median_filter(bsl_concat_evt, frame_window=int(1*samplerate)+1)

			gapfree_bsl.extend(trace_bsl)
			bsl_epochs_all.append(trace_bsl)

			normal=True


	else:
		print('-->The trace are signal, not baseline noise.')
		for i, trace01 in enumerate(trace01_set):

			trace01=np.asarray(trace01)
			# trace=smooth_data(trace, windowlen=1*samplerate)
			# trace=smooth_data(trace, windowlen=1*samplerate)
			# trace=median_filter(trace, frame_window=int(0.5*samplerate)+1)


			gapfree_trace_01.extend(trace01)

			thresh_otsu = skimage.filters.threshold_otsu(trace01)
			thresh_mean = skimage.filters.threshold_mean(trace01)
			thresh_yen = skimage.filters.threshold_yen(trace01)
			thresh_li = skimage.filters.threshold_li(trace01)
			thresh_iso = skimage.filters.threshold_isodata(trace01)

			
			trace_bsl_otsu_bool = trace01 < thresh_otsu
			trace_bsl_mean_bool = trace01 < thresh_mean
			trace_bsl_yen_bool = trace01 < thresh_yen
			trace_bsl_li_bool = trace01 < thresh_li


			trace_bsl_otsu = trace_bsl_otsu_bool*trace01
			trace_bsl_mean = trace_bsl_mean_bool*trace01
			trace_bsl_yen = trace_bsl_yen_bool*trace01
			trace_bsl_li = trace_bsl_li_bool*trace01

			thresh_trace=thresh_otsu
			trace_bsl=trace_bsl_otsu
			trace_thresh_bool=trace_bsl_otsu_bool



			trace_bsl = np.where(trace_bsl==0, np.nan, trace_bsl)
			bsl_datapoint_nonzero = trace_bsl[trace_bsl!=np.nan]

			bsl_concat_evt=[x for x in trace_bsl if x==x]
			# bsl_concat_evt=median_filter(bsl_concat_evt, frame_window=int(1*samplerate)+1)

			nonzeroidx_bsl = np.where(trace_thresh_bool)[0]
			idx_bsl_epoch=grouping_consecutivePoints_into_evt(nonzeroidx_bsl)
			bsl_epochs=find_corresponding_evt_from_groupIdxs(idx_bsl_epoch, trace01, baseline=0, fps=samplerate)	

			gapfree_bsl.extend(bsl_datapoint_nonzero)
			bsl_epochs_all.extend(bsl_epochs)

			normal=False


	gapfree_bsl_nonnan=[x for x in gapfree_bsl if x==x]
	# gapfree_bsl_nonnan=median_filter(gapfree_bsl_nonnan, frame_window=int(1*samplerate)+1)

	fig = plt.figure(facecolor='white', figsize=(20,4*5), dpi=170)

	plt.subplot(6,1,1)
	hist, bin_edges = np.histogram(gapfree_trace_01, bins=15, density=True)
	plt.hist(gapfree_trace_01, len(bin_edges), density=True, color='g', alpha=0.75)
	hist, bin_edges = np.histogram(gapfree_bsl_nonnan, bins=15, density=True)
	plt.hist(gapfree_bsl_nonnan, len(bin_edges), density=True, color='r', alpha=0.75)
	# plt.axvline(x=thresh_mean,color='c')
	# plt.axvline(x=thresh_yen,color='y')
	# plt.axvline(x=thresh_li,color='m')
	plt.axvline(x=thresh_trace,color='r')


	plt.subplot(6,1,2)
	plt.title('otsu')
	plt.plot(gapfree_trace_raw, color='g') 
	plt.plot(gapfree_bsl, color='r') 
	plt.plot(gapfree_bsl_nonnan, color='k', alpha=0.5) 

	plt.subplot(6,1,3)
	plt.title('otsu')
	plt.plot(gapfree_trace_01, color='g') 
	plt.plot(gapfree_bsl, color='r') 
	plt.plot(gapfree_bsl_nonnan, color='k', alpha=0.5) 


	plt.savefig(filepath + filename+'.png', edgecolor='none', transparent=True) #bbox_inches='tight', 
	plt.clf()
	plt.close()  



	return gapfree_bsl_nonnan, bsl_epochs_all, normal




def bootstrapped_twoSamplesComparison(dataGroup1, dataGroup2, fold=10, resample_size=6, stats='Mann-Whitney'):

	


	fold=10 
	votes=[]
	
	for i in range(0,fold):
		sampled_pnts1=np.random.choice(dataGroup1, size=resample_size)
		sampled_pnts2=np.random.choice(dataGroup2, size=resample_size)

		if stats=='Mann-Whitney':
			U_value, p_value = scipy.stats.mannwhitneyu(sampled_pnts1, sampled_pnts2)
		elif stats=='t-test':
			t_value, p_value = scipy.stats.ttest_ind(gc_point_offballmove_sampled, gc_point_onballmove_sampled)
		print('p_value', p_value)

		votes.append(p_value)


	count_p_lessthan_005=sum(map(lambda x : x<0.05, votes))
	print(count_p_lessthan_005, 'out of', fold, 'Mann-Whitney test are <0.05.')
	
	
	p_value_list=[]
	if count_p_lessthan_005<=len(votes)*0.6:
		print('-->GC datapoint is no different between ON and OFF ball')
		p_value_list.append(1)
	else:
		print('-->GC datapoint is significant different between ON and OFF ball')
		count_p_lessthan_0001=sum(map(lambda x : x<0.001, votes))
		count_p_lessthan_001=sum(map(lambda x : x<0.01, votes))-count_p_lessthan_0001
		count_p_lessthan_005=sum(map(lambda x : x<0.05, votes))-count_p_lessthan_0001-count_p_lessthan_001
		
		

		p_vote_list=[count_p_lessthan_005, count_p_lessthan_001, count_p_lessthan_0001]
		max_p_vote_idx=	p_vote_list.index(max(p_vote_list))	
		if max_p_vote_idx==0:
			p_value_list.append(0.04)	
		elif max_p_vote_idx==1:
			p_value_list.append(0.009)	
		elif max_p_vote_idx==2:
			p_value_list.append(0.0009)	


	assigne_p=p_value_list[0]


	return assigne_p














    
