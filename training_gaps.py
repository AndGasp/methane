###======================================================
###Andrea Gaspert
###13/05/2020

###create train/test dataset to fill gaps for a given variable
###get distribution of gaps for given variable
###create artificial gaps for training using same distribution

###======================================================

import numpy as np
import matplotlib.pyplot as plt
import datetime 
from scipy.optimize import curve_fit


var_to_fill = ['FCH4_mean', 'FCH4_median'] 
site_id_list = ['JPBBY','NZKop','CASCB','USUaf','FISi2','FILom','FISii','SEDeg','USLos','USMyb','USOWC','USTw1','USTw4',
'USWPT','USTwt','JPMse','BRNpw']


def geo_fun(x,p,b):
	return p*(1-p)**(x-b)

def lin_fun(x,a,b):
	return b + a*x

def exp_fun(x,a,b):
	return np.exp(b)*np.exp(a*x)

def max_pos(day, time, flux):
	#returns day and time where top 5 max methane flux to position of maximum for yearly and dayly cosine
	return 1

def gap_list_fun(no_data_list):
	#Accepts list of indices where no data and produces list of gap lengths
	data_diff = np.diff(no_data_list) == 1

	gap_l = 1
	gap_list = [] #list of the lengths of gaps

	for i in range(len(data_diff)):
		if data_diff[i]:
			gap_l +=1

		if not data_diff[i]:
			gap_list.append(gap_l)
			gap_l = 1
	return gap_list


def gap_dist(n,site_id,var_to_fill):

	#function to create an array with n lists of status for every half-hourly data point 
	#(0 = real gap, 1 = un-hidden data, 2 = test, 3 = valid or train)

	data = np.load('data/numpy/'+site_id+'.npy')

	data_gaps = data[var_to_fill] #get data for mean methane flux for given site

	data_gaps_no_nan = np.nan_to_num(data_gaps) #convert nan to zeros

	i = 0
	while True:
		if data_gaps_no_nan[i] == 0:
			i+=1

		else:
			earliest = i
			break
	i=1
	while True:
		if data_gaps_no_nan[-i] == 0:
			i+=1

		else:
			latest = -i
			break


	trimmed_data = data_gaps[earliest:latest] #remove data before and after methane sensors online

	ind_if_data = np.argwhere(np.isnan(trimmed_data)==0)[:,0] #indices where data

	ind_to_fill = np.argwhere(np.isnan(trimmed_data))[:,0] #indices where no data

	#get gap length distribution================================================
	gap_list = gap_list_fun(ind_to_fill)

	#visualize data and gap distribution
	# Create two subplots and unpack the output array immediately


	data_min = np.amin(np.nan_to_num(trimmed_data))
	data_max = np.amax(np.nan_to_num(trimmed_data))

	time_axis = np.arange(len(data_gaps)+latest-earliest)
	f, ax1 = plt.subplots(1, 1)

	#visualize data
	ax1.set_ylim([data_min,data_max])
	i=0

	#plot real gaps
	for j in range(len(gap_list)):
		x_0 = ind_to_fill[i]
		x_1 = ind_to_fill[i+gap_list[j]-1]
		i = i+gap_list[j]
		if x_1 != x_0:
			ax1.fill([x_0,x_0,x_1,x_1],[data_min,data_max,data_max,data_min],c='r',alpha=0.3)
		else:
			ax1.plot([x_0,x_0],[data_min,data_max],'r',linewidth=1,alpha=0.3)
	
	
	"""
	#fix this part to get dates in figures instead of index

	dates = data['TIMESTAMP_END'][earliest:latest]
	date_list = []
	for date,i in dates:
		date_str = 


	x_values = [datetime.datetime.strptime(d,"%Y%m%d").date() for d in dates]
	print(x_values)

	formatter = mdates.DateFormatter("%Y-%m-%d")
	ax1.xaxis.set_major_formatter(formatter)

	locator = mdates.DayLocator()
	ax1.xaxis.set_major_locator(locator)
	"""
	#histogram of gap distribution
	ax1.set_title(var_to_fill)
	plt.show()

	#separate 3 regimes of gaps==========================================
	n_gaps = len(gap_list)
	gap_list.sort()
	gap_list_sort = np.array(gap_list)


	n_large = np.sum(gap_list_sort>18)#number of gaps larger than 1/2 day

	gap_list_rest = gap_list_sort[-n_large:]
	gap_list_short = gap_list_sort[:-n_large] #gaps of 1 day or less

	#Study of short gaps=============================================
	n_gaps_short = len(gap_list_short) #total number of short gaps

	#fit geometric distribution to short gap distribution
	y,borns = np.histogram(gap_list_short, bins=18, density=True)
	x = (borns[:-1] + borns[1:])/2

	popt1, pcov1 = curve_fit(geo_fun, x, y, p0=[0.5,0],bounds=(0, [1,0.00000001]))


	#Study of medium gaps (<5 days) ===================================
	n_verylarge = np.sum(gap_list_rest>250) #number of gaps larger than 7 days

	if n_verylarge > 0:
		gap_list_large = gap_list_rest[-n_verylarge:]
		gap_list_medium = gap_list_rest[:-n_verylarge] #gaps 1 day - 7 days
	else:
		gap_list_medium = gap_list_rest
		gap_list_large = []


	n_medium = len(gap_list_medium)
	n_gaps_large = len(gap_list_large)

	print(n_gaps_short,n_medium,n_gaps_large)

	y,borns = np.histogram(gap_list_medium, bins= 18, density=True)
	x=(borns[:-1] + borns[1:])/2

	i_good = np.where(y>0)
	y = y[i_good]
	x = x[i_good]


	popt2, pcov2 = curve_fit(geo_fun, x, y, p0=[0.05,18],bounds=(0, [1,20]))
	

	#show gap distribution and fit
	f, (ax2, ax3) = plt.subplots(1, 2)
	ax2.hist(gap_list_short,bins=18,density=True,label='Actual short gaps')
	ax2.set_xlabel('Gap lengths')

	#show fit
	xx = np.arange(18)+1
	yy = geo_fun(xx,popt1[0],popt1[1])

	ax2.plot(xx,yy,'b',label='fit')
	ax2.legend(loc='best')
	ax3.hist(gap_list_medium,bins=100,density=True,label='Actual medium gaps')

	xxx = np.arange(18,351)
	yyy = geo_fun(xxx,popt2[0],popt2[1])
	ax3.plot(xxx,yyy,'b',label='fit')
	plt.show()


	#Monte Carlo generation of gap dists for test, produce 10 choose closest============================================
	#random dist of lenghts
	n_s = int(1000*n_gaps_short/n_gaps) #number of sim short gaps
	n_m = int(1000*n_medium/n_gaps) #number of sim medium gaps
	print(n_s,n_m)
	if n_m + n_s == 1000:
		nm-=1


	if n_gaps_large>0:
		p_l = n_gaps_large/n_gaps #probability of a gap being long
		min_l = gap_list_large[0]
		max_l = gap_list_large[-1]
	

	#pick n_s  lengths in the corresponding geo. distribution
	#pick n_m lengths in the corresponding geo. disttribution
	#pick 1 length from flat dist. for a long gap for test
	lengths_sim = np.zeros((1000,510))
	lengths_sim[:n_s,:] = np.random.geometric(popt1[0], size=(n_s,510))+int(popt1[1])
	lengths_sim[n_s:n_s+n_m,:] = np.random.geometric(popt2[0], size=(n_m,510))+int(popt2[1])
	lengths_sim[-1,:10] = (max_l - min_l) * np.random.random_sample(size=10) + min_l

	n_l = np.random.random_sample(size=500)
	is_l = n_l<p_l
	lengths_sim[-1,10:] = is_l * ((max_l - min_l) * np.random.random_sample(size=500) + min_l)


	#random positions in time
	#pick 510*1000 random positions in time
	latest = len(data_gaps) + latest
	time_gaps = ((latest - earliest) * np.random.random_sample(size=(1000,510)) + earliest)


	#create status matrix of data (len of data x 510)
	#original matrix 1 if real gap 0 if data
	m_ori = (np.isnan(trimmed_data))
	m_test = np.zeros((len(m_ori),10)) #matrix with zeros where actual gaps, 1 where data, 2 where test gaps

	for j in range(10):
		for i in range(1000):
			end_gaps = int(time_gaps[i,j] + lengths_sim[i,j])
			start_gaps = int(time_gaps[i,j])

			m_test[start_gaps: end_gaps, j] = (m_ori[start_gaps: end_gaps] + 1) 


	m_test = (m_test==1) #one where test gaps
	print(m_test[:,0])
	#print(m_test)
	#m_test = (m_test == 0) #zeros where test position

	#Choose best test gap distribution for accurate testing
	for i in range(10):
		i_test_only = np.argwhere(m_test[:,i])[:,0] #indices where no data
		l_test_only = gap_list_fun(i_test_only)
		cond = (m_ori+m_test[:,i]) > 0
		i_test_and_ori = np.argwhere(cond)[:,0]
		print(l_test_only)
		l_test_and_ori = gap_list_fun(i_test_and_ori)
		print(l_test_and_ori)

		#plot distributions, normalized
		#show gap distribution and fit
		f, ax4 = plt.subplots(1, 1)
		ax4.hist(gap_list_sort,bins=200,range=(0,1000),density=True,alpha = 0.4, label='Actual gaps')
		ax4.hist(l_test_only,bins=200,range=(0,1000),density=True,alpha = 0.4, label='Test gaps only')
		ax4.hist(l_test_and_ori,bins=200,range=(0,1000),density=True,alpha = 0.4, label='Test and original gaps')
		ax4.set_xlabel('Gap lengths')
		ax4.legend(loc='best')
		plt.show()


		#test value


	#compare final gap distribution and only test gap dist. to original gap dist.






	m_else = np.zeros((len(m_ori),500)) #matrix with zeros where actual gaps, 1 where data, 2 where test gaps, 3 where valid/train gap




#visualize gaps for all available variables
for idd in site_id_list:
	gap_dist(20,idd,'FCH4')