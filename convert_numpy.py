###======================================================
###Andrea Gaspert
###12/05/2020

###Data formatting from csv to numpy array for
###Methane deep learning project
###Only need to run this once for given list of interesting properties

###======================================================

import numpy as np
import csv

# list of sites used to test other ml models:
site_id_list = ['JPBBY','NZKop','CASCB','USUaf','FISi2','FILom','FISii','SEDeg','USLos','USMyb','USOWC','USTw1','USTw4',
'USWPT','USTwt','JPMse','BRNpw']

def create_numpy(site_id):

	local_path = 'data/half_hourly/' #change with how locally setup
	file_path = local_path + site_id +'_V2.csv'


	#get list of available variables at this site
	with open(file_path, newline='') as f:
  		reader = csv.reader(f)
  		row1 = next(reader)  # gets the first line

  		avail_var = row1

	#extract data from csv file
	data_to_extract_list = row1[1:31]
	#get column numbers of variables to extract
	columns= []
	for j in range(len(data_to_extract_list)):
		index_of_col = avail_var.index(data_to_extract_list[j])
		columns.append(index_of_col)

	data = np.genfromtxt(file_path, delimiter=',', usecols = columns, dtype=None, names=True)

	#save numpy array to relevent path
	np.save('data/numpy/'+site_id+'.npy',data)

for idd in site_id_list:
	create_numpy(idd)