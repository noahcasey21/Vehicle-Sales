import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import curve_fit
import math
from sklearn import linear_model
import csv
from datetime import datetime

#period (months)
global T
T = 12

global startyear
startyear = 1976
global endyear
endyear = 2019

#-------------------Functions--------------------------------

def next_congruence(num,mod):
    """Obtains the minimum value above num in the congruence class 
        of mod"""
    while num % mod != 0:
        num += 1

    return num

def sinusoid(x,A,offset,omega,phase):
    return A*np.sin(omega*x+phase) + offset

def get_p0(x, y):
    A0 = (max(y[0:T]) - min(y[0:T]))/2
    offset0 = y[0]
    phase0 = 0
    omega0 = 2.*np.pi/T
    return [A0, offset0,omega0, phase0]

def get_peaks(y,metrics):
    n = int(math.ceil(len(y)/T))
    step = 0
    x_peaks = []
    y_peaks = []

    for i in range(0,n):
        peak_index = y.index(metrics(y[step:step+T]))
        x_peaks.append(peak_index)
        y_peaks.append(y[peak_index])
        step = step+T
    return [x_peaks,y_peaks]

def variable_sinusoid(features,omega,phase):
    x = features[0]
    A = features[1]
    offset = features[2]
    return A*np.sin(omega*x+phase) + offset

def variable_get_p0(x, y): 
    phase0 = 0
    omega0 = 2.*np.pi/T
    return [omega0, phase0]

def clean_years(start,years,data):
    #removes entire years from the data, years MUST be in chronological order
    for year in years:
        index = (year - start) * T
        
        for i in range(T):
            del data[index]
            index += 1
        start +=1

    return data

def initialize_labels(start,stop,init_offset,end_offset,exclusions):
    """initializes a list of years to be used as monthly labels
        format: [1990,1991,1991,1991,1991,...]"""
    labels = []

    while start <= stop:
        if start not in exclusions:
            for i in range(T):
                labels.append(start)
        start += 1

    #remove offsets
    labels = labels[init_offset:-end_offset]
    return labels

def last_tick(current,months,step):
    """Determines what last tick on x-axis should be"""
    jump = 0
    origin = current
    for i in range(step - 1):
        current += 1
        if current not in months:
            jump += 1
    origin += jump + step
    return origin

#-------------------------------------------------------------

data = pd.read_excel('TOTALNSA.xlsx','Sheet3')

sales = data['Amount'].tolist()

#clean out recession periods
#recession_years = [1980,1981,1982,1990,1991,1992,2001,2007,2008,2009]
#consideration with no recession
recession_years = []
sales = clean_years(1976,recession_years,sales)

#months indexed at 0
months = np.arange(0, len(sales))

month_labels = initialize_labels(startyear,endyear,0,1,recession_years)

#sinusoidal approximation
#param, covariance = curve_fit(sinusoid, months, sales, p0=get_p0(months,sales))

#variable sinusoid
min_peaks = get_peaks(sales,min)
max_peaks = get_peaks(sales,max)

A = []
offset = []
for i in range(0, len(min_peaks[1])):
    c_a = (max_peaks[1][i] - min_peaks[1][i])/2
    c_offset = min_peaks[1][i] + c_a
    for j in range(0,T):
        A.append(c_a)
        offset.append(c_offset)

#last month of 2019 not available
A = A[:-1]
offset = offset[:-1]

features = [months,A,offset]

param1, covariance1 = curve_fit(variable_sinusoid,features,sales,p0=variable_get_p0(months,sales))

#-----predictions-----
# reshape x_peaks
x_min_peaks = list(map(lambda el:[el], min_peaks[0])) 
x_max_peaks = list(map(lambda el:[el], max_peaks[0])) 

# min model
model_min = linear_model.LinearRegression()
model_min.fit(x_min_peaks,min_peaks[1])

# max model
model_max = linear_model.LinearRegression()
model_max.fit(x_max_peaks,max_peaks[1])

for i in range(T + 2):
    x_min_peaks.append([months[len(months)-1]+i])
    x_max_peaks.append([months[len(months)-1]+i])

y_pred_min = model_min.predict(x_min_peaks)
y_pred_max = model_max.predict(x_max_peaks)

months_pred = np.array(months)
month = months_pred[len(months_pred)-1] + 1

#T+1 for 2020 as well as Dec 2019
for i in range(0,T+1):
    months_pred = np.append(months_pred,month)
    month_labels.append(math.floor(month/12)+startyear+len(recession_years))
    month = month + 1

index = len(max_peaks[0])-1
for j in range(T+1):
    c_a = (y_pred_max[index] - y_pred_min[index])/2
    c_offset = y_pred_min[index] + c_a
    A.append(c_a)
    offset.append(c_offset)
    index += 1

features_pred = [months_pred,A,offset]

#display every 5 years on x-axis
step = 5
x_ticks = np.arange(0, next_congruence(len(months_pred),12*step)+1, step=12*step)
x_labels = []
x_labels.append(startyear)

for i in range(1, len(x_ticks)-1):
    x_labels.append(month_labels[(step*T)*i])

x_labels.append(last_tick(x_labels[-1],month_labels,step))
plt.xticks(x_ticks, x_labels)

#plot the functions
plt.plot(months, sales, color="red", linewidth=1,linestyle='dashed')
#plt.plot(months, variable_sinusoid(features, *param1), color="blue", linewidth=1)
plt.plot(months_pred, variable_sinusoid(features_pred, *param1), color="blue", linewidth=1)

plt.scatter(x_min_peaks, y_pred_min, color="green", linewidth=1,linestyle='dashed')
plt.scatter(x_max_peaks, y_pred_max, color="green", linewidth=1,linestyle='dashed')
#plt.plot(months, sinusoid(months, *param), color="blue", linewidth=1)

plt.grid()
plt.xlabel('Months')
plt.ylabel('Vehicle Sales')

plt.show()

#save data to csv file
table = []
sales_pred = variable_sinusoid(features_pred, *param1)
for i in range(0, len(months_pred)):
    if i % T == 0:
        m = 1
    else:
        m = (i % T) + 1
    date = datetime(month_labels[i],m,1)
    row = { 'Months' : date.strftime("%m-%Y"), 'Sales' : sales_pred[i]}
    table.append(row)

names = table[0].keys()
with open('vehicle_sales_prediction.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, names)
    dict_writer.writeheader()
    dict_writer.writerows(table)
