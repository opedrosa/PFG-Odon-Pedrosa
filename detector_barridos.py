import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from os import listdir

from scipy.io import loadmat

#Functions

#Dos funciones para hacer el smooth de power y freq
def smooth_bad(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#Funcion para obtener los limites de tiempo y freq d un vector con la wdw aplicada
def wdw_edges(start, end, window):
    center= (end-start)/2
    total= (end-start)*window
    total_half= total/2
    start2= center-total_half
    stop2= center+total_half
    if window==1:
        edges= [start, end]
    else:
        edges= [start2, stop2]
    return edges

#Funcion para coger el porcentaje a partir del centro de una señal
def swept_window(center, start, end, window, frequency):
    a= 0
    if center - start < end - center:
        a= center - start
    else:
        a= end - center

    start2= center-a
    stop2= center+a
    total= math.floor((stop2-start2)*window)
    total_half= math.floor(total/2)
    start2= center-total_half
    stop2= center+total_half
    frequency_wdw= []
    while start2 <= stop2:
        frequency_wdw.append(frequency[start2])
        start2 += 1
    return frequency_wdw

def detector_barrido(freq_file, power_file, window):
    #Sampling frequency and sampling time
    fs= 12e3
    ts= 1/fs

    #Reading data from Ettus

    #Por ahora npi de como leer los archivos binarios asi q saco los datos de matlab

    freq= np.fromfile(freq_file, dtype= np.float32)
    power= np.fromfile(power_file, dtype= np.float32)

    #Creating time vectors
    time_aux = [0]

    i= 0
    while i<= ((power.size*ts)-2*ts):
        time_aux.append(i)
        i += ts

    time_rxp= []
    time_rxf= []

    for i in time_aux:
        time_rxp.append(i / 60)

    for i in time_aux:
        time_rxf.append(i / 60)

    #Smoothing the data
    smooth_freq= smooth(freq, 99)
    smooth_power= smooth_bad(power, 99)



    #Once all the data is processed, we start building the dataset
    # first approach find the theoretical centers of all barridos
    # and with a window parameter obtain the separate barridos

    #First we stablish the endpoints of each barrido

 
    

    #Samples vectors

    freq_samples= []

    m= 1
    while m <= len(freq):

        freq_samples.append(m)
        m += 1

    power_samples= []

    m= 1
    while m <= len(power):

        power_samples.append(m)
        m += 1

    e_points= []
    freq_e_points= []
    e_points2= []
    freq_e_points2= []

    #Calculo de los end points de los barridos
    j= 0
    while j < len(smooth_freq)-151:

        if smooth_freq[j]-smooth_freq[j+150] >= 400e3:

            e_points.append(j+120)
            freq_e_points.append(smooth_freq[j+120])
            j+= 400
        j += 1

    #Metodo alternativo para obtener los end points
    '''
    j= 0
    while j <= 2:
        e_points2.append(e_points[j])
        freq_e_points2.append(freq_e_points[j])
        j += 1

    dist_e_points2= e_points[2]-e_points[1]

    while j <= len(e_points)-1:

        e_points2.append((e_points2[j-1])+dist_e_points2)
        freq_e_points2.append(smooth_freq[(e_points2[j-1])+dist_e_points2])
        j += 1
    '''
    e_points.insert(0, 133)
    freq_e_points.insert(0, smooth_freq[133])

    #Calculo de los puntos centrales de los barridos
    s= 0
    c_points= []
    freq_c_points= []
    while s < len(e_points)-1:

        c_points.append(round((e_points[s+1]+e_points[s])/2))
        freq_c_points.append(smooth_freq[c_points[s]])
        s += 1

    #Metodo alternativo para obtener los center points
    total_len= len(smooth_freq)
    n_barridos= len(e_points)-1
    len_half_barrido= round(total_len/(2*n_barridos))
    c_points2= []
    freq_c_points2= []

    s= 0
    while s < len(e_points)-2:

        c_points2.append(len_half_barrido*((2*s)+1))
        freq_c_points2.append(smooth_freq[c_points2[s]])
        s += 1

    #Window parameter for detection of barridos: 0 < wdw < 1
    wdw= window
    freq_edges= wdw_edges(1.5, 5, wdw)
    #Calculo de cada uno de los barridos en power
    barrido= []
    barridos_power= []
    len_barridos_power= []
    s= 1
    while s <= len(c_points)-1:
    #while s <= 1:
        barrido= swept_window(c_points[s], e_points[s], e_points[s+1], wdw, smooth_power)
        len_barridos_power.append(len(barrido))
        barridos_power.append(barrido)
        s += 1

    #Barridos en freq
    time_edges= wdw_edges(0,10,wdw)
    len_barridos_freq= []
    barridos_freq= []
    s= 1
    while s <= len(c_points)-2:
    #while s <= :
        barrido= swept_window(c_points[s], e_points[s], e_points[s+1], wdw, smooth_freq)
        len_barridos_freq.append(len(barrido))
        barridos_freq.append(barrido)
        s += 1


    return barridos_power
