import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from os import listdir

def swept_window_two(len, min_len, frequency):
    a= 0
    start= 0
    end= len
    center= len/2
    window= min_len/len
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
        frequency_wdw.append(frequency[start2.astype(np.int64)])
        start2 += 1
    return frequency_wdw
