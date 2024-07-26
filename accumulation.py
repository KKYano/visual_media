import numpy as np
import pandas as pd
from metavision_core.event_io import RawReader
import cv2
import matplotlib.pyplot as plt
import os
import json 
import scipy.io as scio

src_dir = './event_stream' 
evs_files = sorted(os.listdir(src_dir))

accumulation = np.zeros((3,1280,720))

for i, evs_file in enumerate(evs_files):
        # load .raw files
        if evs_file.endswith('.raw'):
            reader = RawReader(os.path.join(src_dir, evs_file), max_events=int(1e8))
            evs_raw = reader.load_delta_t(1e8)
            num_evs = evs_raw['t'].shape[0]
            evs_arr = np.zeros([num_evs, 4])
            evs_arr[:,0] = evs_raw['t']
            evs_arr[:,1] = evs_raw['x']
            evs_arr[:,2] = evs_raw['y']
            evs_arr[:,3] = evs_raw['p']
        if i == 0:
            lb_time = 1033000 # left time bound
            rb_time = 2466000 # right time bound
            for ev in evs_arr:
                t = (int)(ev[0])
                x = (int)(ev[1])
                y = (int)(ev[2])
                p = (int)(ev[3])
                if (t >= lb_time) and (t <= rb_time):
                    accumulation[i,x,y] += 1
        elif i == 1:
            lb_time = 1066600 # left time bound
            rb_time = 2966000 # right time bound
            for ev in evs_arr:
                t = (int)(ev[0])
                x = (int)(ev[1])
                y = (int)(ev[2])
                p = (int)(ev[3])
                if (t >= lb_time) and (t <= rb_time):
                    accumulation[i,x,y] += 1
        elif i == 2:
            lb_time = 900000 # left time bound
            rb_time = 4000000 # right time bound
            for ev in evs_arr:
                t = (int)(ev[0])
                x = (int)(ev[1])
                y = (int)(ev[2])
                p = (int)(ev[3])
                if (t >= lb_time) and (t <= rb_time):
                    accumulation[i,x,y] += 1

accumulation = accumulation.T
print(accumulation.shape)

plt.imshow((accumulation * 255).astype(np.uint8))
plt.show()