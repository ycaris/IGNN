import json
import os
import io
import numpy as np


# For DIV2K, Set5, Set14, BSD100, Urban100, Manga109
file = io.open('xray_test.json','w',encoding='utf-8')
samples = []

root = './x_ray/test/LR'
sample_list = os.listdir(root)
sample = [sample_list[i][:-5] for i in range(len(sample_list))]
sample_sub = []
for sam in sample:
    if not sam == ".DS_S":
        sample_sub.append(sam)
l = {'name': 'x_ray', 'phase': 'train','sample': sample_sub}

samples.append(l)

js = json.dump(samples, file, sort_keys=True, indent=4)