# In the third step, you need to use the Grad-CAM method to calculate the feature matrix (Mi) of the homologous sequence
# This script is the fourth step, which averages all sequence feature matrices according to the sequence comparison results, using the wild type as the standard, to obtain the functionally relevant evolutionary feature matrix (Me) for the wild type.

import pandas as pd
import numpy as np
import os

step_2_output_file = ""
homologous_seq_path = ""
water_result_path = ""
homologous_gradcam_result_folder = ""
target_seq_len = 278
output_path = ""

a = str("46")  # target GH Family

with open(step_2_output_file) as file:
    line = file.readlines()
id = [i.strip('\n').split()[0][1:].split('.')[0] for i in line[::2]][1:]
seq = [i.strip('\n') for i in line[1::2]][1:]

dict_id_seq = dict(zip(id, seq))

with open(homologous_seq_path) as file2:
    line2 = file2.readlines()
id2 = [i.strip('\n').split()[0][1:].split('.')[0] for i in line2[::2]][1:]
seq2 = [i.strip('\n') for i in line2[1::2]][1:]
dict_id_seq2 = dict(zip(id2, seq2))

print(dict_id_seq2)

p_xin = []
for key, value in dict_id_seq.items():
    url2 = water_result_path + ".%s.water" % key
    url3 = os.path.join(homologous_gradcam_result_folder, "csv-%s" % a, "%s_vactor_mean.csv" % key)

    seq_homo = dict_id_seq2.get(key)
    if seq_homo is None:
        print(f"Missing homologous sequence for {key}")
        continue

    with open(url3) as file3:
        line3 = file3.readlines()[1:-1]

    p = []
    for i in range(len(line3)):
        p.append((seq_homo[i], line3[i].strip('\n').split(',')[1:]))

    with open(url2) as file:
        line = file.readlines()

    xiangxixinxi = []
    for aaa in line:
        if aaa == '\n':
            pass
        elif aaa[0] == '#':
            pass
        else:
            xiangxixinxi.append(aaa)

    number = int(xiangxixinxi[2].split()[1]) - 2
    temp = []

    for j in value.strip('\n'):
        tiaojian = True
        while tiaojian:
            if j == '-':
                temp.append([np.nan] * 20)
                tiaojian = False
            else:
                number += 1
                if number >= len(p):
                    temp.append([np.nan] * 20)
                    tiaojian = False
                else:
                    h = p[number]
                    if j == h[0]:
                        temp.append(h[1])
                        tiaojian = False

    if len(temp) != target_seq_len:
        print(f"Length mismatch for {key}: {len(temp)}")

    p_xin.append(temp)

p_arr = np.array(p_xin, dtype=float)
print(p_arr.shape)
p_arr = np.nanmean(p_arr, axis=0)
df = pd.DataFrame(p_arr)
df.to_csv(output_path, index=False)
