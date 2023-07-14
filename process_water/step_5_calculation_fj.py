import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

target_sequence_file=""
target_gradcam_result_file=""
step_4_output_file=""
plot_path=""
a = 46  #target GH Family

aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
dict_abb_num = {'A': '01', 'C': '02', 'D': '03', 'E': '04', 'F': '05',
                'G': '06', 'H': '07', 'I': '08', 'K': '09', 'L': '10',
                'M': '11', 'N': '12', 'P': '13', 'Q': '14', 'R': '15',
                'S': '16', 'T': '17', 'V': '18', 'W': '19', 'Y': '20'}
x = []
y = []

url_1754 = target_sequence_file
with open(url_1754) as file:
    seq_1754_wt = file.readlines()[1].strip('\n')
# print(seq_1754_wt)
max_index = []
for i in seq_1754_wt:
    max_index.append(int(dict_abb_num.get(i))-1)
# print(max_index)

url1 = target_gradcam_result_file 
df = pd.read_csv(url1, index_col=0)[:-1]
df.columns = aa
# print(df)
# print('%s:' % a)
rownames = list(df.index)
# print(rownames)
number = 0
wt_value = []
for i in range(len(df.values)):
    wt_value.append(df.values[i][max_index[i]])
print(wt_value)
# df1 = df[(df['A'] > number) | (df['C'] > number) | (df['D'] > number) | (df['E'] > number) | (df['F'] > number)
#       | (df['G'] > number) | (df['H'] > number) | (df['I'] > number) | (df['K'] > number) | (df['L'] > number)
#       | (df['M'] > number) | (df['N'] > number) | (df['P'] > number) | (df['Q'] > number) | (df['R'] > number)
#       | (df['S'] > number) | (df['T'] > number) | (df['V'] > number) | (df['W'] > number) | (df['Y'] > number)]
# print(df1)
# rownames = list(df1.index)
# print(rownames)
# # max_index = []
# # wt_index = []
# # for i in range(len(df.values)):
# #     max_index.append(np.argmax(df.values[i]))
# # print(max_index)
# print(df.values[153][max_index[153]])
url2 = step_4_output_file
df2 = pd.read_csv(url2, index_col=0)
# print(df2)
# df2_xin = df2.loc[rownames]
# print(df2_xin)

max_blast_index = []
max_blast_value = []
wt_value_xin = []
for i in range(len(df2.values)):
    max_blast_index.append(np.argmax(df2.values[i]))
    max_blast_value.append(df2.values[i][np.argmax(df2.values[i])])
    wt_value_xin.append(df2.values[i][max_index[i]])
print(max_blast_index)
print(max_blast_value)

value = []
site_blast = []
for i in range(0, len(max_index)-20):
    # print(seq_1754_wt[i: i+21])
    value.append(np.sum(max_blast_value[i: i+21])/np.sum(wt_value[i: i+21]))
    # value.append(np.sum(max_blast_value[i: i+21])/np.sum(wt_value_xin[i: i+21]))
    site_blast.append((i, i+21))
print(value)
b = zip(value, site_blast)
b = list(b)
b = sorted(b, key=lambda x: x[0], reverse=True)
print(b)

x = []
y = []
for i in b:
    # if i[0] >= 1:
    if i[0] >= 2:
        x.append(i[1])
        y.append(i[0])
plt.plot(y, marker='o')
plt.xticks(rotation=45)
# plt.show()

site_xin = []
for i in x:
    a = max_blast_index[i[0]:i[1]]
    b = seq_1754_wt[i[0]:i[1]]
    for j in range(len(a)):
        if a[j] == int(dict_abb_num.get(b[j])) - 1:
            pass
        else:
            if max_blast_value[i[0]+j] <= 0.1:
                pass
            else:
                if '%s%s%s' % (b[j], i[0] + 1 + j, aa[a[j]]) in site_xin:
                    pass
                else:
                    site_xin.append('%s%s%s' % (b[j], i[0] + 1 + j, aa[a[j]]))
# print(site_xin)

site = []
significance = []
for i in range(len(rownames)):
    print('i:%s' % i)
    df_temp = list(df2.loc[rownames[i]])
    for j in range(len(df_temp)):
        if aa[int(max_index[i])] == aa[j]:
            pass
        else:
            if (df_temp[j]+0.001)/(df_temp[int(max_index[i])]+0.001) >= 10:
                site.append('%s%s%s' % (aa[int(max_index[i])], (rownames[i]+1), aa[j]))
                # significance.append(df_temp[j])
                significance.append((df_temp[j]+0.001)/(df_temp[int(max_index[i])]+0.001))
                # significance.append((df_temp[j]-df_temp[int(max_index[i])])/(df_temp[int(max_index[i])]+0.001))


print(len(site))
print(significance)
b = zip(significance, site)
b = list(b)
b = sorted(b, key=lambda x: x[0], reverse=True)
print(b)
x = []
y = []
for i in b:
    # print(i)
    x.append(i[1])
    y.append(i[0])
print(x)
print(y)
url_png = plot_path
title = a
plt.plot(x, y, marker='o')
plt.title(title)
plt.xticks(rotation=45)
# plt.savefig(url_png, dpi=600)
plt.show()

