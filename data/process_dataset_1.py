url = r"E:\cazy_function\CD-hit\xin\train\train_40.txt"
with open(url) as file:
    line = file.readlines()
ID = line[::2]
seq = line[1::2]
dict_gh_number = {}
L = []
seq_all = []
for j in range(1, 169):
    pi = []
    temp = []
    s = 'GH%s' % j
    for i in range(len(ID)):
        if ID[i].strip('\n').split('_')[-1] == s:
            if len(seq[i].strip('\n')) < 200:
                pass
            else:
                pi.append(ID[i].strip('\n'))
                temp.append(seq[i].strip('\n'))
    if len(pi) < 10:
        pass
    else:
        for h in range(len(temp)):
            seq_all.append(temp[h])
        dict_gh_number[s] = len(pi)
        L.append(pi)
print(list(dict_gh_number.keys()))
print(len(L))
print(dict_gh_number)
