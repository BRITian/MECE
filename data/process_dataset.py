import os
import matplotlib.pyplot as plt

dict_abb_num = {'A': '01', 'C': '02', 'D': '03', 'E': '04', 'F': '05',
                'G': '06', 'H': '07', 'I': '08', 'K': '09', 'L': '10',
                'M': '11', 'N': '12', 'P': '13', 'Q': '14', 'R': '15',
                'S': '16', 'T': '17', 'V': '18', 'W': '19', 'Y': '20'}
gh = ['GH1', 'GH2', 'GH3', 'GH4', 'GH5', 'GH6', 'GH7', 'GH8', 'GH9', 'GH10', 'GH11', 'GH12', 'GH13', 'GH14', 'GH15',
      'GH16', 'GH17', 'GH18', 'GH19', 'GH20', 'GH22', 'GH23', 'GH24', 'GH25', 'GH26', 'GH27', 'GH28', 'GH29', 'GH30',
      'GH31', 'GH32', 'GH33', 'GH35', 'GH36', 'GH37', 'GH38', 'GH39', 'GH42', 'GH43', 'GH44', 'GH45', 'GH46', 'GH47',
      'GH49', 'GH50', 'GH51', 'GH53', 'GH55', 'GH56', 'GH57', 'GH62', 'GH63', 'GH64', 'GH65', 'GH66', 'GH68', 'GH71',
      'GH72', 'GH73', 'GH74', 'GH75', 'GH76', 'GH77', 'GH78', 'GH79', 'GH81', 'GH82', 'GH83', 'GH84', 'GH85', 'GH86',
      'GH87', 'GH88', 'GH89', 'GH92', 'GH93', 'GH94', 'GH95', 'GH97', 'GH99', 'GH102', 'GH103', 'GH104', 'GH105',
      'GH106', 'GH108', 'GH109', 'GH110', 'GH113', 'GH114', 'GH115', 'GH116', 'GH117', 'GH120', 'GH123', 'GH125',
      'GH126', 'GH127', 'GH128', 'GH130', 'GH131', 'GH133', 'GH135', 'GH136', 'GH140', 'GH141', 'GH144', 'GH145',
      'GH146', 'GH148', 'GH151', 'GH152', 'GH153', 'GH154', 'GH156', 'GH158', 'GH163', 'GH165', 'GH166']
for e in range(40, 105, 5):
    e = str(e)
    url = r"E:\cazy_function\CD-hit\xin\train\train_%s.txt" % str(e)
    with open(url) as file:
        line = file.readlines()
    ID = line[::2]
    seq = line[1::2]
    dict_gh_number = {}
    L = []
    seq_all = []
    for j in gh:
        pi = []
        temp = []
        s = j
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
            seq_all.append(temp)
            L.append(pi)
            dict_gh_number[s] = len(pi)
    print(e)
    # print(len(L))
    print(dict_gh_number)

    def train(seed):
        path = r"E:\\8\\model\\coding\\"+e+"\\"
        if not os.path.isdir(path):  # 判断是否为路径
            os.mkdir(os.path.split(path)[0])
        url1 =r"E:\\8\\model\\coding\\"+e+"\\Test_num_all_119_%s" % seed
        url2 =r"E:\\8\\model\\coding\\"+e+"\\Train_num_all_119_%s" % seed
        fout1 = open(url1, 'w')
        fout2 = open(url2, 'w')
        for n in range(len(seq_all)):
            for b in range(len(seq_all[n])):
                if str(b)[-1] == seed:
                    temp = str(n) + ','
                    for m in seq_all[n][b].strip('\n'):
                        if dict_abb_num.get(m) == None:
                            temp = temp + '00' + ' '
                        else:
                            temp = temp + str(dict_abb_num.get(m)) + ' '
                    fout1.write(temp)
                    fout1.write(',')
                    fout1.write('\n')
                else:
                    temp = str(n) + ','
                    for m in seq_all[n][b].strip('\n'):
                        if dict_abb_num.get(m) == None:
                            temp = temp + '00' + ' '
                        else:
                            temp = temp + str(dict_abb_num.get(m)) + ' '
                    fout2.write(temp)
                    fout2.write(',')
                    fout2.write('\n')

    for i in range(10):
        train(str(i))

    order = list(range(len(dict_gh_number.keys())))
    dict_gh_order = dict(zip(dict_gh_number.keys(), order))
    print(dict_gh_order)
    d_order = sorted(dict_gh_number.items(), key=lambda x: x[1], reverse=True)
    print(d_order)

