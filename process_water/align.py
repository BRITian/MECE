with open(r"E:\壳聚糖酶\gradcam\1754blastdb90-xin.txt") as file:
    line = file.readlines()
id = line[::2]
print(id)

seq = []
p = []
for b in range(len(id)):
    ourl = r"E:\壳聚糖酶\deeplift\1754-blast\align\align.%s.water" % id[b].split()[0][1:]
# a = 'AWL41499.1'
# a = 'MPY61539.1'
#     ourl = r"E:\壳聚糖酶\gradcam\A01\align\%s.water" % a
    print(id[b].split()[0][1:])
    with open(ourl) as file:
        line = file.readlines()
    # print(line)
    line_a01 = []
    line_xin = []
    for i in line:
        if i[:11] == '# Identity:':
            p.append(i.split()[-1][1:-1])
        if i[0] == "#":
            pass
        elif i == '\n':
            pass
        elif i.split()[0] == 'c1754':
            line_a01.append(i)
            # print(i.split())
            # print(i)
        elif i.split()[0].split('.')[0] == id[b].split()[0][1:].split('.')[0]:
            line_xin.append(i.split()[2])
    print(len(line_a01))
    print(len(line_xin))

    s = ''

    for i in range(len(line_a01)):
        if '-' in line_a01[i].split()[2]:
            index = [m for m, n in enumerate(line_a01[i].split()[2]) if n == '-'][::-1]
            print(index)
            list_line_xin = list(line_xin[i])
            for x in index:
                list_line_xin.pop(x)
            s += ''.join(list_line_xin)
        else:
            s += line_xin[i]
    # print(s)
    if int(line_a01[0].split()[1]) != 1:
        s = '-' * (int(line_a01[0].split()[1])-1) + s
    if int(line_a01[-1].split()[3]) != 278:
        s = s + '-' * (278 - int(line_a01[-1].split()[3]))
    print(s)
    seq.append(s)
with open(r'E:\壳聚糖酶\gradcam\c1754.pro.align', 'w') as file:
    for b in range(len(id)):
        file.write('>%s\t%s\n' % (id[b].split()[0][1:], p[b]))
        file.write('%s\n' % seq[b])