# In the first step, you need to use the WATER tool to make a two by two comparison
# This script is the second step to align the target sequence obtained using water with the results of a two-by-two comparison of homologous sequences.
# the homologous sequences were collected using the online NCBI Blast tool with 60% < identity < 100%, 50% < coverage < 100%, and E-value < e–5. 

homologous_seq_path=""
water_result_path=""
target_protein_name="c1754"
target_seq_len=278
output_path=""

with open(homologous_seq_path) as file:
    line = file.readlines()
id = line[::2]
print(id)

seq = []
p = []
for b in range(len(id)):
    ourl = water_result_path".%s.water" % id[b].split()[0][1:]
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
        elif i.split()[0] == target_protein_name:
            line_a01.append(i)
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
    if int(line_a01[-1].split()[3]) != target_seq_len:
        s = s + '-' * (278 - int(line_a01[-1].split()[3]))
    print(s)
    seq.append(s)
with open(output_path, 'w') as file:
    for b in range(len(id)):
        file.write('>%s\t%s\n' % (id[b].split()[0][1:], p[b]))
        file.write('%s\n' % seq[b])
