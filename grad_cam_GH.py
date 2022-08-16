#!/usr/bin/env python
# coding: utf-8

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import backend as K
import csv
import re
from collections import Counter
tf.disable_eager_execution()
np.set_printoptions(suppress=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm

def save_file(seq_dir,seq,gene_id,heatmap,gene_id_save):
    seq_dir = seq_dir + gene_id_save + ".csv"
    f = open(seq_dir, "w")
    f.write("no.,residue,weight\n")
    for i in range(len(seq)):
        if i < len(heatmap[0]):
            f.write(str(i+1)+","+seq[i]+","+str(heatmap[0][i])+"\n")
        else:
            f.write(str(i+1)+","+seq[i]+",0\n")
    f.close()


def save_csv(seq_dir,gene_id,max_seq_len,gene_seq,heatmap,gene_id_save):
    csv_dir=seq_dir+gene_id_save+".csv"
    f = open(csv_dir, "w")
    writer = csv.writer(f)
    writer.writerow(
        ["", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
         "19"])
    for i in range(max_seq_len):
        line = []
        line.append(i)
        if i + 1 > len(gene_seq):
            print("early_stop_at: " + str(i))
            break
        for s in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
            if gene_seq[i] == s:
                line.append(heatmap[0][i])
            else:
                line.append(0)
        writer.writerow(line)
    f.close()

def save_fa(gene_id,seq,seq_dir,gene_id_save):
    seq_dir = seq_dir + gene_id_save + ".fasta"
    f = open(seq_dir, "w")
    if gene_id[0]!=">":
        gene_id=">"+gene_id
    f.write(gene_id+"\n"+seq+"\n")
    f.close()

def save_chimera(seq_dir,gene_id,heatmap,gene_id_save):
    chimera_dir = seq_dir + gene_id_save + ".attribute"
    loc = 0
    with open(chimera_dir, "w") as f:
        f.write("attribute: "+gene_id+"\nmatch mode: 1-to-1\nrecipient: residues\n")
        for i in heatmap[0]:
            f.write("\t:" + str(loc) + "\t" + str(i) + "\n")
            loc += 1
    f.close()

def grad_cam(seq,save_dir):
    print(os.getcwd())

    gh = ['GH1', 'GH2', 'GH3', 'GH4', 'GH5', 'GH6', 'GH7', 'GH8', 'GH9', 'GH10', 'GH11', 'GH12', 'GH13', 'GH14', 'GH15',
          'GH16', 'GH17', 'GH18', 'GH19', 'GH20', 'GH22', 'GH23', 'GH24', 'GH25', 'GH26', 'GH27', 'GH28', 'GH29',
          'GH30',
          'GH31', 'GH32', 'GH33', 'GH35', 'GH36', 'GH37', 'GH38', 'GH39', 'GH42', 'GH43', 'GH44', 'GH45', 'GH46',
          'GH47',
          'GH49', 'GH50', 'GH51', 'GH53', 'GH55', 'GH56', 'GH57', 'GH62', 'GH63', 'GH64', 'GH65', 'GH66', 'GH68',
          'GH71',
          'GH72', 'GH73', 'GH74', 'GH75', 'GH76', 'GH77', 'GH78', 'GH79', 'GH81', 'GH82', 'GH83', 'GH84', 'GH85',
          'GH86',
          'GH87', 'GH88', 'GH89', 'GH92', 'GH93', 'GH94', 'GH95', 'GH97', 'GH99', 'GH102', 'GH103', 'GH104', 'GH105',
          'GH106', 'GH108', 'GH109', 'GH110', 'GH113', 'GH114', 'GH115', 'GH116', 'GH117', 'GH120', 'GH123', 'GH125',
          'GH126', 'GH127', 'GH128', 'GH130', 'GH131', 'GH133', 'GH135', 'GH136', 'GH140', 'GH141', 'GH144', 'GH145',
          'GH146', 'GH148', 'GH151', 'GH152', 'GH153', 'GH154', 'GH156', 'GH158', 'GH163', 'GH165', 'GH166']


    gene_id, gene_seq = seq[0],seq[1]
    dict_abb_num = {'A': '01', 'C': '02', 'D': '03', 'E': '04', 'F': '05',
                    'G': '06', 'H': '07', 'I': '08', 'K': '09', 'L': '10',
                    'M': '11', 'N': '12', 'P': '13', 'Q': '14', 'R': '15',
                    'S': '16', 'T': '17', 'V': '18', 'W': '19', 'Y': '20'}

    x_pred, pred_data = [], []

    for y in range(len(gene_seq.strip())):
        if dict_abb_num.get(gene_seq[y]) is not None:
            pred_data.append(dict_abb_num.get(gene_seq[y]))
        else:
            pred_data.append(0)

    num_classes = 119
    pred_data = np.array(pred_data, dtype='int').tolist()
    x_pred.append(pred_data)
    x_pred = np.array(x_pred)
    max_seq_len = 735
    x_pred = sequence.pad_sequences(x_pred, maxlen=max_seq_len, padding='post', truncating='post')
    heatmaps = np.zeros([1, 367], dtype=float)
    y_pred_mean=[]

    for model_num in range(1):
        model = load_model("./models/Best_model_C119_T"+str(model_num)+".h5")
        # model.summary()
        y_pred = list(model.predict(x_pred)[0])
        y_pred_mean.append(gh[y_pred.index(max(y_pred))])
        ret = model.output[0, y_pred.index(max(y_pred))]  # set point
        last_conv_layer = model.get_layer("conv1d_2")
        fm = last_conv_layer.output
        grads = K.gradients(ret, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x_pred])
        for i in range(pooled_grads.shape[0]):  # 梯度和特征图加权
            conv_layer_output_value[:, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = heatmap[np.newaxis, :]
        heatmaps = np.append(heatmaps, heatmap, axis=0)
    y_pred_mean = Counter(y_pred_mean).most_common()[0][0]

    heatmap = heatmaps.mean(axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap[np.newaxis, :]
    import cv2
    heatmap = cv2.resize(heatmap, (max_seq_len, 1))
    gene_id_save=gene_id+"pred_"+y_pred_mean
    seq_dir=save_dir+"/"+gene_id_save+"/"
    if os.path.exists(seq_dir)==False:
        os.mkdir(seq_dir)
    # save_csv(seq_dir,gene_id,max_seq_len,gene_seq,heatmap,gene_id_save)
    save_file(seq_dir,gene_seq,gene_id,heatmap,gene_id_save)
    # save_fa(gene_id, gene_seq, seq_dir,gene_id_save)
    # save_chimera(seq_dir, gene_id, heatmap,gene_id_save)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1,len(heatmap[0])+1,1),heatmap[0])
    plt.savefig(seq_dir+gene_id_save+".png", dpi=900, bbox_inches='tight')
    plt.close()
    return y_pred_mean

def check_file(fasta_file):
    seqs = []
    for line in fasta_file:

        line=line.strip()
        if len(line)==0:
            continue
        if line[0] == ">":
            seqs.append([line, ""])
        else:
            seqs[-1][-1] += line
    return seqs





def main(*args):

    save_dir=FLAGS.save_url
    with open(FLAGS.data_url) as f:
        seqs=check_file(f)
    for seq in seqs:
        intab = r'[?*/\|.:><]'
        seq[0] = re.sub(intab, "", seq[0].replace(">", "").replace("|", "_"))
        grad_cam(seq,save_dir)


tf.flags.DEFINE_string('data_url', None, 'fasta file.')
tf.flags.DEFINE_string('save_url', None, 'data save directory.')
FLAGS = tf.flags.FLAGS


if __name__ == '__main__':
    tf.app.run(main=main)