'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
#from __future__ import print_function
import numpy as np
import theano
import keras
import pandas as pd
import shutil, os, sys
import tensorflow as tf
from keras.layers.normalization import BatchNormalization
import random
from group_norm import GroupNormalization
from sklearn.utils import class_weight


cd_rate=sys.argv[1]
rand_id=sys.argv[2]
gpu_id=sys.argv[3]
tar_kernel= int(sys.argv[4])
filters_1= int(sys.argv[5])
filters_2= int(sys.argv[6])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

config = tf.ConfigProto()
config.allow_soft_placement=True 
config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Sequential
from keras.models import load_model
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Activation,Embedding, GRU, RepeatVector,SpatialDropout1D,TimeDistributed
from keras.layers import Conv1D, MaxPooling1D,LSTM
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.preprocessing import sequence
from keras.constraints import maxnorm
from keras import initializers
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
from numpy import genfromtxt
import tensorflow as tf
import gc
from sklearn.metrics import roc_auc_score, roc_curve, auc
theano.config.openmp = True
import datetime as dt
from keras import regularizers
from sklearn.utils import class_weight
# import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy import interp
# matplotlib.use('Agg')
#import tensorflow
#from ..externals import six
#from ..utils.fixes import in1d
from sklearn.metrics import roc_curve, auc,precision_recall_curve,average_precision_score
# from sklearn.metrics import roc_auc_score, accuracy_score
from itertools import cycle
from scipy import interp
np.set_printoptions(threshold=sys.maxsize)

nb_epoch = 30

batch_size = 128
num_classes = 119
#epochs = 20

aan=21

mulu_logs="LOGs_tiaocan/"
mulu_models="MODELs_tiaocan/"

if not os.path.exists(mulu_models):
    os.makedirs(mulu_models)

if not os.path.exists(mulu_logs):
    os.makedirs(mulu_logs)

tar_mulu_model= mulu_models + "MODEL_" + str(cd_rate) + "/"

if not os.path.exists(tar_mulu_model):
	os.makedirs(tar_mulu_model)


mulu_train="/home/liuhanqing/workspace/chitinase/8/coding/65/"

rep_num=1
max_seq_len=735

for tar_r in range(rep_num):

	tar_prefix= "CNN_R" + str(num_classes) + "_mr_" + str(cd_rate)+ "_rand" + rand_id + 'tiaocan' + 'tar_kernel_' +str(tar_kernel) + 'filters_1_' + str(filters_1)+ 'filters_2_' + str(filters_2)

	tar_log_file=mulu_logs + "LOG_" + tar_prefix
	fout=open(tar_log_file, 'w', 0)
	#aa_num=wi+1

	print('Loading data...')

	infile_train=mulu_train + cd_rate + '/Train_num_all_119_' + rand_id
	infile_test=mulu_train + cd_rate + '/Test_num_all_119_' + rand_id

	print infile_train
	print infile_test

	times1 = dt.datetime.now()

	train_data = pd.read_csv(infile_train, index_col = False, header=None)
	test_data = pd.read_csv(infile_test, index_col = False, header=None)

	print train_data.shape
	print test_data.shape

	#index=np.arange(train_data.shape[0])
	#np.random.shuffle(index)
	#np.random.shuffle(train_data)

	y_test_ori=test_data[0]
	x_test_ori=test_data[1]

	x_test=[]
	y_test=[]
	for pi in x_test_ori:
		nr=pi.split(' ')[0:-1]
		ndata=map(int,nr)
		x_test.append(ndata)
	x_test=np.array(x_test)

	for pi in y_test_ori:
		#nr=pi.split(' ')[0:-1]
		ndata=int(pi)
		y_test.append(ndata)
	y_test=np.array(y_test)


	y_train_ori=train_data[0]
	x_train_ori=train_data[1]

	x_train=[]
	y_train=[]
	for pi in x_train_ori:
		nr=pi.split(' ')[0:-1]
		ndata=map(int,nr)
		x_train.append(ndata)
	x_train=np.array(x_train)

	for pi in y_train_ori:
		#nr=pi.split(' ')[0:-1]
		ndata=int(pi)
		y_train.append(ndata)
	y_train=np.array(y_train)

	print x_train.shape
	#sys.exit()

	times2 = dt.datetime.now()

	print('Time spent: '+ str(times2-times1))

# convert class vectors to binary class matrices

	class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)

	y_train = np_utils.to_categorical(y_train, num_classes)
	y_test = np_utils.to_categorical(y_test, num_classes)

	y_real = np.argmax(y_test, axis=1)

	x_train=sequence.pad_sequences(x_train,maxlen=max_seq_len,padding='post',truncating='post')
	x_test=sequence.pad_sequences(x_test,maxlen=max_seq_len,padding='post',truncating='post')



	#class_weights = dict(zip(np.unique(y_train), class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)))

	print class_weights

	print x_train.shape

	for ki in range(1):
		# tar_kernel= 21


		model_filepath= tar_mulu_model + "Best_model_" + tar_prefix + ".h5"

		model = Sequential()

		model.add(Embedding(21, 21, input_length=max_seq_len))
		# model.add(BatchNormalization())

		#model.add(Dropout(0.25))
		model.add(Conv1D(filters=filters_1, kernel_size=tar_kernel, activation='relu',padding='same', kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
		# model.add(Conv1D(filters=256, kernel_size=tar_kernel, activation='relu',padding='same', kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
		# model.add(GroupNormalization(groups = 4,axis=-1))
		#model.add(Dropout(0.5))
		model.add(MaxPooling1D(pool_size=4))
		#model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Conv1D(filters=filters_2, kernel_size=tar_kernel, activation='relu',padding='same',  kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
		# model.add(Conv1D(filters=128, kernel_size=tar_kernel, activation='relu',padding='same',  kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
		# model.add(GroupNormalization(groups = 4,axis=-1))
		#model.add(Dropout(0.5))
		model.add(MaxPooling1D(pool_size=4))
		#model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Flatten())

		model.add(Dense(64, activation='relu', kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),
						bias_constraint=maxnorm(3)))
		model.add(Dropout(0.5))
		model.add(Dense(32, activation='relu', kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),
						bias_constraint=maxnorm(3)))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes, activation='softmax'))

		checkpoint = ModelCheckpoint(model_filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
		callbacks_list = [checkpoint]

		#fout=open(tar_log_file, 'w', 0)

		print('Training')
		#model = load_model("Best_model_CNN_template.h5")

		for re in range(nb_epoch):
		#for i in range(20):
			tar_re=re+1

			print('Epoch', tar_re, '/', nb_epoch)
			#OPTIMIZER=Adam(amsgrad=True)
			#OPTIMIZER=SGD()
			OPTIMIZER=Adam()

			model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])
			Result=model.fit(x_train, y_train, epochs=1, callbacks=callbacks_list, batch_size=batch_size, validation_data=(x_test, y_test), shuffle=True, class_weight=class_weights, verbose = 1)

			loss_and_metrics_train = model.evaluate(x_train, y_train)
			loss_and_metrics_test = model.evaluate(x_test, y_test)
			print >>fout, "Train_Test " + str(tar_re) +" metrics ",loss_and_metrics_train, loss_and_metrics_test

			print loss_and_metrics_train
			print loss_and_metrics_train[1]

			#y_pred_ori = model.predict(x_test)
			#y_pred = np.argmax(y_pred_ori, axis=1)
			#print classification_report(y_real, y_pred)

			#if loss_and_metrics_train[1] >0.5 or loss_and_metrics_test[1] >0.95:
			#	break

			#model.reset_states()
			#K.clear_session()
			#tf.reset_default_graph()


		del model
		model = load_model(model_filepath)
		loss_and_metrics = model.evaluate(x_test, y_test)
		# print >>fout, "\n\nloss and metrics : ",loss_and_metrics,"\n\n"


		y_pred_ori = model.predict(x_test)
		y_pred = np.argmax(y_pred_ori, axis=1) # Convert one-hot to index

		print >>fout, "cd_rate" + "_"+str(cd_rate)
		print >>fout, "kenel size " + str(tar_kernel) +" Final_metrics_Test : ",loss_and_metrics

		loss_and_metrics_train = model.evaluate(x_train, y_train)
		print >>fout, "kenel size " + str(tar_kernel) +" Final_metrics_Train : ",loss_and_metrics_train,"\n\n"

		print >>fout, classification_report(y_real, y_pred,digits =6)

		auc1=roc_auc_score(y_test.flatten(),y_pred_ori.flatten())

		print >>fout, "AUC"
		print >>fout, auc1
		# fpr = dict()
		# tpr = dict()
		# roc_auc = dict()
		# for i in range(num_classes):
		# 	fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_ori[:, i])
		# 	roc_auc[i] = auc(fpr[i], tpr[i])
		# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
		# # Then interpolate all ROC curves at this points
		# mean_tpr = np.zeros_like(all_fpr)
		# for i in range(num_classes):
		# 	mean_tpr += interp(all_fpr, fpr[i], tpr[i])
		# # Finally average it and compute AUC
		# mean_tpr /= num_classes
		# fpr["macro"] = all_fpr
		# tpr["macro"] = mean_tpr
		# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
		#
		# # Plot all ROC curves
		# lw = 2
		# file_name = mulu_logs + tar_prefix + 'ROC.png'
		# plt.figure()
		# plt.plot(fpr["macro"], tpr["macro"],
		# 		 label='macro-average ROC curve (area = {0:0.2f})'
		# 			   ''.format(roc_auc["macro"]),
		# 		 color='navy', linestyle=':', linewidth=4)
		# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
		# plt.xlim([0.0, 1.0])
		# plt.ylim([0.0, 1.05])
		# plt.xlabel('False Positive Rate')
		# plt.ylabel('True Positive Rate')
		# plt.title(str(file_name[:-4]))
		# # plt.title('Some extension of Receiver operating characteristic to multi-class')
		# plt.legend(loc="lower right")
		#
		# plt.savefig(file_name, dpi=600)
		#
		# file_name1 = mulu_logs + "LOG_1_3_" + tar_prefix + 'ROC'
		# file =  open(file_name1, 'w', 0)
		# file.write('fpr\ttpr\n')
		# for i in range(len(list(fpr["macro"]))):
		# 	print(str(list(fpr["macro"])[i]))
		# 	file.write(str(list(fpr["macro"])[i]))
		# 	file.write('\t')
		# 	file.write(str(list(tpr["macro"])[i]))
		# 	file.write('\n')
		#
		# file.close()
		# # ########################ROC########################
		# # # setup plot details
		# # colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue'])
		# # fpr = dict()
		# # tpr = dict()
		# # roc_auc = dict()
		# # class_num = 120
		# # print(y_test.shape)
		# # print(y_pred.shape)
		# # # for i in range(num_classes):
		# # # fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_ori[:, i])
		# # for i in range(class_num):
		# # 	fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_ori[:, i])
		# # 	roc_auc[i] = auc(fpr[i], tpr[i])
		# # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))
		# # # Then interpolate all ROC curves at this points
		# # mean_tpr = np.zeros_like(all_fpr)
		# # for i in range(class_num):
		# # 	mean_tpr += interp(all_fpr, fpr[i], tpr[i])
		# # # Finally average it and compute AUC
		# # mean_tpr /= class_num
		# # fpr["macro"] = all_fpr
		# # tpr["macro"] = mean_tpr
		# # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
		# #
		# # # Plot all ROC curves
		# # lw = 2
		# # plt.figure(figsize=(7, 8))
		# # lines = []
		# # labels = []
		# # l, = plt.plot(fpr["macro"], tpr["macro"],
		# # 			  color='teal', linestyle=':', lw=2)
		# # lines.append(l)
		# # labels.append('micro-average ROC curve (area = {0:0.2f})'
		# # 			  ''.format(roc_auc["macro"]))
		# #
		# # for i, color in zip(range(class_num), colors):
		# # 	l, = plt.plot(fpr[i], tpr[i], color=color, lw=2)
		# # 	lines.append(l)
		# # 	labels.append('ROC for class {0} (area = {1:0.2f})'
		# # 				  ''.format(i, roc_auc[i]))
		# # fig = plt.gcf()
		# # fig.subplots_adjust(bottom=0.25)
		# # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
		# # plt.xlim([0.0, 1.0])
		# # plt.ylim([0.0, 1.05])
		# # plt.xlabel('False Positive Rate')
		# # plt.ylabel('True Positive Rate')
		# # plt.title('ROC curve')
		# # plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
		# # plt.savefig("%s_ROC.pdf" % (cd_rate))
		# #
		# # ######################PR#######################
		# # setup plot details
		# file_pr = open(r'/home/liuhanqing/workspace/chitinase/119zuijin/optimtion/PR_%s_xin.txt' % rand_id, 'w')
		# recall = dict()
		# precision = dict()
		# average_precision = dict()
		# class_num = 119
		# for i in range(class_num):
		# 	precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
		# 														y_pred_ori[:, i])
		# 	average_precision[i] = average_precision_score(y_test[:, i], y_pred_ori[:, i])
		#
		# # A "micro-average": quantifying score on all classes jointly
		# precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
		# 																y_pred_ori.ravel())
		# average_precision["micro"] = average_precision_score(y_test, y_pred_ori,
		# 													 average="micro")
		# list_p = list(precision["micro"])
		# list_r = list(recall["micro"])
		# print(average_precision["micro"])
		# file_pr.write('ap\n')
		# file_pr.write(str(average_precision["micro"]))
		# file_pr.write('\nrecall\tprecision\n')
		# print(list_r)
		# for i in range(len(list_p)):
		# 	# print(i)
		# 	# print(str(list_r[i]))
		# 	file_pr.write(str(list_r[i]))
		# 	file_pr.write('\t')
		# 	file_pr.write(str(list_p[i]))
		# 	file_pr.write('\n')
		# file_pr.close()

			# fig = plt.gcf()
			# fig.subplots_adjust(bottom=0.25)
			# plt.xlim([0.0, 1.0])
			# plt.ylim([0.0, 1.05])
			# plt.xlabel('Recall')
			# plt.ylabel('Precision')
			# plt.title('Extension of Precision-Recall curve to multi-class')
			# plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
			# plt.savefig("%s_PR.pdf" % (cd_rate))

		##############################################
