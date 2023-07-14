
import numpy as np
import theano
import keras
import pandas as pd
import shutil, os, sys
import tensorflow as tf

term = int(sys.argv[1])
a = str(sys.argv[2])
gpu_id=sys.argv[3]

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
from keras.layers import Dense, Dropout, Flatten, Activation,Embedding
from keras.layers import Conv1D, MaxPooling1D,LSTM
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.preprocessing import sequence
from keras.constraints import maxnorm
from keras import initializers
from keras.optimizers import RMSprop
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import classification_report


from keras.callbacks import ModelCheckpoint
import shutil, os, sys
from numpy import genfromtxt

theano.config.openmp = True
import datetime as dt
from keras import regularizers

#import tensorflow

#from ..externals import six
#from ..utils.fixes import in1d

np.set_printoptions(threshold=sys.maxsize)

nb_epoch = 50

batch_size = 128
num_classes = 119
#epochs = 20

# rep_id=int(sys.argv[1])
cut_num=119

aan=21		# 0,1,2,3 A,T,G,C
max_seq_len = 735
# max_seq_len = rep_id
# input image dimensions
#img_rows, img_cols = 80, 105

mulu_train=""

prefix="C119" + "_T" + str(term)

print('Loading data...')

#mulu_train="train_test_R3_split/"

mulu_logs=str(a)+"/LOGs/"
mulu_models=str(a)+"/MODELs/"

#NUM_THREADS=4

tar_mulu_model= mulu_models + "MODEL_" + prefix + "/"

if not os.path.exists(mulu_models):
    os.makedirs(mulu_models)

if not os.path.exists(tar_mulu_model):
    os.makedirs(tar_mulu_model)

if not os.path.exists(mulu_logs):
    os.makedirs(mulu_logs)

tar_log_file=mulu_logs + "LOG_" + prefix
model_filepath= tar_mulu_model + "Best_model_" + prefix + ".h5"

#infile_train=mulu_train + "Test_num_9_1"
#infile_test=mulu_train + "Test_num_9_2"

infile_train=mulu_train + "Train_num_all_"  + str(cut_num)+ "_" + str(term) 
infile_test=mulu_train + "Test_num_all_"+ str(cut_num)+ "_" + str(term) 

print infile_train
print infile_test

#data = genfromtxt(infile_train,delimiter='\t')  # Training data
#test_data = genfromtxt(infile_test,delimiter=',')  # Test data
times1 = dt.datetime.now()

train_data = pd.read_csv(infile_train, index_col = False, header=None)
test_data = pd.read_csv(infile_test, index_col = False, header=None)

print train_data.shape
print test_data.shape

#print test_data
#print len(test_data)
#print test_data.shape[0]
#print test_data.shape[1]

y_test_ori=test_data[0]
x_test_ori=test_data[1]

#y_test = np_utils.to_categorical(y_test, num_classes)
#print y_test

x_test=[]
try:
	for pi in x_test_ori:
		nr=pi.split(' ')[0:-1]
		#print nr
		ndata=map(int,nr)
		#ndata = np_utils.to_categorical(ndata, 21)
		x_test.append(ndata)
except ValueError:
	print(pi)

#print x_test[0]

x_test=np.array(x_test)
#print x_test.shape
#print y_test.shape

#print x_test[0]

#sys.exit()

y_train=train_data[0]
x_train_ori=train_data[1]

#y_train = np_utils.to_categorical(y_train, num_classes)
#print y_train

x_train=[]
for pi in x_train_ori:
	nr=pi.split(' ')[0:-1]
	ndata=map(int,nr)
	#ndata = np_utils.to_categorical(ndata, 21)
	x_train.append(ndata)

x_train=np.array(x_train)
#print x_train.shape

times2 = dt.datetime.now()

print('Time spent: '+ str(times2-times1))

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test_ori, num_classes)

x_train=sequence.pad_sequences(x_train,maxlen=max_seq_len, padding='post', truncating='post')
x_test=sequence.pad_sequences(x_test,maxlen=max_seq_len, padding='post', truncating='post')

model = Sequential()
model.add(Embedding(aan, aan, input_length=max_seq_len))
#model.add(Dropout(0.25))
model.add(Conv1D(filters=64, kernel_size=15, activation='relu',padding='same',kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.20))
model.add(Conv1D(filters=128, kernel_size=15, activation='relu',padding='same',kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
model.add(MaxPooling1D(pool_size=2))
#model.add(LSTM(units=100,kernel_constraint=max_norm(3),recurrent_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
model.add(Dropout(0.20))

model.add(Flatten())

model.add(Dense(64, activation='relu',kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
model.add(Dropout(0.20))
model.add(Dense(32, activation='relu',kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
model.add(Dropout(0.20))
model.add(Dense(num_classes, activation='softmax'))

checkpoint = ModelCheckpoint(model_filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

fout=open(tar_log_file, 'w', 0)

print('Training')
for i in range(nb_epoch):
	print('Epoch', i, '/', nb_epoch)
	OPTIMIZER=Adam()
	model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])
	Result=model.fit(x_train, y_train, epochs=1, callbacks=callbacks_list, batch_size=batch_size, validation_data=(x_test, y_test), class_weight='auto', verbose = 1)

	#tar_model=tar_mulu_model + "Model_1_" + str(i) + ".h5"
	#model.save(tar_model, overwrite=True)
	loss_and_metrics = model.evaluate(x_test, y_test)
	print >>fout, "Epoch ",i," Frac_Cutoff ",cut_num," loss and metrics Test : ",loss_and_metrics

	model.reset_states()

del model

model = load_model(model_filepath)
loss_and_metrics = model.evaluate(x_test, y_test)
print >>fout, "\n\nFinal loss and metrics : ",loss_and_metrics,"\n\n"
y_pred_r = model.predict(x_test)
# print >>fout, "%s, %s" % (y_test, y_pred)

auc=roc_auc_score(y_test.flatten(),y_pred_r.flatten())
print >>fout, "auc"
print >>fout, auc

y_pred = np.argmax(y_pred_r, axis=1) # Convert one-hot to index
y_real = np.argmax(y_test, axis=1)

print >>fout, "classification_report"
print >>fout, classification_report(y_real, y_pred, digits=5)

for i in range(len(y_test_ori)):
	fout.write("%s, %s\n" % (y_test_ori[i], y_pred_r[i]))


fout.close()

sys.exit()
