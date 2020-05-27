
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import json
import random
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import numpy.ma as ma
import scipy.io
import keras
from scipy.stats import pearsonr
from keras.layers import Input,BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Convolution1D, Dense, Flatten,TimeDistributed, GlobalAveragePooling1D,Dropout, MaxPooling1D,Reshape
from keras.models import Model,Sequential,load_model
from keras.callbacks import ModelCheckpoint ,EarlyStopping
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization#from keras.layers import Input, Dense
import librosa

from keras.callbacks import ModelCheckpoint 
from keras.layers.wrappers import Bidirectional
#from keras.utils.generic_utils import Progbar

#from keras.utils.visualize_util import plot
from keras.layers import   MaxPooling2D, Flatten,Reshape,Conv2D,Dropout
from keras.layers import Input, Dense, TimeDistributed, GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def directory(extension):
 list_dir = []
 count = 0
 for file in Mfccfiles:
    
    if (file.find(extension)!=-1) and (count!=8): # eg: '.txt'
      count += 1
      list_dir.append(file)
 return list_dir

def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def Relu_log(In):
    Out_relu=tf.nn.relu(np.abs(In))
    Out_log=tf.log(Out_relu+0.01)
    return Out_log

def subfiles(ext,fold):
        list_dir=directory(ext)
        l1=list_dir[0:2];l2=list_dir[2:4];l3=list_dir[4:6];l4=list_dir[6:]
        if (fold==0):
               l=[l1,l2,l3,l4]
        elif (fold==1):
                l=[l2,l3,l4,l1]
        elif(fold==2):
                l=[l3,l4,l1,l2]
        else:
                l=[l4,l1,l2,l3]
        return l  

'''def build_model():
	model = Sequential()
	model.add(Conv2D(64, (3,3), padding='same', activation = 'relu', input_shape=(100,39,1)))
	model.add(MaxPooling2D(pool_size = 3))
	model.add(Conv2D(64, (3,3), padding='same', activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3,3), padding='same', activation = 'relu'))
	model.add(Conv2D(128, (3,3), padding='same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = 2))
	model.add(Flatten())
	model.add(Dense(64,activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.20))
	model.add(Dense(32,activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.20))
	model.add(Dense(1,activation ='relu'))
	return model'''
	
def phoneme_rate(y,st,et):
      phn=pd.read_csv(Y_dir+y+'.phn',delimiter='\t',header=None)
      phoneme_list=[]
      time_phoneme_1=[]
      time_phoneme_2=[]
      for i in range(0,phn.shape[0]):
                phoneme_list.append(phn[0][i].split(" ")[2])
                time_phoneme_1.append(np.float64(phn[0][i].split(" ")[0])/16000)
                time_phoneme_2.append(np.float64(phn[0][i].split(" ")[1])/16000)
      sn=[]
      for i in range(0,len(time_phoneme_1)):
         if(time_phoneme_1[i]>[st]):
            sn.append(i-1)
            s=sn[0]
            break
      if(len(sn)==0):
          sn.append(len(time_phoneme_1)-1) 
          s=sn[0]
          
      en=[]
      for i in range(0,len(time_phoneme_2)):
          if(time_phoneme_2[i]>[et]):
             en.append(i) 
             e=en[0]
             break
      
      if(len(en)==0):
          en.append(len(time_phoneme_2)-1) 
          e=en[0]
      
      count=0
      for i in range(s,e+1):
          vowels = ['aa', 'ae', 'ah', 'ao', 'aw', 'aax', 'ax', 'axh', 'ax-h','axr','ay','eh','el','em','en','enx','eng','er','ey','ih','ix','iy','ow','oy','uh','uw','ux']
          if phoneme_list[i] in vowels:
             count=count+1 
      num_sy=count
      sen_dur=et-st
      syllab_rate=num_sy/sen_dur    
      return syllab_rate
      
NoUnits=256 
BatchSize=16
#n_mfcc=39
#inputDim=n_mfcc

#epochs=60
EPSILON=0.0000001
co=0;va=0;te=0
VAL_loss = []
train_pearson_coeff = []
val_pearson_coeff = []
test_pearson_coeff = []
y_train = []
y_val = []
y_test = []

Wav_Fs=8000
## Raw waveforms
window_size=int(35*0.001*Wav_Fs) #560 window len
Hop_len=int(10*0.001*Wav_Fs) #160 wind shift
std_frac=0.25
n_mfcc=13
nb_outputs=12 #pred EMA dim

### CNN Spec
CNN1_filters= 16#256#128#64#32 #256#128#64#32#128#64#128#256 ##39 
CNN2_filters=0
CNN1_flength=int(30*0.001*Wav_Fs)
CNN2_flength=8
inputDim=1


def build_model():
   mdninput_Lstm = keras.Input(shape=(100,window_size, inputDim))   
   TCNNOp1BN=TimeDistributed(Convolution1D(nb_filter=CNN1_filters, filter_length=CNN1_flength,activation=Relu_log,input_shape=(window_size, inputDim)))(mdninput_Lstm)
   TCNNOp1=BatchNormalization(axis=-1)(TCNNOp1BN)
   TCNNOp2=TimeDistributed(MaxPooling1D(window_size-CNN1_flength+1))(TCNNOp1)
   #CNNOp=TimeDistributed(Flatten())(TCNNOp2)
   CNNOp = TimeDistributed(Reshape((CNN1_filters,1)))(TCNNOp2)
   c1 = Conv2D(64, (3,3), padding='same', activation = 'relu')(CNNOp)
   m1 = MaxPooling2D(pool_size = 3)(c1)
   b1 = BatchNormalization()(m1)
   c2 = Conv2D(64, (3,3), padding='same', activation = 'relu')(b1)
   c3 = Conv2D(128, (3,3), padding='same', activation = 'relu')(c2)
   m2 = MaxPooling2D(pool_size = 2)(c3)
   f1 = Flatten()(m2)
   d1 = Dense(64,activation = 'relu')(f1)
   b2 = BatchNormalization()(d1)
   dr1 = Dropout(0.20)(b2)
   d2 = Dense(32,activation = 'relu')(dr1)
   b3 = BatchNormalization()(d2)
   dr2 = Dropout(0.20)(b3)
   d3 = Dense(1,activation ='relu')(dr2)
   model = keras.models.Model(mdninput_Lstm,d3)
   return model
model = build_model()
model.summary()

X_dir='/home2/data/jyothi/Data/timit_data/wav/'
Y_dir='/home2/data/jyothi/Data/timit_data/phn/'

Mfccfiles=sorted(os.listdir(X_dir))
phnf=sorted(os.listdir(Y_dir))
subj=[]
ran=np.arange(0,5040,8)
for f in range(len(ran)):
        d=ran[f]
        extension=Mfccfiles[d][4:9]
        subj.append(extension)

for fold in range(0,4): #  no. of folds 
        X_val=[];Y_val=[];X_train=[];Y_train=[];X_test=[];Y_test=[]
        for subf in range(0,3): # for sets train,test,val          
            if (subf==0 ): # for train
                print('FOLD::'+str(fold+1)+'...................appending..' + str(subf)+'..list into X_train')
                for sub in range(0,len(subj)): # for all subjects
                        extension=subj[sub]
                        lsd=subfiles(extension,fold)
                        for j in range(0,2):
                            for i in range(0,2):
                                #MFCC_G =np.loadtxt(X_dir+lsd[j][i])
                                sig, sr = librosa.load(X_dir+lsd[j][i], sr=Wav_Fs);
                                sig = sig/max(abs(sig))
                                #y = sig
                                y = preemphasis(sig)
                                y = (y-np.mean(y))/np.std(y)
                                y_framed = librosa.util.frame(y,window_size, Hop_len).astype(np.float64).T
				#[aW,bW]=y_framed.shape
				#tEnd=np.min([aW,aE])
				#WavFiles.append(y_framed[np.newaxis, :,:,np.newaxis])
                                co=co+1 
                                #se=range(0,MFCC_G.shape[0],50)
                                se = range(0,y_framed.shape[0],50)
                                for m in range(0,len(se)-2):
                                        #temp=MFCC_G[se[m]:se[m+2]]
                                        #X_train.append(np.expand_dims(temp,axis=2))
                                        temp = y_framed[se[m]:se[m+2],:]
                                        X_train.append(np.expand_dims(temp,axis=2))
                                        Y_train.append(phoneme_rate(lsd[j][i][0:-4],se[m]/100 ,se[m+2]/100))
                
            elif(subf==1):
                print('FOLD::'+str(fold+1)+'...................appending..' + str(subf)+'..list into X_val')
                for sub in range(0,len(subj)): # for all subjects
                        extension=subj[sub]
                        lsd=subfiles(extension,fold)
                        for j in range(2,3):
                            for i in range(0,2):
                                #MFCC_G =np.loadtxt(X_dir+lsd[j][i]) 
                                sig, sr = librosa.load(X_dir+lsd[j][i], sr=Wav_Fs);
                                sig = sig/max(abs(sig))
                                #y = sig
                                y = preemphasis(sig)
                                y = (y-np.mean(y))/np.std(y)
                                y_framed = librosa.util.frame(y,window_size, Hop_len).astype(np.float64).T
                                va=va+1 
                                #se=range(0,MFCC_G.shape[0],50)
                                se = range(0,y_framed.shape[0],50)
                                for m in range(0,len(se)-2):
                                        #temp=MFCC_G[se[m]:se[m+2]]
                                        #X_val.append(np.expand_dims(temp,axis=2))
                                        temp = y_framed[se[m]:se[m+2],:]
                                        X_val.append(np.expand_dims(temp,axis=2))
                                        Y_val.append(phoneme_rate(lsd[j][i][0:-4],se[m]/100 ,se[m+2]/100))
            else:
                print('FOLD::'+str(fold+1)+'...................appending..' + str(subf)+'..list into X_test')
                for sub in range(0,630): # for all subjects
                        extension=subj[sub]
                        lsd=subfiles(extension,fold)
                        for j in range(3,4):
                            for i in range(0,2):
                                #MFCC_G =np.loadtxt(X_dir+lsd[j][i]) 
                                sig, sr = librosa.load(X_dir+lsd[j][i], sr=Wav_Fs);
                                sig = sig/max(abs(sig))
                                #y = sig
                                y = preemphasis(sig)
                                y = (y-np.mean(y))/np.std(y)
                                y_framed = librosa.util.frame(y,window_size, Hop_len).astype(np.float64).T
                                te=te+1 
                                #se=range(0,MFCC_G.shape[0],50)
                                se = range(0,y_framed.shape[0],50)
                                for m in range(0,len(se)-2):
                                        #temp=MFCC_G[se[m]:se[m+2]]
                                        #X_test.append(np.expand_dims(temp,axis=2))
                                        temp = y_framed[se[m]:se[m+2],:]
                                        X_test.append(np.expand_dims(temp,axis=2))
                                        Y_test.append(phoneme_rate(lsd[j][i][0:-4],se[m]/100 ,se[m+2]/100)) 
         

        print(len(X_train),co)
        print(len(Y_train))  
        print(len(X_val),va)
        print(len(Y_val))               
        num_test=len(X_test)
        num_train = len(X_train)
        num_val = len(X_val)
        c = list(zip(X_train, Y_train))
        print('Shuffling X_train and Y_train')
        random.shuffle(c)
        X_train, Y_train= zip(*c)
        X_train= [x for x in X_train]
        Y_train= [x for x in Y_train]
        #X_train=change_dims(np.array(X_train))
        X_train = np.array(X_train)
        Y_train=np.array(Y_train)
        #X_val=change_dims(np.array(X_val))
        X_val = np.array(X_val)
        Y_val = np.array(Y_val)
        #X_test=change_dims(np.array(X_test))
        X_test = np.array(X_test)
        Y_test=np.array(Y_test)
        
        
        fName1 = '/home2/data/jyothi/models/raw_waveform/seen_timit/'
        try:
             os.mkdir(fName1)
        except OSError as error:
            print(error)

        fName = fName1+'timit_fold_filters'+str(CNN1_filters)+'_'+str(fold+1)
        model = build_model()
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        checkpointer = ModelCheckpoint(filepath=fName +'_best.h5', monitor='val_loss',save_best_only=True, verbose=0,save_weights_only=True,period=1)
        callbacks = ModelCheckpoint((fName+'_weights.h5'), monitor='val_loss', verbose=0, save_weights_only=True, mode='auto', period=1)
        earlystopper = EarlyStopping(monitor='val_loss', patience=10)
        history=model.fit(X_train, Y_train,epochs=40,batch_size=16,validation_data=(X_val,Y_val),verbose=1, callbacks=[callbacks,checkpointer,earlystopper])

        sr_model=build_model()
        
        print('loading the best model ')
       	sr_model.load_weights(fName+'_best.h5')
        print('Test on samples of test and val ')

        h=sr_model.get_weights()
        scipy.io.savemat(fName+'_out.mat', {'weights':h})

        test_pred = np.zeros(num_test)
        val_pred = np.zeros(num_val)
        train_pred = np.zeros(num_train)
        test_pred = sr_model.predict(X_test)
        val_pred = sr_model.predict(X_val)
        print('Predicting values for '+str(num_train)+' samples of training data ')
        train_pred = sr_model.predict(X_train)

        print('Calculating Pearson Coefficient.....')
        test_corr, _ = pearsonr(np.squeeze(test_pred), np.squeeze(Y_test))
        test_pearson_coeff.append(test_corr)
        val_corr, _ = pearsonr(np.squeeze(val_pred), np.squeeze(Y_val))
        val_pearson_coeff.append(val_corr)
        train_corr, _ = pearsonr(np.squeeze(train_pred), np.squeeze(Y_train))
        train_pearson_coeff.append(train_corr)

        print('True Test Values : ' + str(Y_test))
        print('Test Predictions : ' + str(test_pred))
        print('Test Pearson Coefficient : ' + str(test_pearson_coeff))

        print('True Val Values : ' + str(Y_val))
        print('Val Predictions : ' + str(val_pred))
        print('Val Pearson Coefficient : ' + str(val_pearson_coeff))
	
        print('True Train Values : ' + str(Y_train))
        print('Train Predictions : ' + str(train_pred))
        print('Train Pearson Coefficient : ' + str(train_pearson_coeff))

        y_test.append(test_pred)
        y_val.append(val_pred)
        y_train.append(train_pred)

test_pearson_coeff = np.array(test_pearson_coeff)
test_pearson_coeff_avg = np.mean(test_pearson_coeff)
print('Test Pearson Coefficient Avg: '+str(test_pearson_coeff_avg))

val_pearson_coeff = np.array(val_pearson_coeff)
val_pearson_coeff_avg = np.mean(val_pearson_coeff)
print('Val Pearson Coefficient Avg: '+str(val_pearson_coeff_avg))

train_pearson_coeff = np.array(train_pearson_coeff)
train_pearson_coeff_avg = np.mean(train_pearson_coeff)
print('Train Pearson Coefficient Avg: '+str(train_pearson_coeff_avg))

 
scipy.io.savemat(fName1+str(CNN1_filters)+'_Ytest_pred.mat', {'Ytest_pred': Y_test}, oned_as='row')
scipy.io.savemat(fName1+str(CNN1_filters)+'_test_coeff.mat', mdict={'test_coeff': test_pearson_coeff}, oned_as='row')
scipy.io.savemat(fName1+str(CNN1_filters)+'_Yval_pred.mat', {'Yval_pred': Y_val}, oned_as='row')
scipy.io.savemat(fName1+str(CNN1_filters)+'_val_coeff.mat', mdict={'val_coeff': val_pearson_coeff}, oned_as='row')
scipy.io.savemat(fName1+str(CNN1_filters)+'_Ytrain_pred.mat', {'Ytrain_pred': Y_train}, oned_as='row')
scipy.io.savemat(fName1+str(CNN1_filters)+'_train_coeff.mat', mdict={'train_coeff': train_pearson_coeff}, oned_as='row')
