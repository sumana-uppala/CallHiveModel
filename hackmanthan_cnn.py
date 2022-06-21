#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import joblib
import librosa
import time
import numpy as np


# In[2]:


TRAINING_FILES_PATH = "/Users/supriyauppala/Desktop/hackmanthan/Audio_Speech_Actors_01-24/"
lst=[]
for subdir, dirs, files in os.walk(TRAINING_FILES_PATH):
    for file in files:
        try:
            if(file=='.DS_Store'):
                continue
            X, sample_rate = librosa.load(os.path.join(subdir,file),res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
            file_class = int(file[6:8])
            if(file_class == 1):
                file_class = 0  
            elif(file_class == 2):
                file_class = 0
            elif(file_class == 3):
                file_class = 0
            elif(file_class == 4):
                file_class = 1
            elif(file_class == 5):
                file_class = 2
            elif(file_class == 6):
                file_class = 3
            elif(file_class == 7):
                file_class = 0
            elif(file_class == 8):
                file_class = 0
            arr = mfccs, file_class
            lst.append(arr)
        except ValueError as err:
            print(err)
            continue


# In[3]:


TRAINING_FILES_PATH1 = "/Users/supriyauppala/Desktop/hackmanthan/dataverse_files"
for subdir, dirs, files in os.walk(TRAINING_FILES_PATH1):
    for file in files:
        try:
            if(file=='.DS_Store'):
                continue
            X, sample_rate = librosa.load(os.path.join(subdir,file),res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
            l1=file.split('_')
            class_ = l1[-1]
            file_class = 0
            if(class_=="angry.wav"):
                file_class = 2
            elif(class_=="disgust.wav"):
                file_class = 0
            elif(class_=="fear.wav"):
                file_class = 3
            elif(class_=="happy.wav"):
                file_class = 0
            elif(class_=="neutral.wav"):
                file_class = 0
            elif(class_=="ps.wav"):
                file_class = 0
            elif(class_=="sad.wav"):
                file_class = 1
            else:
                print("Class not found")
                continue
            arr = mfccs, file_class
            lst.append(arr)
        except ValueError as err:
            print(err)
            continue
    


# In[4]:


TRAINING_FILES_PATH2 = "/Users/supriyauppala/Desktop/hackmanthan/AudioData/"
for subdir, dirs, files in os.walk(TRAINING_FILES_PATH2):
    for file in files:
        try:
            if(file=='.DS_Store'):
                continue
            X, sample_rate = librosa.load(os.path.join(subdir,file),res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
            file_class = 0
            if(file[0]=='a'):
                file_class = 2
            elif(file[0]=='d'):
                file_class = 0
            elif(file[0]=='f'):
                file_class = 3
            elif(file[0]=='h'):
                file_class = 0
            elif(file[0]=='n'):
                file_class = 0
            elif(file[0]=='s'):
                if(file[1]=='a'):
                    file_class = 1
                elif(file[1]=='u'):
                    file_class = 0
            else:
                print("=========")
                continue
            arr = mfccs, file_class
            lst.append(arr)
        except ValueError as err:
            print(err)
            continue
    


# In[5]:


X, y = zip(*lst)
X, y = np.asarray(X), np.asarray(y)
print(X.shape, y.shape)


# In[7]:


y1=y.tolist()
set1=set(y1)
print(set1)
frequency = {}
for item in y1:
    if item in frequency:
        frequency[item] += 1
    else:
        frequency[item] = 1
print(frequency)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[9]:


X_train.shape, y_train.shape


# In[10]:


x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)


# In[11]:


x_traincnn.shape, x_testcnn.shape


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

model = Sequential()

model.add(Conv1D(128, 5,padding='same',
                 input_shape=(40,1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
opt = keras.optimizers.RMSprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)


# In[13]:


model.summary()


# In[14]:


model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


# In[15]:


cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=1000, validation_data=(x_testcnn, y_test))


# In[17]:


model_name = 'Emotion_speech_recognition_Model2.hdf5'
save_dir = '/Users/supriyauppala/Desktop/hackmanthan/model/'
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# In[18]:


loaded_model = keras.models.load_model(save_dir+'Emotion_speech_recognition_Model1.hdf5')
loaded_model.summary()


# In[19]:


loss, acc = loaded_model.evaluate(x_testcnn, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[20]:


lst1=[]
X, sample_rate = librosa.load("/Users/supriyauppala/Desktop/hackmanthan/dataverse_files/OAF_back_angry.wav",res_type='kaiser_fast')
mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
file_class = 0
arr = mfccs, file_class
lst1.append(arr)


# In[21]:


X1, y1 = zip(*lst1)
X1, y1 = np.asarray(X1), np.asarray(y1)
print(X1.shape, y1.shape)


# In[22]:


x_traincnn1 = np.expand_dims(X1, axis=2)
ans_y = np.argmax(loaded_model.predict(x_traincnn1), axis=-1)


# In[23]:


print(ans_y) # ['neutral', 'sad', 'angry', 'fearful'] 

