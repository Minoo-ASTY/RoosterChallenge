#!/usr/bin/env python
# coding: utf-8

# # Rooster challenge 
# ## Audio classification using neural network concept
# 
#                             ~-.
#           ,,,;            ~-.~-.~-
#          (.../           ~-.~-.~-.~-.~-.
#          } o~`,         ~-.~-.~-.~-.~-.~-.
#          (/    \      ~-.~-.~-.~-.~-.~-.~-.
#           ;    \    ~-.~-.~-.~-.~-.~-.~-.
#          ;     {_.~-.~-.~-.~-.~-.~-.~
#         ;:  .-~`    ~-.~-.~-.~-.~-.
#        ;.: :'    ._   ~-.~-.~-.~-.~-
#         ;::`-.    '-._  ~-.~-.~-.~-
#         
#          ;::. `-.    '-,~-.~-.~-.
#           ';::::.`''-.-'
#             ';::;;:,:'
#                '||"
#                / |
#              ~` ~"'
#              

# ### Train Convolutional Neural Network (CNN) model from ESC-50 data set

# In[22]:



import os
os.sys.path


# #load the zip file and unzip and before check the GPU

# In[23]:


import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# #load the zip file and unzip and before check the GPU

# In[26]:


# import dependencies
import requests, zipfile, io
from glob import glob
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
seed = 7
import pandas as pd
np.random.seed(seed)
import os


# In[27]:


zip_file_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip' # link: ESC-50 Datset


# In[28]:


if not os.path.exists('sound'):
    os.makedirs('sound')


# In[30]:


r = requests.get(zip_file_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('sound/')
z.close()


# In[31]:


glob('sound/ESC-50-master/audio/*')


# # Define a function to covert the image based on calculate log scaled mel-spectrograms and their corresponding deltas from a sound clip.

# Regarding fixed size input, we will divide each sound clip into segments of 60x41 (60 rows and 41 columns). The mel-spec and their deltas will become two channels, which we will be fed into CNN

# In[32]:


get_ipython().system('pip install librosa')
import librosa


# In[33]:


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)

def extract_features(bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for fn in tqdm(glob('sound/ESC-50-master/audio/*')):
        sound_clip,s = librosa.load(fn) # 5sec
        sound_clip   = np.concatenate((sound_clip,sound_clip),axis=None) # make it 10s
        label = fn.split("/")[-1].split("-")[-1].split(".")[0]
        for (start,end) in windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.core.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
                labels.append(label)
            
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    return np.array(features), np.array(labels,dtype = np.int)


# In[34]:


features,labels = extract_features()


# In[35]:


# label category names
df = pd.read_csv(glob('sound/ESC-50-master/meta/esc50.csv')[0])
df = df[['target','category']]
df = df.drop_duplicates().reset_index(drop=True)
df = df.sort_values(by=['target']).reset_index(drop=True)
df.head()


# In[36]:


my_dict = {}
for i in range(len(df)):
  my_dict[df['target'][i]] = df['category'][i]
my_dict


# In[37]:


seed = 4
rng = np.random.RandomState(seed)
from keras.utils import to_categorical


# In[38]:


onehot_labels = to_categorical(labels,num_classes=50)


# In[39]:


# Create train test Dataset

rnd_indices = np.random.rand(len(labels)) < 0.70

X_train = features[rnd_indices]
y_train = onehot_labels[rnd_indices]
X_test  = features[~rnd_indices]
y_test  = onehot_labels[~rnd_indices]


# In[40]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape, 


# # CNN Model

# In[41]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten,InputLayer
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint


# In[42]:


def basemodel():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape=(60,41,2), activation='relu', padding='same'))
  model.add(Dropout(0.2))
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
  model.add(Dropout(0.2))
  model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
  model.add(Dropout(0.2))
  model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dropout(0.2))
  model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
  model.add(Dropout(0.2))
  model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
  model.add(Dropout(0.2))
  model.add(Dense(50, activation='softmax'))
  # Compile model
  epochs = 25
  lrate = 0.01
  decay = lrate/epochs
  sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  return model


# In[ ]:


import os
if not os.path.exists('model'):
    os.makedirs('model')
    
filepath="model/weights_0.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[45]:


model = basemodel()
print(model.summary())


# # Training with Data Augmentation

# One of the major reasons for overfitting is that we don’t have enough data to train our network. Apart from regularization, another very effective way to counter Overfitting is Data Augmentation. It is the process of artificially creating more images from the images you already have by changing the size, orientation etc of the image. It can be a tedious task but fortunately, this can be done in Keras using the ImageDataGenerator instance.

# In[46]:


from keras.preprocessing.image import ImageDataGenerator


# In[48]:


datagen = ImageDataGenerator(
              width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
              height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
              horizontal_flip=True,  # randomly flip images
              vertical_flip=False  # randomly flip images
          )


# In[49]:


# init the batch size and epochs

'''
Note: Due to Memory Error like Buffered data was truncated after reaching the output size limit. What i did is that Save the model in for example 60th epoch and close current program and run new program and restore saved model and train model from 61 epoch to 120 epoch and 
save that and close program and repeat this work for your interested epoch For this [100,50] three times repeat 

'''
batch_size = 50
epochs = 100


# In[41]:


# fit the model
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                              steps_per_epoch=int(np.ceil(X_train.shape[0] / float(batch_size))),
                              epochs=epochs,
                              validation_data=(X_test, y_test),
                              verbose=1,callbacks=callbacks_list)
 


# Note: Due to Memory Error like Buffered data was truncated after reaching the output size limit.
# What i did is that Save the model in for example 60th epoch and close current program and run new program and restore saved model and train model from 61 epoch to 120 epoch and save that and close program and repeat this work for your interested epoch 
# For this [100,50] three times repeat 
# 

# In[50]:


# evaluate model
model.evaluate(X_test, y_test)


# # Classification Report and Confusion Matrix

# In[52]:


from sklearn.metrics import classification_report, confusion_matrix


# In[53]:


y_pred = model.predict_classes(X_test)


# In[54]:


target_name = np.array(df['category'])


# In[55]:


print(classification_report(np.argmax(y_test,axis=1),y_pred,target_names=target_name))


# In[56]:


print(confusion_matrix(np.argmax(y_test,axis=1),y_pred))


# # Test with Real time Data of 10s Sound Clip to the Model

# In[57]:



class convertSound2image:
  
  def __init__(self,sourcePath):
    '''
    Insert the source path of sound 10s 
    '''
    self.sourcePath = sourcePath
  
  def windows(self,data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)

  def extract_features(self,bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    sound_clip,s = librosa.load(self.sourcePath) # 10sec
    for (start,end) in self.windows(sound_clip,window_size):
        if(len(sound_clip[start:end]) == window_size):
            signal = sound_clip[start:end]
            melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
            logspec = librosa.core.amplitude_to_db(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features)


# In[70]:


get_ipython().system('pip install -U -q PyDrive')
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import os
import pandas as pd
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


# In[76]:


listed = drive.ListFile({'q': "title contains 'rooster_competition.wav'"}).GetList()
for file in listed:
 print('title {}, id {}'.format(file['title'], file['id']))


# In[ ]:


download_path = os.path.expanduser('~/sample')
try:
  os.makedirs(download_path)
except OSError:
  pass


# In[ ]:


sample = os.path.join('rooster_competition.wav')


# In[84]:


sound_clip,s = librosa.load(sample)
sound_clip,s


# # No Backends Error of librosa

# In[85]:


#!apt install libav-tools


# 
# #Some visualiztion from the roosters voice 
# 
# ### Loading file

# In[86]:


import librosa
audio_data = 'rooster_competition.wav'
x , sr = librosa.load(audio_data)
print(type(x), type(sr))#<class 'numpy.ndarray'> <class 'int'>print(x.shape, sr)#(94316,) 22050


# In[87]:


librosa.load(audio_data, sr=44100)


# Disable resampling

# In[90]:


librosa.load(audio_data, sr=None)


# In[91]:


import IPython.display as ipd
ipd.Audio(audio_data)


# In[92]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# In[93]:


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()


# In[94]:


librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()


# Create an Audio Signal:
# 
# 

# In[102]:


import numpy as np
import soundfile as sf
sr = 22050 # sample rate
T = 5.0    # seconds
t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
x = 0.5*np.sin(2*np.pi*220*t)# pure sine wave at 220 Hz
#Playing the audio
ipd.Audio(x, rate=sr) # load a NumPy array
#Saving the audio
sf.write('tone_220.wav', x, sr)


# librosa.feature.spectral_centroid computes the spectral centroid for each frame in a signal
# 

# In[ ]:


import sklearn
import matplotlib.pyplot as plt
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape(775,)
# Computing the time variable for visualization
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')


# Spectral Rolloff: Computing the rolloff frequency for each frame in a signal
# 

# In[ ]:


spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
plt.figure(figsize=(12, 4))librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')


# Spectral Bandwidth

# In[ ]:


spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
plt.figure(figsize=(15, 9))librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
plt.legend(('p = 2', 'p = 3', 'p = 4'))


# Zero-Crossing Rate

# In[ ]:


x, sr = librosa.load('/../../gruesome.wav')
#Plot the signal:
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
# Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()


# Zoom in

# In[ ]:


n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()


# In[ ]:


zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))#16


# Mel-Frequency Cepstral Coefficients(MFCCs)

# In[ ]:


mfccs = librosa.feature.mfcc(x, sr=fs)
print(mfccs.shape)
(20, 97)
#Displaying  the MFCCs:
plt.figure(figsize=(15, 7))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')


# Chroma feature

# In[ ]:


chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')

