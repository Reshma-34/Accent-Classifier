import logging
import os
from collections import defaultdict
from pathlib import Path
from scipy.ndimage.morphology import binary_dilation
from typing import Optional, Union
import webrtcvad
import struct
from scipy import stats
import seaborn as sn
import matplotlib.pyplot as plt

import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import fbank
import scipy.io.wavfile as wav

import itertools

from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import LSTM, Dense, Dropout, Flatten,LeakyReLU, Input, SpatialDropout1D, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras.utils import np_utils
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix
import time
date           = '1003'
np.random.seed(1337)  # for reproducibility

# from constants import SAMPLE_RATE, NUM_FBANKS
# from utils import find_files, ensures_dir

logger = logging.getLogger(__name__)
path_to_midway = '/scratch/midway2/reshmask/'

# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30  # In milliseconds
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out. 
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6

## Audio volume normalization
audio_norm_target_dBFS = -30
int16_max              = (2 ** 15) - 1

speaker_dir = 'common_voice_wav_/home/reshmask/Desktop/common_voice_wav/'

def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    vad_window_length = 30 
    sampling_rate     = 16000
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def normalize_frames(m,Scale=True):
    if Scale:
        return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))

def get_wav(filenames):
    '''
    Load wav file from disk
    :param language_num (list): list of file names
    :return (numpy array): wav files
    '''
    wav, sr = librosa.load(speaker_dir+'{}'.format(filenames))
#     rescaling_max = 0.9
#     wav         = wav / np.abs(wav).max() * rescaling_max
    return(wav)

def get_kaldi_features(wav_, y_, X_):
    '''
    Get Kaldi - Discrete FFT features
    :param wav_: list of trimmed wav file
    :param y   : Array of accents
    :param filename: Array of filenames 
    :return (numpy array): array of (mfcc, filter_banks, delta_1, delta_2), accent array (utternace level), dict(filename,number of frames)
    '''
    n_mfcc   = 13
    n_filt   = 32
    features = []
    target   = []
    f_len    = defaultdict(list)
    for wav, accent, x_arr in (zip(wav_, y_, np.array(X_))):
        if len(wav) > 0:
            mfcc_                  = mfcc(wav, samplerate=16000, winlen=0.025, winstep=0.01, numcep=n_mfcc)
            filter_banks, energies = fbank(wav, samplerate=16000, nfilt=n_filt)
            filter_banks           = 20 * np.log10(np.maximum(filter_banks,1e-5))
            delta_1                = delta(filter_banks, N=1)
            delta_2                = delta(delta_1, N=1)

            filter_banks = normalize_frames(filter_banks, Scale=True)
            delta_1      = normalize_frames(delta_1, Scale=True)
            delta_2      = normalize_frames(delta_2, Scale=True)
            accent_      = list(itertools.repeat(accent, len(mfcc_)))
            dummies      = list(itertools.repeat(x_arr[1:], len(mfcc_)))
            frames_features = np.hstack([mfcc_, filter_banks, delta_1, delta_2, dummies])
            features.append(frames_features)
            target.append(accent_)
            f_len[x_arr[0]] = [len(mfcc_),accent]# num of frames
    features = np.vstack(features)
    target   = np.hstack(target)
    df = pd.DataFrame.from_dict(f_len,orient='index').reset_index()
    df.columns = ['filename', 'frame_len', 'accent']
    return features, target, df

#def get_kaldi_features(wav, accent, x_arr):
    '''
    Get Kaldi - Discrete FFT features
    :param wav_: list of trimmed wav file
    :param y   : Array of accents
    :param filename: Array of filenames 
    :return (numpy array): array of (mfcc, filter_banks, delta_1, delta_2), accent array (utternace level), dict(filename,number of frames)
    '''
#    time.sleep(5)
#    n_mfcc   = 13
#    n_filt   = 64
#    features = []
#    target   = []
#    f_len    = defaultdict(list)
#    if len(wav) > 0:
#        mfcc_                  = mfcc(wav, samplerate=16000, winlen=0.025, winstep=0.01, numcep=n_mfcc)
#        filter_banks, energies = fbank(wav, samplerate=16000, nfilt=n_filt)
#        filter_banks           = 20 * np.log10(np.maximum(filter_banks,1e-5))
#        delta_1                = delta(filter_banks, N=1)
#        delta_2                = delta(delta_1, N=1)

#        filter_banks = normalize_frames(filter_banks, Scale=True)
#        delta_1      = normalize_frames(delta_1, Scale=True)
#        delta_2      = normalize_frames(delta_2, Scale=True)
#        accent_      = list(itertools.repeat(accent, len(mfcc_)))
#        dummies      = list(itertools.repeat(x_arr[1:], len(mfcc_)))
#        frames_features = np.hstack([mfcc_, filter_banks, delta_1, delta_2, dummies])
#        features.append(frames_features)
#        target.append(accent_)
#        f_len[x_arr[0]] = [len(mfcc_),accent]# num of frames
    #features = np.vstack(features)
    #target   = np.hstack(target)
    #df = pd.DataFrame.from_dict(f_len,orient='index').reset_index()
    #df.columns = ['filename', 'frame_len', 'accent']
#    return features, target, f_len

#def convert_feat_form(arr):
#    features= np.vstack([ i[0] for i in arr])
#    target  = np.hstack([ i[1] for i in arr])
#    f_len   = {list(i[2].keys())[0]:list(i[2].values())[0] for i in arr}
#    df      = pd.DataFrame.from_dict(f_len,orient='index').reset_index()
#    df.columns = ['filename', 'frame_len', 'accent']
#    return features, target, df


def train_conv1D_model(train_X, Y_train, val_X, Y_val, input_shape):
    '''
    Get convoluted 1D neural network trained model
    :param train_X     : Predictor array for train data (mfccs, filterbanks, delta_1, delta_2)
    :param Y_train     : Target array in training (Frame level)
    :param val_X       : Predictor array for validation data (mfccs, filterbanks, delta_1, delta_2)
    :param Y_val       : Target array in validation (Frame level)
    :param input_shape : Input shape for trained data (model parameter)
    :return            : convolution1D trained neural network model
    '''
    batch_size  = 3000 
    epochs      = 20
    nb_classes  = 5 

    model1 = Sequential()
    model1.add(Conv1D(32, kernel_size=(3),activation='linear',input_shape= input_shape,padding='same'))
    model1.add(LeakyReLU(alpha=0.1))
    model1.add(MaxPooling1D((2),padding='same'))
    model1.add(Conv1D(128, (3), activation='linear',padding='same'))
    model1.add(LeakyReLU(alpha=0.1))
    model1.add(MaxPooling1D(pool_size=(2),padding='same'))
    model1.add(Conv1D(128, (3), activation='linear',padding='same'))
    model1.add(LeakyReLU(alpha=0.1))                  
    model1.add(MaxPooling1D(pool_size=(2),padding='same'))
    model1.add(Flatten())
    model1.add(Dense(128, activation='linear'))
    model1.add(LeakyReLU(alpha=0.1))                  
    model1.add(Dense(nb_classes, activation='softmax'))
    
    model1.compile(loss = 'categorical_crossentropy',
              optimizer= "Adam",
              metrics=[convo_f1_score])
    # Stops training if accuracy does not change at least 0.005 over 3 epochs
    es = EarlyStopping(monitor="val_loss", min_delta=.005, patience=3, verbose=1, mode='auto') #monitor='accuracy'
    
    history = model1.fit(train_X, Y_train,
          batch_size=batch_size,
          epochs= epochs,
          verbose=1,callbacks = [es,], 
          validation_data=(val_X, Y_val))
    return (history, model1)

def convo_f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def train_blstm_model(train_X, Y_train, val_X, Y_val, input_shape):
    '''
     Get Bidirectional LSTM neural network trained model
    :param train_X     : Predictor array for train data (mfccs, filterbanks, delta_1, delta_2)
    :param Y_train     : Target array in training (Frame level)
    :param val_X       : Predictor array for validation data (mfccs, filterbanks, delta_1, delta_2)
    :param Y_val       : Target array in validation (Frame level)
    :param input_shape : Input shape for trained data (model parameter)
    :return            : Bidirectional LSTM trained neural network model
    '''
    batch_size  = 5000 
    epochs      = 20
    nb_classes  = 5 
    model       = Sequential()

    # note that it is necessary to pass in 3d batch_input_shape if stateful=True
    model.add(Bidirectional(LSTM(64, return_sequences=True, stateful=False,
           batch_input_shape= (batch_size, input_shape))))
    model.add(Bidirectional(LSTM(64, return_sequences=True, stateful=False)))
    model.add(Bidirectional(LSTM(64, stateful=False)))

    # add dropout to control for overfitting
    model.add(Dropout(.25))

    # squash output onto number of classes in probability space
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=[convo_f1_score])

    history = model.fit(train_X, Y_train, batch_size=batch_size, nb_epoch=20, validation_data=(val_X, Y_val))
    return (history, model)


def get_speaker_pred(file_len_, y_preds):
    '''
    Get spekaer level prediction my calculating mode(frame level prediction)
    :param file_len_ : dict of filename as key and length of frames as a value
    :param y_preds   : Array of prediction class for frame
    :return (numpy array): numpy array of prediction class at speaker level
    '''
    sum_ = 0 
    bins = []
    y_pred_cls = []
    for v in file_len_:
        sum_ = sum_ + v
        bins.append(sum_)
    bins = [0] + bins
    for idx in range(1, len(bins)):
        y_pred_cls.append(stats.mode(y_preds[bins[idx-1]:bins[idx]])[0][0])
        
    return y_pred_cls

def save_report(name, y_true, y_pred):
    
    report      = classification_report(y_true, y_pred)
    report_path = name+".txt"

    text_file   = open(report_path, "w")
    n           = text_file.write(report)
    text_file.close()
    
def save_cm(name, y_true, y_pred, all_classes):
    plt.figure(figsize=(10,6))
    cm    = confusion_matrix(y_true, y_pred)
    cm    = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, columns = all_classes, index=all_classes)
    sn.set(font_scale=1) # for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size":12}, cmap="YlGnBu") # font size  {"size": 35 / np.sqrt(len(corrmat))}},
    plt.savefig(name.format(date),bbox_inches='tight')