### Predict an accent of a speaker

####Problem Statement:
Despite the progress of automatic speech recognition (ASR) systems, accent is still an issue in developing robust ASR systems. Research has shown that speaking in our native accent with these ASR systems typically end up with not much success. Accent classifier can  deliver high performance speech recognition across diverse user groups. Since accent is such a crucial aspect in ASR, we were inspired to build an accent classification ML model to the identify accent for better performance for multiple applications. 


####Datasets:
We used common voice mozilla datasets (https://voice.mozilla.org/en/datasets). Common voice collects data from users where users can submit their own records and/or review other records for accuracy and manual translation. Datasets include audio (.mp3) files,  transcript of the phrase, as well as the accent, age, and sex of the speaker. There were 16 accents, we have selected top 5 accents in this study.

####Data Description:
There were ~41000 audio files for US, England, Canada, India, Australia accents which includes 64% of Male and 365 of females.

####Methodology:
Feature Extractions: 
Step 1:  Loading wav files through librosa 
Step 2:  Trimming long silences in the wav file 
Step 3:  Extracting features using Kaldi-toolkit
                   Features: MFCC + Filterbank + delta_1 +  delta_2
                   Target Variable: Frame level accent vector
Step 4:  Appending Gender and Age along with voice features
Step 5: Fitting train data Convolutional/Bidirectional LSTM
Step 6: Predicting accents by frames
Step 7: Label majority vote of accent from frames to speaker

####Modelling:
We built two models using CNN and BLSTM. CNN model outperformed BLSTM model. 
Frame level model accuracy in train is 80%  while in test it is 73%. Speaker level model accuracy is 92% in train whereas 84% in test. 
