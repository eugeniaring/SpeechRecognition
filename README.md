# Speech command recognition using CNN andCNN-LSTM Neural Networks

I built a keyword spotting system that classifies 36 keywords including noise. 
The reference dataset is released by Google in August 2017.
There are 105,836 one second long utterances of 35 words + noise, where the audio duration is one second. 

# 1) Features extraction

A well established feature representation for speech is the ”logmel-spectrum”

Steps:
1.  Define a 30 ms analysis window.
2.  Segment the signal into short frames by shifting the window 10ms .
3.  Obtain 98 time frames.
4.  Calculate **Fast Fourier Transformation** for each frame to obtain the frequency features.
5.  Apply **logarithmic Mel-Scale filter bank** to the Fouriertransformed frames.
6.  Calculate **Discrete Cosine Transformation**(DCT) to obtain the 40-dimensional coefficients vector

# 2)  Hybrid VGG+LSTM architecture

1. Import libraries/datasets 
2. Visualize features extracted
3. Train an VGG+LSTM Model
4. Assess/evaluate trained model performance
