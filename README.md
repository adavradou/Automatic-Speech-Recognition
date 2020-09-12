# Automatic-Speech-Recognition

This is an Automatic Speech Recognition (ASR) system. 
It takes a recording of 4 to 10 digits (between 0 and 9) as an input, and gives the recognized digit in text form as an output.
The system is developed in such way, that it is not biased from the speaker's voice characteristics.

Both a Fully Connected Neural Network (FCN) and a Convolutional Neural Network (CNN) were developed, with the latter having way better performance. 

Change of sampling rate, filtering and trimming were applied as a preprocessing step for both networks, and additionally, Fourier transform and Mel-Frequency Cepstral
Coefficients (MFCC) were applied only to the FCN.
