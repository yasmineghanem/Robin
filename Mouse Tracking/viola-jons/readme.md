# python implementation is the worst , it worser than worst c++ implementation without any optmizations

# basic implemntation following the paper without any code optimization

for 20 point
training time is 1400 sec

# before memory optimization with normal adaboost

2000 point
time for loading data is : 34.554 s
Training classifier layer 1
erorr in layer : 0 is : 0.194
Training time: 173.963 s
saving model to model2.txt
Accuracy: 0.73875
Error rate: 0.26125
False positive rate: 0.055
False negative rate: 0.4675

# after memory optimization with normal adaboost

2000 point
time for loading data is : 22.2578 s
Training classifier layer 1
erorr in layer : 0 is : 0.194
Classifier trained!
Training time: 145.427 s
saving model to model2.txt
Accuracy: 0.73875
Error rate: 0.26125
False positive rate: 0.055
False negative rate: 0.4675

# face detector "viola-jones algorithm " without working with validation and without optimization memory

2000 point
total training set =2000
memory usage 3.8 GB ram
Yo=0.5 , Yl=0.5, Bl=0.5
time for loading data is : 29.1783 s
.... Training Face Detector ....
erorr in layer : 0 is : 0.194
layer 1 is trained
false positive rate: 0.234
false negative rate: 0.154
Classifier trained!
Training time: 182.046 s
saving model to face1
Accuracy: 0.747942
Error rate: 0.252058
False positive rate: 0.05
False negative rate: 0.466102

# face detector "viola-jones algorithm " without working with validation but with optmization

2000 point
Yo=0.5 , Yl=0.5, Bl=0.5
memory 1.2GB
time for loading data is : 10.5066 s
.... Training Face Detector ....
adaboost number : 1 layer number : 1 shift: 0 false positive rate : 0.234 false negative rate : 0.154
layer 1 is trained
false positive rate: 0.234
false negative rate: 0.154
Classifier trained!
Training time: 90.4031 s
saving model to face1
Accuracy: 0.747942
Error rate: 0.252058
False positive rate: 0.05
False negative rate: 0.466102

for n=11K "full data"
time : 660 sec
memory 8GB

# face detector "viola-jones algorithm " without working with validation but with optmization and multithreading

2000 points
training time 15 sec

for n=11K "full data"
time for loading data is : 44.7206 s
.... Training Face Detector ....
availabel threads : 12
adaboost number : 1 layer number : 1 shift: 0 false positive rate : 0.164483 false negative rate : 0.0843969
layer 1 is trained
false positive rate: 0.164483
false negative rate: 0.0843969
Classifier trained!
Training time: 132.839 s
saving model to face1
Accuracy: 0.897675
