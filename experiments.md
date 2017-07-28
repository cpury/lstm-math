# EXPERIMENTS ON MATH AND LSTMs

## 1.
LSTM(2) -> DENSE(1)
recurrent_activation='hard_sigmoid'
optimizer='adam'

### Result:
Epoch 83/200
1s - loss: 0.3850 - val_loss: 0.4438

Some examples:
"4 + 8" = 11.378273010253906 (12)
"7 + 0" = 7.418281555175781 (7)
"8 + 2" = 9.624608039855957 (10)
"4 + 0" = 4.910335540771484 (4)
"4 + 9" = 12.510830879211426 (13)
"8 + 3" = 10.72602367401123 (11)
"2 + 8" = 9.02862548828125 (10)
"6 + 2" = 7.825167179107666 (8)
"7 + 4" = 10.902262687683105 (11)
"8 + 1" = 8.627217292785645 (9)

### Interpretation
Can learn simple addition with 2 LSTM units. Would have converged further.
Learning rate can probably be improved.



## 2.
Same but with recurrent_activation='relu'

### Result:
Epoch 89/200
0s - loss: 0.3442 - val_loss: 0.7308

Some examples:
"3 + 1" = 3.48122239112854 (4)
"5 + 1" = 6.603221416473389 (6)
"8 + 1" = 8.824166297912598 (9)
"4 + 1" = 6.1089653968811035 (5)
"1 + 7" = 7.156800270080566 (8)
"2 + 0" = 2.802712917327881 (2)
"2 + 9" = 12.44596004486084 (11)
"0 + 9" = 8.189338684082031 (9)
"1 + 8" = 7.226089000701904 (9)
"5 + 7" = 11.940388679504395 (12)

### Interpretation
recurrent_activation doesn't need to be relu, it seems.



## 3.
Back to 1.
optimizer = RMSprop(
    lr=0.001,
    clipnorm=1.,
)

### Result
Epoch 30/200
1s - loss: 5.4777 - val_loss: 7.8369

Some examples:
"5 + 6" = 8.59000015258789 (11)
"3 + 4" = 6.820000171661377 (7)
"2 + 3" = 6.519999980926514 (5)
"1 + 2" = 6.039999961853027 (3)
"1 + 5" = 6.96999979019165 (6)
"6 + 4" = 6.690000057220459 (10)
"4 + 0" = 4.639999866485596 (4)
"5 + 8" = 10.0 (13)
"3 + 6" = 7.210000038146973 (9)
"6 + 2" = 6.039999961853027 (8)

### Interpretation
Terrible... Either RMSprop sucks for this, or learning rate, or gradient clipping.



## 4.
Same as 3. but without gradient clipping

### Result
Epoch 152/200
1s - loss: 0.3278 - val_loss: 2.1981

Some examples:
"0 + 3" = 4.610000133514404 (3)
"7 + 6" = 13.520000457763672 (13)
"0 + 1" = 3.4600000381469727 (1)
"8 + 4" = 12.020000457763672 (12)
"0 + 6" = 6.730000019073486 (6)
"8 + 9" = 20.34000015258789 (17)
"5 + 6" = 11.0600004196167 (11)
"3 + 7" = 9.930000305175781 (10)
"3 + 0" = 4.010000228881836 (3)
"4 + 6" = 10.0 (10)

### Interpretation
RMSprop: Slower and just not as good



## 5.
Completely different:
LSTM(4, no activation) -> DROPOUT(.5) -> DENSE(1)
'adam'

### Results
Epoch 200/200
1s - loss: 6.7605 - val_loss: 2.0614

Some examples:
"8 + 7" = 12.0 (15)
"8 + 6" = 11.6899995803833 (14)
"1 + 8" = 7.400000095367432 (9)
"3 + 7" = 8.579999923706055 (10)
"7 + 2" = 8.59000015258789 (9)
"7 + 0" = 6.590000152587891 (7)
"8 + 4" = 10.0600004196167 (12)
"0 + 9" = 8.010000228881836 (9)
"0 + 1" = 3.4000000953674316 (1)
"1 + 0" = 3.0899999141693115 (1)

### Interpretation
Not as good as 1. lol... Would have converged further, though. Why so slow?
But very clean conversion.



## 6.
Same as 5. but with relu activation in LSTM layer

### Results
Epoch 125/400
1s - loss: 14.8930 - val_loss: 9.3832

Some examples:
"6 + 0" = 6.539999961853027 (6)
"0 + 7" = 6.420000076293945 (7)
"4 + 1" = 6.320000171661377 (5)
"3 + 7" = 7.619999885559082 (10)
"6 + 8" = 9.729999542236328 (14)
"9 + 4" = 9.90999984741211 (13)
"8 + 5" = 8.84000015258789 (13)
"4 + 2" = 6.380000114440918 (6)
"7 + 1" = 7.730000019073486 (8)
"0 + 9" = 6.78000020980835 (9)

### Interpretation
Much worse to use RELU for some reason.


## 7.
Same as 1. but without relu in LSTM and longer training

### Results
Epoch 400/400
1s - loss: 0.0134 - val_loss: 0.0251

Some examples:
"2 + 1" = 2.7899999618530273 (3)
"8 + 7" = 15.100000381469727 (15)
"4 + 8" = 12.1899995803833 (12)
"6 + 9" = 15.010000228881836 (15)
"1 + 4" = 4.820000171661377 (5)
"0 + 4" = 3.9600000381469727 (4)
"6 + 8" = 14.079999923706055 (14)
"4 + 1" = 5.079999923706055 (5)
"2 + 5" = 6.909999847412109 (7)
"6 + 7" = 13.130000114440918 (13)

### Interpretation
Close to perfect! This seems to be the best setup.


## 8.
Same as 7., but with operations = '-'. This is much harder! There can be
negative numbers...

### Results
Epoch 80/400
1s - loss: 11.7426 - val_loss: 11.9316

Some examples:
"9 - 3" = 4.0 (6)
"2 - 7" = 0.0 (-5)
"9 - 8" = 1.559999942779541 (1)
"2 - 1" = 0.0 (1)
"2 - 3" = 0.0 (-1)
"5 - 3" = 1.3899999856948853 (2)
"5 - 2" = 0.0 (3)
"7 - 1" = 3.2699999809265137 (6)
"6 - 3" = 3.5899999141693115 (3)
"6 - 0" = 3.3299999237060547 (6)

### Interpretation
Can't do negative results. This is logical, as relu can't even do negative, lol
I could add a separate output neuron that would indicate which sign.



## 9.
Same as 7., but with MAX_NUMBER = 999

### Results
Epoch 103/400
256s - loss: 2.4368 - val_loss: 2.2159

Some examples:
"39 + 33" = 71.98999786376953 (72)
"7 + 87 " = 94.93000030517578 (94)
"9 + 18 " = 26.799999237060547 (27)
"71 + 25" = 96.05999755859375 (96)
"71 + 45" = 116.05000305175781 (116)
"90 + 85" = 174.61000061035156 (175)
"45 + 11" = 57.2599983215332 (56)
"88 + 83" = 172.7899932861328 (171)
"47 + 53" = 100.0999984741211 (100)
"11 + 59" = 70.4800033569336 (70)


## Further Experiments
- Does it scale with the numbers? Train up to 50, then test on 50-60.
  For that, it really needs to understand basic math and decimal system.
- For arbitrary results, we'll have to use the same method as the aXb guy...
  Meaning, we sample the result.
