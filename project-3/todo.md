Project 3 TODO
==============

1 Mandatory

*  [X] 1 Train the 3-state model (from class) on parts 0-8 of the training data using training-by-counting.
    Show the obtained model parameters (transition, emission, and start probilities).

*  [X] 2 Redo step 1 with the 4-state model (from class). Recall that for this model, the given annotations does
    not correspond immediately to sequences of hidden states, as there are two states that we interpret as being in
    transmembrane helix (annotation M).

*  [ ] 3 Make a 10-fold experiment using the 3-state model, training-by-counting, and Viterbi decoding for prediction.
    Show the AC compute by compare_tm_pred.py for each fold, and show the mean and variance of the ACs over all 10 folds.

*  [ ] 4 Redo step 3 with the 4-state model.

*  [ ] 5 Redo step 3 and 4 using Posterior decoding. How does the results obtained by posterior decoding compare to the
    results obtained by Viterbi decoding?

*  [ ] 6 Redo steps 3-5 for any other models that you find relevant, e.g. the ones we talked about in class. What is the
    best AC (i.e. best mean over a 10-fold experiment) that you can obtain? How does your best model look like?

2 Optional

*  [ ] 7 Redo steps 3-6 using Viterbi-training instead of training-by-counting (i.e. you ignore the annotations in the
    training data.)

*  [ ] 8 Redo steps 3-6 using EM-training instead of training-by-counting.

*  [ ] 9 If you have implemented the forward and backward algorithms using both scaling and log-transform as explained in
    class, you can make a comparison of their running times, e.g. by measuring the time it takes to make the
    posterior decodings in the 10-fold experiments.

*  [ ] 10 Compare your best prediction method against the [THMMM program] (http://www.cbs.dtu.dk/services/TMHMM/)



