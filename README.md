# lstm-math

Train a Seq2Seq network to predict the result of math equations on the
character level.
Configuration through global variables because I'm lazy.

Written by Max Schumacher (@cpury) in Summer 2017.


## Preview

Will learn how to complete simple math formulas. E.g. during training, you will
get samples like this:

```
Examples:
  67 + 38 =  108   (expected:  105)
  15 + 49 =   69   (expected:   64)
  84 - 91 =   -5   (expected:   -7)
  71 + 53 =  123   (expected:  124)
  72 -  1 =   75   (expected:   71)
```

And ideally, after an half an hour CPU-time or so, it will learn (almost)
perfectly to get the right results :)


## Details

This uses a Seq2Seq model based on LSTMs in Keras. Depending on the complexity
of equations you choose, it will train on ca. 25% of the complete equation
space and validate on another, different 25%. So all the equations you see
in the example above have not been seen by the network before.


## Running it yourself

Please note that I wrote this for Python 3.6+. It will probably work with 2.7+,
but I don't offer any guarantees on that.

0. Set up a virtual env: `virtualenv venv; source venv/bin/activate`
1. Install the requirements: `pip install -r requirements.txt`
2. Optional: Open `main.py` and play around with the global config values.
3. `python main.py`

You can cancel the training any time with `ctrl+c`. It will always output some
examples at the end and store the model in `model.h5`, even if you cancel.

Actually, since it's not stopping when overfitting has begun, it might actually
be good to hit `ctr+c` once you're happy with the results.


## Playing with the config

All the config values at the top of `main.py` should be fairly
self-explanatory. You could e.g.

* Change the maximum possible number to a higher or lower number
* Work with negative numbers by setting `MIN_NUMBER` to something negative
* Add multiplication by adding `*` to `OPERATIONS`
* Have more operations per equation by increasing `N_OPERATIONS`
* etc.


## Feedback

Feel free to submit issues if you find bugs or room for improvements.


Thanks!

Max
