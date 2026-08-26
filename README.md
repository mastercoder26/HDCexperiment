# Generic HDC Baseline

A Hyperdimensional Computing (HDC) baseline for learning from categorical data.
## Run

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python main.py --dimensions 10000 --seed 42
~~~

## What it demonstrates

- random bipolar hypervectors containing `-1` and `+1`;
- binding, bundling, permutation, and similarity;
- encoding categorical records into hypervectors;
- bundling training examples into one prototype per class;
- choosing the most similar prototype for prediction;
- counting operation calls and vector work;
- optional analytical energy and latency estimates.

The `class_a` and `class_b` records are a generic demonstration of the HDC pipeline.

## Files

~~~text
main.py       example data, settings, training, evaluation, and output
hdc/core.py   HDC operations and simulator counters
hdc/model.py  categorical encoding, prototypes, and prediction
tests/        automated checks
~~~

## Customize it

Change dimensions or the repeatable random seed:

~~~bash
.venv/bin/python main.py --dimensions 5000 --seed 7
~~~

Add placeholder analytical costs:

~~~bash
.venv/bin/python main.py --dimensions 5000 \
    --energy-cost 0.1 --latency-cost 0.2
~~~

Edit `TRAINING_RECORDS` and `TEST_RECORDS` near the top of `main.py` to try
different labels, fields, and categorical values. Python experiments can also
pass different energy or latency values for each operation directly to `HDC`.


## Tests

~~~bash
.venv/bin/python -m unittest discover -v
.venv/bin/coverage run --branch --source=hdc -m unittest discover
.venv/bin/coverage report -m
~~~
