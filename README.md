# Generic HDC Baseline

A compact Hyperdimensional Computing (HDC) research baseline for categorical
data. It supports single runs, experiment sweeps, JSON datasets, timing, and
analytical operation-cost profiles without hardware-specific assumptions.

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
- measured training and inference time;
- overall accuracy, per-class accuracy, and average prediction margin;
- dataset size, class names, model-vector counts, and memory usage;
- optional operation-specific energy and latency estimates;
- dimension and seed sweeps for repeatable comparisons.

The `class_a` and `class_b` records are a generic demonstration of the HDC pipeline.

## Files

~~~text
main.py       example data, settings, training, evaluation, and output
hdc/core.py   HDC operations and simulator counters
hdc/costs.py  named analytical cost profiles
hdc/data.py   validated JSON dataset loading
hdc/model.py  categorical encoding, prototypes, and prediction
tests/        automated checks
~~~

## Customize it

Change dimensions or the repeatable random seed:

~~~bash
.venv/bin/python main.py --dimensions 5000 --seed 7
~~~

Run four configurations and save their reports:

~~~bash
.venv/bin/python main.py \
    --sweep-dimensions 1000,5000 \
    --sweep-seeds 7,42 \
    --output sweep.json
~~~

Use the illustrative operation-specific cost profile:

~~~bash
.venv/bin/python main.py --cost-profile example_edge
~~~

`example_edge` is a demonstration profile, not a hardware measurement. Uniform
cost overrides remain available through `--energy-cost` and `--latency-cost`.

Load another categorical dataset:

~~~bash
.venv/bin/python main.py --dataset dataset.json --output result.json
~~~

The JSON file contains `training` and `test` lists. Each item has a string
`label` and a non-empty `features` object. Without `--dataset`, the example
records near the top of `main.py` are used.
## Tests

~~~bash
.venv/bin/python -m pytest -v
.venv/bin/python -m pytest --cov=hdc --cov-report=term-missing
~~~

The coverage configuration requires at least 90% branch-aware coverage. The
tests remain compatible with Python's built-in `unittest` runner as well.
