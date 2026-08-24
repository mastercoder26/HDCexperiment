# Beginner Guide to This HDC Project

## 1. The project in one sentence

This program turns a small IoT record into a long list of `-1` and `+1`, learns
one representative list for each class, and predicts the class whose list looks
most similar.

That long list is called a **hypervector**.

You do not need to understand all of HDC research to understand this code. The
whole project is built from five vector operations and one simple classifier.

## 2. Why are the vectors so long?

A normal vector might contain two or three numbers. An HDC vector often contains
thousands. This project defaults to 10,000 dimensions:

~~~text
[-1, +1, +1, -1, ..., +1]
 1   2   3   4       10,000
~~~

Random long vectors are usually close to unrelated. That gives us room to combine
ideas while still comparing them later.

The default is not forced. If you run:

~~~bash
.venv/bin/python main.py --dimensions 30 --seed 42
~~~

the `HDC` object stores `self.dimensions = 30`, so every vector it creates has 30
positions. The number 10,000 is only a fallback for when no command-line value is
provided.

## 3. The five operations

All five live in `hdc/core.py`.

### Random vector

`random_vector()` creates a new list of random `-1` and `+1` values. A token such
as `field:protocol` gets one of these vectors.

The seed makes the randomness repeatable. Seed 42 produces the same sequence on
every run, which is useful for experiments.

### Bind

Binding joins two concepts. This project multiplies matching positions:

~~~text
left:    [+1, -1, +1, -1]
right:   [-1, -1, +1, +1]
bound:   [-1, +1, +1, -1]
~~~

We use this to join a field with its value:

~~~text
bind(vector for "field:protocol", vector for "value:protocol:mqtt")
~~~

Binding is reversible in this bipolar representation. Binding the result with
one input recovers the other because each position is either `-1` or `+1`.

### Bundle

Bundling places several concepts into one vector. At each position, it takes a
majority vote:

~~~text
vector A:  [+1, -1, +1]
vector B:  [+1, +1, -1]
vector C:  [-1, +1, +1]
bundle:    [+1, +1, +1]
~~~

The result is normally somewhat similar to each input. This lets one record
remember several field-value pairs, or one class prototype remember several
training examples.

If a vote is tied, this beginner version chooses `+1`. That rule is simple and
keeps every result bipolar.

### Permute

Permutation rotates the positions:

~~~text
before: [+1, -1, -1, +1]
after:  [+1, +1, -1, -1]
~~~

It can represent order in future sequence experiments. The toy IoT classifier
does not currently need it, but it is included as a baseline HDC operation.

### Similarity

Similarity asks how alike two vectors are. A score near `+1` means very similar,
near `0` means mostly unrelated, and near `-1` means opposite.

The classifier compares a test vector to every class prototype and chooses the
largest score.

## 4. How a dictionary becomes a hypervector

Suppose the input is:

~~~python
{
    "protocol": "mqtt",
    "encryption": "on",
    "rate": "low",
}
~~~

`HDCClassifier.encode()` performs these steps:

1. Get a stable vector for `field:protocol`.
2. Get a stable vector for `value:protocol:mqtt`.
3. Bind those two vectors.
4. Repeat for `encryption` and `rate`.
5. Bundle the three bound vectors.

In compact form:

~~~text
protocol pair   = bind(field:protocol, value:protocol:mqtt)
encryption pair = bind(field:encryption, value:encryption:on)
rate pair       = bind(field:rate, value:rate:low)

record vector = bundle(protocol pair, encryption pair, rate pair)
~~~

Field names are sorted before encoding, so changing dictionary order does not
change the result.

## 5. What item memory means

`item_memory` is just a Python dictionary:

~~~text
token                         -> hypervector
"field:protocol"              -> [-1, +1, ...]
"value:protocol:mqtt"         -> [+1, +1, ...]
"value:encryption:on"         -> [-1, -1, ...]
~~~

The first time the classifier sees a token, it creates a random vector and saves
it. Later it returns the saved vector. Without this memory, `mqtt` would mean
something different every time it appeared.

## 6. How training works

Training records are pairs of a label and a dictionary:

~~~python
("normal", {"protocol": "mqtt", "encryption": "on", "rate": "low"})
~~~

The classifier:

1. Encodes every training record.
2. Groups encoded vectors by label.
3. Bundles each group.
4. Saves the result in `prototypes`.

After training, the data structure looks like:

~~~text
prototypes = {
    "normal":  one bundled hypervector,
    "anomaly": one bundled hypervector,
}
~~~

A prototype is a rough summary of all examples in that class.

## 7. How prediction works

For a new record:

1. Encode it using the same process.
2. Calculate its similarity to the normal prototype.
3. Calculate its similarity to the anomaly prototype.
4. Choose the larger score.

Example:

~~~text
normal similarity  = 0.64
anomaly similarity = 0.10
prediction         = normal
margin             = 0.64 - 0.10 = 0.54
~~~

The margin tells us how far the winning score was ahead. It is not a calibrated
probability or a guarantee of confidence.

## 8. The simulator part

This is an **algorithm-level simulator**. It does not pretend to be a Raspberry
Pi, microcontroller, FPGA, or ASIC.

Every basic operation records:

- how many times it was called;
- how many vector positions it processed, called `work_units`;
- optional estimated energy;
- optional estimated latency.

For example, binding two 30-dimensional vectors counts 30 work units. Binding
two 10,000-dimensional vectors counts 10,000.

The cost formula is deliberately simple:

~~~text
estimated energy = work units * energy cost per work unit
estimated latency = work units * latency cost per work unit
~~~

The values default to zero because we do not yet have a professor-approved paper
or real-device measurement. A zero estimate means **unconfigured**, not free.

Python code can assign a different cost to each operation:

~~~python
hdc = HDC(
    dimensions=10_000,
    energy_costs={"bind": 0.5, "bundle": 0.8},
    latency_costs={"bind": 2.0, "bundle": 3.0},
)
~~~

This is the modular part your professor requested: the HDC operations remain the
same while the assumptions can change.

## 9. Every runtime file

### `hdc/core.py`

Owns the mathematical foundation:

- `DEFAULT_DIMENSIONS` is the fallback vector length;
- `HDC.__init__` stores dimensions, seed, and optional costs;
- `_check_vector` rejects invalid input;
- `_record` updates simulator counters;
- the five public methods implement HDC operations;
- `report` turns counters into a plain dictionary.

The key idea is that dimensions belong to the `HDC` object. There is no separate
`vectors.py` with its own hard-coded dimension anymore.

### `hdc/model.py`

Owns the learning task:

- `item_memory` maps tokens to stable random vectors;
- `prototypes` maps labels to learned class vectors;
- `encode` converts one record;
- `train` builds prototypes;
- `predict` compares one record;
- `evaluate` calculates accuracy;
- `memory_bytes` counts stored NumPy vector bytes.

### `main.py`

Owns the demonstration:

- tiny training and test records;
- command-line arguments;
- creation of `HDC` and `HDCClassifier`;
- training and evaluation;
- readable output and optional JSON saving.

It should contain example choices, while `core.py` and `model.py` contain reusable
logic.

### `hdc/__init__.py`

This tiny file makes three names easy to import:

~~~python
from hdc import DEFAULT_DIMENSIONS, HDC, HDCClassifier
~~~

### `tests/`

Tests are short examples that also protect behavior. They check operations,
encoding, training, prediction, costs, dimensions, and the command-line path.

## 10. What the project proves today

It proves that:

- a configurable HDC pipeline runs from input records to predictions;
- a command-line dimension reaches the vector generator correctly;
- the random seed makes runs repeatable;
- HDC operations keep valid bipolar vectors;
- item memory, encoding, prototypes, and nearest-prototype prediction work;
- operation work and optional costs can be changed and reported;
- results can be saved for later comparison.

It does **not** prove that:

- the toy accuracy will transfer to a real dataset;
- these encodings are best for the professor's target task;
- energy or latency estimates match real hardware;
- HDC beats another machine-learning method.

## 11. Sensible next research steps

Do not add more architecture just to make the repository look advanced. First
ask the professor to choose:

1. A baseline HDC paper to reproduce.
2. A real task and dataset.
3. Binary or bipolar representation.
4. The main metric: accuracy, memory, latency, energy, or robustness.
5. An edge platform or paper for hardware cost values.

Then add features in this order:

1. A loader for the selected dataset.
2. An encoder that matches its numeric, categorical, or sequential inputs.
3. Train/test splitting without leakage.
4. Runs across several seeds and vector dimensions.
5. CSV or JSON result tables.
6. One professor-approved algorithm change at a time.
7. Real hardware calibration only after a target is selected.

That keeps the baseline understandable and gives each later experiment a clear
reason to exist.
