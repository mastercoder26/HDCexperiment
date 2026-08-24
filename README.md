# HDCbase

A small, beginner-friendly Hyperdimensional Computing (HDC) simulator. It learns
to label tiny IoT records as `normal` or `anomaly` and counts the work performed
by the HDC algorithm.

This is an algorithm experiment, not a hardware emulator and not a real anomaly
detector yet.

## Run it

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python main.py --dimensions 10000 --seed 42
~~~

Try the exact command that caused confusion earlier:

~~~bash
.venv/bin/python main.py --dimensions 30 --seed 42
~~~

The vectors will have 30 entries. `DEFAULT_DIMENSIONS = 10_000` in `core.py` is
used only when you omit `--dimensions`. The path is:

~~~text
command line --dimensions 30
        -> args.dimensions in main.py
        -> HDC(dimensions=30)
        -> self.dimensions in core.py
        -> every new vector has size self.dimensions
~~~

Save the result as JSON:

~~~bash
.venv/bin/python main.py --dimensions 10000 --seed 42 \
    --output baseline-result.json
~~~

Add example analytical costs:

~~~bash
.venv/bin/python main.py --dimensions 1000 --seed 42 \
    --energy-cost 0.1 --latency-cost 0.2
~~~

Those costs are adjustable placeholders, not measurements from real hardware.

## Only three runtime files

~~~text
main.py       example data, command-line settings, train/test run, printing
hdc/core.py   vectors, bind, bundle, permute, similarity, work counters
hdc/model.py  record encoding, prototypes, prediction, accuracy, memory
~~~

`hdc/__init__.py` only exposes the three useful Python names. The `tests/`
directory checks behavior but is not part of the simulator's runtime.

## What one run does

1. Assigns stable random vectors to field names and values.
2. Binds each field name to its value.
3. Bundles the field pairs into one vector for the whole record.
4. Bundles training records into a `normal` and an `anomaly` prototype.
5. Compares each test record to both prototypes.
6. Chooses the label with the higher similarity.
7. Reports accuracy, memory, operation counts, and optional cost estimates.

## Tests

~~~bash
.venv/bin/python -m unittest discover -v
.venv/bin/coverage run --branch --source=hdc -m unittest discover
.venv/bin/coverage report -m
~~~

## Read next

- [Beginner HDC and code guide](docs/HDC_RESEARCH_GUIDE.md)
- [Professor meeting brief](docs/PROFESSOR_MEETING_BRIEF.md)
