# HDCbase

A modular bipolar Hyperdimensional Computing simulator for reproducible algorithm experiments and replaceable analytical edge-device cost assumptions.

The included baseline encodes categorical IoT records, trains one normal and one anomaly prototype, classifies four toy test records, and reports similarities, prediction margins, memory, operation counts, abstract work units, measured NumPy runtime, and modeled cost.

This is an algorithm-level research baseline. It is not a cycle-accurate hardware simulator, and its toy accuracy is not evidence of real-world IoT performance.

## Quick start

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python main.py --dimensions 10000 --seed 42
~~~

Save the full JSON report:

~~~bash
.venv/bin/python main.py \
    --dimensions 10000 \
    --seed 42 \
    --output baseline-result.json
~~~

Run with example analytical coefficients:

~~~bash
.venv/bin/python main.py \
    --dimensions 1000 \
    --seed 42 \
    --energy-pj-per-work-unit 0.1 \
    --latency-ns-per-work-unit 0.2
~~~

Those command-line coefficients are examples, not measured hardware values. Research reports should use a named profile derived from a paper or target-device measurement.

## What is implemented

- Configurable hypervector dimension and seed.
- Bipolar -1/+1 representation.
- Binding by element-wise multiplication.
- Tie-safe multi-vector bundling.
- Positive, negative, and seeded-random tie policies.
- Cyclic permutation.
- Normalized dot-product similarity.
- Input and dimension validation.
- Stable item memory.
- Nearest-match associative memory.
- Categorical role-value record encoding.
- One-prototype-per-class classification.
- Prediction scores and margins.
- Per-operation calls, work units, and host runtime.
- Replaceable operation-specific energy and latency profiles.
- JSON-compatible experiment reports.
- Unit and command-line integration tests.

## Project structure

~~~text
main.py                 command-line baseline
hdc/
  config.py             immutable HDC configuration
  vectors.py            bipolar vector generation
  ops.py                bind, bundle, permute, similarity
  costs.py              analytical cost profiles
  simulator.py          operation instrumentation and reports
  memory.py             item and associative memories
  encoding.py           categorical role-value encoder
  experiment.py         prototype classifier and toy IoT run
tests/                  unit and integration tests
docs/
  HDC_RESEARCH_GUIDE.md
  PROFESSOR_MEETING_BRIEF.md
~~~

## Test and coverage

~~~bash
.venv/bin/python -m unittest discover -v
.venv/bin/coverage erase
.venv/bin/coverage run --branch -m unittest discover
.venv/bin/coverage report -m
~~~

The current suite contains 39 passing tests and reports 94% branch-aware coverage across the hdc package.

## Research notes

- [HDC research guide and simulator plan](docs/HDC_RESEARCH_GUIDE.md)
- [Professor meeting brief](docs/PROFESSOR_MEETING_BRIEF.md)

## Important interpretation boundaries

- Measured runtime is Python/NumPy time on the host machine.
- Modeled latency and energy come only from the configured analytical profile.
- An unconfigured profile correctly reports zero modeled cost.
- NumPy stores each current bipolar component as int8. Packed binary hardware can have a different memory cost.
- The built-in dataset is deliberately tiny and only validates the end-to-end mechanics.
- A real research result requires an agreed paper, dataset, encoder, multi-seed evaluation, and sourced hardware assumptions.
