# Professor Meeting Brief

## What to say first

> I built a small bipolar HDC baseline for categorical IoT records. It creates
> stable hypervectors for fields and values, binds and bundles them into record
> vectors, learns one prototype per class, and predicts using similarity. It also
> counts algorithm work and accepts replaceable energy and latency assumptions.
> It is intentionally an algorithm-level baseline, not a hardware simulator. The
> toy result verifies that the pipeline works; it is not a research accuracy
> claim.

## Live demonstration

~~~bash
.venv/bin/python main.py --dimensions 10000 --seed 42
~~~

Then show that dimensions are actually configurable:

~~~bash
.venv/bin/python main.py --dimensions 30 --seed 42
~~~

Point out that the second command prints `dimensions: 30`. The command-line value
is passed to `HDC(dimensions=30)`, and that object creates 30-position vectors.

Optional JSON report:

~~~bash
.venv/bin/python main.py --dimensions 10000 --seed 42 \
    --output baseline-result.json
~~~

## What is ready

- configurable dimension and repeatable seed;
- bipolar `-1/+1` hypervectors;
- random generation, binding, bundling, permutation, and similarity;
- stable item memory for fields and values;
- categorical record encoding;
- one prototype per class;
- nearest-prototype prediction with scores and margin;
- accuracy and vector-memory reporting;
- operation calls and work-unit reporting;
- replaceable per-operation energy and latency assumptions;
- JSON output and automated tests.

The runtime has only three important files:

~~~text
hdc/core.py   HDC operations and work/cost counters
hdc/model.py  encoder, training, and prediction
main.py       tiny IoT demonstration and command line
~~~

## How one record is processed

For this record:

~~~text
protocol=mqtt, encryption=on, rate=low
~~~

the model does:

~~~text
protocol pair   = bind(field:protocol, value:protocol:mqtt)
encryption pair = bind(field:encryption, value:encryption:on)
rate pair       = bind(field:rate, value:rate:low)

record vector = bundle(protocol pair, encryption pair, rate pair)
~~~

Training bundles normal record vectors into a normal prototype and anomalous
record vectors into an anomaly prototype. Prediction selects the prototype with
the higher similarity.

## What “simulator” means here

The program simulates the HDC algorithm and counts how much vector work it does.
It can estimate costs using:

~~~text
work units * supplied energy or latency cost
~~~

Current default costs are zero because they have not been calibrated to hardware.
This avoids presenting invented numbers as measurements. Once a device or paper
is chosen, its assumptions can be inserted without rewriting the classifier.

## Be honest about limitations

- The built-in dataset is tiny and handcrafted.
- Its accuracy is only a pipeline check.
- The categorical encoder may not match the final research dataset.
- Python/NumPy behavior is not edge-device performance.
- Cost estimates need a named hardware source or measurement.
- There is no proposed novel HDC improvement yet.

## Questions to ask the professor

1. Which exact HDC paper should I reproduce as the baseline?
2. Which task and dataset should we use?
3. Should the representation stay bipolar, or match another paper?
4. Is an algorithm workload model enough for now?
5. Which edge device or published source should provide cost values?
6. Which result matters most: accuracy, memory, latency, energy, or robustness?
7. After reproduction, which module should we try changing first?

## Proposed next milestone

> Once we select one paper and dataset, I will implement its matching encoder,
> run multiple seeds and vector dimensions, save a result table, and compare the
> baseline metrics. If we select hardware assumptions, I will add them as a named
> cost configuration.

Bring the repository, this brief, one saved JSON result, and the selected paper
if your professor has already named it. The most useful meeting outcome is a
precise paper, dataset, metric, and hardware target—not more code layers.
