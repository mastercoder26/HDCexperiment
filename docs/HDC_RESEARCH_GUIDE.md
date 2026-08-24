# HDC Research Guide and Simulator Plan

This document explains the project from first principles: what Hyperdimensional Computing is, how its core operations work, how data becomes a hypervector, how classification works, what the current repository already implements, and how to turn the current demonstration into a modular research simulator.

The project uses the abbreviation **HDC** for Hyperdimensional Computing. HDC is also closely related to the broader family of methods called **Vector Symbolic Architectures**, or VSAs.

## 1. The research goal

The long-term research area is machine learning on edge and Internet of Things devices. Those devices often have limited memory, power, compute capacity, and communication bandwidth. A useful on-device method should therefore be:

- computationally simple;
- reasonably accurate for its target task;
- inexpensive to store and execute;
- robust when sensor data is noisy;
- adaptable when new observations arrive;
- measurable in terms of accuracy, latency, memory, and energy.

HDC is interesting in this context because it represents information using very wide, low-precision vectors and manipulates those vectors with simple operations such as element-wise multiplication, addition or majority voting, permutation, and similarity search. The simplicity of those operations makes HDC a candidate for lightweight classification and specialized hardware. It does **not** automatically make every Python HDC implementation energy efficient. Energy efficiency depends on the representation, implementation, memory layout, processor, and hardware architecture.

The immediate goal from the professor meeting is narrower:

> Build a correct baseline HDC algorithm simulator whose parts can be changed independently, whose experiments can be repeated, and whose operation costs can later be adjusted to represent different edge-device assumptions.

The first version is therefore an **algorithm simulator with an analytical cost model**, not a cycle-accurate hardware simulator.

## 2. The central idea: computing with hypervectors

### 2.1 Ordinary feature vectors versus hypervectors

An ordinary machine-learning feature vector might contain a small number of directly meaningful values:

~~~text
[temperature, humidity, packet_rate]
[72.4,        0.63,     18.0]
~~~

In HDC, the complete object is represented by a much wider vector:

~~~text
[-1, +1, +1, -1, ..., +1]
~~~

The current project uses **bipolar hypervectors**, meaning that every component is either -1 or +1. A common dimension is 10,000, although the correct dimension is an experimental choice rather than a law. Binary 0/1, real-valued, complex-valued, and other hypervector models also exist. Different models use different definitions of binding, bundling, and similarity.

For this project, keeping the first baseline entirely bipolar is the cleanest choice:

~~~text
Hypervector space: {-1, +1}^D
Default dimension: D = 10,000
Binding: element-wise multiplication
Bundling: element-wise sum followed by a bipolar threshold
Permutation: cyclic shift
Similarity: normalized dot product or cosine similarity
~~~

### 2.2 Why high dimension matters

Suppose A and B are independent random bipolar hypervectors. For each position i, the product AᵢBᵢ is equally likely to be -1 or +1. Their normalized dot product is:

~~~text
similarity(A, B) = (1 / D) × Σ AᵢBᵢ
~~~

Its expected value is 0. Its standard deviation is approximately:

~~~text
1 / √D
~~~

At D = 10,000, that standard deviation is approximately 0.01. Most independently generated vectors are therefore close to zero similarity. They are not exactly geometrically orthogonal, but they are **quasi-orthogonal** for practical use.

This gives HDC a large supply of distinguishable symbols. Random hypervectors can represent concepts such as:

~~~text
protocol
mqtt
packet_rate
high
normal
anomaly
position_1
position_2
~~~

The symbols do not receive meaning from the random values themselves. Meaning comes from how vectors are combined and compared.

### 2.3 Distributed representation

Information is spread across the full vector instead of being stored in one position. No single component means “MQTT” or “normal.” Each component contributes a very small amount to the complete representation.

This distribution has useful consequences:

- Flipping a small number of components changes similarity gradually.
- Related structures can remain similar after being combined.
- The representation can tolerate some noise or approximation.
- Operations can be performed independently across dimensions, creating parallelism opportunities.

It also has limits:

- Bundling too many items can cause the individual items to become difficult to recover.
- A smaller dimension can increase accidental similarity and interference.
- Robustness must be measured for the actual encoder and task; it should not be assumed.

## 3. The four baseline operations

The baseline should support generation, binding, bundling, permutation, and similarity. Generation creates atomic symbols; the remaining operations build and inspect structured representations.

### 3.1 Generation

A random bipolar hypervector is sampled independently:

~~~text
Hᵢ ∈ {-1, +1}, with equal probability
~~~

The generator needs:

- a configurable dimension;
- a seeded random-number generator;
- a known data type, such as NumPy int8;
- validation that the result is one-dimensional and bipolar.

A seed matters because an experiment without a seed cannot be reproduced reliably. The seed belongs in the experiment configuration and in every saved result.

### 3.2 Binding: associate two things

For bipolar vectors, binding is element-wise multiplication:

~~~text
C = A ⊙ B
Cᵢ = AᵢBᵢ
~~~

Binding is useful for role-value or key-value relationships:

~~~text
bind(HV("protocol"), HV("mqtt"))
bind(HV("packet_rate"), HV("high"))
~~~

If A and B are independent random vectors, the bound result is normally dissimilar to both. It represents the relationship rather than either input by itself.

A particularly helpful bipolar property is self-inversion:

~~~text
(A ⊙ B) ⊙ A = B
~~~

This works because every Aᵢ is -1 or +1, so Aᵢ² = 1. Binding with A a second time is therefore also an unbinding operation.

The result may contain noise when unbinding from a record that bundles several relationships. A cleanup or associative memory is then used to select the known item most similar to the noisy recovered vector.

### 3.3 Bundling: place several things in one representation

Bundling forms a superposition:

~~~text
S = A + B + C
bundle(A, B, C) = sign(S)
~~~

The bundled result should remain similar to its constituents. This is useful for representing sets, records, and class prototypes.

For example:

~~~text
device = bundle(
    bind(HV("protocol"), HV("mqtt")),
    bind(HV("encryption"), HV("enabled")),
    bind(HV("packet_rate"), HV("low"))
)
~~~

#### The tie problem

The existing project bundles exactly two vectors using NumPy sign:

~~~text
sign(A + B)
~~~

When Aᵢ and Bᵢ disagree, their sum is zero, and NumPy returns sign(0) = 0. That creates a ternary vector containing -1, 0, and +1 even though the rest of the project describes the vectors as bipolar.

The implemented simulator supports and documents three tie policies:

- **seeded random:** replace every zero with a seeded random -1 or +1;
- **positive:** always replace zero with +1;
- **negative:** always replace zero with -1;
- **odd bundle size:** an algorithm-design option that avoids exact ties when the algorithm permits it;
- **unthresholded accumulator:** a possible future training option that keeps integer sums and thresholds later.

The command-line baseline defaults to positive tie-breaking. This makes repeated encoding of the same even-sized record deterministic without consuming additional random values. Seeded random and negative tie-breaking remain selectable. The current toy records and class prototypes use odd bundle sizes, so their normal path does not need tie resolution.

### 3.4 Permutation: represent order or position

A permutation rearranges vector components without changing their values. A simple baseline uses a cyclic shift:

~~~text
ρₖ(H) = roll(H, k positions)
~~~

Important properties include:

- The permutation is deterministic.
- It is reversible using the inverse shift.
- It preserves similarity when the same permutation is applied to both inputs.
- Different shifts of a random vector are normally dissimilar.

Permutation gives HDC a way to distinguish order:

~~~text
sequence("A", "B", "C") =
    bundle(
        ρ₀(HV("A")),
        ρ₁(HV("B")),
        ρ₂(HV("C"))
    )
~~~

Without roles or permutations, bundling behaves like an unordered set. The records planned for the first IoT baseline use explicit field-role binding, so permutation is not required for every record. It should still exist as a core operation so later sequence or time-series encoders can use it.

### 3.5 Similarity: decide what a vector resembles

The current project uses cosine similarity:

~~~text
cosine(A, B) = (A · B) / (||A|| ||B||)
~~~

For equal-length bipolar vectors, every vector has norm √D, so this simplifies to:

~~~text
similarity(A, B) = (A · B) / D
~~~

Interpretation:

| Similarity | Meaning |
|---:|---|
| Near +1 | Almost identical |
| Near 0 | Unrelated or quasi-orthogonal |
| Near -1 | Nearly opposite |

Similarity should validate that:

- both values are one-dimensional vectors;
- both have the same dimension;
- both contain values allowed by the selected vector model;
- neither has zero norm.

## 4. Memories and data structures

HDC uses several kinds of memory with different responsibilities.

### 4.1 Item memory

Item memory maps atomic symbols to stable random hypervectors:

~~~text
"field:protocol"       → HV₁
"value:mqtt"           → HV₂
"field:packet_rate"    → HV₃
"value:high"           → HV₄
~~~

Requirements:

- Requesting the same symbol twice returns the same vector.
- Different symbols receive independent vectors.
- The mapping is scoped to an experiment and seed.
- Returned arrays should not be accidentally mutated.
- Token names should include namespaces so a field and value cannot collide accidentally.

Possible token conventions:

~~~text
field:protocol
categorical:protocol:mqtt
field:packet_rate
categorical:packet_rate:high
class:normal
class:anomaly
~~~

### 4.2 Associative memory

Associative memory stores known hypervectors, usually one or more prototypes per class:

~~~text
"normal"  → prototype_normal
"anomaly" → prototype_anomaly
~~~

A query compares the encoded input with every stored prototype:

~~~text
prediction = argmax over classes c of similarity(query, prototype_c)
~~~

The query result should contain more than a label:

~~~text
predicted label
winning similarity
similarity for every class
margin between first and second place
~~~

The margin is useful because a prediction that barely wins is less convincing than a clearly separated one. It is not automatically a calibrated probability.

### 4.3 Class accumulators and prototypes

For each class c, training can maintain an integer accumulator:

~~~text
accumulator_c = Σ encode(training sample x where label(x) = c)
~~~

The class prototype is:

~~~text
prototype_c = sign(accumulator_c)
~~~

This is a simple one-pass or few-pass form of training. It does not use gradient descent. Later improvements may include:

- retraining on misclassified examples;
- subtracting an example from an incorrect prototype;
- multiple centroids per class;
- weighted examples;
- online updates;
- prototype pruning or quantization.

Those are later research modules. The first baseline should use one accumulator and one prototype per class.

## 5. Turning raw data into hypervectors

The encoder is usually the most task-specific part of an HDC system. The core operations can be correct while the classifier still performs badly because the encoder fails to preserve useful relationships.

### 5.1 Categorical record encoder

For a first IoT baseline, use a record such as:

~~~text
label: normal
features:
    protocol: mqtt
    encryption: enabled
    packet_rate: low
~~~

For each feature:

~~~text
pair_protocol =
    bind(HV("field:protocol"), HV("categorical:protocol:mqtt"))

pair_encryption =
    bind(HV("field:encryption"), HV("categorical:encryption:enabled"))

pair_rate =
    bind(HV("field:packet_rate"), HV("categorical:packet_rate:low"))
~~~

Then:

~~~text
record = bundle(pair_protocol, pair_encryption, pair_rate)
~~~

Sort feature names before encoding or otherwise guarantee a deterministic order. Bundling is mathematically commutative, but deterministic ordering makes instrumentation, logs, and tests stable.

### 5.2 Numeric features

Continuous sensor values need an explicit encoding strategy. Common starting strategies include:

1. **Quantization:** divide a numeric range into bins and assign a hypervector to each bin.
2. **Level hypervectors:** create an ordered series of vectors where nearby numeric levels are more similar than distant levels.
3. **Random projection:** project a numeric feature vector into high-dimensional space and threshold it.
4. **Thermometer encoding:** progressively activate dimensions as the value increases.

Do not add all four strategies to the first version. Begin with categorical or quantized inputs, then make the numeric encoder a replaceable interface.

### 5.3 Sequences and time-series

For ordered data, permutation can represent time:

~~~text
window =
    bundle(
        ρ₀(encode(sample at t)),
        ρ₁(encode(sample at t-1)),
        ρ₂(encode(sample at t-2))
    )
~~~

An alternative is an n-gram encoder that binds permuted symbols. Sequence encoders should be a later module because they introduce additional design choices: window size, position convention, temporal weighting, and streaming update strategy.

## 6. End-to-end classification

The complete baseline pipeline is:

~~~text
Training data
    |
    v
Encode each record
    |
    v
Group encoded records by class
    |
    v
Bundle each group into a class prototype
    |
    v
Store prototypes in associative memory

Test record
    |
    v
Encode record with the same item memory and encoder
    |
    v
Compare against every class prototype
    |
    v
Choose highest similarity
    |
    v
Report prediction, class scores, accuracy, and simulator metrics
~~~

Two rules are essential:

1. Training and test records must use the same item memory and encoding configuration.
2. Test labels must never affect encoding, prototype creation, or parameter selection.

## 7. What the repository now contains

The original scaffold has been expanded into a working modular baseline:

| File | Implemented role |
|---|---|
| hdc/config.py | Immutable dimension, seed, and tie-policy configuration |
| hdc/vectors.py | Configurable seeded bipolar vector generation |
| hdc/ops.py | Validated bind, multi-vector bundle, permute, and similarity operations |
| hdc/costs.py | Named, replaceable analytical energy and latency profiles |
| hdc/simulator.py | Instrumented operation wrapper and immutable report snapshots |
| hdc/memory.py | Defensive item memory and nearest-match associative memory |
| hdc/encoding.py | Deterministic categorical role-value record encoder |
| hdc/experiment.py | Prototype classifier, toy IoT dataset, evaluation, and structured results |
| main.py | Command-line experiment runner and JSON export |
| tests/ | Unit and end-to-end tests for the complete baseline |

### 7.1 Current readiness

The algorithm-level version 0.1 baseline is implemented. It now provides:

- deterministic seeded runs;
- configurable dimensions;
- validated bipolar operations;
- tie-safe multi-vector bundling;
- configurable positive, negative, or seeded-random tie handling;
- cyclic permutation;
- item and associative memories;
- categorical field-value encoding;
- one-prototype-per-class training;
- nearest-prototype inference;
- accuracy, similarities, margins, memory, work units, and host runtime;
- named analytical cost profiles;
- JSON output;
- automated unit and command-line integration tests.

The remaining research work is different from missing simulator infrastructure. It includes selecting the professor’s baseline paper and real dataset, implementing the matching task-specific encoder, sourcing target-device cost coefficients, running multi-seed experiments, and comparing proposed algorithm changes.

## 8. Simulator architecture

The simulator should be composed of small modules:

~~~text
ExperimentRunner
├── ExperimentConfig
├── RecordEncoder
│   ├── ItemMemory
│   └── HDCSimulator
├── PrototypeClassifier
│   ├── AssociativeMemory
│   └── HDCSimulator
└── SimulationReport
    ├── Algorithm metrics
    ├── Measured host runtime
    └── Modeled hardware costs
~~~

### 8.1 Configuration

The experiment configuration should be immutable after a run begins.

~~~text
HDCConfig
    dimensions: integer
    seed: integer
    vector_type: "bipolar"
    tie_breaker: "positive", "negative", or "random"

ExperimentConfig
    name: string
    dataset_name: string
    train_fraction or split identifier
    hdc: HDCConfig
    cost_profile_name: string
~~~

Validation should reject:

- nonpositive dimensions;
- unsupported vector types;
- unsupported tie policies;
- invalid seeds;
- malformed cost parameters.

### 8.2 Pure operation layer

The mathematical operation functions should:

- accept arrays and explicit parameters;
- return new arrays rather than mutate inputs;
- validate shape and representation;
- avoid hidden global state;
- contain no file I/O;
- contain no knowledge of datasets or classes.

This layer makes the math independently testable.

### 8.3 Simulator layer

The simulator wraps each operation:

~~~text
start timer
run pure operation
stop timer
calculate abstract work units
look up modeled operation cost
update aggregate statistics
return result
~~~

The simulator should own the seeded random generator so that vector generation and random tie-breaking follow one reproducible sequence.

### 8.4 Cost model

The professor asked for values such as energy cost and latency to be changeable. Those values should live in a separate profile:

~~~text
OperationCost
    fixed_energy_pj
    energy_pj_per_work_unit
    fixed_latency_ns
    latency_ns_per_work_unit
~~~

For an operation:

~~~text
modeled energy =
    fixed energy per call
    + work units × energy per work unit

modeled latency =
    fixed latency per call
    + work units × latency per work unit
~~~

Suggested abstract work-unit definitions:

| Operation | Work units |
|---|---:|
| Random generation | D |
| Bind two vectors | D |
| Bundle N vectors | (N - 1) × D |
| Permute one vector | D |
| Compare two vectors | D |

These work units are an abstraction. Real latency does not necessarily scale as their simple sum because hardware may execute dimensions in parallel, incur memory-transfer costs, or fuse operations. The value of this model is that the assumptions are explicit and replaceable.

Keep three categories separate:

1. **Algorithmic work:** calls, dimensions, and work units.
2. **Measured host runtime:** elapsed time on the current Python and NumPy machine.
3. **Modeled device cost:** latency and energy calculated from a named hardware or paper-derived profile.

Never label measured Mac or laptop NumPy time as edge-device latency. Never label an arbitrary coefficient as measured energy.

### 8.5 Metrics

Per-operation statistics:

~~~text
OperationStats
    calls
    work_units
    measured_runtime_ns
    modeled_energy_pj
    modeled_latency_ns
~~~

Experiment statistics:

~~~text
accuracy
correct_predictions
total_predictions
per_class_accuracy
confusion_matrix
average_winning_similarity
average_prediction_margin
training_runtime
inference_runtime
prototype_memory_bytes
item_memory_bytes
operation totals
~~~

The first baseline needs overall accuracy, predictions, class similarities, operation totals, and runtime. Per-class metrics and a confusion matrix become important as soon as the dataset is larger than a toy example.

### 8.6 Result format

Save every run as JSON or CSV. A JSON result could conceptually contain:

~~~json
{
  "experiment": "iot_categorical_baseline",
  "dataset": "toy_iot_v1",
  "config": {
    "dimensions": 10000,
    "seed": 42,
    "vector_type": "bipolar",
    "tie_breaker": "positive"
  },
  "results": {
    "accuracy": 0.0,
    "predictions": []
  },
  "metrics": {
    "operations": {},
    "totals": {}
  },
  "cost_profile": {
    "name": "unconfigured"
  }
}
~~~

Use null or “unconfigured” rather than a fabricated energy value.

For reproducibility, eventually add:

- timestamp;
- Python and NumPy versions;
- operating system;
- repository commit identifier;
- dataset version or hash;
- training/test split identifier.

## 9. Implemented baseline version 0.1

The smallest credible baseline is not a hardware simulator and does not need a large public dataset. It should prove that the architecture works end to end.

### 9.1 Baseline features

Version 0.1 now contains:

- configurable dimension;
- seeded random generation;
- binding;
- multi-input bipolar bundling with a documented tie policy;
- cyclic permutation;
- cosine similarity;
- item memory;
- categorical record encoder;
- one prototype per class;
- nearest-prototype classification;
- operation counters;
- replaceable cost-profile data structure;
- JSON result export;
- mathematical and integration tests.

### 9.2 Toy IoT task

Use a deliberately small normal-versus-anomaly dataset:

~~~text
Normal examples:
    protocol=mqtt, encryption=enabled, packet_rate=low
    protocol=coap, encryption=enabled, packet_rate=low

Anomaly examples:
    protocol=telnet, encryption=disabled, packet_rate=high
    protocol=ftp, encryption=disabled, packet_rate=high
~~~

Test examples should differ slightly from the training records so the experiment is not only memorization. For example, a test record may share two of three features with one class and contain one unseen or changed feature.

The point of this dataset is to validate mechanics. High accuracy on four handcrafted records is not evidence that the method works on real IoT traffic.

### 9.3 What the baseline prints and saves

The command-line program reports:

~~~text
Experiment name
Dimension and seed
Training and test counts
True label
Predicted label
Similarity to each class
Accuracy
Operation count by type
Measured training and inference runtime
Modeled energy and latency, only if a sourced profile is configured
~~~

### 9.4 Definition of done

The current baseline satisfies these conditions:

- running twice with the same seed produces identical algorithmic results;
- every stored hypervector remains bipolar;
- the complete toy experiment runs through training and prediction;
- results include configuration and operation statistics;
- cost coefficients can be replaced without changing HDC math;
- the tests pass;
- the README describes exactly what is measured and what is modeled.

## 10. Implemented test coverage

The repository contains 39 automated tests covering the following behaviors. A branch-aware coverage run currently reports 94% across the hdc package.

### 10.1 Vector and operation tests

- A fixed seed produces a fixed vector.
- Different seeds normally produce different vectors.
- Generated vectors contain only -1 and +1.
- Generated vectors have the configured dimension.
- Binding returns a bipolar vector with the same dimension.
- bind(bind(A, B), A) exactly recovers B.
- Bundling never produces zeros.
- A bundle of several vectors is positively similar to its constituents.
- Permuting by k and then by -k restores the original vector.
- Similarity of a vector with itself is 1.
- Similarity of a vector with its negative is -1.
- Similarity of independent large random vectors is close to 0.
- Mismatched dimensions and non-bipolar inputs are rejected.

### 10.2 Memory and encoding tests

- The same token always returns the same item-memory vector.
- Different tokens receive different vectors.
- Encoding a record is independent of dictionary insertion order.
- Unbinding a simple role-value pair retrieves a vector closest to the correct value.
- Querying associative memory returns the nearest stored prototype.
- Predicting before classifier training produces a clear error.

### 10.3 Simulator tests

- Each wrapper increments the correct operation counter.
- Work units scale with dimension.
- Bundling work scales with the number of vectors.
- Modeled costs match a small hand-calculated example.
- Resetting metrics does not silently change configuration.
- A report can be serialized to JSON.
- Measured host time and modeled device time are separate fields.

### 10.4 Experiment tests

- A complete training and inference run succeeds.
- The same configuration produces the same predictions.
- Test records are not used to build class prototypes.
- Accuracy is calculated correctly.
- The report includes the full configuration.

## 11. Completed implementation sequence

### Phase 1: make the mathematical core reliable

Status: completed.

Estimated effort: 1–2 focused hours.

1. Pass dimension and random generator explicitly.
2. Add validation helpers.
3. Correct bundling ties.
4. Extend bundling to multiple vectors.
5. Add permutation.
6. Write mathematical property tests.

Deliverable: a deterministic and tested operation library.

### Phase 2: add symbolic data structures

Status: completed.

Estimated effort: 1–2 focused hours.

1. Add item memory.
2. Add associative memory.
3. Add a categorical record encoder.
4. Demonstrate key-value binding and cleanup lookup.

Deliverable: structured records can be encoded and queried.

### Phase 3: add the classifier and experiment

Status: completed.

Estimated effort: 2–3 focused hours.

1. Define labeled sample data.
2. Build class accumulators.
3. Threshold prototypes.
4. Predict by nearest similarity.
5. Calculate accuracy and prediction margins.

Deliverable: one end-to-end IoT classification baseline.

### Phase 4: add instrumentation

Status: completed.

Estimated effort: 1–2 focused hours.

1. Wrap operations in the simulator.
2. Record calls, work units, and host runtime.
3. Add replaceable cost profiles.
4. Export a structured run report.

Deliverable: experiments can compare algorithms and cost assumptions.

### Phase 5: document and verify

Status: completed for the toy baseline; repeat for the professor-selected dataset.

Estimated effort: 1 hour.

1. Run all tests.
2. Run the baseline with at least two dimensions.
3. Save results.
4. Document limitations.
5. Prepare a small result table for the next meeting.

## 12. Experiments after version 0.1

### 12.1 Dimension sweep

Run the same seeded dataset across:

~~~text
D = 1,000
D = 2,000
D = 5,000
D = 10,000
~~~

Measure:

- accuracy;
- average prediction margin;
- runtime;
- memory;
- work units;
- modeled cost.

Do not assume larger is always better. Recent research has explicitly investigated situations where much smaller hypervectors retain competitive accuracy, so dimension should be treated as a design-space variable.

### 12.2 Noise robustness

Flip a controlled percentage of query components:

~~~text
0%
1%
5%
10%
20%
~~~

Plot accuracy and similarity margin against noise. Repeat across multiple random seeds and report mean and variation.

### 12.3 Encoder comparison

Keep the classifier and cost model fixed while changing only the encoder:

~~~text
categorical binding-and-bundling
quantized numeric levels
random projection
sequence or n-gram encoding
~~~

This modular comparison matches the professor’s idea of replacing one component and observing how the result changes.

### 12.4 Cost-profile comparison

Keep the algorithmic trace fixed and calculate cost under multiple named profiles:

~~~text
unconfigured analytical profile
paper-derived CPU profile
paper-derived FPGA profile
paper-derived in-memory profile
measured target-device profile
~~~

Each profile must cite its source and state whether costs are per component, per operation, or measured for a complete kernel.

### 12.5 Real dataset

After the professor selects the research direction, replace the toy dataset with an appropriate task:

- network-device or traffic classification;
- human activity recognition;
- wearable sensor classification;
- anomaly detection;
- keyword or language recognition;
- another dataset used by the selected paper.

The paper and dataset should determine the encoder. Do not choose an encoder only because it is easy to implement.

## 13. Common mistakes to avoid

### 13.1 Calling an operation-only demo a simulator

The original scaffold only demonstrated operations. The current version adds configuration, an experiment, metrics, cost profiles, and repeatability. Future versions should keep those qualities as new modules are added.

### 13.2 Mixing vector models

Binary 0/1 and bipolar -1/+1 representations use related but different operators. For example:

| Concept | Binary baseline | Bipolar baseline |
|---|---|---|
| Bind | XOR | Multiplication |
| Bundle | Majority vote | Sum then sign |
| Similarity | Hamming-based | Dot or cosine |

Pick one model for the baseline and validate it consistently.

### 13.3 Allowing zero-valued “bipolar” vectors

NumPy sign can introduce zero on ties. Resolve ties explicitly or retain a separate accumulator representation.

### 13.4 Using global randomness

Global random calls make experiments difficult to reproduce. Give the simulator its own seeded generator.

### 13.5 Mutating vectors held by memory

If a caller changes an array stored in item or associative memory, later results can change silently. Treat stored vectors as immutable or return copies.

### 13.6 Reporting toy accuracy as research evidence

A toy dataset proves the pipeline works. It does not establish real-world accuracy or generalization.

### 13.7 Reporting arbitrary energy values

An energy number without a device, measurement method, or cited model is not meaningful. Label analytical values clearly.

### 13.8 Confusing host runtime with modeled latency

Python overhead, NumPy implementation, CPU caches, threading, and the host machine all affect measured runtime. Keep it separate from target-device estimates.

### 13.9 Comparing methods under different conditions

When comparing modules, hold the following constant unless it is the variable being tested:

- dataset split;
- seed set;
- dimension;
- vector representation;
- encoder;
- cost profile;
- hardware and software environment.

### 13.10 Optimizing before establishing correctness

Do not add GPU, FPGA, bit packing, multiprocessing, or a GUI until the operation properties and end-to-end baseline are verified.

## 14. Questions for the professor

Ask these before committing to the next major implementation:

1. Which exact paper should define the baseline algorithm?
2. Which dataset or application should be the first target?
3. Should the baseline use bipolar, binary, or another hypervector model?
4. Does “simulator” mean an algorithm-level experiment runner, an analytical hardware model, or eventually both?
5. Which operations need energy and latency estimates?
6. What hardware platform or publication should supply the cost coefficients?
7. Which result matters most initially: accuracy, energy, latency, memory, robustness, or a tradeoff among them?
8. Should class prototypes use one-pass bundling, retraining, or the method from the shared paper?
9. Which module does the professor believe is the best research target for modification?
10. What preliminary result would he consider sufficient for the next milestone?

## 15. What success looks like

A useful first milestone is:

> Given a configuration, a small labeled dataset, and a replaceable cost profile, the program deterministically encodes samples into bipolar hypervectors, trains class prototypes, predicts using associative similarity, and produces a report containing accuracy, operation counts, runtime, and clearly labeled modeled costs.

That milestone creates a stable platform for research. New encoders, retraining rules, vector dimensions, bundling methods, and hardware-cost profiles can then be changed one at a time.

## 16. Glossary

| Term | Meaning in this project |
|---|---|
| HDC | Hyperdimensional Computing |
| VSA | Vector Symbolic Architecture |
| Hypervector | A wide distributed vector representing a symbol or structure |
| Bipolar | Components are restricted to -1 and +1 |
| Dimension, D | Number of components in one hypervector |
| Item memory | Mapping from atomic symbols to stable hypervectors |
| Binding | Operation that associates roles and values |
| Bundling | Operation that superposes several vectors |
| Permutation | Reversible rearrangement used for order or position |
| Similarity | Numerical measure of how closely vectors match |
| Associative memory | Store that retrieves the nearest known vector or prototype |
| Prototype | Vector representing a class |
| Cleanup memory | Associative lookup that maps a noisy vector to the nearest known item |
| Encoder | Task-specific conversion from raw data to a hypervector |
| Work unit | Abstract count used by the analytical cost model |
| Host runtime | Measured execution time on the machine running Python |
| Modeled latency | Estimated time under an explicit cost profile |

## 17. References and further reading

1. Pentti Kanerva, “Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors,” 2009. [Manuscript PDF](https://www.rctn.org/pkanerva/papers/kanerva09-hyperdimensional.pdf)
2. Lulu Ge and Keshab K. Parhi, “Classification using Hyperdimensional Computing: A Review,” 2020. [arXiv](https://arxiv.org/abs/2004.11204)
3. Manuel Schmuck, Luca Benini, and Abbas Rahimi, “Hardware Optimizations of Dense Binary Hyperdimensional Computing,” 2018. [arXiv](https://arxiv.org/abs/1807.08583)
4. Mike Heddes et al., “Torchhd: An Open Source Python Library to Support Research on Hyperdimensional Computing and Vector Symbolic Architectures,” 2022. [arXiv](https://arxiv.org/abs/2205.09208)
5. TorchHD project, “Getting Started.” [GitHub documentation](https://github.com/hyperdimensional-computing/torchhd/blob/main/docs/getting_started.rst)
6. Zhanglu Yan et al., “Efficient Hyperdimensional Computing,” 2023. [arXiv](https://arxiv.org/abs/2301.10902)
7. William Andrew Simon et al., “HDTorch: Accelerating Hyperdimensional Computing with GP-GPUs for Design Space Exploration,” 2022. [arXiv](https://arxiv.org/abs/2206.04746)

The references describe multiple HDC models. Their operators and results should not be combined without checking that their vector representation, encoder, dataset, and evaluation assumptions match.
