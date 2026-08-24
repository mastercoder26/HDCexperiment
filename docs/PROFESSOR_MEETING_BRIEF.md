# Professor Meeting Brief: HDC Baseline

Use this as a short discussion document for the meeting. It separates the implemented toy baseline from the real-dataset and hardware-calibration work that still needs research decisions.

## 1. One-minute explanation

> I expanded the original operation demo into a modular bipolar HDC baseline. It now has deterministic configuration, validated binding and bundling, permutation, item and associative memory, categorical IoT encoding, prototype classification, operation accounting, JSON reports, and replaceable analytical energy and latency profiles. On the four-record toy test with dimension 10,000 and seed 42, it classified all four records correctly. That result validates the pipeline, not real-world accuracy. I now want to confirm the target paper, dataset, representation, and hardware-cost source before adapting the baseline to the actual research task.

## 2. Current status

Currently implemented:

- configurable bipolar hypervectors;
- seeded, repeatable random generation;
- binding through element-wise multiplication;
- tie-safe multi-vector bundling;
- positive, negative, and seeded-random tie policies;
- cyclic permutation;
- cosine similarity;
- item memory;
- associative memory;
- role-value categorical encoding;
- one-prototype-per-class training;
- nearest-prototype prediction;
- accuracy, score margin, memory, runtime, and work-unit reporting;
- replaceable per-operation cost profiles;
- readable command-line output and JSON export;
- 39 automated tests.

The original bundling flaw is fixed. Zero-valued ties are now handled explicitly, and stored hypervectors remain bipolar.

Still to research or implement:

- the professor-selected paper and real dataset;
- the task-specific numeric, signal, or sequence encoder;
- paper-derived or measured hardware cost coefficients;
- multi-seed, dimension, and noise experiments on real data;
- any proposed improvement beyond the baseline algorithm.

## 3. Working demonstration for tomorrow

From the repository root:

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python main.py --dimensions 10000 --seed 42
~~~

To save the complete structured report:

~~~bash
.venv/bin/python main.py \
    --dimensions 10000 \
    --seed 42 \
    --output baseline-result.json
~~~

The demonstration:

1. Creates stable item-memory vectors for categorical roles and values.
2. Binds each field role to its value.
3. Bundles each record into one bipolar hypervector.
4. Bundles training records into normal and anomaly prototypes.
5. Encodes four test records, including unseen protocol values.
6. Compares each query with both prototypes.
7. Prints labels, similarities, prediction margins, operation counts, work units, runtime, and cost-profile status.

## 4. Implemented baseline version 0.1

The first baseline is an **algorithm simulator with analytical costs**, not a cycle-accurate hardware simulator.

~~~text
Categorical IoT record
        |
        v
Role-value HDC encoder
        |
        v
Bipolar record hypervector
        |
        v
One prototype per class
        |
        v
Cosine similarity
        |
        v
Prediction + operation report
~~~

Features:

| Area | Baseline choice |
|---|---|
| Representation | Bipolar -1/+1 |
| Dimension | Configurable; start at 10,000 |
| Binding | Element-wise multiplication |
| Bundling | Multi-vector sum with documented tie handling |
| Permutation | Cyclic shift |
| Similarity | Cosine or normalized dot product |
| Encoding | Bind field roles to categorical values, then bundle |
| Training | Bundle encoded records into one prototype per class |
| Inference | Nearest-prototype associative lookup |
| Metrics | Accuracy, similarity margin, calls, work units, runtime |
| Cost | Replaceable per-operation energy/latency profile |
| Reproducibility | Seed and configuration saved with every run |

## 5. Worked example

Input record:

~~~text
protocol = mqtt
encryption = enabled
packet_rate = low
~~~

Encoding:

~~~text
protocol_pair =
    bind(HV("field:protocol"), HV("value:mqtt"))

encryption_pair =
    bind(HV("field:encryption"), HV("value:enabled"))

rate_pair =
    bind(HV("field:packet_rate"), HV("value:low"))

record =
    bundle(protocol_pair, encryption_pair, rate_pair)
~~~

Training:

~~~text
normal_prototype =
    bundle(all encoded normal training records)

anomaly_prototype =
    bundle(all encoded anomaly training records)
~~~

Prediction:

~~~text
normal_score = similarity(query, normal_prototype)
anomaly_score = similarity(query, anomaly_prototype)

prediction = label with the larger score
~~~

This example proves the end-to-end mechanism. A real dataset is still needed before drawing conclusions about accuracy or edge-device performance.

## 6. What the simulator reports

Algorithm results:

- predicted and true label;
- similarity to each class;
- prediction margin;
- overall and per-class accuracy.

Algorithmic workload:

- random-vector generation calls;
- bind calls;
- bundle work;
- permutation calls;
- similarity calls;
- dimensions processed.

Performance:

- measured Python/NumPy training runtime;
- measured Python/NumPy inference runtime;
- memory used by item and class memories;
- modeled latency and energy only when a sourced cost profile exists.

## 7. Decisions requested from the professor

The most important outcome of the meeting is agreement on these points:

1. **Baseline paper:** Which paper should be reproduced first?
2. **Task:** What application or dataset should drive the encoder?
3. **Representation:** Bipolar, binary, or another HDC model?
4. **Simulation level:** Algorithm-level only now, or an analytical hardware model as well?
5. **Cost source:** Which hardware platform or paper provides energy and latency values?
6. **Primary metric:** Accuracy, latency, energy, memory, robustness, or a combined tradeoff?
7. **Research target:** Which module should be modified after the baseline works?

## 8. Suggested next milestone

Propose the following:

> The modular toy baseline is complete. For the next milestone, I will adapt it to one agreed paper and dataset, implement the matching encoder, run multiple seeds and dimensions, and report accuracy, memory, and runtime. If we agree on a hardware source, I will add those coefficients as a named analytical profile rather than presenting arbitrary energy values.

Expected artifacts:

- real dataset adapter and task-specific encoder;
- reproduced baseline configuration from the selected paper;
- multi-seed result table;
- JSON or CSV results;
- a comparison across dimensions;
- one proposed module change to test next.

## 9. Verified toy-baseline results

These are single seed-42 runs on the handcrafted four-record test set. Instrumented runtime is the sum of timed NumPy HDC operations on the development machine, not full wall-clock time or edge-device latency.

| Dimension | Accuracy | Avg. margin | Instrumented op runtime | NumPy item + prototype memory | Modeled energy |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 4/4 | 0.634 | 0.627 ms | 16,000 bytes | Unconfigured |
| 10,000 | 4/4 | 0.689 | 1.497 ms | 160,000 bytes | Unconfigured |

Across seeds 0 through 9, the toy test remained 4/4 at dimensions 1,000, 5,000, and 10,000. This only shows that the intentionally simple toy classes are separable.

## 10. What to bring tomorrow

Bring or have open:

1. The repository and its working main.py HDC IoT demonstration.
2. This meeting brief.
3. The full HDC research guide in docs/HDC_RESEARCH_GUIDE.md.
4. The paper or papers the professor previously shared.
5. A notebook or notes page containing the seven decisions above.
6. One saved JSON report from the command in section 3.

If there is time tonight, understand these three equations well enough to explain them:

~~~text
Binding:
C = A ⊙ B

Bundling:
S = sign(Σ Hⱼ)

Prediction:
ŷ = argmax over classes c of similarity(encode(x), prototype_c)
~~~

## 11. What not to claim yet

Do not claim:

- that energy has been measured;
- that the Python implementation represents hardware latency;
- that the toy dataset proves real IoT accuracy;
- that 10,000 is automatically the optimal dimension;
- that a specific encoder is appropriate before the dataset is selected.

It is credible to say:

- the operation-only scaffold has become a working algorithm simulator;
- the original bundling correctness issue is fixed;
- the baseline is deterministic for a fixed configuration;
- modules and cost assumptions are replaceable;
- 39 tests pass with 94% branch-aware coverage across the hdc package;
- the toy baseline produces repeatable preliminary mechanics and accounting results.

## 12. Meeting checklist

- [ ] Run the IoT baseline from the command line.
- [ ] Show one prediction and explain its two similarity scores.
- [ ] Explain how the zero-tie bundling issue was fixed.
- [ ] Show the implemented architecture and JSON report.
- [ ] Walk through the categorical IoT encoding example.
- [ ] State clearly that the 4/4 result is from a toy dataset.
- [ ] Confirm the baseline paper.
- [ ] Confirm the dataset.
- [ ] Confirm the vector representation.
- [ ] Confirm what “simulator” means for this project.
- [ ] Confirm the energy/latency source.
- [ ] Agree on the primary evaluation metric.
- [ ] Agree on the next delivery date and expected artifact.
