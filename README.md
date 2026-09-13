# HDC Baseline

I built this project to understand how Hyperdimensional Computing can be used for a small classification problem. It turns categorical data into long vectors, learns one prototype for each class, and predicts the closest match.

## Why I built this

I cold-emailed more than 100 professors before hearing back from a professor at Texas State University. He was exploring how hyperdimensional computing could be used on Internet of Things devices, such as sensors in agricultural or medical applications. As part of that work, he asked me to create a baseline simulator. This project is one of the baselines I have been working on.

## What is Hyperdimensional Computing (HDC)?

Hyperdimensional Computing is a brain-inspired AI approach based on high-dimensional vectors (hypervectors, typically 10,000 numbers of `-1` and `+1`).

Unlike standard deep neural networks that require heavy backpropagation, thousands of matrix multiplications, and high-power GPUs:
- **Fast one-pass learning**: Patterns are encoded and bundled in a single pass without iterative training.
- **Ultra-lightweight math**: Operations are primarily element-wise multiplication (binding) and vector addition (bundling).
- **Edge & IoT friendly**: Extremely low compute, energy, and memory footprints, making it ideal for microcontrollers and wearable sensors.

## Try the demo

Download the ZIP for your computer from the [latest release](https://github.com/mastercoder26/HDCexperiment/releases/latest), extract it, and open a terminal in that folder. Python is not required.

Windows:

```powershell
.\hdc-baseline.exe --demo
```

macOS or Linux:

```bash
./hdc-baseline --demo
```

Whenever you run it, the program:
1. **Explains what it is doing**: Outlines the dataset, classes, and how hypervectors encode features.
2. **Executes the training and predictions**: Shows an accuracy bar, confidence margins, class breakdown, prediction grid, and individual predictions.
3. **Interprets the results in plain English**: Explains what the accuracy, confidence margin, and hardware efficiency numbers mean for real-world edge deployment.

If macOS blocks the unsigned file, run:

```bash
xattr -d com.apple.quarantine hdc-baseline
```

On Windows, choose **More info**, then **Run anyway** if SmartScreen appears. On Linux, run `chmod +x hdc-baseline` if the file is not executable.

## How it works

Each field and value gets a random vector made of `-1` and `+1` values. The program then:

1. **Binds** each field to its value using element-wise multiplication.
2. **Bundles** the fields into one vector for the full record using addition.
3. **Bundles** training records into an average prototype vector for each class.
4. **Compares** unseen test records with the prototypes using cosine similarity.

The class with the highest similarity score becomes the prediction. The random seed is fixed by default, so the same command gives repeatable results.

## How to interpret the output

When you run the classifier, each section tells you something specific:

| Output Section | What It Tells You |
| --- | --- |
| **Intro & Description** | Summarizes HDC principles, the classes being learned, and the number of examples. |
| **Confidence & Accuracy** | The percentage of test records correctly identified, visual progress bar, and F1 score. |
| **Average Margin** | The difference in similarity between the winning class and the runner-up. Margins above `0.3` indicate high decision confidence. |
| **Breakdown by Type** | Precision, recall, and F1 score for every individual category. |
| **Prediction Grid** | A confusion matrix displaying actual types as rows and guessed types as columns. |
| **Performance & Hardware** | Estimated work units, latency, energy consumption, and memory footprint. |
| **Interpretation** | Plain-language summary explaining what the scores mean and why the efficiency numbers matter for edge hardware. |

The included dataset is intentionally small. Its accuracy shows that the pipeline works, not that it is ready for a real-world classification task. The `example_edge` energy and latency numbers are also estimates, not hardware measurements.

## Other commands

```bash
# Run with default settings (explains HDC, trains, and interprets results)
./hdc-baseline

# Show the installed release version
./hdc-baseline --version

# Change the vector size and random seed
./hdc-baseline --dimensions 5000 --seed 7

# Compare several configurations and save the full results
./hdc-baseline --sweep-dimensions 1000,5000 --sweep-seeds 7,42 --output sweep.json

# Export a compact sweep summary for a spreadsheet or plotting tool
./hdc-baseline --sweep-dimensions 1000,5000 --sweep-seeds 7,42 --csv-output sweep.csv

# Load your own categorical dataset and export prediction rows
./hdc-baseline --dataset dataset.json --output result.json --csv-output predictions.csv
```

Use `./hdc-baseline --help` to see every option. For a normal run, the CSV contains one row per prediction with every class score. For a sweep, it contains one row per dimensions/seed combination with accuracy, macro F1, margin, memory, operation totals, cost estimates, and timing.

## Challenges

The hardest part was understanding how ordinary field/value data could be represented with hypervectors. I also had to make the random vectors repeatable, validate custom datasets, and package NumPy into standalone builds for each operating system.

## What I want to add next

- Test the classifier on a larger dataset
- Compare more dimensions and encoding choices
- Add charts for experiment results
- Build a small desktop interface for people who do not use the terminal

## Run from source

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python main.py --demo
.venv/bin/python -m pytest
```

The main implementation is in `hdc/core.py`, `hdc/model.py`, and `main.py`. GitHub Actions builds the Windows, macOS, and Linux release files with PyInstaller.
