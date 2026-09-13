# HDC Baseline

I built this project to understand how Hyperdimensional Computing can be used for a small classification problem. It turns categorical data into long vectors, learns one prototype for each class, and predicts the closest match.

## Why I built this

I cold-emailed more than 100 professors before hearing back from a professor at Texas State University. He was exploring how hyperdimensional computing could be used on Internet of Things devices, such as sensors in agricultural or medical applications. As part of that work, he asked me to create a baseline simulator. This project is one of the baselines I have been working on.

## What is Hyperdimensional Computing?

Hyperdimensional Computing (HDC) is a brain-inspired way of doing machine learning using long vectors of numbers (hypervectors, usually 10,000 numbers of -1 and +1).

Instead of training deep neural networks with backpropagation and heavy GPU compute:
- It learns in a single pass without iterative training loops.
- It uses simple math like element-wise multiplication and vector addition.
- It takes very little memory and power, which makes it great for small microcontrollers and sensors.

## Try the demo

1. Download the ZIP for your computer from the [latest release](https://github.com/mastercoder26/HDCexperiment/releases/latest).
2. Extract the ZIP file.
3. Open your terminal and change into the folder where you extracted the download. For example on a Mac:

```bash
cd ~/Downloads/hdc-baseline-macos-arm64
```

### Running on macOS
Apple will block the file from running with a warning because it is not signed with an Apple developer certificate. To get rid of that warning, run this in your terminal while inside the extracted folder:

```bash
xattr -cr hdc-baseline
```

Then run the demo:

```bash
./hdc-baseline --demo
```

### Running on Windows
Open PowerShell in the extracted folder and run:

```powershell
.\hdc-baseline.exe --demo
```

If Windows SmartScreen shows a popup, click **More info** and then **Run anyway**.

### Running on Linux
Make sure the file is executable, then run it:

```bash
chmod +x hdc-baseline
./hdc-baseline --demo
```

## What the program does when you run it

When you run `./hdc-baseline` or `./hdc-baseline --demo`, it walks you through three things:

1. **Description**: It explains what HDC is, lists the training examples, and shows the test records.
2. **Execution**: It trains the model in one pass, evaluates the test records, and prints accuracy, confidence margins, and a confusion matrix.
3. **Interpretation**: It gives a plain English breakdown of what the accuracy and confidence margins mean, plus how much memory and simulated energy the run took.

## How it works

Every category name and value gets its own random vector of -1 and +1 values. The program then:

1. Binds each field to its value using multiplication.
2. Bundles the fields together into one vector for the record using addition.
3. Bundles the training records together into an average prototype vector for each class.
4. Compares test records to those class prototypes using cosine similarity.

The class with the highest similarity wins. The random seed is set to 42 by default so you get the same result every time you run it.

## Understanding the output

- **Accuracy & Confidence Bar**: Shows how many test examples it got right.
- **Average Margin**: How much higher the winning class score was compared to the runner-up. Anything over 0.3 means high confidence.
- **Breakdown by Type**: Precision, recall, and F1 score for each class.
- **Prediction Grid**: A confusion matrix where rows are actual classes and columns are what the model guessed.
- **Performance**: Total work units, estimated energy, and runtime.
- **Interpretation**: A quick summary explaining the results and why the low compute footprint matters for edge devices.

The included example dataset is small on purpose so you can trace every step. The energy and latency numbers are analytical estimates, not direct hardware measurements.

## Other commands

```bash
# Run with default settings (explains HDC, trains, and interprets results)
./hdc-baseline

# Show version
./hdc-baseline --version

# Change the vector size and seed
./hdc-baseline --dimensions 5000 --seed 7

# Compare different vector dimensions and seeds
./hdc-baseline --sweep-dimensions 1000,5000 --sweep-seeds 7,42 --output sweep.json

# Save a CSV summary of a sweep
./hdc-baseline --sweep-dimensions 1000,5000 --sweep-seeds 7,42 --csv-output sweep.csv

# Run on your own JSON dataset and export predictions to CSV
./hdc-baseline --dataset dataset.json --output result.json --csv-output predictions.csv
```

Run `./hdc-baseline --help` to see all available flags.

## Challenges

The hardest part was figuring out how normal categorical data maps into hypervectors. I also had to make sure the random vectors stayed reproducible, add dataset validation, and package NumPy into standalone binaries for Windows, Linux, and macOS without needing Python installed.

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

The core code is in `hdc/core.py`, `hdc/model.py`, and `main.py`. GitHub Actions builds the binaries with PyInstaller on every tagged release.
