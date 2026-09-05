# HDC Baseline

I built this project to understand how Hyperdimensional Computing can be used for a small classification problem. It turns categorical data into long vectors, learns one prototype for each class, and predicts the closest match.

## Why I built this

I cold-emailed more than 100 professors before hearing back from a professor at Texas State University. He was exploring how hyperdimensional computing could be used on Internet of Things devices, such as sensors in agricultural or medical applications. As part of that work, he asked me to create a baseline simulator. This project is one of the baselines I have been working on.

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

The demo starts by showing its goal, all six training examples, and the four held-out test records in plain language. It then prints labeled predictions, an accuracy bar, similarity margins, dataset and model details, operation counts, and run time. Experiment sweeps list every run and highlight the best-performing configuration, and saved JSON files are confirmed in the terminal.

If macOS blocks the unsigned file, run:

```bash
xattr -d com.apple.quarantine hdc-baseline
```

On Windows, choose **More info**, then **Run anyway** if SmartScreen appears. On Linux, run `chmod +x hdc-baseline` if the file is not executable.

## How it works

Each field and value gets a random vector made of `-1` and `+1` values. The program then:

1. Binds each field to its value.
2. Bundles the fields into one vector for the full record.
3. Bundles training records into a prototype for each class.
4. Compares test records with the prototypes using cosine similarity.

The class with the highest similarity becomes the prediction. The random seed is fixed by default, so the same command gives repeatable results.

The included dataset is intentionally small. Its accuracy shows that the pipeline works, not that it is ready for a real-world classification task. The `example_edge` energy and latency numbers are also estimates, not hardware measurements.

## Other commands

```bash
# Show the installed release version
./hdc-baseline --version

# Change the vector size and random seed
./hdc-baseline --dimensions 5000 --seed 7

# Compare several configurations and save the results
./hdc-baseline --sweep-dimensions 1000,5000 --sweep-seeds 7,42 --output sweep.json

# Load your own categorical dataset
./hdc-baseline --dataset dataset.json --output result.json
```

Use `./hdc-baseline --help` to see every option.

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
