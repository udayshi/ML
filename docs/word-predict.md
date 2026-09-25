# Next-Word Prediction: Jump-Start Guide

This example trains a TensorFlow/Keras LSTM model to predict the word that is most likely to follow a word in a text corpus. It demonstrates a complete jump-start workflow:

1. Install the Python dependencies.
2. Add UTF-8 text files to the training corpus directory.
3. Clean the corpus and convert its words to numeric IDs.
4. Create input-word and next-word training pairs.
5. Build and train an LSTM classifier.
6. Save the best model and its matching tokenizer.
7. Load both resources and generate words interactively.

## Install the packages

After initializing the uv project from the main [README](../README.md), install the packages used by this example:

```bash
uv add tensorflow numpy
```

The code uses these Python packages:

- `tensorflow` — tokenizes text and builds, trains, saves, and loads the Keras LSTM model.
- `numpy` — represents token IDs as arrays and selects the most likely predicted word.

## 1. Add the training text

The training script reads every `.txt` file below `data/`, including nested directories. This project includes example files in `data/predict/`:

```text
data/
└── predict/
    ├── data-1.txt
    └── data-2.txt
```

Use UTF-8 text files. The script reads them in sorted path order, joins their contents, removes a UTF-8 byte-order mark when present, replaces standard punctuation with spaces, and collapses repeated whitespace. It does not change letter case, so `Hello` and `hello` become different vocabulary entries.

At least two recognized words are required. Training text should be representative of the wording and style you want the model to generate; the model learns only from the supplied corpus.

## 2. Build and save the model

Run the training script from the project directory:

```bash
uv run word_predict_build.py
```

### Create next-word examples

`Tokenizer` assigns an integer ID to every word in the cleaned corpus. Given these token IDs:

```text
[1, 2, 3]
```

the script creates one-word inputs and the words that follow them:

```text
Inputs:  [[1], [2]]
Labels:  [2, 3]
```

The labels are one-hot encoded with one output position per vocabulary word. Token ID `0` remains reserved for padding, so the vocabulary size is the number of learned words plus one.

### Build and train the LSTM

The model receives a sequence containing one word ID and produces a probability for every vocabulary word:

```python
model = Sequential(
    [
        Embedding(vocabulary_size, 10),
        LSTM(1000, return_sequences=True),
        LSTM(1000),
        Dense(1000, activation="relu"),
        Dense(vocabulary_size, activation="softmax"),
    ]
)
```

The embedding layer learns a 10-number representation for each word. The two LSTM layers and dense hidden layer each use 1,000 units. The softmax output returns a probability distribution across the vocabulary.

The model uses Adam with a learning rate of `0.001`, categorical cross-entropy loss, batches of 64 examples, and up to 1,000 epochs. `ModelCheckpoint` writes `udays.keras` whenever the training loss improves. `ReduceLROnPlateau` reduces the learning rate by a factor of `0.2` after three epochs without a loss improvement, without reducing it below `0.0001`.

The fitted tokenizer is saved to `tokenizer.pkl`. It must stay paired with `udays.keras`, because it contains the word-to-ID mapping used during training.

## 3. Generate words interactively

After training successfully, run:

```bash
uv run word_predict.py
```

Enter a seed phrase when prompted:

```text
Enter your line: your seed words
```

For each prediction, the script converts the current text into token IDs, passes only the final recognized word ID to the model, chooses the ID with the highest probability, and converts that ID back into a word. It appends the prediction to the seed text and repeats for at most 1,000 words.

The script prints each generated word on its own line and stops early when it cannot predict a word or when it predicts the same word twice in a row. A phrase containing unknown words can still work if its final recognized word is in the corpus; a phrase with no recognized words cannot produce a prediction.

## Why use this approach?

This example introduces the main parts of a word-level language-model workflow:

- A tokenizer turns text into the numeric IDs required by neural-network layers.
- Consecutive words create supervised examples without manually labeling the corpus.
- An embedding lets the model learn numeric representations of words.
- LSTM layers are designed to process ordered sequences.
- A softmax output can select one likely next word from a fixed vocabulary.
- Saving the tokenizer with the model preserves the vocabulary required for inference.

## Important limitations

Despite using LSTM layers, this script trains and predicts from only one input word at a time. The earlier words in a seed phrase are discarded before prediction, so the model cannot use phrase-level or sentence-level context. It effectively learns how often one word follows another word in the corpus.

The default network is very large for a small learning corpus: it has two 1,000-unit LSTMs, a 1,000-unit dense layer, and 1,000 epochs. Training can take substantial time and memory, and it can overfit a small corpus. For an experiment, consider reducing `LSTM_UNITS`, `HIDDEN_UNITS`, and `EPOCHS` first.

Generation is greedy: `np.argmax` always chooses the single highest-probability word. This can make output repetitive and does not provide varied alternatives. The corpus is also the sole source of the model's vocabulary and writing style; it has no general language knowledge.

`tokenizer.pkl` uses Python pickle. Load it only from a trusted source, because unpickling untrusted data can execute arbitrary code.

## Common problems

### `ModuleNotFoundError: No module named 'tensorflow'`

Install TensorFlow for the current Python environment:

```bash
uv add tensorflow
```

### `Training data directory was not found: data`

Run the command from the project directory, or create the expected `data/` directory.

### `No .txt training files were found in: data`

Add at least one UTF-8 file ending in `.txt` below `data/`. Files may be placed in subdirectories such as `data/predict/`.

### `The training corpus must contain at least two recognized words`

Add more text. The model needs one word as an input and a later word as its target.

### `Model was not found: udays.keras` or `Tokenizer was not found: tokenizer.pkl`

Run `uv run word_predict_build.py` first. Keep both generated files together; replacing either one independently can make predictions incorrect.

### Training is slow or runs out of memory

The default model is large. Reduce `LSTM_UNITS`, `HIDDEN_UNITS`, and `EPOCHS` in `word_predict_build.py`, then train again so the saved model and tokenizer match.

## Project files

```text
.
├── data/
│   └── predict/
│       ├── data-1.txt          # Example training text
│       └── data-2.txt          # Example training text
├── docs/
│   └── word-predict.md         # This walkthrough
├── word_predict_build.py       # Cleans text, trains, and saves the model
├── word_predict.py             # Loads resources and generates words
├── udays.keras                 # Best saved model after training
├── tokenizer.pkl               # Saved word-to-ID mapping
└── README.md                   # Project overview and setup
```

## Suggested next steps

1. Train on input sequences of several words so predictions can use meaningful context.
2. Reserve validation data and monitor validation loss instead of training loss alone.
3. Lower the model size and epochs for small corpora, then compare quality and training time.
4. Add temperature sampling or top-k sampling to generate less repetitive alternatives.
5. Save the model and tokenizer in a dedicated `models/` directory with a versioned corpus description.
