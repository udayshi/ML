"""Train a simple LSTM model that predicts the next word in a text corpus."""

import pickle  # Saves the fitted tokenizer so prediction uses the same word IDs.
import string  # Provides the standard punctuation characters for text cleanup.
from pathlib import Path  # Represents corpus, model, and tokenizer file locations.

import numpy as np  # Converts token IDs into arrays that Keras can train on.
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau  # Saves models and adjusts learning rate.
from tensorflow.keras.layers import Dense, Embedding, LSTM  # Defines word, sequence, and output layers.
from tensorflow.keras.models import Sequential  # Creates an ordered stack of neural-network layers.
from tensorflow.keras.optimizers import Adam  # Updates model weights during training.
from tensorflow.keras.preprocessing.text import Tokenizer  # Converts words into numeric IDs.
from tensorflow.keras.utils import to_categorical  # Converts target IDs into one-hot classification labels.


DATA_DIRECTORY = Path("data")  # Reads every .txt training file inside this directory.
MODEL_PATH = Path("udays.keras")  # Saves the best trained model in native Keras format.
TOKENIZER_PATH = Path("tokenizer.pkl")  # Saves the tokenizer used by predict.py.
EMBEDDING_SIZE = 10  # Represents each word with ten learned numeric features.
LSTM_UNITS = 1000  # Uses one thousand memory units in each LSTM layer.
HIDDEN_UNITS = 1000  # Uses one thousand neurons in the dense hidden layer.
EPOCHS = 1000  # Repeats training over the corpus one thousand times.
BATCH_SIZE = 64  # Trains on sixty-four next-word examples at a time.


# Loads all text files and converts their contents into one clean training string.
def load_corpus(data_directory: Path) -> str:
    """Read and normalize every text file in ``data_directory``."""
    # Stops early when the expected training directory is unavailable.
    if not data_directory.is_dir():
        # Explains exactly which directory must be created or supplied.
        raise FileNotFoundError(f"Training data directory was not found: {data_directory}")

    # Finds .txt files recursively and sorts them for repeatable training order.
    text_files = sorted(data_directory.rglob("*.txt"))
    # Stops early when the directory contains no usable training documents.
    if not text_files:
        # Explains that at least one .txt file is required for training.
        raise FileNotFoundError(f"No .txt training files were found in: {data_directory}")

    # Reads each UTF-8 file and joins documents with a space between them.
    text = " ".join(path.read_text(encoding="utf8") for path in text_files)
    # Builds a character map that replaces punctuation with spaces; example: "hello,world" becomes "hello world".
    punctuation_to_spaces = str.maketrans(string.punctuation, " " * len(string.punctuation))
    # Removes an invisible UTF-8 BOM character; example: "\ufeffhello" becomes "hello".
    without_bom = text.replace("\ufeff", "")
    # Replaces punctuation through str.translate; example: "hello, world!" becomes "hello  world ".
    normalized_text = without_bom.translate(punctuation_to_spaces)
    # Collapses repeated whitespace; example: "hello   world" becomes "hello world".
    return " ".join(normalized_text.split())


# Converts cleaned words into one-word inputs and their following-word labels.
def create_training_data(text: str) -> tuple[np.ndarray, np.ndarray, Tokenizer]:
    """Create one-word inputs, next-word labels, and the fitted tokenizer."""
    # Creates a tokenizer that learns a unique integer ID for each word.
    tokenizer = Tokenizer()
    # Learns the vocabulary from the complete cleaned training corpus.
    tokenizer.fit_on_texts([text])
    # Converts corpus words to IDs; example: "hello world" can become [1, 2].
    token_ids = np.array(tokenizer.texts_to_sequences([text])[0])
    # Requires an input word and a following target word for next-word training.
    if token_ids.size < 2:
        # Gives a helpful error instead of training on an invalid corpus.
        raise ValueError("The training corpus must contain at least two recognized words.")

    # Uses each word except the last as input; example: [1, 2, 3] becomes [[1], [2]].
    inputs = token_ids[:-1].reshape(-1, 1)
    # Adds one because token ID zero is reserved for padding.
    vocabulary_size = len(tokenizer.word_index) + 1
    # Uses each following word as the label; example: [1, 2, 3] targets IDs [2, 3].
    labels = to_categorical(token_ids[1:], num_classes=vocabulary_size)
    # Returns the prepared data and vocabulary mapping used by both scripts.
    return inputs, labels, tokenizer


# Creates the neural network that maps one input word to the likely next word.
def build_model(vocabulary_size: int) -> Sequential:
    """Build the LSTM network used to predict the next word."""
    # Creates a sequential model where each listed layer feeds into the next layer.
    model = Sequential(
        [
            # Converts a word ID into ten learned features.
            Embedding(vocabulary_size, EMBEDDING_SIZE),
            # Reads the one-word sequence while preserving a sequence output for the next LSTM.
            LSTM(LSTM_UNITS, return_sequences=True),
            # Produces one summary vector from the LSTM sequence.
            LSTM(LSTM_UNITS),
            # Learns nonlinear relationships between the input and next word.
            Dense(HIDDEN_UNITS, activation="relu"),
            # Produces one probability for every word in the vocabulary.
            Dense(vocabulary_size, activation="softmax"),
        ]
    )
    # Uses categorical loss and Adam to learn the correct next-word probability.
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
    # Returns the compiled model so the caller can train it.
    return model


# Saves the learned word-to-ID mapping for prediction after training completes.
def save_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    """Save ``tokenizer`` for use by the prediction script."""
    # Opens the target file in binary write mode for pickle serialization.
    with path.open("wb") as tokenizer_file:
        # Writes the fitted tokenizer so prediction uses training vocabulary IDs.
        pickle.dump(tokenizer, tokenizer_file)


# Coordinates corpus loading, data preparation, training, and model persistence.
def main() -> None:
    """Train the model and save the best checkpoint and tokenizer."""
    # Loads and cleans every text file located below data/.
    corpus = load_corpus(DATA_DIRECTORY)
    # Creates next-word examples and a tokenizer from the cleaned corpus.
    inputs, labels, tokenizer = create_training_data(corpus)
    # Saves the tokenizer before training so predict.py can use the same mapping.
    save_tokenizer(tokenizer, TOKENIZER_PATH)

    # Builds a model with one output class for every vocabulary word.
    model = build_model(labels.shape[1])
    # Reports how many word-pair examples are available for training.
    print(f"Training on {len(inputs)} next-word pairs.")
    # Prints the model layers and parameter counts for the demo.
    model.summary()

    # Saves the checkpoint with the lowest training loss.
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor="loss", save_best_only=True, verbose=1)
    # Reduces the learning rate when training loss stops improving.
    learning_rate_reducer = ReduceLROnPlateau(
        monitor="loss", factor=0.2, min_lr=0.0001, patience=3, verbose=1
    )
    # Trains the model using the prepared examples and training callbacks.
    model.fit(
        inputs,
        labels,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, learning_rate_reducer],
        epochs=EPOCHS,
    )


# Runs training only when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    # Starts the complete training workflow.
    main()
