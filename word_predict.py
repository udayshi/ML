"""Generate next-word predictions with the model trained by ``build_model.py``."""

import pickle  # Loads the saved tokenizer that maps words to training IDs.
from pathlib import Path  # Represents the model and tokenizer file locations.

import numpy as np  # Selects the word ID with the highest predicted probability.
from tensorflow.keras.models import Sequential, load_model  # Loads and types the trained Keras model.
from tensorflow.keras.preprocessing.text import Tokenizer  # Types the saved tokenizer used for word conversion.


MODEL_PATH = Path("udays.keras")  # Points to the trained Keras model saved by build_model.py.
TOKENIZER_PATH = Path("tokenizer.pkl")  # Points to the tokenizer saved during training.
WORDS_TO_GENERATE = 1000  # Limits one demo run to one thousand generated words.


# Loads the trained model and tokenizer that must use the same word-ID mapping.
def load_prediction_resources() -> tuple[Sequential, Tokenizer]:
    """Load the trained model and its matching tokenizer."""
    # Checks that training created the model file before attempting to load it.
    if not MODEL_PATH.is_file():
        # Explains how to create the missing model file.
        raise FileNotFoundError(f"Model was not found: {MODEL_PATH}. Run build_model.py first.")
    # Checks that training created the tokenizer file before attempting to load it.
    if not TOKENIZER_PATH.is_file():
        # Explains how to create the missing tokenizer file.
        raise FileNotFoundError(f"Tokenizer was not found: {TOKENIZER_PATH}. Run build_model.py first.")

    # Loads inference-only model state; compile=False avoids unnecessary training metrics.
    model = load_model(MODEL_PATH, compile=False)
    # Opens the tokenizer file in binary read mode for pickle deserialization.
    with TOKENIZER_PATH.open("rb") as tokenizer_file:
        # Restores the word-to-ID mapping learned from the training corpus.
        tokenizer = pickle.load(tokenizer_file)
    # Returns both matching resources for next-word prediction.
    return model, tokenizer


# Predicts one word from the final recognized word in a seed phrase.
def predict_next_word(model: Sequential, tokenizer: Tokenizer, text: str) -> str:
    """Return the most likely word after the final recognized word in ``text``."""
    # Converts input text to known token IDs; example: "hello world" can become [1, 2].
    token_ids = tokenizer.texts_to_sequences([text])[0]
    # Stops safely when none of the supplied words occur in the training vocabulary.
    if not token_ids:
        # Returns an empty result so the caller can display a helpful message.
        return ""

    # Selects the final word and adds the required batch and sequence dimensions; example: [2] becomes [[2]].
    model_input = np.array(token_ids[-1:]).reshape(1, 1)
    # Gets one probability per vocabulary word without printing Keras progress output.
    probabilities = model.predict(model_input, verbose=0)
    # Selects the ID with the largest probability; example: [0.1, 0.8, 0.1] selects ID 1.
    predicted_id = int(np.argmax(probabilities[0]))
    # Converts the predicted ID back into a word, or returns empty when unavailable.
    return tokenizer.index_word.get(predicted_id, "")


# Prompts for seed text and repeatedly appends predicted words for the command-line demo.
def main() -> None:
    """Prompt for a seed phrase and print a sequence of predicted words."""
    # Loads the model and tokenizer once before generating any words.
    model, tokenizer = load_prediction_resources()
    # Reads a seed phrase and removes leading or trailing spaces.
    text = input("Enter your line: ").strip()
    # Stores the prior prediction so repeated-word loops can stop early.
    old_prediction = ""
    # Requires at least one non-space seed word from the user.
    if not text:
        # Tells the user why generation cannot begin.
        print("Please enter at least one word from the training corpus.")
        # Ends the demo run after invalid input.
        return

    # Generates no more words than the configured demo limit.
    for _ in range(WORDS_TO_GENERATE):
        # Predicts the word most likely to follow the current seed text.
        predicted_word = predict_next_word(model, tokenizer, text)
        # Stops when the model cannot map the seed to a known next word.
        if not predicted_word:
            # Explains why the demo cannot continue.
            print("No next word could be predicted from the supplied text.")
            # Ends generation when no valid word is available.
            return

        # Appends the prediction so it becomes context for the next prediction.
        text = f"{text} {predicted_word}"
        # Prints a new prediction while avoiding immediate repeated output.
        if predicted_word != old_prediction:
            # Shows the next generated word to the user.
            print(predicted_word)
        # Stops when the model starts repeating the same word consecutively.
        else:
            # Ends the loop to prevent a repetitive prediction sequence.
            break

        # Remembers this word so the next loop can detect repetition.
        old_prediction = predicted_word


# Runs the interactive demo only when this file is executed directly.
if __name__ == "__main__":
    # Starts the command-line prediction workflow.
    main()
