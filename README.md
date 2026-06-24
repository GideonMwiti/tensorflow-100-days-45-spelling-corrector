# Project 45: Spelling Correction Model (Norvig + TensorFlow Lookup)

This project implements a basic spelling correction model inspired by Peter Norvig’s classic approach. The implementation utilizes a Python-based edit-distance candidates generator and leverages TensorFlow's `tf.lookup.StaticHashTable` to query and rank word probabilities in a graph-friendly and scalable way.

---

## Conceptual & Mathematical Overview

Spelling correction can be modeled as a Bayesian inference problem. Given a misspelled word $w$, we seek the candidate correction $c$ that maximizes the posterior probability:
$$P(c \mid w) = \frac{P(w \mid c) P(c)}{P(w)}$$

Since $P(w)$ is constant for all candidate corrections, the objective simplifies to:
$$\hat{c} = \arg\max_{c} P(w \mid c) P(c)$$

Where:
1. **$P(c)$ (Language Model / Prior)**:
   The probability that the word $c$ occurs in a typical text corpus. We estimate this from a corpus of standard text:
   $$P(c) = \frac{\text{count}(c)}{\text{total corpus tokens}}$$
2. **$P(w \mid c)$ (Error Model / Likelihood)**:
   The probability that a writer types $w$ when they intended $c$. Rather than using a complex keyboard transition matrix, Norvig's algorithm implements a simplified distance hierarchy:
   - **Edit Distance 0**: The candidate is $w$ itself (highest priority).
   - **Edit Distance 1**: Candidates that are one edit away (deletion, transposition, replacement, insertion).
   - **Edit Distance 2**: Candidates that are two edits away.
   - **Out of Vocabulary**: Fall back to the original word $w$.

### Edit Operations (Distance 1)
For any string $w$, the candidates of distance 1 are generated using:
- **Splits**: Partitioning the word into left and right halves: $\{(L, R) \mid L + R = w\}$.
- **Deletions**: Removing a single character: $\{L + R[1:]\}$.
- **Transpositions**: Swapping two adjacent characters: $\{L + R[1] + R[0] + R[2:]\}$.
- **Replacements**: Swapping one character for a letter from the alphabet: $\{L + \text{char} + R[1:]\}$.
- **Insertions**: Inserting a new character from the alphabet: $\{L + \text{char} + R\}$.

---

## TensorFlow-Friendly Integration

To query candidate probabilities in a TensorFlow graph-compatible environment, we represent the vocabulary frequency dictionary using a `tf.lookup.StaticHashTable`:
- **Key-Value Initializer**: Initialized with `tf.lookup.KeyValueTensorInitializer(keys, values)`, where keys are strings (vocabulary words) and values are integer frequencies.
- **Lookup Query**: Executes lookup `table.lookup(keys_tensor)` which is highly optimized and compatible with standard TensorFlow model graphs, serving as a clean embedding or classification routing layer.

---

## File Structure

- **[main.py](file:///c:/Users/Dart%20Technologies/OneDrive/Desktop/tensorflow_100_projects/45_spelling_corrector/main.py)**: Spelling corrector pipeline:
  - Processes a raw English corpus to extract word count statistics.
  - Prepares the `tf.lookup.StaticHashTable` with vocabulary frequencies.
  - Implements the recursive edit-distance candidate generator in Python.
  - Performs candidate lookup and ranking inside a TensorFlow-friendly execution block.
  - Generates a metrics and correction comparison panel `spelling_corrector_results.png`.
- **[README.md](file:///c:/Users/Dart%20Technologies/OneDrive/Desktop/tensorflow_100_projects/45_spelling_corrector/README.md)**: Conceptual guide and mathematical details.
- **[.gitignore](file:///c:/Users/Dart%20Technologies/OneDrive/Desktop/tensorflow_100_projects/45_spelling_corrector/.gitignore)**: standard gitignore configurations.

---

## Installation & Running

Ensure the environment has TensorFlow 2.x and Matplotlib:
```powershell
pip install tensorflow matplotlib numpy
```

Run the pipeline:
```powershell
python main.py
```
