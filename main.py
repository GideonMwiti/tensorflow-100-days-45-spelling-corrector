import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Define Corpus Text representing typical English + Tech/ML vocabulary
# We duplicate some sentences/words to create specific frequency hierarchies
corpus_text = """
tensorflow is an open source machine learning framework developed by the google brain team.
tensorflow tensorflow tensorflow tensorflow tensorflow tensorflow
python is a high level programming language that is widely used for artificial intelligence and deep learning models.
python python python python python
neural networks are composed of nodes that process incoming data and pass it to fully connected dense layers.
neural neural neural
networks networks networks
dense dense
convolutional networks are used for computer vision, image classification, segmentation, and object detection.
computer computer computer
learning models requires optimizing variables using gradient descent algorithms like adam and loss metrics.
learning learning learning learning learning
models models models
gradient gradient
algorithms algorithms
we develop new tools for artificial intelligence, computer science, and engineering applications.
intelligence intelligence
science science
databases store structured tables of records that can be queried using database queries.
database database database
stadium athlete coach tournament championship referee football basketball tennis soccer are sports.
recipe kitchen restaurant chef ingredient baking spices sauce delicious flavor are related to cooking.
the there these three
the the the the the the the the the the the the the the the the the the the the the the the the
there there there there there there there there there there there there there there
these these these these these these these these these
three three three three three
"""

def get_word_frequencies(text):
    """Tokenize the corpus and return a dictionary of word frequencies."""
    words = re.findall(r'\w+', text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq

def edits1(word):
    """Generate all edits that are one edit distance away from `word`."""
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    """Generate all edits that are two edit distances away from `word`."""
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def main():
    print("====================================================")
    print("Project 45: Spelling Correction Model (Norvig + TF)")
    print("Goal: Basic spelling correction with StaticHashTable lookup")
    print("====================================================\n")

    # 2. Extract word frequency statistics
    freq_dict = get_word_frequencies(corpus_text)
    vocab = list(freq_dict.keys())
    vocab_size = len(vocab)
    print(f"Corpus tokenization complete. Unique vocabulary words: {vocab_size}")
    print(f"Total corpus tokens processed: {sum(freq_dict.values())}\n")

    # 3. Create TensorFlow-Friendly StaticHashTable for word frequency lookup
    keys = tf.constant(vocab, dtype=tf.string)
    values = tf.constant([freq_dict[w] for w in vocab], dtype=tf.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys, values)
    table = tf.lookup.StaticHashTable(initializer, default_value=0)

    # Helper function to check which candidates are present in our vocabulary
    def filter_known_candidates(candidates):
        cand_list = list(candidates)
        if not cand_list:
            return []
        # Convert candidate list to string tensor and query table
        cand_tensor = tf.constant(cand_list, dtype=tf.string)
        freqs = table.lookup(cand_tensor).numpy()
        # Keep candidates with a frequency greater than 0
        known = [cand_list[i] for i, f in enumerate(freqs) if f > 0]
        return known

    # Candidate search hierarchy
    def get_candidates(word):
        # 1. Edit distance 0
        e0 = filter_known_candidates({word})
        if e0:
            return e0, 0
        
        # 2. Edit distance 1
        e1 = filter_known_candidates(edits1(word))
        if e1:
            return e1, 1
        
        # 3. Edit distance 2
        e2 = filter_known_candidates(edits2(word))
        if e2:
            return e2, 2
        
        # 4. Fallback (no match found)
        return [word], -1

    # Main spelling correction logic
    def correct_spelling(word):
        candidates, distance = get_candidates(word)
        if distance == -1:
            return word, [word], -1, [0]
        
        # Query TF table for the candidate frequencies
        cand_tensor = tf.constant(candidates, dtype=tf.string)
        freqs = table.lookup(cand_tensor).numpy()
        
        # Select the candidate with the highest frequency
        best_idx = np.argmax(freqs)
        correction = candidates[best_idx]
        
        return correction, candidates, distance, freqs

    # 4. Evaluate on test misspellings
    test_words = [
        "tenstrflow",
        "pythn",
        "computr",
        "learnin",
        "datbase",
        "netwrk",
        "thee"
    ]

    print("Evaluating Test Misspellings:")
    eval_results = []
    for test_word in test_words:
        correction, candidates, dist, freqs = correct_spelling(test_word)
        print(f"  Input: '{test_word:12s}' -> Correction: '{correction:12s}' (Edit Dist: {dist}, Candidates Found: {len(candidates)})")
        
        # Store for visualization
        eval_results.append({
            "input": test_word,
            "correction": correction,
            "candidates": candidates,
            "dist": dist,
            "freqs": freqs,
            "selected_freq": int(freq_dict.get(correction, 0))
        })

    # 5. Generate and Save Spelling Correction Dashboard
    print("\nGenerating visualization dashboard...")
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.8])

    # Left Subplot: Probability Distribution for "thee"
    ax_bar = fig.add_subplot(gs[0])
    
    # Extract data for "thee" candidates
    thee_data = [res for res in eval_results if res["input"] == "thee"][0]
    thee_candidates = thee_data["candidates"]
    thee_freqs = thee_data["freqs"]
    thee_winner = thee_data["correction"]
    
    # Calculate probabilities
    total_thee_freq = sum(thee_freqs)
    thee_probs = [f / total_thee_freq for f in thee_freqs]
    
    # Stylized colors
    bar_colors = ["#2ecc71" if c == thee_winner else "#e74c3c" for c in thee_candidates]
    
    bars = ax_bar.bar(thee_candidates, thee_probs, color=bar_colors, edgecolor='black', width=0.5, zorder=3)
    ax_bar.set_title("Candidate Probabilities for Misspelled Word: 'thee'", fontsize=11, fontweight="bold", pad=12)
    ax_bar.set_ylabel("Normalized Candidate Probability P(c | w)", fontsize=10)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.grid(True, linestyle="--", alpha=0.3, zorder=0)

    # Annotate bars
    for bar, freq in zip(bars, thee_freqs):
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"Count: {freq}\n({yval:.1%})",
                    ha='center', va='bottom', fontsize=9, fontweight='semibold')

    # Right Subplot: Results Summary Table
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    ax_table.set_title("Spelling Correction Verification Panel", fontsize=12, fontweight="bold", pad=15)

    # Format table data
    table_data = []
    for res in eval_results:
        table_data.append([
            res["input"],
            res["correction"],
            str(res["dist"]),
            str(len(res["candidates"])),
            str(res["selected_freq"])
        ])

    headers = ["Misspelled Input", "Suggested Correction", "Edit Distance", "Candidates Found", "Corpus Frequency"]
    
    # Draw table
    tb = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#2c3e50"] * len(headers),
        colWidths=[0.22, 0.22, 0.16, 0.20, 0.20]
    )
    
    # Format table style
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.5)
    tb.scale(1.0, 1.6)

    # Color header text white
    for col_idx in range(len(headers)):
        cell = tb[0, col_idx]
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")

    # Style cells with alternating backgrounds
    for row_idx in range(1, len(table_data) + 1):
        bg_color = "#f8f9fa" if row_idx % 2 == 0 else "#ffffff"
        for col_idx in range(len(headers)):
            cell = tb[row_idx, col_idx]
            cell.set_facecolor(bg_color)
            if col_idx == 1:
                cell.get_text().set_weight("bold")
                cell.get_text().set_color("#27ae60")

    plt.tight_layout()
    output_filename = "spelling_corrector_results.png"
    plt.savefig(output_filename, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"\nSuccess! Results panel saved as '{output_filename}'.")
    print("====================================================")

if __name__ == "__main__":
    main()
