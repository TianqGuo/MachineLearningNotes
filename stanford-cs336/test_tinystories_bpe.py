#!/usr/bin/env python3
"""Test script for BPE training on TinyStories dataset."""

import time
import psutil
import os
import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'assignment1-basics', 'cs336_basics'))
from bpe_train import train_bpe

def test_tinystories_bpe():
    """Test BPE training on TinyStories dataset with vocab size 10,000."""

    # File paths
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_path = "./tinystories_vocab.pkl"
    merges_path = "./tinystories_merges.pkl"

    print("Starting BPE training on TinyStories dataset...")
    print(f"Input file size: {os.path.getsize(input_path) / (1024**3):.2f} GB")

    # Monitor memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024**3)  # GB
    print(f"Initial memory usage: {initial_memory:.2f} GB")

    # Start timing
    start_time = time.time()

    # Train BPE tokenizer
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        use_parallel=True
    )

    end_time = time.time()
    training_time = end_time - start_time

    # Check final memory usage
    final_memory = process.memory_info().rss / (1024**3)  # GB
    peak_memory = final_memory  # This is a simple approximation

    print(f"\n=== Training Complete ===")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Peak memory usage: {peak_memory:.2f} GB")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    # Verify special token is in vocabulary
    special_token_found = False
    special_token_id = None
    for token_id, token_bytes in vocab.items():
        if token_bytes == b"<|endoftext|>":
            special_token_found = True
            special_token_id = token_id
            break

    print(f"Special token '<|endoftext|>' found: {special_token_found}")
    if special_token_found:
        print(f"Special token ID: {special_token_id}")

    # Find longest token
    longest_token = max(vocab.values(), key=len)
    longest_length = len(longest_token)
    print(f"Longest token length: {longest_length} bytes")
    print(f"Longest token: {longest_token}")

    # Try to decode as UTF-8 for display
    try:
        longest_token_str = longest_token.decode('utf-8')
        print(f"Longest token (decoded): '{longest_token_str}'")
    except UnicodeDecodeError:
        print(f"Longest token (hex): {longest_token.hex()}")

    # Save results
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)

    print(f"\nResults saved to:")
    print(f"  Vocabulary: {vocab_path}")
    print(f"  Merges: {merges_path}")

    # Performance analysis
    print(f"\n=== Performance Analysis ===")
    if training_time <= 30 * 60:  # 30 minutes
        print("✓ Training time meets requirement (≤ 30 minutes)")
    else:
        print("✗ Training time exceeds requirement (> 30 minutes)")

    if peak_memory <= 30:  # 30 GB
        print("✓ Memory usage meets requirement (≤ 30GB)")
    else:
        print("✗ Memory usage exceeds requirement (> 30GB)")

    return vocab, merges, training_time, peak_memory

if __name__ == "__main__":
    vocab, merges, training_time, peak_memory = test_tinystories_bpe()