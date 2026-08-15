import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Example text
text = "Hello world! I'll tokenize this."

# Step 1: Pre-tokenization with regex
print("=== Pre-tokenization ===")
matches = re.findall(PAT, text)
print("Regex matches:", matches)
# Output: ['Hello', ' world', '!', ' I', "'ll", ' tokenize', ' this', '.']

# Step 2: Convert each pre-token to bytes
print("\n=== Converting to bytes ===")
pre_tokens = []
for match in re.finditer(PAT, text):
    token_str = match.group()
    token_bytes = token_str.encode('utf-8')
    byte_list = list(token_bytes)
    print(f"'{token_str}' -> {token_bytes} -> {byte_list}")
    pre_tokens.append(byte_list)

print(f"\nFinal pre_tokens structure: {pre_tokens}")

# Step 3: Why this structure matters for BPE
print("\n=== Why list of lists? ===")
print("❌ If we flattened to single list:")
flat_list = [byte for sublist in pre_tokens for byte in sublist]
print(f"Flat: {flat_list}")
print("Problem: We'd lose boundary info and merge across word boundaries!")

print("\n✅ With list of lists:")
print(f"Structured: {pre_tokens}")
print("Benefit: We can merge within each pre-token but never across boundaries")

# Example of merging within boundaries
print("\n=== Example: Finding pairs within boundaries ===")
from collections import defaultdict

# Count pairs within each pre-token (correct way)
pair_counts = defaultdict(int)
for pre_token in pre_tokens:
    if len(pre_token) >= 2:
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i + 1])
            pair_counts[pair] += 1

print("Pairs found within pre-token boundaries:")
for pair, count in list(pair_counts.items())[:5]:  # Show first 5
    char1 = chr(pair[0]) if 32 <= pair[0] <= 126 else f"byte({pair[0]})"
    char2 = chr(pair[1]) if 32 <= pair[1] <= 126 else f"byte({pair[1]})"
    print(f"  ({char1}, {char2}): {count}")