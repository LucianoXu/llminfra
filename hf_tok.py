from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers
from pathlib import Path
import json

def train_tokenizer(
    train_files,
    vocab_size=50000,
    min_frequency=2,
    output_path="tokenizer.json",
):
    # Initialize a BPE tokenizer without UNK token
    tokenizer = Tokenizer(models.BPE())  # No unk_token specified

    # Use a sequence of pre-tokenizers
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(),
        pre_tokenizers.ByteLevel()
    ])  # type: ignore

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,  # type: ignore
        min_frequency=min_frequency,    # type: ignore
        special_tokens=["<|endoftext|>"],   # type: ignore
        show_progress=True  # type: ignore
    )

    # Ensure all training files exist
    train_files = [Path(f) for f in train_files]
    if not all(f.exists() for f in train_files):
        raise ValueError("Some training files do not exist")

    # Train the tokenizer
    tokenizer.train(files=[str(f) for f in train_files], trainer=trainer)

    # Save the tokenizer and vocabulary
    tokenizer.save(str(output_path))

    return tokenizer

def test_tokenizer(tokenizer, test_texts):
    """
    Test the trained tokenizer with detailed output.
    """
    for text in test_texts:
        print(f"\nOriginal text: {text}")
        
        # Show pre-tokenization (before BPE)
        pre_tokenized = tokenizer.pre_tokenizer.pre_tokenize_str(text)
        print(f"Pre-tokenization (before BPE): {pre_tokenized}")
        
        # Show final tokenization
        encoding = tokenizer.encode(text)
        print(f"Final tokens: {encoding.tokens}")
        print(f"Token IDs: {encoding.ids}")
        print(f"Decoded text: {tokenizer.decode(encoding.ids)}")
        
        # Show token-to-id mapping for this text
        vocab = tokenizer.get_vocab()
        print("\nToken to ID mapping for this text:")
        for token in encoding.tokens:
            if token in vocab:
                print(f"{token}: {vocab[token]}")

# Example usage and testing
if __name__ == "__main__":
    # Example training data
    train_files = ["data/TinyStoriesV2-GPT4-train.txt"]
    
    # Train the tokenizer
    tokenizer = train_tokenizer(
        train_files=train_files,
        vocab_size=10000,
        min_frequency=1,
    )