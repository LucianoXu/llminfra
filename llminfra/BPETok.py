from functools import lru_cache
import regex as re
from typing import Iterable, Iterator, Optional
from tqdm import tqdm
import os
from pathlib import Path
import time
import sys
import json
import base64

    
def pretokenize(text: str) -> list[str]:
    '''
    Pre-tokenize a string of text into a list of tokens.

    Returns:
        list[int]: A list of token ids.
    '''
    # pretokenize the text
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(PAT, text)

def token_statistics(tokens: list[str]) -> dict[str, int]:
    '''
    Compute statistics on a list of tokens.

    Returns:
        dict[str, int]: A dictionary mapping tokens to their counts.
    '''
    pretokenized_fac: dict[str, int] = {}
    for token in tqdm(tokens, desc='Token Statistics'):
        if token in pretokenized_fac:
            pretokenized_fac[token] += 1
        else:
            pretokenized_fac[token] = 1

    return pretokenized_fac


def train_bpe(path: str | os.PathLike, vocab_size: int, special_tokens: list[str], load_chunk_size: int = 1024*1024*1024) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    '''
    train the BPE tokenizer on the text file at path

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.
    '''

    print('Training BPE tokenizer...')
    start_time = time.time()

    # the BPE vocab: a mapping from token id to bytes
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    normal_vocab_size = vocab_size -len(special_tokens)

    # accumulator for the pretokenized tokens and their counts
    pretokenized_fac : dict[str, int] = {}

    # read the text file by chunks and pretokenize it
    print('Reading text file and pretokenize...')
    sys.stdout.flush()
    # Get the total file size to set up the tqdm progress bar
    file_size = os.path.getsize(path)

    with open(path, 'r') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Processing Chuncks") as pbar:
            while True:
                text = f.read(load_chunk_size)  # Read in chunks
                if not text:
                    break
                new_pretokenized_fac = token_statistics(pretokenize(text))
                for token, count in new_pretokenized_fac.items():
                    if token in pretokenized_fac:
                        pretokenized_fac[token] += count
                    else:
                        pretokenized_fac[token] = count

                pbar.update(len(text))  # Update progress bar with the bytes read

    # transfrom the pretokenized tokens into a list of bytes and a list of counts                    
    pretokenized_list : list[tuple[int, ...]] = [tuple(token.encode('utf-8')) for token in pretokenized_fac]
    pretokenized_count : list[int] = [count for count in pretokenized_fac.values()]
    print('Pretokenize done.')

    # build the byte pairs countings and pair appearances
    # pair appearances is a dictionary that maps a pair to a set of indices where it appears
    print('Building byte pairs...', end='')
    sys.stdout.flush()
    pair_counts: dict[tuple[int, int], int] = {}
    pair_appearances: dict[tuple[int, int], set[int]] = {}
    for i, token in enumerate(pretokenized_list):
        for j in range(len(token) - 1):
            pair = (token[j], token[j + 1])
            if pair in pair_counts:
                pair_counts[pair] += pretokenized_count[i]
                pair_appearances[pair].add(i)
            else:
                pair_counts[pair] = pretokenized_count[i]
                pair_appearances[pair] = {i}
    print('done.')

    # BPE Merging
    for i in tqdm(range(256, normal_vocab_size), desc="BPE Merging"):
        # find the most common pair. If there are multiple pairs with the same count, choose the one that is lexicographically largest
        if len(pair_counts) == 0:
            raise ValueError('The BPE training data is too small to generate the requested vocabulary size.')
        max_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], (vocab[pair[0]], vocab[pair[1]])))

        # merge this pair, add it to the vocab
        vocab[i] = vocab[max_pair[0]] + vocab[max_pair[1]]
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))

        # update the pair counts and appearances
        appearances = pair_appearances[max_pair].copy()
        for appear_id in appearances:

            # remove the count influence from the old pair
            pretoken_appear = pretokenized_list[appear_id]
            for j in range(len(pretoken_appear) - 1):
                pair = (pretoken_appear[j], pretoken_appear[j + 1])
                pair_counts[pair] -= pretokenized_count[appear_id]
                if pair_counts[pair] == 0:
                    pair_counts.pop(pair)
                    pair_appearances[pair].remove(appear_id)

            # replace the old pair with the new pair
            new_appear = list(pretoken_appear)
            j = 0
            while j < len(new_appear) - 1:
                if new_appear[j] == max_pair[0] and new_appear[j + 1] == max_pair[1]:
                    new_appear[j: j+2] = [i]
                else:
                    j += 1

            pretokenized_list[appear_id] = tuple(new_appear)                      

            # add the count influence from the new pair
            for j in range(len(new_appear) - 1):
                pair = (new_appear[j], new_appear[j + 1])
                if pair in pair_counts:
                    pair_counts[pair] += pretokenized_count[appear_id]
                    pair_appearances[pair].add(appear_id)
                else:
                    pair_counts[pair] = pretokenized_count[appear_id]
                    pair_appearances[pair]={appear_id}

    # add the special tokens to the vocab
    for i, token in enumerate(special_tokens, start=normal_vocab_size):
        vocab[i] = token.encode('utf-8')

    end_time = time.time()
    print('Finished training BPE tokenizer, took', end_time - start_time, 'seconds.')

    return vocab, merges



class BPETok:
    '''
    Speical tokens are handled by direct mapping before tokenization.
    '''
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Optional[list[str]]):
        real_special_tokens = special_tokens.copy() if special_tokens is not None else []
        self.special_token_ids = {
            token: next(k for k, v in vocab.items() if v == token.encode('utf-8')) for token in real_special_tokens
        }
        # special tokens are ordered in reverse order
        self.special_match_order = real_special_tokens[::-1]

        self.vocab = vocab.copy()
        self.vocab_bytes2id = {v: k for k, v in vocab.items()}
        self.merges = merges.copy()
        self.merge_rules = {merge: i for i, merge in enumerate(merges)}
        self.vocab_size = len(vocab)

    @staticmethod
    def from_train(path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
        vocab_bytes, merges = train_bpe(path, vocab_size, special_tokens)
        return BPETok(vocab_bytes, merges, special_tokens)
        
    def save(self, file_path: str | os.PathLike):
        
        # Convert folder_path to a Path object
        file_path = Path(file_path)

        # Ensure the directory exists
        folder_path = file_path.parent
        folder_path.mkdir(parents=True, exist_ok=True)

        # prepare the object for serialization
        obj = {
            'vocab_size': self.vocab_size,
            'special_tokens': list(self.special_token_ids.keys()),
            'special_token_ids': self.special_token_ids,
            'vocab': {k: base64.b64encode(v).decode('utf-8') for k, v in self.vocab.items()},
            'merges': [(base64.b64encode(a).decode('utf-8'), base64.b64encode(b).decode('utf-8')) for a, b in self.merges]
        }

        with open(file_path, 'w') as f:
            f.write(json.dumps(obj, indent=4))

    @staticmethod
    def from_local(file_path: str | os.PathLike):
        file_path = Path(file_path)
        with open(file_path, 'r') as f:
            obj = json.load(f)
            vocab = {int(k): base64.b64decode(v.encode('utf-8')) for k, v in obj['vocab'].items()}
            merges = [(base64.b64decode(a.encode('utf-8')), base64.b64decode(b.encode('utf-8')) ) for a, b in obj['merges']]
            special_tokens = obj['special_tokens']

        return BPETok(vocab, merges, special_tokens)
    
    def encode(self, text: str)-> list[int]:
        '''
        Encode a string of text into a list of token ids.
        '''
        if len(self.special_match_order) == 0:
            spec_split_list = [text]
        else:
            # compose the spliter regex
            splitter = "(" + "|".join(map(re.escape, self.special_match_order)) + ")"

            spec_split_list = re.split(splitter, text)

        # # split the pretoken by special tokens first
        # spec_split_list : list[str|int] = [text]

        # for spec in self.special_match_order:
        #     i = 0
        #     while i < len(spec_split_list):
        #         token = spec_split_list[i]

        #         if isinstance(token, int):
        #             i += 1
        #             continue

        #         # check if the token contains any special tokens, and split it
        #         # NOTICE: the order for matching is defined by the order of special tokens
        #         idx = token.find(spec)
        #         if idx != -1:
        #             spec_split_list[i] = self.special_token_ids[spec]
        #             # suffix
        #             if idx + len(spec) < len(token):
        #                 spec_split_list.insert(i+1, token[idx + len(spec):])
        #             # prefix
        #             if idx > 0:
        #                 spec_split_list.insert(i, token[:idx])
        #                 i += 1

        #         i += 1


        res_ids : list[int] = []


        for textpiece in tqdm(spec_split_list, desc='Encoding'):
            if textpiece in self.special_token_ids:
                res_ids.append(self.special_token_ids[textpiece])
            else:
                        
                # pretokenize the text
                pretokens = pretokenize(textpiece)

                # encode the pretokens
                pretoken_encodings = encode_rewrite(pretokens, self.vocab_bytes2id, self.merge_rules)
                
                res_ids += [token for pretoken in pretoken_encodings for token in pretoken]

        return res_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Encode an iterable of strings into an iterator of token ids.
        '''
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        '''
        Decode a list of token ids into a string.
        '''
        bytestring = bytes()
        for token in ids:
            bytestring += self.vocab[token]
        return bytestring.decode('utf-8', errors='replace')


def encode_one_pretoken(pretoken: str, vocab_bytes2id: dict[bytes, int], merge_rules: dict[tuple[bytes, bytes], int]) -> list[int]:
    '''
    Encode a single pretoken into a list of token ids.
    '''

    # encode the pretoken
    tokens = list(map(lambda b: bytes([b]), pretoken.encode('utf-8')))

    # apply the merges
    while True:
        len_tokens = len(tokens)
    
        merge_id = -1
        replace_idx = -1

        for i in range(len_tokens - 1):
            pair = (tokens[i], tokens[i + 1])
            current_id = merge_rules.get(pair, -1)
            if current_id != -1 and (current_id < merge_id or merge_id == -1):
                merge_id = current_id
                replace_idx = i

        if merge_id != -1:
            tokens[replace_idx: replace_idx + 2] = [tokens[replace_idx] + tokens[replace_idx + 1]]

        else:
            break

    # convert the tokens to ids
    return [vocab_bytes2id[token] for token in tokens]

def encode_rewrite(pretokens: list[str], vocab_bytes2id: dict[bytes, int], merge_rules: dict[tuple[bytes, bytes], int]) -> list[list[int]]:
    '''
    Encode a list of pretokens into a list of token ids.
    '''
    # encode the pretokens
    cached_encoder = lru_cache(maxsize=50000)(lambda pretoken: encode_one_pretoken(pretoken, vocab_bytes2id, merge_rules))
    return list(map(cached_encoder, pretokens))