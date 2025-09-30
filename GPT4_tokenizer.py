"""
Implementation of the gpt-4 tokenizers as a light weight wrwpper arround regextokenizer
"""
import tiktoken
from regex_tokenizer import RegexTokenizer
from base import render_token

def bpe(mergeable_ranks, token, max_rank):
    # this helper function will further help in reconstructing the merge forest
    # this will help to reconstruct the immediate childre of merged tok

    # mergeable_ranks>>> dict mapping( token byte sequence-> rank/id: rank means when the tok was created smaller is the eraliest)-->from tiktioken
    # token: a bytes object that represents one final merged token, which we are analyzing ex b"ABC"
    # max_rank: only consider merges with rank < max_Rank will reconstruct the state just before token
    parts = [bytes([b]) for b in token] # will split the tokens bytes into a list of single byte tokens {b"ABC"--> [b"A",b"B",b"C"]}
    # repeatedly tracking the best lowest rank adjacent pair
    while True:

        min_idx = None
        min_rank = None
        # enumerate over parts-> the 2 lists, without last ele and another with last ele
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                # checking if the current rank is less that the min_rank, store it
                # rank--> order in which that pair was learned(lower is earlier)
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                # stoping if there are no mergeable pair or the best has a rank more than > max_Rank
                break
        assert min_idx is not None # incase of no merges found, min_idx would remain None
        # parts = the slice of list of byte partss before the mrge position + actual merging, i.e the left and the right ele + the slice of everything after the merged pair
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx +1]] + parts[min_idx + 2:]
    return parts

def recover_merges(mergeable_ranks):
    # recover_merges --> reverse engineering
    # the merges are the byte sequences in their merged state
    # now we will recover the orignal pairing , which wis done by-
    #  a bpe training run on all the tokens in their order
    # references->
    # https://github.com/openai/tiktoken/issues/60
    merges = {}

    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            # case to skip b"A", b"B" ... the base vocabs
            continue
        pair = tuple(bpe(mergeable_ranks,token,max_rank=rank))
        # for a merged token, the will get the orignal parents using bpe
        assert len(pair) == 2 # check for valid pair
        # rank recovery
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        # rank of the children
        merges[(ix0,ix1)] = rank
        # of format: (left_child, right_child)
    return merges

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

GPT4_SPECIAL_TOKENS = {'<|endoftext|>': 100257,'<|fim_prefix|>': 100258,'<|fim_middle|>': 100259,'<|fim_suffix|>': 100260,'<|endofprompt|>': 100276}


class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        enc = tiktoken.get_encoding("cl100k_base") # getting the official tokenizer and its merges
        mergeable_ranks = enc._mergeable_ranks # merges are from gpt4 ,we will recover them
        self.merges = recover_merges(mergeable_ranks)
        # reconstruction of vocabs from merges
        vocab = {idx: bytes([idx]) for idx in range(256)} # initial vocab of 256 words
        # building new tokens according to merge rules

        for (p0,p1) , idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        # Note the tokens corresponding to individual bytes are permuted
        # in a differennt order, hence need to be dealt
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)} # will map each raw byte value to its shuffled id from tiktoken
        # BPE merges-> recovered from tiktokens are defined over token ids
        # i.e. if 65 is passed to loop bpe this would not match the gpt-4 merger rules
        # hence remaping of each raw byte is done
        self.inverse_byte_shuffle = {v: k for k,v in self.byte_shuffle.items()}
        # spl tokens register
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        # permute text_bytes
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)
        return ids

    def decode(self,ids):
        # combining the bytes for each token id into a long byte string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # unpermuting every single value back  to real worls values
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        # real bytes into a python string
        # This decode assumes every idx is either:
        # a base/merged token in self.vocab
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def save_vocab(self, vocab_file):
        # outputing the gpt-4 tokenizers for visualization purpose
        # in the exact same format as the base class would.
        vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256) }
        # for the 256 base tokens each bbyte id to be converted back into true raw byte
        # if raw 65("A") -> shiffled id 200, then inverse_byte_shuffle[200] -= 65 and hence vocab[200] == 65\
        # making sure that actual bytes are pronted rather than shuffled ones

        for (p0,p1) , idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
            # recreation of byte seq from merges tokens , i.e for each learned merge, foe each
            # learned merge (p0, p1) -> idx, set the parents pytes to the concatenation
            # of its childrens bytes like if :(200, 17) -> 500 and vocab[200]=b"A", vocab[17]=b"B", then vocab[500]=b"AB"
        # merging of shuffled bytes and writing to file
        # merging the shuffled bytes , write to file
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        # inverse mapping so that, inverse mapping of idx->pair(p0,p1) so when printing
        # a token ID ... we can tell if its a merged token or or leaf
        with open(vocab_file, "w", encoding ="utf-8") as f:
            # getting the outputfile, and iteration over all idx -> bytes entries in temp vocab
            for idx, token in vocab.items():
                s = render_token(token) # will decode bytes to a printable string , in a readable format
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx] # getting the inverted merges
                    # pretty print
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}]{idx}\n")
                else:
                    f.write(f"[{s}]{idx}\n")

# def decode(self, ids):
#     part_bytes = []
#     for idx in ids:
#         if idx in self.vocab:
#             part_bytes.append(self.vocab[idx])
#         elif hasattr(self, "inverse_special_tokens") and idx in self.inverse_special_tokens:
#             part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
#         else:
#             raise ValueError(f"invalid token id: {idx}")
#     text_bytes = b"".join(part_bytes)
#     text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
#     return text_bytes.decode("utf-8", errors="replace")







