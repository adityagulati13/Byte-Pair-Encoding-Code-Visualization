"""
unlike BasicTokenizer-
this will handle and optional regex spliting patterns.
and also handles optional special tokens.
"""
import regex as re
from base import Tokenizer, get_stats, merge

# the main GPT text split patterns
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
# refernce- https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
# these are the gpt tokenization regexes, from tiktoken, they aim to split text into matural
#"chunks" before bpe (more about this is discussed in the blog post associated to this )

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern= None):
        """- pattern : optional string to override the default split pattern
            - special_tokens : str -> int dictionary of special tokens
            example: {'<|endoftext|> : 100257}
            This __init__ sets up the regex tokenizer to
            (1) inherit the byte-level basics from Tokenizer,
             (2) decide which regex pattern to use for text splitting,
              (3) pre-compile it, and
              (4) prepare placeholders for optional special tokens.

        """
        super().__init__()

        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        # will call the parent tokenizer which setsup:self.merges, self.pattern, self.special_tokens, self.vocabs
        # calls the base constructor, sets up base 256 byte tokens, empy merges/specials,
        #and builds the default ccab
        # chooses a split pattern , gpt 4 by default, or a custom of passed
        #compiles the patterns once
        # initializes 2 special token maps
        # initializes inverse_special_tokens: id -> str
        # during decode
    #og
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)
        # will split the input string into regex chunks(words, numbers, etc)
        # this is the main step which differentiated basic onee with regex one
        ids = [list(ch.encode("utf-8")) for ch in text_chunks] # each chunk is encoded as UTF-8 bytes then
        # turned into a list
        # hence ids is a list of sequences once per chunk
        # not a single long sequence
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        # merges map and a local vocab starting at 256 base bytes
        for i in range(num_merges):
            # count the number of times every consecutive pairs appears
            stats = {}
            for chunk_ids in ids:
                # passing in the stats will update it inplace, adding up counts
                get_stats(chunk_ids, stats)
            if not stats:
                if verbose:
                    print(f"stopping early at merge {i+1}/{num_merges}: no pairs left")
                    break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # replace all occurences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # this will record the rule and define the new tokens bytes as concatenation
            # of its children
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        self.merges = merges
        self.vocab = vocab

    # # special tokens registeration
    def register_special_tokens(self, special_tokens):
        # input a spl token format-> dict of str->int
        # for example-> {"|endoftext|" : 100257}
        self.special_tokens = special_tokens # direct reference
        self.inverse_special_tokens = {v: k for k , v in special_tokens.items()}
        # built an inverse map, (ids->string) by swapping keys and values via comprehension
        # stores the mapping fo exact match special tokens and builds an inverse map for decode
        # these ids should not collide with learned is, hence placed with the largest learned id

    def decode(self, ids):
        """"given ids(list of integers), returns python string
                ids = [258, 32, 260, 50000]
                part_bytes = [b"ABB", b" ", b"ACB", b"<|eot|>"]
                text_bytes = b"ABB ACB<|eot|>"
                text = "ABB ACB<|eot|>"

        """
        part_bytes = [] # will collect byte sequences
        for idx in ids:
            # case-1 if the id is in the vocab dictionary,append it to part_bytes
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            # if the id corresponds to a spl token, look it in inversemap,then encode it to utf-8
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes =b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # returns the token ids
        # converts all bytes to integers in range-0 to 255

        ids = list(text_bytes)
        # find the pair with lowest merge index
        while len(ids) >= 2:
            stats = get_stats(ids) # current pair count
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) # min i.e. the lowest merge index, earliest learned rule
            # if there are no more merges available the key will result in an
            # in an inf for every single pair, and the min will
            # be the first pair in the list
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    def encode_ordinary(self,text):
        """encoding func that ignores any special tokens"""
        # split text into chunks of text , using the regex patterns
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks-> encoded seperately, then the results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids) # concatenate all resulting ids
            # this mimics the behaviour , applying bpe per token-like chunk, it improves merge stats

        return ids
    def encode(self, text, allowed_special ="none_raise"):
        """this func will handle spl tokens
         allowed_special: can be all none, none_raise or custom as well
         NOTE -: if none_raise, then an error is raised, if any spl_token is encountered in text[this is the default
         tiktoken behaviour]
         all: allow any registered special ,
         none: treats specials as normal text,
         none_raise: asssertion that no special string occurs, if any registered spl string appears, it should raise
         
         """
        special = None
        if allowed_special == "all":
            special = self.special_tokens
            # will allow every registered spl tokens , exactly same mapas registered earlier
        elif allowed_special == "none":
            special = {}
            # will treate no token as spl, everythig in text is encoded normally
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
            # will also no token as special, but will make sure that none of the registered spl token anywhere in the regstered token

        elif isinstance(allowed_special,set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        # allows only the "set" and any other registerd specials are ignored
        if not special:
            # if no special tokens, just is the ordinary encoding,
            # means skipping all special handling and a normal encode: RegexTokenizer, BasicTokenizer
            return self.encode_ordinary(text)

        # otherwise we have to be careful with potential special tokens in text
        # special tokens are handled by splitting text, based on the occurance of
        # any exact match with any of the special tokens, we can use re.split for this

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")" # ensures spl strings are not treated as regex
        special_chunks = re.split(special_pattern, text)
        # the special tokens will be in the result list
        # ex-: text--> "hi <|eot|> there", will chunk to ["hi ", "<|eot|>", " there"]

        ids = []
        # final look: [ ...ids for "Hello "..., 50000, ...ids for " world "..., 50001 ]
        # loop each part in the spl chunks, if it matches a registered special, directly append
        # its predifined ID, otherwise normal BPE on that string
        for part in special_chunks:
            if part in special:
                # spl token ,hence encode it seperately as a special case
                # if a part is exactly one of the allowed spl tokens, append it to pre assigned id
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it
                ids.extend(self.encode_ordinary(part))
        return ids
















































