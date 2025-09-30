"""the algorithimatical implementation of bpe
get_stats(ids)--> will count adjacent pairs
merges(ids, pair, idx)--> replaces all occurances of that pair with a new id"""
from base import Tokenizer, get_stats, merge
class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    # training--> build merges, vocab from raw text)
    def train(self, text, vocab_size, verbose=False):
        # making sure to have atleast 256 byte tokens
        # also, the number f learned tokens. merges is vocab_size - 256
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        # input text preprocessing, converts the wholeinput text to UTF-8    buytes and the the ids will be a sequence of integers
        #each in 0...255
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        # if text is empty or a single byte, there are no pairs to merge
        if num_merges > 0 and len(ids) < 2:
            if verbose:
                print("no pairs to merge, input too short!!!")
                # set empty merges and vocab
                self.merges = {}
                self.vocab = {idx: bytes([idx]) for idx in range(256)}
                return
        # local structures
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # execution of classic BPE steps, [counting all adjacent pairs in the current sequence, then choose one wit max count]
        for i in range(num_merges):
            # counting the number of times every consecutive pair appears
            stats = get_stats(ids)
            if not stats:
                if verbose:
                    print(f"stopping early at merge{i+1}/{num_merges} : no pair left")
                    break
            # will find the pair with the highest count
            pair = max(stats, key=stats.get)
            # minting of a new token, assign it the next availab;e id
            idx = 256 + i
            # replacing the occurences of pair in idss with idx
            ids = merge(ids, pair, idx)
            # every set of (p0, p1) occurance will collapse into idx
            # the i+=2 constraint will make it possible
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # vocab[pair[0]]--> bytes for left child
            # vocab[pair[1]]--> bytes for right child
            #concatenate-->  + will work
            # builds the byte sequence for the new token by concatenating the childres bytes
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        """
        decode() takes token IDs → looks up their
        byte sequences in the vocab → concatenates them into one byte
        stream → decodes that stream into a Python string.
        id's -> text, list of token ids (int)
        map->join->utf-8 decode
        What would happen-
        self.vocab[65]   = b"A"
        self.vocab[66]   = b"B"
        self.vocab[67]   = b"C"
        self.vocab[256]  = b"AB"
        >>decode([256, 67])
        - Lookup: [self.vocab[256], self.vocab[67]] = [b"AB", b"C"]
        - b"AB" + b"C" = b"ABC"
        - UTF-8 decode "ABC"
        """
        # given we had--> ids (list integers)
        #look up each id's byte sequence in self.vocab , joint into one bytes
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        #each idx is mapped to its byts sequence stored in self.vocab
        # decode from UTF-* to get a string
        ##join all byte chunks to a continuous byte obj
        text = text_bytes.decode("utf-8", errors="replace")
        # Interprets the full byte sequence as UTF-8 and converts it to a Python string.
        # errors="replace" ensures that if there are invalid byte sequences, they are replaced by "�" instead of raising an error.

        return text

    def encode(self, text):
        """What will take place
        during training- (65,66) -> 256 i.e. "A""B"-> "AB"
        and (66,67) -> 257 are the merges discovered during training
        example text-> "ABBC"
        - starting with ids [65,66,66,67]
        - get_stats-> will find pairs like, (65,66),(66,66),(66,67)
        - will pick the pair with lowest learned id among those present
        - (65,66)--256, (66,67)--257 is 257 hence select (65,66)--256
        - merge: (65,66) -> ids = [256, 66, 67]
        now recompute pairs which will be --> (256,66) and (66,67)
            - now only , (66,67) is learnt
            - (66,67) -> 257
            - merge: (256,257)
            = no more learned pairs, hence stop
            result: ids=[256, 257]
        _____________________________________________________________________
        decoding:
                [256, 257]
               > vocab[256] = b"AB", vocab[257] = b"BC"
               > join b"ABBC" -> UTF-8 decode -> "ABBC"

        """
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pairs with lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # what happened?-> pair with smallest merge id was selected, self.merges.get(p, inf) pairs that were never learned get inf, hence will not be chosen
            # merge order enforced during merge
            # if there are no more merges available, the key will
            # result in and inf for every single pair, and the min will be the
            # first pair in the list
            if pair not in self.merges:
                break # IF no pair in in the current sequence appears in self.merges, no possible merges
            # merging the best pairs(with lowest merge index  )
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids






    

