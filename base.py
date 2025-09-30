import unicodedata

def get_stats(ids, counts=None):
    """list of integers--> return a dictionary of counts of consective pairs"""
    counts = {} if counts is None else counts #counts-> dictionary to keep track on how many times each pair of tokens appear
    #creating consecutive pairs(ids[i], ids[i+1]) and countiong how often they occur
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
        #zip(ids, ids[1:]) creates consecutive pairs od (ids[i], ids[i+1])
        #count how often each adjacent pair appears
        #if pair exist in dict-> return the cont of that pair and increment else return 0 and add 1 as new pair added
    return counts

#merging a specific pair

def merge(ids, pair, idx):
    """in the list of ids, replaces all consecutive occurances of pairs
    with the new token idx for example (1,2) two tuple to collapse, idx new token id"""
    newids = []  # to store new ids
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            # i+=2--> as we found a match and hence we must consume both i and i+1 tokens
            i += 2
        else:
            newids.append(ids[i]) # copying the current token and move to nextstep
            i += 1
    return newids

#function to replace control charcters--> purpose scan s and replace any unicode control char with safe escaped form

def replace_control_characters(s: str) -> str:
    #collecting chars into a list for efficiency at lat joining it back in to one
    chars = []
    for ch in s :
        #consider this --> the function unicodedata.category(ch) gives 2 letter category code for the chars (Lu- uppercase, Ll for lower letter, Nd for dec number, Cc for cntrol chars like \n
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
            # normal char  append it to char list
        else:
            chars.append(f"\\u{ord(ch):04x}")
            # explanation to chars.append(f"\\u{ord(ch):04x}")
            # ord(ch) will give the unicode codepoint , an integer
            # :04x--> a 4 digit lowercase hex number padded with 0
            # example ord("\n") = 10, 04x--> 0x000a--> \u000a
            #also if \\u ensure that it becomes a literal text sequence ,not an actuall new line
    return "".join(chars)

#byte rendering
# why tokens stored in bytes--not all byte sequences are valid utf8, decoding them would fail, keeping them as raw bytes will preserve the
#exavt sequence on which model was trained
#the below function will convert bytes to str with escapess, A tokenizer designed this way is universal:Works on English, Chinese, emojis, source code, binary blobs.
def render_token(t: bytes) -> str:
    #t--> token represented as raw bytes, output a string version of token that is :
    #decoded into human readable form , also save escape from conrol chars
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# defining the base Tokenizer class

class Tokenizer():
    """Base Tokenizer class"""
    def __init__(self):
        # default vocab of size
        self.merges = {} #(int,int) -> int
        self.pattern = "" # str
        self.special_tokens = {} #{"str" : int}
        self.vocab = self._build_vocab() # int -> bytes
    def train(self, text, vocab_size, verbose=False):
        #subclass--> implements how merges / splits are trained
        raise NotImplementedError
    def encode(self, text):
        #implements--> string to ids
        raise NotImplementedError
    def decode(self, ids):
        #impelemts ids to strings
        raise NotImplementedError
    def _build_vocab(self):
        # vocab --> derived simply from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # for every integer idx from 0 to 255, bytes([idx]) creates a single byte sequence, hence the base vocab always contains all raw bytes
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
            #for every learned merges, (p0,p1)->idx define merged tokens
            #byte sequence as the concatenation of its byte sequence
            #this assumes that merges were added in the dict preserved the insertion
            #so vocab[p0]/vocab[p1] already exists when use
            #example-> suppose, merge(65,66):256 i.e. "A" + "B" new token 256
            #then vocab[256] = vocab[65] + vocab[66]== b"A" + B"B" = b"AB"
        for special, idx in self.special_tokens.items():
            #iterates over spl tokens-->like-->(<|endoftext|>)
            #keys= string--
            #values-- integer ID
            #example--
            #{"<|endoftext|>": 100257} --> vocab[100257] = b"<|endoftext|>"
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """saves--> 2files
        file_prefix.vocab - a human readable, .vocab for inspection purposes
        file_prefix.model - .model, used to load()"""
        model_file = file_prefix + ".model" # path to the model
        with open(model_file, 'w',encoding= "utf-8") as f:
            #writes into the file , writes the version also .
            f.write("minbpe v1\n")
            # writes the regex patterns used byt he tokenizer,if none then empty
            f.write(f"{self.pattern}\n")
            # writes the count of special tokens , so load function is aware of how many lines to parse

            f.write(f"{len(self.special_tokens)}\n")

            for special, idx in self.special_tokens.items():
                #write the count of special tokens , then each-"token_string token_id
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                #write each merge pair as two int per line
                #now we iterate over dict to get keys here (idx1,idx2)
                #note-- the new id assigned to this pair isnt saved as it
                #is reconstructedduring load() by counting after 256 in same order
                f.write(f"{idx1} {idx2}\n")

        vocab_file = file_prefix + ".vocab"
        # builds a map between new token id--> and its child pair
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        #an inverse map, new_id--> (left,right)
        with open(vocab_file, "w", encoding= "utf-8") as f:
            for idx, token in self.vocab.items():
               #iterate over-> every entry of the form-> id->bytes
                s = render_token(token)
                #THE render_token() will convert raw bytes to safe printable strings
                # open utf-8 file to display tokens
                # iterate all id --> bytes
                if idx in inverted_merges:
                    # if idx came from--> inverted_merges--> then printing it in [child0][child1]-->[merged]id
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                    #if this token was created by a merge, display it as [left][right]-> [merged] id
                else:
                    #incase of leaf token i.e. between 0...255 or special tokens
                    f.write(f"[{s}] {idx}\n")
                    #otherwise it is a l
                    # eaf (likely one of the 0-255 byte tokens or a spl)

    def load(self, model_file):
        """ loads the model file, will read tthe same format as stored using above func"""
        assert model_file.endswith(".model")
        #local scratch dicts
        merges = {}
        special_tokens = {}
        # start assigning merged token ids from 256
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            # validting the version header
            version = f.readline().strip()
            assert version == "minbpe v1"
            #read the pattern lune(regex or empty) ad store it
            self.pattern = f.readline().strip()
            # getting the number of special tokens--> then storing each string-id pair
            num_special = int(f.readline().strip()) # number of special tokens to expect
            for _ in range(num_special):
                # each line must be :token string  token id
                special, special_idx = f.readline().strip().split()
                # store a str-> int mapping special_tokens
                special_tokens[special] = int(special_idx)
                #for each remaining merge pair, assign a new id sequentially
                #starting at 256
                #this will mirrir the training order and exactly recreates the
                #learnt vocab without storing the new ids

            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
            # the _build_vocab() func call will regenerate the full mapping token_id->bytes
            #will start with 256 byte tokens
            #replaying merges
            #adding spl tokens as utf-8\













