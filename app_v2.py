

from typing import Dict, List, Tuple, Optional
import inspect
import io
# will  help in creating downloadable textfile in memory
import streamlit as st
from graphviz import Digraph # for graph (nodes and edges)

#  imports

from basic import BasicTokenizer
from base import get_stats, merge, render_token
from regex_tokenizer import RegexTokenizer, GPT4_SPLIT_PATTERN
from GPT4_tokenizer import GPT4Tokenizer


# ---------- Helpers ----------


# Simulate Basic.encode() and return snapshots after each merge that applied
def simulate_encode_steps_basic(text: str, merges: Dict[Tuple[int, int], int]) -> List[List[int]]:
    # will return list of id list like ->raw bytes first then ids after merge rule that actually applied
    from collections import Counter

    def _get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
        # will count adjacent pairs
        c = Counter()
        for a, b in zip(ids, ids[1:]):
            c[(a, b)] += 1
        return c

    def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        # will replace every occurance of pair with the new
        out, i, n = [], 0, len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id); i += 2
            else:
                out.append(ids[i]); i += 1
        return out

    ids = list(text.encode("utf-8"))
    snapshots = [ids.copy()]
    if not merges or len(ids) < 2:
        return snapshots

    for pair, new_id in sorted(merges.items(), key=lambda kv: kv[1]):
        stats = _get_stats(ids)
        if pair not in stats:
            continue
        ids = _merge(ids, pair, new_id)
        snapshots.append(ids.copy())
    # helper to give a merges list , i.e. pairing-> merging-> final tokens
    return snapshots


def _ascii_preview(b: bytes) -> str:
    # for the conversion-> b"A\x01B" → "A.B"
    return "".join(chr(x) if 32 <= x < 127 else "." for x in b)


def render_token_gpt4(b: bytes, mode: str = "smart") -> str:
    # example- b"Hi\n"
    if mode == "hex":
        # hex → "48 69 0a"
        return " ".join(f"{x:02x}" for x in b)
    elif mode == "ascii":
        # ascii → "Hi."
        return _ascii_preview(b)
    elif mode == "bytes":
        # bytes → "72 105 10"
        return " ".join(str(x) for x in b)
    else:
        return render_token(b)


def preview_table(
    ids_list: List[int], vocab_dict: Dict[int, bytes], renderer=None
) -> List[Dict[str, object]]:
    if renderer is None:
        renderer = render_token
    rows = []
    for i, tid in enumerate(ids_list):
        token_bytes = vocab_dict.get(tid, b"")
        rows.append({"pos": i, "id": tid, "token": renderer(token_bytes)})
    # rows will be-> rows: {"pos": i, "id": tid, "token": renderer(vocab_dict[tid])}
    return rows



# Partial-merge(for growing trees with a slider)
def merges_up_to_step(merges: Dict[Tuple[int, int], int], step: int) -> Dict[Tuple[int, int], int]:
    # keeps the merge with new_id, now if ids are from 256..258..300 anf step is 3 the cutoff will be at 258
    if step <= 0:
        return {}
    cutoff = 256 + (step - 1)
    return {pair: idx for pair, idx in merges.items() if idx <= cutoff}


def build_vocab_from_merges(filtered_merges: Dict[Tuple[int, int], int]) -> Dict[int, bytes]:
    # will replay merges in new_id order to reconstruct bytes from each new token, ex-> 256:(97,98)
    vocab = {i: bytes([i]) for i in range(256)}
    for (p0, p1), new_id in sorted(filtered_merges.items(), key=lambda kv: kv[1]):
        vocab[new_id] = vocab[p0] + vocab[p1]
    return vocab


def encode_with_merges_basic(text: str, filtered_merges: Dict[Tuple[int, int], int]) -> List[int]:
    # will encode based on the merges-> base bytes and earliest merges
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        if not stats:
            break
        pair = min(stats, key=lambda p: filtered_merges.get(p, float("inf")))
        if pair not in filtered_merges:
            break
        idx = filtered_merges[pair]
        ids = merge(ids, pair, idx)
    return ids


def encode_with_merges_regex(text: str, filtered_merges: Dict[Tuple[int, int], int], pattern: str) -> List[int]:
    # text splitting based on-> regex first, then as same as bpe
    import regex as re
    compiled = re.compile(pattern)
    chunks = re.findall(compiled, text)
    out: List[int] = []
    for ch in chunks:
        ids = list(ch.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            if not stats:
                break
            pair = min(stats, key=lambda p: filtered_merges.get(p, float("inf")))
            if pair not in filtered_merges:
                break
            idx = filtered_merges[pair]
            ids = merge(ids, pair, idx)
        out.extend(ids)
    return out


# ---- Graphviz (browser-rendered) trees/forests ----
def _label_bytes_for_tree(tid: int, vocab: Dict[int, bytes], show_mode: str) -> str:
    b = vocab.get(tid, b"")
    body = render_token(b)
    #id's + bytes = 257/n[aba] or bytes only-> "[aba]"
    return f"{tid}\\n[{body}]" if show_mode == "id + bytes" else f"[{body}]"


def show_token_tree(
    token_id: int,
    filtered_merges: Dict[Tuple[int, int], int],
    vocab: Dict[int, bytes],
    show_mode: str = "id + bytes",
    layout: str = "Left-right",
):
    # recursively expanding tokn ids to token children
    merged_to_pair = {new_id: pair for pair, new_id in filtered_merges.items()}
    g = Digraph(format="svg")
    g.attr(rankdir="LR" if layout == "Left-right" else "TB")
    g.attr("node", shape="box", fontsize="10", margin="0.05,0.03")
    g.attr("edge", arrowsize="0.6", penwidth="0.7")

    def _build(tid, counter=[0]):
        nid = f"n{counter[0]}"; counter[0] += 1
        g.node(nid, label=_label_bytes_for_tree(tid, vocab, show_mode))
        if tid in merged_to_pair:
            l, r = merged_to_pair[tid]
            ln = _build(l, counter)
            rn = _build(r, counter)
            g.edge(ln, nid); g.edge(rn, nid)
        return nid

    _build(token_id)

    st.graphviz_chart(g.source, use_container_width=True)

# will draw a forest , single tiny token tree per token in current encoded sequence

# for encoded id [256,257] will draw 2 roots (token 256 tree and 257 tree)

def show_merge_forest(

    ids: List[int],
    filtered_merges: Dict[Tuple[int, int], int],
    vocab: Dict[int, bytes],
    show_mode: str = "id + bytes",
    layout: str = "Left-right",
):

    merged_to_pair = {new_id: pair for pair, new_id in filtered_merges.items()}
    g = Digraph(format="svg")
    g.attr(rankdir="LR" if layout == "Left-right" else "TB")
    g.attr("node", shape="box", fontsize="9", margin="0.04,0.02")
    g.attr("edge", arrowsize="0.5", penwidth="0.6")

    counter = [0]
    seen_nodes: Dict[int, str] = {}

    def add_node(tid: int) -> str:
        if tid in seen_nodes:
            return seen_nodes[tid]
        nid = f"n{counter[0]}"; counter[0] += 1
        g.node(nid, label=_label_bytes_for_tree(tid, vocab, show_mode))
        seen_nodes[tid] = nid
        if tid in merged_to_pair:
            l, r = merged_to_pair[tid]
            ln = add_node(l); rn = add_node(r)
            g.edge(ln, nid); g.edge(rn, nid)
        return nid

    for t in ids:
        add_node(t)

    st.graphviz_chart(g.source, use_container_width=True)


# Trainers
def train_with_logs_basic(train_text: str, vocab_size: int):
    # will print the logs for while training, ex- training script, a and n
    # has 2 occurance, (97,110) new id will be 256

    tok = BasicTokenizer()
    assert vocab_size >= 256
    num_merges_requested = vocab_size - 256

    ids = list(train_text.encode("utf-8"))
    merges: Dict[Tuple[int, int], int] = {}
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    logs, pretty = [], []

    if len(ids) < 2 or num_merges_requested <= 0:
        tok.merges = {}
        tok.vocab = vocab
        return tok, logs, pretty

    for i in range(num_merges_requested):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        new_id = 256 + i
        ids = merge(ids, pair, new_id)

        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        cnt = stats[pair]
        token_repr = repr(vocab[new_id])

        pretty_line = f"merge {i+1}/{num_merges_requested}: {pair} -> {new_id} ({token_repr}) had {cnt} occurrences"
        pretty.append(pretty_line)
        logs.append({
            "step": i + 1, "total": num_merges_requested,
            "pair": pair, "new_id": new_id, "token": token_repr, "count": cnt,
        })

    tok.merges = merges
    tok.vocab = vocab
    return tok, logs, pretty
# the final output will look like this-merge 1/N: (97, 110) -> 256 (b'an') had 2 occurrences


def train_with_logs_regex(train_text: str, vocab_size: int, pattern: str):
    tok = RegexTokenizer(pattern=pattern)
    assert vocab_size >= 256
    num_merges_requested = vocab_size - 256

    import regex as re
    compiled = re.compile(pattern)
    text_chunks = re.findall(compiled, train_text)
    ids_list = [list(ch.encode("utf-8")) for ch in text_chunks]

    merges: Dict[Tuple[int, int], int] = {}
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    logs, pretty = [], []

    if num_merges_requested <= 0 or len(ids_list) == 0:
        tok.merges = {}
        tok.vocab = vocab
        return tok, logs, pretty

    for i in range(num_merges_requested):
        stats: Dict[Tuple[int, int], int] = {}
        for chunk in ids_list:
            get_stats(chunk, stats)
        if not stats:
            break

        pair = max(stats, key=stats.get)
        new_id = 256 + i
        ids_list = [merge(chunk, pair, new_id) for chunk in ids_list]

        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        cnt = stats[pair]
        token_repr = repr(vocab[new_id])

        pretty_line = f"merge {i+1}/{num_merges_requested}: {pair} -> {new_id} ({token_repr}) had {cnt} occurrences"
        pretty.append(pretty_line)
        logs.append({
            "step": i + 1, "total": num_merges_requested,
            "pair": pair, "new_id": new_id, "token": token_repr, "count": cnt,
        })

    tok.merges = merges
    tok.vocab = vocab
    return tok, logs, pretty


#  TRAINING VISUALIZATION HELPERS
def compute_training_steps_basic(train_text: str, vocab_size: int, topk: int = 10):
    # will simulate training and report winner pair, count

    ids = list(train_text.encode("utf-8"))
    steps = []
    if vocab_size < 257 or len(ids) < 2:
        return steps

    num_merges_requested = vocab_size - 256
    merges: Dict[Tuple[int, int], int] = {}
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    # ids.. before and after
    for i in range(num_merges_requested):
        stats = get_stats(ids)
        if not stats:
            break
        sorted_pairs = sorted(stats.items(), key=lambda kv: (-kv[1], kv[0]))[:topk]
        pair = sorted_pairs[0][0]
        cnt = stats[pair]
        new_id = 256 + i
        ids_before = ids.copy()
        ids_after = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        ids = ids_after
        #pretty print merge line
        pretty_line = f"merge {i+1}/{num_merges_requested}: {pair} -> {new_id} ({repr(vocab[new_id])}) had {cnt} occurrences"
        steps.append({
            "step": i + 1, "winner_pair": pair, "winner_count": cnt,
            "new_id": new_id, "top_pairs": sorted_pairs,
            "ids_before": ids_before, "ids_after": ids_after, "pretty": pretty_line,
        })
    return steps


def compute_training_steps_regex(train_text: str, vocab_size: int, pattern: str, topk: int = 10):
    import regex as re
    compiled = re.compile(pattern)
    chunks = re.findall(compiled, train_text)
    ids_list = [list(ch.encode("utf-8")) for ch in chunks]
    steps = []
    if vocab_size < 257 or len(ids_list) == 0:
        return steps

    num_merges_requested = vocab_size - 256
    merges: Dict[Tuple[int, int], int] = {}
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    for i in range(num_merges_requested):
        stats: Dict[Tuple[int, int], int] = {}
        for chunk in ids_list:
            get_stats(chunk, stats)
        if not stats:
            break
        sorted_pairs = sorted(stats.items(), key=lambda kv: (-kv[1], kv[0]))[:topk]
        pair = sorted_pairs[0][0]
        cnt = stats[pair]
        new_id = 256 + i

        before_concat = []
        for ch in ids_list: before_concat.extend(ch)
        ids_list = [merge(ch, pair, new_id) for ch in ids_list]
        after_concat = []
        for ch in ids_list: after_concat.extend(ch)

        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

        pretty_line = f"merge {i+1}/{num_merges_requested}: {pair} -> {new_id} ({repr(vocab[new_id])}) had {cnt} occurrences"
        steps.append({
            "step": i + 1, "winner_pair": pair, "winner_count": cnt,
            "new_id": new_id, "top_pairs": sorted_pairs,
            "ids_before": before_concat, "ids_after": after_concat, "pretty": pretty_line,
        })
    return steps


# ---------- Streamlit page ----------
st.set_page_config(page_title="BPE Visualizer", layout="wide")
st.title("🧠 Byte Pair Encoding — Visual Learning tool")

with st.sidebar:
    st.header("Settings")

    # --- big demo corpus string (unchanged) ---
    demo_corpus = (
        "Once upon a time, in a small village surrounded by rolling hills and dense forests, there lived a storyteller who was known by everyone.\n"
        "Each evening, as the sun set and the sky turned shades of orange and purple, people from the village would gather beneath a giant tree.\n"
        "Children sat cross-legged at the front, while the elders leaned on their walking sticks and listened patiently.\n"
        "The storyteller spoke of heroes and dragons, of magical lands and forgotten kingdoms.\n"
        "His voice carried not only words but also imagination that painted vivid pictures in the minds of his listeners.\n"
        "\n"
        "He told tales of a brave knight who fought for justice, of a farmer who discovered treasure beneath his field, of a young girl who befriended the animals of the forest, and of travelers who crossed deserts in search of hidden cities.\n"
        "Sometimes the stories were happy, filled with laughter and joy.\n"
        "At other times, they carried lessons, reminding the villagers of kindness, courage, and honesty.\n"
        "The storyteller repeated many tales, yet each time he told them, they felt new, as if shaped fresh by the wind of creativity.\n"
        "\n"
        "The children especially loved the story of a dragon that lived atop the highest mountain.\n"
        "The dragon was not cruel or wicked, but lonely, waiting for a friend.\n"
        "The brave knight ventured to meet the dragon, not with sword but with an open heart.\n"
        "Their friendship grew strong, teaching everyone that fear can often be replaced by understanding.\n"
        "This story became a favorite, told again and again, until even the youngest children could recite parts of it.\n"
        "\n"
        "Beyond the village, life continued with its daily rhythm.\n"
        "Farmers worked in their fields, sowing seeds and harvesting grain.\n"
        "Merchants carried goods to the market, where the air buzzed with voices bargaining and laughing.\n"
        "Blacksmiths hammered metal, bakers kneaded dough, and potters shaped clay into vessels.\n"
        "The storyteller observed these details carefully, weaving them into his tales.\n"
        "By adding pieces of real life, he made each story more believable and close to the hearts of his listeners.\n"
        "\n"
        "As years passed, the storyteller grew older.\n"
        "His hair turned silver, and his voice became soft, but the sparkle in his eyes never faded.\n"
        "He taught young apprentices the art of weaving stories, passing down the gift of imagination.\n"
        "Even when he could no longer walk to the great tree, children came to his home, filling the room with excitement and laughter.\n"
        "They listened to his voice, now fragile but still magical, and carried his tales to others.\n"
        "\n"
        "One day, the storyteller spoke his final story.\n"
        "It was a tale of beginnings and endings, of seasons turning, of rivers flowing endlessly to the sea.\n"
        "When he finished, the village remained silent for a long time.\n"
        "Then, one child whispered, “We will carry your stories forever.”\n"
        "And indeed, they did.\n"
        "The tales became part of the village, told from one generation to another, shaping not only entertainment but also wisdom and tradition.\n"
        "In that way, the storyteller lived on, not as a person, but as the spirit of storytelling itself.\n"
        "\n"
        "The legacy of stories is powerful.\n"
        "Words, when arranged with care, can inspire, heal, and transform.\n"
        "They are bridges between hearts, across time and space.\n"
        "Stories explain mysteries, capture history, and spark dreams.\n"
        "They remind us that even in the simplest of places — a quiet village, a humble home, a circle beneath a tree — imagination has the power to light the world.\n"
        "The storyteller of the small village proved this truth, and so do countless others who continue the tradition.\n"
    )

    default_train = (
        "Hello , Enter your your training corpus in here....\n"
        "or load default training corpus\n"
    )

    # --- Initialize state BEFORE widgets are created
    if "train_text" not in st.session_state:
        st.session_state.train_text = default_train
    if "input_text" not in st.session_state:
        st.session_state.input_text = "enter the sentence to encode"

    # --- Button callback updates session_state safely
    def _load_demo():
        st.session_state.train_text = demo_corpus

    # Place the button BEFORE the textarea (important!)
    st.button("📚 Load demo corpus", on_click=_load_demo)

    # Textarea bound to session_state key
    st.text_area(
        "Training text (for Basic/Regex)",
        key="train_text",
        height=240
    )

    # Vocab size slider
    vocab_size = st.slider(
        "Vocab size",
        min_value=256, max_value=1024, value=300, step=1,
        help="Total vocab = 256 base bytes + learned merges"
    )

    # Input text (also bound to session_state)
    st.text_input(
        "Text to encode/decode",
        key="input_text"
    )

    retrain = st.button("🔁 Retrain tokenizers")


# cache / persist trained tokenizers in session_state
def _cache_key(prefix: str, train: str, vs: int, cli_mode: bool) -> str:
    return f"{prefix}:{len(train)}:{vs}:{cli_mode}:{train[:128]!r}"


def get_or_train_basic_with_log(train: str, vs: int, cli_mode: bool, force: bool = False):
    key = _cache_key("basic_log", train, vs, cli_mode)
    kt, kl, kp = key + "::tok", key + "::log", key + "::pretty"
    if force or kt not in st.session_state:
        tok, logs, pretty = train_with_logs_basic(train, vs)
        st.session_state[kt] = tok
        st.session_state[kl] = logs
        st.session_state[kp] = pretty
    return st.session_state[kt], st.session_state[kl], st.session_state[kp], key


def get_or_train_regex_with_log(train: str, vs: int, cli_mode: bool, force: bool = False):
    key = _cache_key("regex_log", train, vs, cli_mode)
    kt, kl, kp = key + "::tok", key + "::log", key + "::pretty"
    if force or kt not in st.session_state:
        tok, logs, pretty = train_with_logs_regex(train, vs, GPT4_SPLIT_PATTERN)
        st.session_state[kt] = tok
        st.session_state[kl] = logs
        st.session_state[kp] = pretty
    return st.session_state[kt], st.session_state[kl], st.session_state[kp], key


def get_gpt4_tok() -> Optional[GPT4Tokenizer]:
    key = "gpt4_tok"
    if key not in st.session_state:
        try:
            st.session_state[key] = GPT4Tokenizer()
        except Exception as e:
            st.session_state[key] = None
            st.warning(f"GPT4Tokenizer unavailable: {e}")
    return st.session_state[key]


# Normalize training & input for CLI match
train_effective = st.session_state["train_text"]
input_effective = st.session_state["input_text"]


# Train / fetch
basic_tok, basic_logs, basic_pretty, basic_key = get_or_train_basic_with_log(
    train_effective, vocab_size, False, force=retrain
)
regex_tok, regex_logs, regex_pretty, regex_key = get_or_train_regex_with_log(
    train_effective, vocab_size, False, force=retrain
)

gpt4_tok = get_gpt4_tok()




# ---------- Tabs ----------
perf_tab, train_vis_tab, learn_tab, compare_tab, trees_tab, log_tab, compare_logs_tab = st.tabs([
    "⚡ Performance",
    "🔬 Train BPE (How merges are computed)",
    "🎬 Learn BPE (Animated apply)",
    "🆚 Compare Tokenizers",
    "🌳 Token Trees",
    "📜 Merge Log",
    "📑 Compare Logs",
])

with perf_tab:
    st.subheader("Quick Benchmark (time + peak memory)")
    import tracemalloc, time

    def time_mem(fn, *args, **kw):
        tracemalloc.start()
        t0 = time.perf_counter()
        out = fn(*args, **kw)
        dt = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return out, dt, peak / (1024 * 1024)

    _, t_train_basic, m_train_basic = time_mem(basic_tok.train, train_effective, vocab_size)
    _, t_enc_basic, m_enc_basic = time_mem(basic_tok.encode, input_effective)
    _, t_train_regex, m_train_regex = time_mem(regex_tok.train, train_effective, vocab_size)
    _, t_enc_regex, m_enc_regex = time_mem(regex_tok.encode, input_effective)

    data = [
        {"Tokenizer": "Basic", "Train (s)": f"{t_train_basic:.3f}", "Train (MB)": f"{m_train_basic:.1f}",
         "Encode (s)": f"{t_enc_basic:.3f}", "Encode (MB)": f"{m_enc_basic:.1f}"},
        {"Tokenizer": "Regex", "Train (s)": f"{t_train_regex:.3f}", "Train (MB)": f"{m_train_regex:.1f}",
         "Encode (s)": f"{t_enc_regex:.3f}", "Encode (MB)": f"{m_enc_regex:.1f}"},
    ]
    st.table(data)
    st.caption("Measured on current corpus & input. Shows rough timing and peak memory.")


with train_vis_tab:
    st.subheader("How merges are computed during training")
    mode = st.radio("Trainer", ["Basic", "Regex"], horizontal=True)
    topk = st.slider("Show top-K pairs per step", 3, 25, 10)

    if mode == "Basic":
        train_steps = compute_training_steps_basic(train_effective, vocab_size, topk=topk)
    else:
        train_steps = compute_training_steps_regex(train_effective, vocab_size, GPT4_SPLIT_PATTERN, topk=topk)

    if not train_steps:
        st.info("Not enough text or merges to visualize. Increase training text or vocab size.")
    else:
        slider_key = "train_step_basic" if mode == "Basic" else "train_step_regex"
        max_step = len(train_steps)
        step_idx = st.slider("Training step", 1, max_step, 1, key=slider_key)
        step = train_steps[step_idx - 1]

        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown("**Top-K pairs this step (counts)**")
            rows = [{"rank": i + 1, "pair": p, "count": c} for i, (p, c) in enumerate(step["top_pairs"])]
            st.dataframe(rows, use_container_width=True)
            st.success(f"Winner: {step['winner_pair']}  (count={step['winner_count']}) → new_id={step['new_id']}")
            st.code(step["pretty"], language="text")

        with c2:
            st.markdown("**Token stream before → after this step**")
            st.code(f"ids_before ({len(step['ids_before'])}): {step['ids_before']}", language="text")
            st.code(f"ids_after  ({len(step['ids_after'])}):  {step['ids_after']}", language="text")

        st.caption("This page shows the training loop (count pairs → pick most frequent → merge). "
                   "The '🎬 Learn BPE' page animates the apply/encode phase on your input text.")


with learn_tab:
    st.subheader("Animated Apply Phase — BasicTokenizer")
    st.write("**Number of merges learned (Basic):**", len(basic_tok.merges))

    steps = simulate_encode_steps_basic(input_effective, basic_tok.merges)
    total_steps = max(1, len(steps))

    if "anim_step" not in st.session_state:
        st.session_state["anim_step"] = 0
    st.session_state["anim_step"] = min(st.session_state["anim_step"], total_steps - 1)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Text:**")
        st.code(input_effective)

        st.write("**Number of merge steps:**", total_steps - 1)

        if len(steps) > 1:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("⏮ Start"):
                    st.session_state["anim_step"] = 0; st.rerun()
            with c2:
                if st.button("Step ▶"):
                    if st.session_state["anim_step"] < total_steps - 1:
                        st.session_state["anim_step"] += 1; st.rerun()
            with c3:
                if st.button("⏭ End"):
                    st.session_state["anim_step"] = total_steps - 1; st.rerun()

            if "anim_slider" not in st.session_state:
                st.session_state["anim_slider"] = st.session_state["anim_step"]

            slider_val = st.slider("Animation step", 0, total_steps - 1,
                                   value=st.session_state["anim_step"], key="anim_slider")
            if slider_val != st.session_state["anim_step"]:
                st.session_state["anim_step"] = slider_val; st.rerun()
        else:
            st.info("No merges available. Try more training text or smaller vocab.")
            st.session_state["anim_step"] = 0

    with col2:
        current_ids = steps[st.session_state["anim_step"]]
        st.markdown("**Current token IDs at this step:**")
        st.write(current_ids)
        st.markdown("**Preview (first 30 tokens):**")
        try:
            st.dataframe(preview_table(current_ids, basic_tok.vocab), use_container_width=True)
        except Exception as e:
            st.info(f"Preview unavailable: {e}")

    st.divider()
    st.markdown("**Final encoding/decoding (BasicTokenizer)**")
    basic_ids_final = basic_tok.encode(input_effective)
    st.write({
        "ids": basic_ids_final,
        "num_tokens": len(basic_ids_final),
        "num_bytes": len(input_effective.encode('utf-8')),
        "compression (bytes/tokens)": round(len(input_effective.encode('utf-8')) / max(1, len(basic_ids_final)), 3),
    })
    st.write("Decoded:", basic_tok.decode(basic_ids_final))


with compare_tab:
    st.subheader("Compare Tokenizers: Basic vs Regex vs GPT-4")
    st.write("**Merges learned (Basic):**", len(basic_tok.merges))
    st.write("**Merges learned (Regex):**", len(regex_tok.merges))

    basic_ids = basic_tok.encode(input_effective)
    regex_ids = regex_tok.encode(input_effective)
    gpt4_ids = gpt4_tok.encode(input_effective) if gpt4_tok is not None else []

    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("### BasicTokenizer")
        st.write({
            "ids": basic_ids, "tokens": len(basic_ids),
            "bytes": len(input_effective.encode('utf-8')),
            "bytes/tok": round(len(input_effective.encode('utf-8')) / max(1, len(basic_ids)), 3),
        })
        st.markdown("**Full token list (pos, id, token)**")
        st.dataframe(preview_table(basic_ids, basic_tok.vocab), use_container_width=True)
        st.caption("IDs map to bytes via `basic_tok.vocab[id]` (rendered).")

    with colB:
        st.markdown("### RegexTokenizer (GPT-style splits)")
        st.write({
            "ids": regex_ids, "tokens": len(regex_ids),
            "bytes": len(input_effective.encode('utf-8')),
            "bytes/tok": round(len(input_effective.encode('utf-8')) / max(1, len(regex_ids)), 3),
        })
        st.markdown("**Full token list (pos, id, token)**")
        st.dataframe(preview_table(regex_ids, regex_tok.vocab), use_container_width=True)
        st.caption("Regex pre-splits text, then BPE within chunks.")

    with colC:
        st.markdown("### GPT4Tokenizer (tiktoken)")
        if gpt4_tok is not None:
            st.write({
                "ids": gpt4_ids, "tokens": len(gpt4_ids),
                "bytes": len(input_effective.encode('utf-8')),
                "bytes/tok": round(len(input_effective.encode('utf-8')) / max(1, len(gpt4_ids)), 3),
            })

            gpt4_view = st.selectbox(
                "Token view (GPT-4)", options=["ascii", "hex", "bytes"], index=0,
                help="How to display each token's bytes"
            )
            st.markdown("**Full token list (pos, id, token)**")
            st.dataframe(
                preview_table(gpt4_ids, gpt4_tok.vocab, renderer=lambda b: render_token_gpt4(b, gpt4_view)),
                use_container_width=True
            )
            st.caption("These are tiktoken’s shuffled-byte tokens; choose ASCII/hex/bytes view.")
        else:
            st.error("GPT4Tokenizer unavailable (see warning above).")


with trees_tab:
    st.subheader("Merge Trees / Forests per Token")
    st.caption("Pick a token and see how it was built from byte merges, or view the whole forest.")

    which_tok = st.radio("Choose tokenizer", ["Basic", "Regex"], index=0, horizontal=True)

    total_merge_steps = len(basic_tok.merges) if which_tok == "Basic" else len(regex_tok.merges)
    tree_step = st.slider("Show merges up to step", 0, total_merge_steps, min(total_merge_steps, 10),
                          key=f"tree_step_{which_tok}")

    if which_tok == "Basic":
        filtered = merges_up_to_step(basic_tok.merges, tree_step)
        vocab_partial = build_vocab_from_merges(filtered)
        ids_partial = encode_with_merges_basic(input_effective, filtered)
    else:
        filtered = merges_up_to_step(regex_tok.merges, tree_step)
        vocab_partial = build_vocab_from_merges(filtered)
        ids_partial = encode_with_merges_regex(input_effective, filtered, GPT4_SPLIT_PATTERN)

    left, right = st.columns([1, 2])

    with left:
        st.markdown(f"**Merges included:** {len(filtered)}  |  **Tokens at this step:** {len(ids_partial)}")
        display_mode = st.radio("Display", ["Whole forest (compact)"], index=0)
        rankdir = st.radio("Layout", ["Top-down", "Left-right"], index=1, horizontal=True)
        show_mode = st.radio("Labels", ["id + bytes", "bytes only"], index=0, horizontal=True)



    with right:
        if display_mode.startswith("Per-token"):
            if len(ids_partial) == 0:
                st.info("No tokens to display at this step.")
            else:
                token_id = ids_partial[pos]
                bseq = vocab_partial.get(token_id, b"")
                st.markdown(f"**Selected token** at pos **{pos}** → id **{token_id}**  \nBytes: `{render_token(bseq)}`")
                show_token_tree(
                    token_id=token_id,
                    filtered_merges=filtered,
                    vocab=vocab_partial,
                    show_mode=show_mode,
                    layout=rankdir,
                )
                with st.expander("path", expanded=False):
                    merged_to_pair = {new_id: pair for pair, new_id in filtered.items()}
                    stack, path = [token_id], []
                    while stack:
                        t = stack.pop()
                        if t in merged_to_pair:
                            l, r = merged_to_pair[t]
                            path.append((t, "←", l, "+", r))
                            stack.extend([r, l])
                        else:
                            path.append((t, "←", "byte"))
                    st.code("\n".join(str(x) for x in path), language="text")
        else:
            show_merge_forest(
                ids=ids_partial,
                filtered_merges=filtered,
                vocab=vocab_partial,
                show_mode=show_mode,
                layout=rankdir,
            )

        with st.expander("Full token list (pos, id, token bytes)"):
            st.dataframe(preview_table(ids_partial, vocab_partial), use_container_width=True)


with log_tab:
    st.subheader("📜 Merge Log ")
    which_log = st.radio("Select tokenizer", ["Basic", "Regex"], index=0, horizontal=True)

    if which_log == "Basic":
        logs, pretty_lines, tok = basic_logs, basic_pretty, basic_tok
        fname = "merge_log_basic.txt"
    else:
        logs, pretty_lines, tok = regex_logs, regex_pretty, regex_tok
        fname = "merge_log_regex.txt"

    if logs:
        pretty_text = "\n".join(pretty_lines)
        st.code(pretty_text, language="text")

        st.markdown("**Table view**")
        rows = [{
            "step": r["step"], "of": r["total"], "pair": r["pair"],
            "new_id": r["new_id"], "token": r["token"], "count": r["count"],
        } for r in logs]
        st.dataframe(rows, use_container_width=True)

        buf = io.StringIO(); buf.write(pretty_text + "\n")
        st.download_button(f"⬇️ Download {fname}", data=buf.getvalue(), file_name=fname, mime="text/plain")

        st.divider()
        st.markdown("**Encoded IDs & Decoded**")
        _ids = tok.encode(input_effective)
        st.code(f"Encoded IDs: {_ids}", language="text")
        st.code(f"Decoded: {tok.decode(_ids)}", language="text")
    else:
        st.info("No merges were recorded (training text too short or vocab size too small). Try retraining.")


with compare_logs_tab:
    st.subheader("📑 Training Logs — Basic vs Regex (side-by-side)")
    st.caption("Left: BasicTokenizer merges. Right: RegexTokenizer merges. Below: step-aligned diff summary.")

    colL, colR = st.columns(2)
    with colL:
        st.markdown("### BasicTokenizer ")
        if basic_pretty:
            st.code("\n".join(basic_pretty), language="text")
        else:
            st.info("No Basic log available.")

    with colR:
        st.markdown("### RegexTokenizer ")
        if regex_pretty:
            st.code("\n".join(regex_pretty), language="text")
        else:
            st.info("No Regex log available.")

    st.divider()

    colL2, colR2 = st.columns(2)
    with colL2:
        st.markdown("#### Basic — table")
        if basic_logs:
            basic_rows = [{"step": r["step"], "of": r["total"], "pair": r["pair"],
                           "new_id": r["new_id"], "token": r["token"], "count": r["count"]}
                          for r in basic_logs]
            st.dataframe(basic_rows, use_container_width=True)
        else:
            st.info("No Basic log.")
    with colR2:
        st.markdown("#### Regex — table")
        if regex_logs:
            regex_rows = [{"step": r["step"], "of": r["total"], "pair": r["pair"],
                           "new_id": r["new_id"], "token": r["token"], "count": r["count"]}
                          for r in regex_logs]
            st.dataframe(regex_rows, use_container_width=True)
        else:
            st.info("No Regex log.")

    st.divider()
    st.markdown("### Step-aligned summary")
    if basic_logs and regex_logs:
        n = min(len(basic_logs), len(regex_logs))
        diff_rows = []
        for i in range(n):
            b = basic_logs[i]; r = regex_logs[i]
            diff_rows.append({
                "step": i + 1,
                "basic_pair": b["pair"], "regex_pair": r["pair"],
                "pairs_match": b["pair"] == r["pair"],
                "basic_count": b["count"], "regex_count": r["count"],
                "counts_match": b["count"] == r["count"],
                "basic_new_id": b["new_id"], "regex_new_id": r["new_id"],
                "basic_token": b["token"], "regex_token": r["token"],
            })
        st.dataframe(diff_rows, use_container_width=True)

        buf = io.StringIO()
        for row in diff_rows: buf.write(str(row) + "\n")
        st.download_button("⬇️ Download diff_summary.txt", data=buf.getvalue(),
                           file_name="diff_summary.txt", mime="text/plain")
    else:
        st.info("Need both Basic and Regex logs to build a comparison.")
