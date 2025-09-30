# bench.py
import time
import tracemalloc
from basic import BasicTokenizer
from regex_tokenizer import RegexTokenizer, GPT4_SPLIT_PATTERN


def time_mem(fn, *args, **kw):
    """Measure wall-clock time and peak memory (MB) for a function call."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    dt = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, dt, peak / (1024 * 1024)


def bench(train_text: str, input_text: str, vocab_size: int = 400):
    basic = BasicTokenizer()
    regex = RegexTokenizer(GPT4_SPLIT_PATTERN)

    # --- Training
    _, t_train_basic, m_train_basic = time_mem(basic.train, train_text, vocab_size)
    _, t_train_regex, m_train_regex = time_mem(regex.train, train_text, vocab_size)

    # --- Encoding
    _, t_enc_basic, m_enc_basic = time_mem(basic.encode, input_text)
    _, t_enc_regex, m_enc_regex = time_mem(regex.encode, input_text)

    print("==== Benchmark Results ====")
    print(f"{'Tokenizer':<10} {'Train (s)':>10} {'Train (MB)':>12} {'Encode (s)':>12} {'Encode (MB)':>12}")
    print(f"{'Basic':<10} {t_train_basic:10.3f} {m_train_basic:12.1f} {t_enc_basic:12.3f} {m_enc_basic:12.1f}")
    print(f"{'Regex':<10} {t_train_regex:10.3f} {m_train_regex:12.1f} {t_enc_regex:12.3f} {m_enc_regex:12.1f}")


# if __name__ == "__main__":
#     # Build a larger corpus for meaningful timings
#     para = (
#         "the brave boy told a magical story about a dragon and a castle. "
#         "everyone in the village gathered under the big tree to listen. "
#     )
#     train = para.lower() * 5000   # simulate a bigger training corpus
#     test = para.lower() * 50      # test sentence repeated
#
#     bench(train, test, vocab_size=600)
