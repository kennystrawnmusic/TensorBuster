import json
import re
from functools import lru_cache
from pathlib import Path

import requests
from transformers.utils.logging import tqdm


@lru_cache
def btou() -> dict:
    """
    Creats map of UTF-8 bytes to their corresponding Unicode strings.
    """

    bytes = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = bytes[:]

    i = 0;
    for b in range(256):
        if b not in bytes:
            bytes.append(b)
            chars.append(256 + i)
            i += 1

    chars = [chr(c) for c in chars]
    return dict(zip(bytes, chars))

def get_pairs(token: tuple[str, ...]) -> set:
    """
    Returns a list of character pairs in the token.
    """
    pairs = set()
    prev_char = token[0]
    for char in token[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class BytePairEncoder:
    def __init__(self, encoder, bpe_merges: list[tuple[str, ...]], errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in encoder.items()}
        self.errors = errors
        self.byte_enc = btou()
        self.byte_dec = {v: k for k, v in self.byte_enc.items()}
        self.ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.regex = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", flags=re.IGNORECASE)

    def bpe(self, tok: str) -> str:
        if tok in self.cache:
            return self.cache[tok]

        word = tuple(tok)
        pairs = get_pairs(word)

        while True:
            bigram = min(pairs, key=lambda pair: self.ranks.get(pair, float("inf")))
            if bigram not in self.ranks:
                break
            first, _ = bigram
            next_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    next_word.extend(word[i:j])
                    i = j
                except ValueError:
                    next_word.extend(word[i:])
                    break
            next_word = tuple(next_word)
            word = next_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        tok = "".join(word)
        self.cache[tok] = tok
        return tok

    def encode(self, text):
        bpe_tokens = []
        for tok in re.findall(self.regex, text):
            tok = "".join(self.byte_enc[c] for c in tok.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(tok).split(" "))
        return bpe_tokens

    def decode(self, bpe_tokens):
        text = "".join([self.decoder[c] for c in bpe_tokens])
        text = bytearray([self.byte_dec[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

def get_encoder(model: str, dir: str) -> BytePairEncoder:
    with open(Path(dir) / model / "encoder.json") as f:
        encoder = json.load(f)
    with open(Path(dir) / model / "vocab.bpe") as f:
        bpe_data = f.read();

    bpe_merges = [tuple(line.split()) for line in bpe_data.split("\n")[1:-1]]
    return BytePairEncoder(encoder=encoder, bpe_merges=bpe_merges)

def fetch_vocab():
    # Using pathlib.Path as a platform-agnostic solution
    dir = Path("gpt2-oss")
    if not dir.exists():
        dir.mkdir()

    for fname in ["encoder.json", "vocab.bpe"]:
        req = requests.get(f"https://openaipublic.blob.core.windows.net/gpt-2/models/117M/{fname}", stream=True)

        with open(Path(dir) / fname, "wb") as f:
            fsize = int(req.headers["content-length"])
            chunk_size = 1024
            with tqdm(ncols=100, desc=f"Downloading {fname}", total=fsize, unit_scale=True) as pbar:
                for chunk in req.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)
