"""Tokenizador BPE a nivel de byte, 16,384 piezas, con pretokenización alineada a lexemas de Python.

Ninguna pieza cruza la frontera entre identificador, número, corrida de puntuación o salto de línea con su sangría.
"""
from __future__ import annotations

from pathlib import Path

ESPECIALES = ["<|endoftext|>", "<fim_prefix>", "<fim_suffix>", "<fim_middle>", "<pad>"]
VOCAB = 16384
# identificador | número | salto(s) de línea + sangría | espacios antes de no-espacio | puntuación | resto de espacios
PATRON = r" ?[A-Za-z_][A-Za-z0-9_]*| ?[0-9]+(?:\.[0-9]+)?|\n+[ \t]*| ?[^\sA-Za-z0-9_]+| +(?=\S)|\s+"


def _nuevo():
    from tokenizers import Tokenizer, Regex, decoders, models, pre_tokenizers
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATRON), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    return tk


def entrenar(datos: Path, chars: int):
    from tokenizers import trainers, pre_tokenizers
    from .data import repos
    tk = _nuevo()

    def corpus():
        n = 0
        for r in repos(datos, "train"):
            for f in r["files"]:
                yield f["content"]
                n += len(f["content"])
                if n >= chars:
                    return

    tr = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=ESPECIALES, min_frequency=2,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tk.train_from_iterator(corpus(), trainer=tr)
    (datos / "tok").mkdir(exist_ok=True)
    tk.save(str(datos / "tok" / "tokenizer.json"))
    print("tokenizador:", tk.get_vocab_size(), "piezas", flush=True)


def cargar(datos: Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(datos) / "tok" / "tokenizer.json"))
