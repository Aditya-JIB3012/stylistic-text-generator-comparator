"""Sports commentary corpus preparation and preprocessing."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_COMMENTARY_CORPUS = [
    "Welcome to an electrifying evening under the stadium floodlights as the derby kicks off.",
    "The striker picks up the loose ball on the half-turn and drives with menacing pace towards the box.",
    "He beats one defender with a dazzling stepover and curls it majestically towards the top corner.",
    "What an absolute thunderbolt of a strike, kissing the crossbar on its way into the net!",
    "The goalkeeper stretches out a desperate hand, but there is no stopping that rocket of a shot.",
    "Listen to the deafening roar of the home supporters as the net bulges in sheer disbelief.",
    "A pinpoint delivery from the corner flag finds the towering center-back rising above the crowd.",
    "He powers a thumping header straight into the bottom corner, leaving the keeper utterly rooted to the spot.",
    "A swift counter-attack breaks at breakneck speed across the pristine turf.",
    "The midfielder threads a glorious through ball that dissects the defensive line with surgical precision.",
    "A delicate chip over the rushing goalkeeper bounces gracefully over the line for a sublime finish.",
    "Sensational footwork on the wing leaves the full-back completely stranded and grasping at shadows.",
    "The referee blows the whistle and points directly to the penalty spot amid furious protests.",
    "He steps up under immense pressure, sends the keeper the wrong way, and calmly rolls it into the corner.",
    "A stoppage-time equalizer sparks absolute pandemonium in the packed away end!",
    "What drama, what passion, what an unbelievable turnaround in the dying moments of the match!",
    "The winger cuts inside onto his favored left foot and unleashes a ferocious dipping drive.",
    "A heroic last-ditch sliding tackle prevents what looked like a certain goal for the visitors.",
    "The playmaker sprays a breathtaking diagonal pass forty yards across the pitch into open space.",
    "The crowd rises to their feet in appreciation of that breathtaking piece of individual brilliance.",
    "He chests the ball down with composure, swivels on the volley, and rattles the woodwork with venom.",
    "A breathless ninety minutes of relentless end-to-end football comes to a dramatic conclusion.",
    "The captain lifts the trophy aloft as golden ticker tape rains down from the stadium rafters.",
    "A masterclass in tactical discipline and clinical finishing from the reigning champions today.",
    "The keeper pulls off an astonishing double save at point-blank range to preserve the clean sheet."
]


def clean_text(text: str) -> str:
    """Normalize whitespace and clean up commentary text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str, lower: bool = True) -> list[str]:
    """Tokenize text into words and punctuation tokens."""
    if lower:
        text = text.lower()
    # Separate punctuation while keeping words intact
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens


def detokenize(tokens: list[str]) -> str:
    """Reconstruct natural readable text from a list of tokens."""
    if not tokens:
        return ""
    text = ""
    for token in tokens:
        if not text or token in {",", ".", "!", "?", ";", ":", "'", ")"}:
            if text and token == "'" and text[-1] != " ":
                text += token
            elif text and token in {",", ".", "!", "?", ";", ":", ")"}:
                text += token
            else:
                text += token
        elif token in {"("}:
            text += " " + token
        else:
            text += (" " if text and text[-1] not in {"(", "$"} else "") + token
    # Capitalize first letter of sentences
    sentences = re.split(r"([.!?]\s*)", text)
    capitalized = "".join(
        part.capitalize() if i % 2 == 0 else part
        for i, part in enumerate(sentences)
    )
    return capitalized.strip()


class CommentaryCorpus:
    """Manages preparation, preprocessing, and vocabulary creation for sports commentary."""

    def __init__(self, raw_lines: list[str] | None = None):
        self.raw_lines = [clean_text(line) for line in (raw_lines or DEFAULT_COMMENTARY_CORPUS) if line.strip()]
        self.sentences: list[list[str]] = [tokenize(line, lower=True) for line in self.raw_lines]
        self.tokens: list[str] = [t for s in self.sentences for t in s]
        
        # Build vocabulary with special tokens
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        unique_words = sorted(list(set(self.tokens)))
        self.vocab = self.special_tokens + [w for w in unique_words if w not in self.special_tokens]
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Convert text into token IDs."""
        tokens = tokenize(text, lower=True)
        unk_idx = self.word2idx["<unk>"]
        ids = [self.word2idx.get(t, unk_idx) for t in tokens]
        if add_special_tokens:
            ids = [self.word2idx["<bos>"]] + ids + [self.word2idx["<eos>"]]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to human-readable text."""
        words = []
        for i in ids:
            word = self.idx2word.get(i, "<unk>")
            if skip_special_tokens and word in self.special_tokens:
                continue
            words.append(word)
        return detokenize(words)

    def save(self, filepath: str | Path) -> None:
        """Save raw corpus and metadata to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_sentences": self.num_sentences,
            "num_tokens": self.num_tokens,
            "vocab_size": self.vocab_size,
            "raw_lines": self.raw_lines,
            "vocab": self.vocab,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, filepath: str | Path) -> CommentaryCorpus:
        """Load corpus from a text or JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            lines = data.get("raw_lines", [])
            return cls(raw_lines=lines)
        else:
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return cls(raw_lines=lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_lines": self.raw_lines,
            "sentences": self.sentences,
            "tokens": self.tokens,
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "num_sentences": self.num_sentences,
            "num_tokens": self.num_tokens,
        }
