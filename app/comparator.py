"""Qualitative comparison interface: N-gram vs LoRA-LLM side by side."""
from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field

from app.corpus import CommentaryCorpus
from app.ngram import NGramModel


@dataclass
class ModelOutput:
    model_name: str
    prompt: str
    text: str
    perplexity: float = 0.0
    tokens: int = 0
    extra_info: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    prompt: str
    ngram_output: ModelOutput
    lora_output: ModelOutput
    analysis: dict = field(default_factory=dict)


def compute_ngram_perplexity(ngram_model: NGramModel, text: str) -> float:
    """Compute n-gram perplexity of text."""
    return ngram_model.perplexity(text)


def count_unique_words(text: str) -> int:
    """Count unique words in a text."""
    words = text.lower().split()
    return len(set(words))


def vocab_richness_score(text: str) -> float:
    """Type-token ratio: a quick lexical diversity measure."""
    words = text.lower().split()
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 3)


def style_markers(text: str) -> dict:
    """
    Detect stylistic markers typical of live sports commentary:
    - Exclamation marks (excitement level)
    - Action verbs (from common sports verbs set)
    - Superlatives / intensifiers
    - Sentence length (avg words per sentence)
    """
    words = text.split()
    lower = text.lower()

    action_verbs = {
        "strike", "scores", "goal", "shoots", "saves", "tackles", "dribbles",
        "runs", "kicks", "passes", "headers", "blocks", "volleys", "chips",
        "crosses", "curls", "powers", "rattles", "nets", "thunders", "sprints"
    }
    intensifiers = {
        "absolutely", "brilliant", "thunderbolt", "dazzling", "breathtaking",
        "astonishing", "sensational", "electrifying", "magnificent", "stunning",
        "spectacular", "extraordinary", "incredible", "wonderful", "glorious"
    }
    
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    avg_sent_len = round(len(words) / max(len(sentences), 1), 1)
    
    return {
        "exclamations": text.count("!"),
        "action_verb_hits": sum(1 for w in words if w.lower().rstrip(",.!") in action_verbs),
        "intensifier_hits": sum(1 for w in words if w.lower().rstrip(",.!") in intensifiers),
        "avg_sentence_length": avg_sent_len,
        "word_count": len(words),
        "unique_words": count_unique_words(text),
        "vocab_richness": vocab_richness_score(text),
    }


def compare_outputs(
    ngram_model: NGramModel,
    lora_model,
    corpus: CommentaryCorpus,
    prompt: str = "The striker",
    max_tokens_ngram: int = 30,
    max_tokens_lora: int = 30,
    temperature: float = 0.8,
    seed: int = 42,
) -> ComparisonResult:
    """
    Run both models for the same prompt and return a structured comparison.
    """
    from app.generator import generate_lora_commentary
    
    # N-gram generation
    ngram_text = ngram_model.generate(prompt=prompt, max_tokens=max_tokens_ngram, seed=seed)
    ngram_perp = ngram_model.perplexity(ngram_text)
    ngram_markers = style_markers(ngram_text)
    
    ngram_output = ModelOutput(
        model_name="N-gram (trigram)",
        prompt=prompt,
        text=ngram_text,
        perplexity=round(ngram_perp, 2),
        tokens=ngram_markers["word_count"],
        extra_info=ngram_markers,
    )
    
    # LoRA LLM generation
    lora_text = generate_lora_commentary(
        model=lora_model,
        corpus=corpus,
        prompt=prompt,
        max_new_tokens=max_tokens_lora,
        temperature=temperature,
        seed=seed,
    )
    lora_perp = ngram_model.perplexity(lora_text)  # Cross-evaluate using n-gram perplexity for comparison
    lora_markers = style_markers(lora_text)
    
    lora_output = ModelOutput(
        model_name="LoRA-adapted LLM",
        prompt=prompt,
        text=lora_text,
        perplexity=round(lora_perp, 2),
        tokens=lora_markers["word_count"],
        extra_info=lora_markers,
    )
    
    # Analysis summary
    analysis = {
        "perplexity_winner": "ngram" if ngram_perp < lora_perp else "lora",
        "vocab_richness_winner": (
            "ngram" if ngram_markers["vocab_richness"] > lora_markers["vocab_richness"] else "lora"
        ),
        "excitement_winner": (
            "ngram" if ngram_markers["exclamations"] > lora_markers["exclamations"] else "lora"
        ),
        "action_winner": (
            "ngram" if ngram_markers["action_verb_hits"] > lora_markers["action_verb_hits"] else "lora"
        ),
        "ngram_markers": ngram_markers,
        "lora_markers": lora_markers,
    }
    
    return ComparisonResult(
        prompt=prompt,
        ngram_output=ngram_output,
        lora_output=lora_output,
        analysis=analysis,
    )


def render_comparison_report(result: ComparisonResult) -> str:
    """Render a human-readable side-by-side comparison report."""
    sep = "=" * 72
    thin = "-" * 72
    
    def wrap(text: str, width: int = 68, indent: int = 2) -> str:
        prefix = " " * indent
        return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=prefix)
    
    lines = [
        sep,
        "  STYLISTIC TEXT GENERATOR COMPARATOR - Side-by-Side Report",
        sep,
        f"  Prompt : \"{result.prompt}\"",
        thin,
        "",
        "  [MODEL 1] N-gram (Trigram)",
        wrap(result.ngram_output.text),
        "",
        f"    Words  : {result.ngram_output.tokens}",
        f"    Perp.  : {result.ngram_output.perplexity:.2f}",
        f"    Vocab richness : {result.ngram_output.extra_info['vocab_richness']:.3f}",
        f"    Action verbs   : {result.ngram_output.extra_info['action_verb_hits']}",
        f"    Intensifiers   : {result.ngram_output.extra_info['intensifier_hits']}",
        "",
        thin,
        "",
        "  [MODEL 2] LoRA-adapted LLM",
        wrap(result.lora_output.text),
        "",
        f"    Words  : {result.lora_output.tokens}",
        f"    Perp.  : {result.lora_output.perplexity:.2f}",
        f"    Vocab richness : {result.lora_output.extra_info['vocab_richness']:.3f}",
        f"    Action verbs   : {result.lora_output.extra_info['action_verb_hits']}",
        f"    Intensifiers   : {result.lora_output.extra_info['intensifier_hits']}",
        "",
        sep,
        "  Analysis",
        thin,
        f"    Fluency (lower perplexity wins)   : {result.analysis['perplexity_winner'].upper()}",
        f"    Lexical diversity                  : {result.analysis['vocab_richness_winner'].upper()}",
        f"    Excitement style                   : {result.analysis['excitement_winner'].upper()}",
        f"    Action-word usage                  : {result.analysis['action_winner'].upper()}",
        sep,
        "",
    ]
    
    return "\n".join(lines)
