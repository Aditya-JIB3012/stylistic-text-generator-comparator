"""N-gram language model for stylistic sports commentary generation and evaluation."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Sequence

from app.corpus import clean_text, detokenize, tokenize


class NGramModel:
    """An N-gram language model with backoff, smoothing, and probabilistic sampling."""

    def __init__(self, n: int = 3, laplace_alpha: float = 0.01):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.laplace_alpha = laplace_alpha
        # Maps context tuple of length (k-1) to Counter of next word for k in 1..n
        self.tables: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            k: defaultdict(Counter) for k in range(1, n + 1)
        }
        self.vocab: set[str] = set()
        self.total_words: int = 0

    def train(self, sentences: Sequence[Sequence[str]]) -> None:
        """Train the model on a list of tokenized sentences."""
        for sentence in sentences:
            tokens = list(sentence)
            if not tokens:
                continue
            self.vocab.update(tokens)
            self.total_words += len(tokens)
            
            # Pad with (n - 1) <bos> tokens and one <eos>
            padded = ["<bos>"] * (self.n - 1) + tokens + ["<eos>"]
            self.vocab.add("<bos>")
            self.vocab.add("<eos>")
            
            # Count for all orders 1..n
            for k in range(1, self.n + 1):
                for i in range(len(padded) - k + 1):
                    ngram = tuple(padded[i : i + k])
                    context = ngram[:-1]
                    word = ngram[-1]
                    self.tables[k][context][word] += 1

    def get_distribution(
        self, context: tuple[str, ...], temperature: float = 1.0
    ) -> tuple[list[str], list[float]]:
        """Get the probability distribution over next words given a context."""
        # Trim context to at most (n - 1)
        ctx = context[-(self.n - 1) :] if self.n > 1 else ()
        
        # Backoff search from current context length down to 0 (unigram)
        chosen_counter: Counter[str] | None = None
        while len(ctx) >= 0:
            k = len(ctx) + 1
            if ctx in self.tables[k] and len(self.tables[k][ctx]) > 0:
                chosen_counter = self.tables[k][ctx]
                break
            if not ctx:
                break
            ctx = ctx[1:]
            
        if not chosen_counter:
            # Fallback to uniform over vocabulary
            words = sorted(list(self.vocab - {"<bos>"}))
            if not words:
                return ["<eos>"], [1.0]
            prob = 1.0 / len(words)
            return words, [prob] * len(words)
            
        words = list(chosen_counter.keys())
        counts = [chosen_counter[w] for w in words]
        
        # Apply temperature
        if temperature <= 0.01:
            # Greedy: highest count gets prob 1.0
            max_idx = counts.index(max(counts))
            probs = [1.0 if i == max_idx else 0.0 for i in range(len(words))]
            return words, probs
            
        # Logits scaled by temperature
        logits = [math.log(max(c, 1e-6)) / temperature for c in counts]
        max_logit = max(logits)
        exp_logits = [math.exp(lg - max_logit) for lg in logits]
        total_exp = sum(exp_logits)
        probs = [e / total_exp for e in exp_logits]
        return words, probs

    def generate(
        self,
        prompt: str | None = None,
        max_tokens: int = 35,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> str:
        """Generate commentary text starting from a prompt or scratch."""
        if seed is not None:
            random.seed(seed)
            
        if prompt:
            prompt_tokens = tokenize(prompt, lower=True)
            generated = list(prompt_tokens)
        else:
            generated = []
            
        context = ["<bos>"] * max(0, (self.n - 1) - len(generated)) + generated
        
        for _ in range(max_tokens):
            curr_context = tuple(context[-(self.n - 1) :]) if self.n > 1 else ()
            words, probs = self.get_distribution(curr_context, temperature=temperature)
            
            # Sample next word
            r = random.random()
            cumulative = 0.0
            chosen = words[-1]
            for w, p in zip(words, probs):
                cumulative += p
                if r <= cumulative:
                    chosen = w
                    break
                    
            if chosen == "<eos>":
                break
                
            generated.append(chosen)
            context.append(chosen)
            
        return detokenize(generated)

    def perplexity(self, text: str) -> float:
        """Compute the perplexity of the given text under the N-gram model."""
        tokens = tokenize(text, lower=True)
        if not tokens:
            return 1.0
            
        padded = ["<bos>"] * (self.n - 1) + tokens + ["<eos>"]
        log_prob_sum = 0.0
        n_eval = 0
        v_size = max(len(self.vocab), 1)
        
        for i in range(self.n - 1, len(padded)):
            ctx = tuple(padded[max(0, i - (self.n - 1)) : i])
            target_word = padded[i]
            
            k = len(ctx) + 1
            counter = self.tables[k].get(ctx, Counter())
            ctx_total = sum(counter.values())
            target_count = counter.get(target_word, 0)
            
            # Add-alpha smoothing
            prob = (target_count + self.laplace_alpha) / (ctx_total + self.laplace_alpha * v_size)
            log_prob_sum += math.log(prob)
            n_eval += 1
            
        if n_eval == 0:
            return 1.0
        return math.exp(-log_prob_sum / n_eval)
