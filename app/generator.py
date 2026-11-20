"""Text generation utilities for the LoRA-adapted LLM."""
from __future__ import annotations

import random
import torch
import torch.nn as nn

from app.corpus import CommentaryCorpus


def generate_lora_commentary(
    model: nn.Module,
    corpus: CommentaryCorpus,
    prompt: str | None = None,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_k: int | None = 25,
    seed: int | None = None,
) -> str:
    """Generate sports commentary text from the LoRA-adapted language model."""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    model.eval()

    if prompt and prompt.strip():
        # Encode prompt without special tokens
        token_ids = corpus.encode(prompt.strip(), add_special_tokens=False)
        if not token_ids:
            token_ids = [corpus.word2idx["<bos>"]]
    else:
        token_ids = [corpus.word2idx["<bos>"]]

    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    eos_id = corpus.word2idx.get("<eos>")

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
        )

    generated_ids = output_ids[0].tolist()
    return corpus.decode(generated_ids, skip_special_tokens=True)
