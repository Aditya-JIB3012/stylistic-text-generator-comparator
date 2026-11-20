"""Your code. One stub per milestone; build them in order and keep each one small."""

from app.corpus import CommentaryCorpus
from app.ngram import NGramModel

# --- Milestone 1: Prepare a small corpus of sports commentary text.
# Hint: Data preprocessing
def milestone_1(save_path: str = "data/commentary_corpus.json") -> dict:
    """Prepare and preprocess a small corpus of sports commentary text."""
    corpus = CommentaryCorpus()
    corpus.save(save_path)
    result = corpus.to_dict()
    result["saved_path"] = save_path
    result["corpus_instance"] = corpus
    return result


# --- Milestone 2: Implement the n-gram model and generate text from it.
# Hint: N-gram modeling
def milestone_2(
    corpus_dict: dict | None = None,
    n: int = 3,
    prompt: str = "The striker",
    max_tokens: int = 25,
    seed: int = 42,
) -> dict:
    """Train an n-gram model on the commentary corpus and generate text."""
    if corpus_dict is None:
        corpus_dict = milestone_1()
    
    sentences = corpus_dict["sentences"]
    model = NGramModel(n=n)
    model.train(sentences)
    
    generated_text = model.generate(prompt=prompt, max_tokens=max_tokens, seed=seed)
    unprompted_text = model.generate(prompt=None, max_tokens=max_tokens, seed=seed + 1)
    sample_eval = sentences[0] if sentences else ["goal"]
    eval_text = " ".join(sample_eval)
    perp = model.perplexity(eval_text)
    
    return {
        "n": n,
        "model": model,
        "prompt": prompt,
        "generated_text": generated_text,
        "unprompted_text": unprompted_text,
        "sample_perplexity": perp,
        "vocab_size": len(model.vocab),
    }


# --- Milestone 3: Load a small pre-trained LLM and implement LoRA adapter injection.
# Hint: LoRA adapter injection
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
# Hint: LoRA adapter training
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Generate text from the LoRA-adapted LLM.
# Hint: Text generation
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
# Hint: Text comparison
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
