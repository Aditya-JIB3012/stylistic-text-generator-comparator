"""Your code. One stub per milestone; build them in order and keep each one small."""

from app.corpus import CommentaryCorpus
from app.ngram import NGramModel
from app.model import SmallCausalLM, LMConfig
from app.lora import inject_lora, get_trainable_parameters, LoRALinear
from app.trainer import train_lora
from app.generator import generate_lora_commentary

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
def milestone_3(
    vocab_size: int = 300,
    rank: int = 4,
    alpha: float = 8.0,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj", "fc1"),
) -> dict:
    """Initialize a small pre-trained LLM and inject LoRA adapter matrices."""
    config = LMConfig(vocab_size=vocab_size, n_embd=64, n_layer=2, n_head=4)
    model = SmallCausalLM(config)
    
    before_stats = get_trainable_parameters(model)
    model = inject_lora(model, rank=rank, alpha=alpha, target_modules=target_modules)
    after_stats = get_trainable_parameters(model)
    
    return {
        "model": model,
        "config": config,
        "rank": rank,
        "alpha": alpha,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "trainable_percent": after_stats["trainable_percent"],
    }


# --- Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
# Hint: LoRA adapter training
def milestone_4(
    corpus_dict: dict | None = None,
    lora_dict: dict | None = None,
    epochs: int = 12,
    lr: float = 3e-3,
    save_adapter_path: str = "data/lora_adapter.pt",
) -> dict:
    """Fine-tune the LoRA adapter on the sports commentary corpus."""
    if corpus_dict is None:
        corpus_dict = milestone_1()
    corpus = corpus_dict["corpus_instance"]
    
    if lora_dict is None:
        lora_dict = milestone_3(vocab_size=corpus.vocab_size)
    model = lora_dict["model"]
    
    # Fine-tune LoRA
    losses = train_lora(
        model=model,
        corpus=corpus,
        epochs=epochs,
        lr=lr,
        save_path=save_adapter_path,
    )
    
    return {
        "model": model,
        "corpus": corpus,
        "losses": losses,
        "initial_loss": losses[0] if losses else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "adapter_path": save_adapter_path,
    }


# --- Milestone 5: Generate text from the LoRA-adapted LLM.
# Hint: Text generation
def milestone_5(
    lora_trained_dict: dict | None = None,
    prompt: str = "What an absolute thunderbolt",
    max_new_tokens: int = 25,
    temperature: float = 0.8,
    seed: int = 42,
) -> dict:
    """Generate text from the fine-tuned LoRA-adapted LLM."""
    if lora_trained_dict is None:
        lora_trained_dict = milestone_4(epochs=8)
    
    model = lora_trained_dict["model"]
    corpus = lora_trained_dict["corpus"]
    
    generated_text = generate_lora_commentary(
        model=model,
        corpus=corpus,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )
    
    unprompted_text = generate_lora_commentary(
        model=model,
        corpus=corpus,
        prompt=None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed + 1,
    )
    
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "unprompted_text": unprompted_text,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "model": model,
        "corpus": corpus,
    }


# --- Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
# Hint: Text comparison
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
