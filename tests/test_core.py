"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

from app import core
from app.corpus import CommentaryCorpus

# Milestone 1: Prepare a small corpus of sports commentary text.
def test_milestone_1(tmp_path):
    save_file = str(tmp_path / "corpus.json")
    result = core.milestone_1(save_path=save_file)
    
    assert "sentences" in result
    assert "tokens" in result
    assert "vocab" in result
    assert result["num_sentences"] >= 20
    assert result["vocab_size"] > 100
    assert result["num_tokens"] > 200
    
    # Check special tokens
    assert "<pad>" in result["vocab"]
    assert "<unk>" in result["vocab"]
    assert "<bos>" in result["vocab"]
    assert "<eos>" in result["vocab"]
    
    # Test encoding / decoding
    corpus = result["corpus_instance"]
    encoded = corpus.encode("What a goal!", add_special_tokens=True)
    decoded = corpus.decode(encoded, skip_special_tokens=True)
    assert "what a goal" in decoded.lower()
    
    # Test saved file on disk
    loaded = CommentaryCorpus.load(save_file)
    assert loaded.num_sentences == result["num_sentences"]


# Milestone 2: Implement the n-gram model and generate text from it.
def test_milestone_2():
    res = core.milestone_2(n=3, prompt="The striker", max_tokens=20, seed=123)
    
    assert res["n"] == 3
    assert "generated_text" in res
    assert len(res["generated_text"]) > 0
    assert res["generated_text"].lower().startswith("the striker")
    assert res["vocab_size"] > 50
    assert res["sample_perplexity"] > 0.0
    
    # Test generation without prompt
    assert len(res["unprompted_text"]) > 0
    
    # Test backoff and perplexity on unseen phrase
    model = res["model"]
    unseen_perp = model.perplexity("A totally alien sentence never seen in sports!")
    assert unseen_perp > 0.0


import torch
from app.lora import LoRALinear

# Milestone 3: Load a small pre-trained LLM and implement LoRA adapter injection.
def test_milestone_3():
    res = core.milestone_3(vocab_size=200, rank=4, alpha=8.0)
    
    assert "model" in res
    assert "before_stats" in res
    assert "after_stats" in res
    
    # Trainable percentage should be significantly reduced (e.g. < 10%)
    assert res["before_stats"]["trainable_percent"] == 100.0
    assert res["after_stats"]["trainable_percent"] < 10.0
    assert res["after_stats"]["trainable_params"] > 0
    
    # Check that model contains LoRALinear modules
    model = res["model"]
    lora_modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(lora_modules) > 0
    
    # Check that base parameters are frozen while lora_A/B require gradients
    for lora in lora_modules:
        assert not lora.base_layer.weight.requires_grad
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad
        
    # Forward pass works correctly
    dummy_input = torch.tensor([[5, 10, 15, 20]])
    logits, _ = model(dummy_input)
    assert logits.shape == (1, 4, 200)


from pathlib import Path
from app.lora import load_lora_weights

# Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
def test_milestone_4(tmp_path):
    adapter_file = str(tmp_path / "test_lora.pt")
    res = core.milestone_4(epochs=8, lr=5e-3, save_adapter_path=adapter_file)
    
    assert "losses" in res
    assert len(res["losses"]) == 8
    # Loss should decrease significantly during training
    assert res["final_loss"] < res["initial_loss"]
    
    # Check that adapter file was saved on disk
    assert Path(adapter_file).exists()
    
    # Check that adapter can be loaded back into a model
    model = res["model"]
    load_lora_weights(model, adapter_file)


# Milestone 5: Generate text from the LoRA-adapted LLM.
def test_milestone_5():
    res = core.milestone_5(prompt="The striker", max_new_tokens=20, seed=99)
    
    assert "generated_text" in res
    assert len(res["generated_text"]) > 0
    assert "the striker" in res["generated_text"].lower()
    
    # Test unprompted generation
    assert "unprompted_text" in res
    assert len(res["unprompted_text"]) > 0


from app.comparator import ComparisonResult

# Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
def test_milestone_6():
    res = core.milestone_6(prompt="The striker", max_tokens=15, seed=1, print_report=False)
    
    assert "comparison" in res
    assert "report" in res
    assert "ngram_text" in res
    assert "lora_text" in res
    assert "analysis" in res
    
    comp = res["comparison"]
    assert isinstance(comp, ComparisonResult)
    assert len(comp.ngram_output.text) > 0
    assert len(comp.lora_output.text) > 0
    
    report = res["report"]
    assert "COMPARATOR" in report
    assert "N-gram" in report
    assert "LoRA" in report
    assert comp.prompt in report

