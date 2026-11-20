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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Load a small pre-trained LLM and implement LoRA adapter injection.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Generate text from the LoRA-adapted LLM.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

