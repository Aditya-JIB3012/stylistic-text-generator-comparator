"""Training pipeline for fine-tuning LoRA adapters on sports commentary."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.corpus import CommentaryCorpus
from app.lora import save_lora_weights


class CommentaryDataset(Dataset):
    """PyTorch Dataset for sports commentary text sequences."""

    def __init__(self, corpus: CommentaryCorpus, max_seq_len: int = 64):
        self.corpus = corpus
        self.max_seq_len = max_seq_len
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        pad_id = corpus.word2idx["<pad>"]

        for raw_line in corpus.raw_lines:
            token_ids = corpus.encode(raw_line, add_special_tokens=True)
            if len(token_ids) < 2:
                continue
            # Crop if exceeds max length
            token_ids = token_ids[: max_seq_len + 1]

            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]

            self.samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate_commentary_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences into uniform batch tensors."""
    max_len = max(len(inp) for inp, _ in batch)
    inputs, targets = [], []

    for inp, tgt in batch:
        pad_size = max_len - len(inp)
        inp_padded = torch.cat([inp, torch.full((pad_size,), pad_id, dtype=torch.long)])
        # Use -100 for ignored index in CrossEntropyLoss
        tgt_padded = torch.cat([tgt, torch.full((pad_size,), -100, dtype=torch.long)])
        inputs.append(inp_padded)
        targets.append(tgt_padded)

    return torch.stack(inputs), torch.stack(targets)


def train_lora(
    model: nn.Module,
    corpus: CommentaryCorpus,
    epochs: int = 15,
    lr: float = 3e-3,
    batch_size: int = 8,
    save_path: str | Path | None = "data/lora_adapter.pt",
) -> list[float]:
    """Fine-tune the LoRA adapter parameters on the sports commentary dataset."""
    dataset = CommentaryDataset(corpus, max_seq_len=64)
    pad_id = corpus.word2idx["<pad>"]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_commentary_batch(b, pad_id=pad_id),
    )

    # Train only LoRA parameters that require grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    model.train()
    loss_history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for input_ids, target_ids in loader:
            optimizer.zero_grad()
            _, loss = model(input_ids, targets=target_ids)
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)
        loss_history.append(round(avg_loss, 4))

    if save_path:
        save_lora_weights(model, save_path)

    return loss_history
