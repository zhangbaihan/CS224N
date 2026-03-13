#!/usr/bin/env python3

'''
Improved Paraphrase Detection.
Key changes:
  - Consistent train/test prompt (via improved datasets.py)
  - LR warmup + linear decay
  - Gradient clipping
  - Weight decay with proper param grouping
  - Data augmentation (swapped pairs)
'''

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import random
import torch
import math

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lora_linear import LoRALinear

# Import from local improved datasets
from improved.datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mark_only_lora_and_head_trainable(gpt: nn.Module, head: nn.Module):
    for p in gpt.parameters():
        p.requires_grad = False
    for m in gpt.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad = True
            m.B.requires_grad = True
    for p in head.parameters():
        p.requires_grad = True


class ParaphraseGPT(nn.Module):
    """GPT-2 Model for paraphrase detection."""

    def __init__(self, args):
        super().__init__()
        fine_tune_mode = getattr(args, 'fine_tune_mode', 'full-model')
        use_lora = fine_tune_mode == 'lora'
        lora_alpha = getattr(args, 'lora_alpha', 16.0)
        self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l,
                                             num_heads=args.num_heads, use_lora=use_lora,
                                             lora_alpha=lora_alpha)
        self.paraphrase_detection_head = nn.Linear(args.d, 2)

        if fine_tune_mode == 'last-linear-layer':
            for param in self.gpt.parameters():
                param.requires_grad = False
        elif fine_tune_mode == 'full-model':
            for param in self.gpt.parameters():
                param.requires_grad = True
        elif fine_tune_mode == 'lora':
            mark_only_lora_and_head_trainable(self.gpt, self.paraphrase_detection_head)

    def forward(self, input_ids, attention_mask):
        output = self.gpt(input_ids, attention_mask)
        last_tokens = output["last_token"]
        pred = self.paraphrase_detection_head(last_tokens)
        logits = last_tokens.new_full((last_tokens.size(0), self.gpt.config.vocab_size), float("-inf"))
        logits[:, 3919] = pred[:, 0]  # no
        logits[:, 8505] = pred[:, 1]  # yes
        return logits


def save_model(model, optimizer, args, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def get_lr_scale(step, warmup_steps, total_steps):
    """Linear warmup then linear decay."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def get_grouped_params(model, weight_decay):
    """Separate params into decay and no-decay groups."""
    no_decay = ['bias', 'layer_norm', 'LayerNorm', 'final_layer_norm']
    grouped = [
        {
            'params': [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]
    return grouped


def train(args):
    """Train GPT-2 for paraphrase detection on the Quora dataset."""
    if args.use_gpu and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Load data (augmentation controlled by --augment flag)
    augment = getattr(args, 'augment', False)
    para_train_data = load_paraphrase_data(args.para_train, augment=augment)
    para_dev_data = load_paraphrase_data(args.para_dev)

    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    args = add_arguments(args)
    model = ParaphraseGPT(args)
    model = model.to(device)

    lr = args.lr
    grouped_params = get_grouped_params(model, weight_decay=0.01)
    optimizer = AdamW(grouped_params, lr=lr)

    # LR schedule
    total_steps = len(para_train_dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    best_dev_acc = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, labels, reduction='mean')
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )

            # LR schedule update
            lr_scale = get_lr_scale(global_step, warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr * lr_scale

            optimizer.step()
            global_step += 1

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")


@torch.no_grad()
def test(args):
    """Evaluate model on the dev and test datasets."""
    if args.use_gpu and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath, weights_only=False)

    model = ParaphraseGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model to test from {args.filepath}")

    para_dev_data = load_paraphrase_data(args.para_dev)
    para_test_data = load_paraphrase_data(args.para_test, split='test')

    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
    para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)
    para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_test_data.collate_fn)

    dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
    print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
    test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

    with open(args.para_dev_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
            f.write(f"{p}, {s} \n")

    with open(args.para_test_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(test_para_sent_ids, test_para_y_pred):
            f.write(f"{p}, {s} \n")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--fine_tune_mode", type=str,
                        choices=('last-linear-layer', 'full-model', 'lora'), default='full-model')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--model_size", type=str,
                        help="The model size as specified on hugging face.",
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
    parser.add_argument("--augment", action='store_true',
                        help="Enable data augmentation (swapped sentence pairs)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha scaling factor")

    args = parser.parse_args()
    return args


def add_arguments(args):
    """Add arguments that are deterministic on model size."""
    if args.model_size == 'gpt2':
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == 'gpt2-medium':
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == 'gpt2-large':
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    else:
        raise Exception(f'{args.model_size} is not supported.')
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'
    seed_everything(args.seed)
    train(args)
    test(args)
