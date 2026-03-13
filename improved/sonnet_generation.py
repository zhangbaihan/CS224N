#!/usr/bin/env python3

'''
Improved Sonnet Generation.
Key changes:
  - LR warmup + linear decay
  - Gradient clipping
  - Weight decay with proper param grouping
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
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import SonnetsDataset
from models.gpt2 import GPT2Model
from lora_linear import LoRALinear

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


class SonnetGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        fine_tune_mode = getattr(args, 'fine_tune_mode', 'full-model')
        use_lora = fine_tune_mode == 'lora'
        lora_alpha = getattr(args, 'lora_alpha', 16.0)
        self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l,
                                             num_heads=args.num_heads, use_lora=use_lora,
                                             lora_alpha=lora_alpha)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if fine_tune_mode == 'full-model':
            for param in self.gpt.parameters():
                param.requires_grad = True
        elif fine_tune_mode == 'lora':
            for param in self.gpt.parameters():
                param.requires_grad = False
            for m in self.gpt.modules():
                if isinstance(m, LoRALinear):
                    m.A.requires_grad = True
                    m.B.requires_grad = True

    def forward(self, input_ids, attention_mask):
        output = self.gpt(input_ids, attention_mask)
        hidden_states = output['last_hidden_state']
        logits = self.gpt.hidden_state_to_token(hidden_states)
        return logits

    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    @torch.no_grad()
    def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
        token_ids = encoding.to(self.get_device())
        attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

        for _ in range(max_length):
            logits_sequence = self.forward(token_ids, attention_mask)
            logits_last_token = logits_sequence[:, -1, :] / temperature

            probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= top_p
            top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
            top_p_mask[..., 0] = True
            filtered_probs = sorted_probs * top_p_mask
            filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

            sampled_index = torch.multinomial(filtered_probs, 1)
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

            if sampled_token.item() == self.tokenizer.eos_token_id:
                break

            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
            )

        generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
        return token_ids, generated_output


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
    """Train GPT-2 for sonnet generation."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                   collate_fn=sonnet_dataset.collate_fn)

    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)
    model = SonnetGPT(args)
    model = model.to(device)

    lr = args.lr
    grouped_params = get_grouped_params(model, weight_decay=0.01)
    optimizer = AdamW(grouped_params, lr=lr)

    # LR schedule
    total_steps = len(sonnet_dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask = batch['token_ids'], batch['attention_mask']
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
            labels = b_ids[:, 1:].contiguous().flatten()
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
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
        print('Generating several output sonnets...')
        model.eval()
        for batch in held_out_sonnet_dataset:
            encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
            output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
            print(f'{batch[1]}{output[1]}\n\n')

        save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

    model = SonnetGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
        sonnet_id = batch[0]
        encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
        output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
        decoded_output = model.tokenizer.decode(output)
        full_sonnet = f'{decoded_output}\n\n'
        generated_sonnets.append((sonnet_id, full_sonnet))
        print(f'{decoded_output}\n\n')

    with open(args.sonnet_out, "w+") as f:
        f.write(f"--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])


@torch.no_grad()
def generate_and_eval_dev(args):
    """Generate sonnets from dev held-out prompts and compute CHRF score."""
    from evaluation import test_sonnet

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

    model = SonnetGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    dev_dataset = SonnetsDataset('data/sonnets_held_out_dev.txt')
    dev_out = 'predictions/generated_sonnets_dev.txt'

    generated_sonnets = []
    for batch in dev_dataset:
        sonnet_id = batch[0]
        encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
        output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
        decoded_output = model.tokenizer.decode(output)
        full_sonnet = f'{decoded_output}\n\n'
        generated_sonnets.append((sonnet_id, full_sonnet))

    with open(dev_out, "w+") as f:
        f.write(f"--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])

    chrf_score = test_sonnet(test_path=dev_out, gold_path='data/TRUE_sonnets_held_out_dev.txt')
    print(f"Dev CHRF score :: {chrf_score:.3f}")
    return chrf_score


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
    parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
    parser.add_argument("--top_p", type=float, help="Cumulative probability for nucleus sampling.",
                        default=0.9)

    parser.add_argument("--fine_tune_mode", type=str,
                        choices=('full-model', 'lora'), default='full-model')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--model_size", type=str,
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha scaling factor")

    args = parser.parse_args()
    return args


def add_arguments(args):
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
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-sonnet.pt'
    seed_everything(args.seed)
    train(args)
    generate_submission_sonnets(args)
    generate_and_eval_dev(args)
