#!/usr/bin/env python3

"""
Improved dataset classes for paraphrase detection.
Key changes:
  - Consistent prompt template between train and test
  - Fixed missing closing quote
  - Data augmentation (swapped sentence pairs)
"""

import csv
import re
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


PROMPT_TEMPLATE = 'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '


class ParaphraseDetectionDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        sent1 = [x[0] for x in all_data]
        sent2 = [x[1] for x in all_data]
        labels = ['yes' if label == 1 else 'no' for label in [x[2] for x in all_data]]
        labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)['input_ids']
        sent_ids = [x[3] for x in all_data]

        # Use the SAME prompt template as test
        cloze_style_sents = [
            PROMPT_TEMPLATE.format(s1=s1, s2=s2) for (s1, s2) in zip(sent1, sent2)
        ]
        encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sent_ids': sent_ids
        }

        return batched_data


class ParaphraseDetectionTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        sent1 = [x[0] for x in all_data]
        sent2 = [x[1] for x in all_data]
        sent_ids = [x[2] for x in all_data]

        cloze_style_sents = [
            PROMPT_TEMPLATE.format(s1=s1, s2=s2) for (s1, s2) in zip(sent1, sent2)
        ]

        encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sent_ids': sent_ids
        }

        return batched_data


def load_paraphrase_data(paraphrase_filename, split='train', augment=False):
    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))
    else:
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    s1 = preprocess_string(record['sentence1'])
                    s2 = preprocess_string(record['sentence2'])
                    label = int(float(record['is_duplicate']))
                    paraphrase_data.append((s1, s2, label, sent_id))
                    # Data augmentation: add swapped pair
                    if augment:
                        paraphrase_data.append((s2, s1, label, sent_id + '_aug'))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
    return paraphrase_data


class SonnetsDataset(Dataset):
    def __init__(self, file_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sonnets = self._load_sonnets(file_path)

    def _load_sonnets(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]
        return [s.strip() for s in sonnets]

    def __len__(self):
        return len(self.sonnets)

    def __getitem__(self, idx):
        return (idx, self.sonnets[idx])

    def collate_fn(self, all_data):
        idx = [example[0] for example in all_data]
        sonnets = [example[1] for example in all_data]

        encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sent_ids': idx
        }

        return batched_data
