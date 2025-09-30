import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Dict, List

class OverlapDataset(Dataset):
    """
    cases: List[List[str]]  (each case is list of visits)
    card_labels, dig_labels: List[int] (0/1)
    tokenizer: transformers tokenizer (not called here; used in collate)
    """
    def __init__(self, cases: List[List[str]], card_labels: List[int], dig_labels: List[int]):
        assert len(cases) == len(card_labels) == len(dig_labels)
        self.cases = cases
        self.card_labels = card_labels
        self.dig_labels = dig_labels

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "visits": self.cases[idx],          # list[str]
            "card_label": torch.tensor(self.card_labels[idx], dtype=torch.float32),
            "dig_label": torch.tensor(self.dig_labels[idx], dtype=torch.float32),
        }

# collate: tokenizes per-case visits, pads visits->seq_len and cases->max_n_visits in batch
def collate_cases(batch: List[Dict], tokenizer, max_length: int = 512):
    """
    Returns:
      input_ids: (B, M, S) long
      attention_mask: (B, M, S) long
      card_labels: (B,) float
      dig_labels: (B,) float
      n_visits: (B,) long
    """

    B = len(batch)
    # tokenize per-case: each case's visits -> a dict with tensors (n_visits, seq_len_case)
    tokenized_cases = []
    n_visits_list = []
    for item in batch:
        visits = item["visits"]
        # tokenizer will return tensors shaped (n_visits, seq_len_case)
        enc = tokenizer(
            visits,
            truncation=True,
            padding='longest',    # pad visits inside the case to the longest visit in this case
            max_length=max_length,
            return_tensors='pt'
        )
        tokenized_cases.append(enc)
        n_visits_list.append(enc['input_ids'].size(0))

    max_n_visits = max(n_visits_list)
    max_seq_len = max(enc['input_ids'].size(1) for enc in tokenized_cases)

    input_ids_batch = []
    attn_batch = []
    for enc in tokenized_cases:
        n_visits, seq_len = enc['input_ids'].shape
        # pad seq_len to max_seq_len if needed (shouldn't be if padding='longest' per case, but different cases -> different seq lens)
        if seq_len < max_seq_len:
            pad = max_seq_len - seq_len
            enc_input = F.pad(enc['input_ids'], (0, pad), value=tokenizer.pad_token_id)
            enc_attn  = F.pad(enc['attention_mask'], (0, pad), value=0)
        else:
            enc_input = enc['input_ids']
            enc_attn  = enc['attention_mask']

        # pad visits dimension to max_n_visits
        if n_visits < max_n_visits:
            visits_pad = max_n_visits - n_visits
            enc_input = torch.cat([enc_input, torch.zeros((visits_pad, max_seq_len), dtype=torch.long)], dim=0)
            enc_attn  = torch.cat([enc_attn, torch.zeros((visits_pad, max_seq_len), dtype=torch.long)], dim=0)

        input_ids_batch.append(enc_input.unsqueeze(0))   # (1, M, S)
        attn_batch.append(enc_attn.unsqueeze(0))

    input_ids = torch.cat(input_ids_batch, dim=0)        # (B, M, S)
    attention_mask = torch.cat(attn_batch, dim=0)        # (B, M, S)
    card_labels = torch.stack([item['card_label'] for item in batch])
    dig_labels  = torch.stack([item['dig_label'] for item in batch])
    n_visits = torch.tensor(n_visits_list, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "card_labels": card_labels,
        "dig_labels": dig_labels,
        "n_visits": n_visits
    }