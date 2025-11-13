import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Dict, List, Union
from datetime import datetime

class OverlapDataset(Dataset):
    """
    cases: List[List[str]]  (each case is list of visits)
    # labels, dig_labels: List[int] (0/1)
    tokenizer: transformers tokenizer (not called here; used in collate)
    """
    def __init__(self, cases: List[List[str]], labels: List[int]):
        self.cases = cases
        self.labels = labels

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "visits": self.cases[idx],          # list[str]
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# collate: tokenizes per-case visits, pads visits->seq_len and cases->max_n_visits in batch
def collate_cases(batch: List[Dict], tokenizer, max_length: int = 512):
    """
    Returns:
      input_ids: (B, M, S) long
      attention_mask: (B, M, S) long
      labels: (B,) float
      n_visits: (B,) long
    """

    B = len(batch)
    # tokenize per-case: each case's visits -> a dict with tensors (n_visits, seq_len_case_str)
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
    labels = torch.stack([item['label'] for item in batch])
    n_visits = torch.tensor(n_visits_list, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "n_visits": n_visits
    }


def dates_to_log_deltas(case_dates: List[datetime], unit: str = "days"):
    """
    Convert one case's ordered dates into two differnce arrays:
      - log_prev: log1p(delta since previous visit)  (first visit -> 0)
      - log_start: log1p(delta since first visit)    (first visit -> 0)

    Returns:
      list of tuples [(log_prev0, log_start0), ...] length == len(case_dates)
    """
    if len(case_dates) == 0:
        return []

    first = case_dates[0]
    prev = case_dates[0]

    out = []
    for dt in case_dates:
        # delta from previous (in days, possibly fractional)
        delta_prev_seconds = (dt - prev).total_seconds()
        if unit == "days":
            delta_prev = delta_prev_seconds / 86400.0
        elif unit == "hours":
            delta_prev = delta_prev_seconds / 3600.0
        else:
            raise ValueError("Unsupported unit; use 'days' or 'hours'")

        # delta from first
        delta_start_seconds = (dt - first).total_seconds()
        if unit == "days":
            delta_start = delta_start_seconds / 86400.0
        else:
            delta_start = delta_start_seconds / 3600.0

        # first visit: if dt == first then delta_prev may be 0.0, keep that
        log_prev = float(torch.log1p(torch.tensor(delta_prev, dtype=torch.float32)).item())
        log_start = float(torch.log1p(torch.tensor(delta_start, dtype=torch.float32)).item())

        out.append((log_prev, log_start))
        prev = dt

    return out

class DateVisitsDataset(Dataset):
    """
    cases: List[List[str]]  (each case is list of visits)
    dates: List[List[datetime]] (each case's visit datetimes, same order as visits)
    labels: List[int] (0/1) or float
    """
    def __init__(self, cases: List[List[str]], dates: List[List[datetime]], labels: List[int]):
        assert len(cases) == len(dates) == len(labels)
        self.cases = cases
        self.dates = dates
        self.labels = labels

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "visits": self.cases[idx],          # list[str]
            "dates": self.dates[idx],          # list[datetime]
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# collate: tokenizes per-case visits, pads visits->seq_len and cases->max_n_visits in batch
def collate_cases_time(batch: List[Dict], tokenizer, max_length: int = 512, time_unit: str = "days"):
    """
    Returns:
      input_ids: (B, M, S) long
      attention_mask: (B, M, S) long
      labels: (B,) long
      n_visits: (B,) long
      visit_time_diffs: (B, M, 2) float (log_prev, log_start)
    """

    B = len(batch)
    # tokenize per-case: each case's visits -> a dict with tensors (n_visits, seq_len_case_str)
    tokenized_cases = []
    n_visits_list = []
    visit_times_per_case = []

    for item in batch:
        visits = item["visits"]
        enc = tokenizer(
            visits,
            truncation=True,
            padding='longest',    # pad visits inside the case to the longest visit in this case
            max_length=max_length,
            return_tensors='pt'
        )
        tokenized_cases.append(enc)
        n_visits = enc['input_ids'].size(0)
        n_visits_list.append(n_visits)

        # compute log time differences for this case (list of tuples)
        case_dates = item.get("dates", [])
        # ensure lengths match
        if len(case_dates) != n_visits:
            # If the tokenizer dropped or merged visits somehow, raise to catch mismatch.
            raise ValueError(f"Number of dates ({len(case_dates)}) and visits ({n_visits}) must match for a case.")
        visit_time_diffs = dates_to_log_deltas(case_dates, unit=time_unit)  # list[(log_prev, log_start)]
        # convert to tensor (n_visits, 2)
        if len(visit_time_diffs) > 0:
            visit_time_diffs = torch.tensor(visit_time_diffs, dtype=torch.float32)
        else:
            visit_time_diffs = torch.zeros((0, 2), dtype=torch.float32)
        visit_times_per_case.append(visit_time_diffs)

    max_n_visits = max(n_visits_list)
    max_seq_len = max(enc['input_ids'].size(1) for enc in tokenized_cases)

    input_ids_batch = []
    attn_batch = []
    visit_times_batch = []
    for enc, visit_times in zip(tokenized_cases, visit_times_per_case):
        n_visits, seq_len = enc['input_ids'].shape
        # pad seq_len to max_seq_len if needed
        if seq_len < max_seq_len:
            pad = max_seq_len - seq_len
            enc_input = F.pad(enc['input_ids'], (0, pad), value=tokenizer.pad_token_id)
            enc_attn  = F.pad(enc['attention_mask'], (0, pad), value=0)
        else:
            enc_input = enc['input_ids']
            enc_attn  = enc['attention_mask']

        # pad visits dimension to max_n_visits for input & attn
        if n_visits < max_n_visits:
            visits_pad = max_n_visits - n_visits
            enc_input = torch.cat([enc_input, torch.zeros((visits_pad, max_seq_len), dtype=torch.long)], dim=0)
            enc_attn  = torch.cat([enc_attn, torch.zeros((visits_pad, max_seq_len), dtype=torch.long)], dim=0)

        input_ids_batch.append(enc_input.unsqueeze(0))   # (1, M, S)
        attn_batch.append(enc_attn.unsqueeze(0))

        # pad time differences along visits dimension to max_n_visits
        if visit_times.shape[0] < max_n_visits:
            pad_vis = max_n_visits - visit_times.shape[0]
            pad_tensor = torch.zeros((pad_vis, 2), dtype=torch.float32)
            visit_times_padded = torch.cat([visit_times, pad_tensor], dim=0)
        else:
            visit_times_padded = visit_times
        # ensure shape (max_n_visits, 2)
        visit_times_batch.append(visit_times_padded.unsqueeze(0))  # (1, M, 2)

    input_ids = torch.cat(input_ids_batch, dim=0)        # (B, M, S)
    attention_mask = torch.cat(attn_batch, dim=0)        # (B, M, S)
    labels = torch.stack([item['label'] for item in batch])
    n_visits = torch.tensor(n_visits_list, dtype=torch.long)
    visit_times = torch.cat(visit_times_batch, dim=0)  # (B, M, 2)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "n_visits": n_visits,
        "visit_times": visit_times
    }