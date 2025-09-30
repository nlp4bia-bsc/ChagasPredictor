import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from typing import Dict
import logging
import time
from accelerate import Accelerator
from pathlib import Path
import operator

from utils.evaluation import multilabel_eval
# Define which comparison to use for each metric
comparison_ops = {
    'val_loss': operator.lt,
    'val_accuracy': operator.gt,
    'val_f1': operator.gt
}

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    card_losses = 0.0
    dig_losses = 0.0
    n_batches = len(data_loader)

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        card_labels = batch['card_labels'].to(device)
        dig_labels = batch['dig_labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        output, card_loss, dig_loss = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            card_labels=card_labels,
            dig_labels=dig_labels)

        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        # keep scheduler.step() here if you intend per-step scheduling; otherwise call per-epoch outside
        if scheduler is not None:
            scheduler.step()

        total_loss += output.loss.item()
        card_losses += card_loss.item()
        dig_losses += dig_loss.item()

    return (total_loss / n_batches,
            card_losses / n_batches,
            dig_losses / n_batches)


def evaluate_model(model, data_loader, device):
    model.eval()
    card_predictions = []
    dig_predictions = []
    card_actual = []
    dig_actual = []
    total_loss = 0.0
    n_batches = len(data_loader)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            card_labels = batch['card_labels'].to(device)
            dig_labels = batch['dig_labels'].to(device)

            # Forward pass
            output, _, _ = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                card_labels=card_labels,
                dig_labels=dig_labels)
            
            total_loss += output.loss.item()

            # Get predictions (assuming binary classification with logits)
            card_probs = torch.sigmoid(output.logits[0])
            dig_probs = torch.sigmoid(output.logits[1])
            card_pred = (card_probs >= 0.5).long().squeeze()
            dig_pred = (dig_probs >= 0.5).long().squeeze()

            card_predictions.extend(card_pred.tolist())
            dig_predictions.extend(dig_pred.tolist())
            card_actual.extend(card_labels.cpu().numpy().tolist())
            dig_actual.extend(dig_labels.cpu().numpy().tolist())

    avg_loss = total_loss / n_batches
    return (avg_loss,
            np.array(card_predictions), np.array(dig_predictions),
            np.array(card_actual), np.array(dig_actual))