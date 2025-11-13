import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import operator
import pickle

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
    n_batches = len(data_loader)

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        visit_times = batch['visit_times'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        output, _ = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            visit_times=visit_times,
            labels=labels)

        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        # keep scheduler.step() here if you intend per-step scheduling; otherwise call per-epoch outside
        if scheduler is not None:
            scheduler.step()

        total_loss += output.loss.item()

        del output
        torch.cuda.empty_cache()

    return total_loss / n_batches

def evaluate_model(model, data_loader, device, test=False):
    model.eval()
    predictions = []
    actual = []
    attn_weights_list = []
    total_loss = 0.0
    n_batches = len(data_loader)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            visit_times = batch['visit_times'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            output, attn_weights = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                visit_times=visit_times,
                labels=labels,
                get_attention_weights=test
            )
            
            total_loss += output.loss.item()
            if test:
                attn_weights_list += attn_weights

            # Get predictions (assuming multi-class classification with logits)
            probs = nn.Softmax(dim=1)(output.logits[0])  # (B, num_classes)
            pred = torch.argmax(probs, dim=1)  # (B,)
            pred = pred.cpu()
            predictions.extend(pred.tolist())
            actual.extend(labels.cpu().numpy().tolist())
    
    with open("attention_weights.pickle", 'wb') as f:
        pickle.dump(attn_weights_list, f)
    avg_loss = total_loss / n_batches
    return (avg_loss, np.array(predictions), np.array(actual))