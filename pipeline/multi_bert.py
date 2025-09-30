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



class MuBertPipeline:
    def __init__(
            self, 
            config: Dict, 
            model: nn.Module, 
            accelerator: Accelerator,
            optimizer: torch.optim.Optimizer=None,
            scheduler: torch.optim.lr_scheduler._LRScheduler=None,
            logger: logging.Logger=None
        ):
        """

        Initialize the pipeline.

        Args:
            config (Dict): Configuration dictionary.
            model (nn.Module): Model to train.
            accelerator (Accelerator): Accelerator to use.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            logger (logging.Logger): Logger.
        """

        self.config = config

        self.accelerator = accelerator
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            model, optimizer, scheduler
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'best_val_loss': float('inf'),
            'best_val_f1': 0.0,
            'test_loss': float('inf'),
            'best_epoch': 0,
            'best_classification_report': None
        }
        self.epochs = config['training']['num_epochs']
        self.early_stopping_patience = config['training']['early_stopping_patience']
        if not logger:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.best_model_placeholder_path = Path(config['model']['model_save_path'] + 'placeholder/')

    
    def train(
            self, 
            train_loader: torch.utils.data.DataLoader, 
            val_loader: torch.utils.data.DataLoader
        ) -> None:
        """

        Train the model on the training set and validate on the validation set. Update the history dictionary
        with the training and validation losses and classification report. Save the best model based on the validation
        loss.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training set.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation set.
        """

        self.logger.info(f"Starting training on device: {self.accelerator.device}")
        train_loader, val_loader = self.accelerator.prepare(train_loader, val_loader)
        early_stopping_counter = 0
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.epochs} [Train]')
            for batch_idx, batch in enumerate(train_pbar):
                self.optimizer.zero_grad()

                outputs = self.model(batch['input_ids'], batch['attention_mask'], batch['labels'])

                self.accelerator.backward(outputs['loss'])
                self.optimizer.step()
                
                train_loss += outputs['loss'].item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': train_loss / (batch_idx + 1),
                })
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            total_labels = []
            total_preds = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{self.epochs} [Val]')
                for batch_idx, batch in enumerate(val_pbar):
                    outputs = self.model(batch['input_ids'], batch['attention_mask'], batch['labels'])
                    # save labels and predictions for classification report
                    outputs_prob = torch.sigmoid(outputs['logits'])        # multilabel probabilities
                    predictions = (outputs_prob > 0.5).float()     # threshold

                    total_labels.extend(batch['labels'].tolist())
                    total_preds.extend(predictions.tolist())
                    
                    val_loss += outputs['loss'].item()
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        'loss': val_loss / (batch_idx + 1),
                    })
            
            avg_val_loss = val_loss / len(val_loader)
            metric_dict = multilabel_eval(np.array(total_labels), np.array(total_preds), self.logger, print_metrics=False)
            
            # Update learning rate scheduler
            self.scheduler.step(avg_val_loss)
            
            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_f1'].append(metric_dict['Subset']['weighted avg']['f1-score'])

            es_metric = self.config['training']['early_stopping_metric']
            compare = comparison_ops[es_metric]

            if compare(self.history[es_metric][-1], self.history[f'best_{es_metric}']):
                self.history[f'best_{es_metric}'] = self.history[es_metric][-1]
                self.history['best_epoch'] = epoch
                self.history['best_classification_report'] = metric_dict['Subset']
                _ = multilabel_eval(np.array(total_labels), np.array(total_preds), self.logger, print_metrics=True)
                early_stopping_counter = 0
                self.logger.info(f"Saving best model at epoch {epoch + 1} on directory: {self.best_model_placeholder_path}")
                self.save_model()
            else:
                early_stopping_counter += 1
            
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Time: {epoch_time:.2f}s - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {avg_val_loss:.4f} - "
                f"Accuracy: {metric_dict['Subset']['accuracy']:.4f} - "
                f"Precision: {metric_dict['Subset']['weighted avg']['precision']:.4f} - "
                f"Recall: {metric_dict['Subset']['weighted avg']['recall']:.4f} - "
                f"F1-Score: {metric_dict['Subset']['weighted avg']['f1-score']:.4f}"
            )
                
            if early_stopping_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        self.logger.info(f"Training finished after {epoch + 1} epochs")
        self.load_model()
        
    
    def evaluate(
            self,
            test_loader: torch.utils.data.DataLoader
        ) -> None:

        self.model.eval()
        test_loss = 0
        y_true = []
        y_pred = []
        
        test_loader = self.accelerator.prepare(test_loader)
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc='Testing')
            for batch in test_pbar:
                outputs = self.model(batch['input_ids'], batch['attention_mask'], batch['labels'])
                outputs_prob = torch.sigmoid(outputs['logits'])
                predictions = (outputs_prob > 0.5).float()

                y_true.extend(batch['labels'].tolist())
                y_pred.extend(predictions.tolist())
                test_loss += outputs['loss'].item()

        logging.info(f"Classification report:\n")
        metric_dict = multilabel_eval(np.array(y_true), np.array(y_pred), self.logger, print_metrics=True)
        avg_test_loss = test_loss / len(test_loader)
        self.history['test_loss'] = avg_test_loss
        logging.info(f"Test Loss: {avg_test_loss:.4f}")
        return metric_dict
        
        
    def save_model(self) -> None:
        self.best_model_placeholder_path.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), self.best_model_placeholder_path / 'best_model.pth')

    def load_model(self) -> None:
        self.model.load_state_dict(torch.load(self.best_model_placeholder_path / 'best_model.pth', weights_only=True))
    
    def get_history(
            self
        ) -> Dict:

        return self.history
    
    def get_model(
            self
        ) -> nn.Module:
        return self.model
    


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