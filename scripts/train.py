import logging
import torch
import numpy as np
from typing import Dict
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, RobertaConfig, get_scheduler
import operator
import os

from models.overlap_bert import MeanBERT, LSTMBERT
from data.dataset import OverlapDataset, collate_cases
from data.load_data import load_data
from pipeline.multi_bert import train_epoch, evaluate_model
from utils.evaluation import multilabel_eval

comparison_ops = {
    'val_loss': operator.lt,
    'ham_loss': operator.lt,
    'val_accuracy': operator.gt,
    'val_macro_f1': operator.gt,
    'val_weighted_f1': operator.gt,
}

def main(logger: logging.Logger, config: Dict, method: str):
    train_df, test_df, val_df = load_data(config['general']['dataset'], config['general']['label_names'])
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer_path'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = os.path.join(config['training']['output_dir'], method)

    # Create datasets and dataloaders
    train_dataset = OverlapDataset(train_df['visits'].to_list(), train_df['card'].to_list(), train_df['dig'].to_list())
    val_dataset = OverlapDataset(val_df['visits'].to_list(), val_df['card'].to_list(), val_df['dig'].to_list())
    test_dataset = OverlapDataset(test_df['visits'].to_list(), test_df['card'].to_list(), test_df['dig'].to_list())

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=lambda b: collate_cases(b, tokenizer, max_length=512)
        )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        collate_fn=lambda b: collate_cases(b, tokenizer, max_length=512)
        )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        collate_fn=lambda b: collate_cases(b, tokenizer, max_length=512)
        )


    # Create RobertaConfig for the hf pretrained model
    hf_config = RobertaConfig.from_pretrained(config['model']['tokenizer_path'])
    hf_config.loss_fct = config['training']['loss']
    hf_config.output_dim = config['model']['output_dim']
    hf_config.num_tasks = config['model']['num_tasks']
    hf_config.freeze_bert = config['model']['freeze_bert']
    hf_config.classifier_dropout = config['model']['dropout']
    if method == 'mean':
        model = MeanBERT(hf_config, pretrained_model=config['model']['pretrained_model'])
    elif method == 'lstm':
        hf_config.lstm_hidden = 256
        model = LSTMBERT(hf_config, pretrained_model=config['model']['pretrained_model'])
    elif method == 'lstm-attn':
        hf_config.lstm_hidden = 256
        hf_config.attn_dim = 128
        model = LSTMBERT(hf_config, pretrained_model=config['model']['pretrained_model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize training methods
    accelerator = Accelerator()
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])

    num_epochs = config['training']['num_epochs']
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )

    # Training loop
    best_es_metric_val = 0
    es_drag = 0
    es_metric = config['training']['early_stopping_metric']
    es_patience = config['training']['early_stopping_patience']
    es_compare = comparison_ops[es_metric]
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 50)
        
        # Train
        train_loss, card_train_loss, dig_train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        logger.info(f"Train Loss: {train_loss:.4f} (card: {card_train_loss:.4f}, dig: {dig_train_loss:.4f})")
        
        # Evaluate
        val_loss, card_preds, dig_preds, card_actual, dig_actual = evaluate_model(
            model, val_loader, device
        )
        logger.info(f"Val Loss: {val_loss:.4f}")

        val_metrics = multilabel_eval(
            np.array([(int(c), int(d)) for c, d in zip(card_actual, dig_actual)]),
            np.array([(c, d) for c, d in zip(card_preds, dig_preds)]),
            logger, 
            target_names=['Cardiological', 'Digestive'], 
            multi_task_target_names=['Neither', 'Cardiological Only', 'Digestive Only', 'Mixed']
        )
        val_metrics['es']['val_loss'] = val_loss  

        # Save best model based on average F1 score
        if es_compare(val_metrics['es'][es_metric], best_es_metric_val) or epoch == 0:
            best_es_metric_val = val_metrics['es'][es_metric]
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"New best model saved in {output_dir}!")
            es_drag = 0
        else:
            es_drag += 1
            if es_drag >= es_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

    accelerator.wait_for_everyone()
    best_config = RobertaConfig.from_pretrained(output_dir)
    if method == 'mean':
        best_model = MeanBERT.from_pretrained(output_dir, config=best_config)
    if method == 'lstm' or method =='lstm-attn':
        best_model = LSTMBERT.from_pretrained(output_dir, config=best_config)
    best_model.to(device)
    # test
    test_loss, card_preds, dig_preds, card_actual, dig_actual = evaluate_model(
        best_model, test_loader, device
    )
    # Final evaluation
    logger.info("\nTest Set Evaluation:")
    logger.info(f"Val Loss: {test_loss:.4f}")
    val_metrics = multilabel_eval(
        np.array([(int(c), int(d)) for c, d in zip(card_actual, dig_actual)]),
        np.array([(c, d) for c, d in zip(card_preds, dig_preds)]),
        logger, 
        target_names=['Cardiological', 'Digestive'], 
        multi_task_target_names=['Neither', 'Cardiological Only', 'Digestive Only', 'Mixed']
    )
if __name__ == "__main__":
    main()