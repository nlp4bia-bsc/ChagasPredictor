import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

class MeanBERT(RobertaForSequenceClassification):
    def __init__(self, config: RobertaConfig, pretrained_model: str=None, **kwargs):
        super().__init__(config)
        self.config = config
        if pretrained_model:
            base_model = RobertaModel.from_pretrained(
                pretrained_model,
                config=config,
                local_files_only=kwargs.pop("local_files_only", False),
            )

            self.roberta.load_state_dict(base_model.state_dict(), strict=False)

        if len(self.config.freeze_bert) != 0: 
            self._freeze_bert_layers(self.config.freeze_bert)

        self.config.problem_type = "multi_label_classification"
        self.loss_fct = getattr(nn, self.config.loss_fct)()

        self.dropout = nn.Dropout(
            config.classifier_dropout if hasattr(config, 'classifier_dropout') else config.hidden_dropout_prob
        )
        self.classifier = nn.ModuleList([
            nn.Linear(self.roberta.config.hidden_size, self.config.output_dim)
            for _ in range(self.config.num_tasks)
        ])

    def _freeze_bert_layers(self, freeze_layer_names: list):
        # freeze by parameter name or layer index e.g. ['embeddings', 'encoder.layer.0']
        for name, param in self.roberta.named_parameters():
            if any(fl in name for fl in freeze_layer_names):
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, card_labels: torch.Tensor, dig_labels: torch.Tensor, visit_chunk_size: int = 128):
        """
        input_ids: (B, M, S) long
        attention_mask: (B, M, S) long
        card_labels, dig_labels: (B,) float (0./1.)
        visit_chunk_size: int, number of visits to process at once to control memory
        
        where B = batch size (number of cases), M = max number of visits in batch, S = max seq len per visit
        """
        B, M, S = input_ids.shape

        # flatten visits => (B*M, S)
        flat_input = input_ids.view(B * M, S)
        flat_attn  = attention_mask.view(B * M, S)

        # mask which flattened visits are real (sum attn > 0)
        visit_exists = (flat_attn.sum(dim=-1) > 0).to(torch.float32)  # (B*M,)

        # process visits in chunks to control memory; collect pooled outputs
        pooled_chunks = []
        for start in range(0, B * M, visit_chunk_size):
            end = min(B * M, start + visit_chunk_size)
            out = self.roberta(input_ids=flat_input[start:end], attention_mask=flat_attn[start:end], return_dict=True)
            # use CLS from last_hidden_state
            last_hidden = out.last_hidden_state          # (batch*M (or visit_chunk_size), seq_len, hidden)
            cls_vec = last_hidden[:, 0, :]  
            pooled_chunks.append(cls_vec)  

        pooled_flat = torch.cat(pooled_chunks, dim=0)      # (B*M, hidden)
        hidden_size = pooled_flat.size(-1)
        pooled_visits = pooled_flat.view(B, M, hidden_size)  # (B, M, hidden)

        # mask padded visits (B, M)
        visit_mask = visit_exists.view(B, M).to(pooled_visits.dtype)  # 1.0 for real visits, 0.0 for padded visits
        visit_mask_unsq = visit_mask.unsqueeze(-1)  # (B, M, 1)

        summed = (pooled_visits * visit_mask_unsq).sum(dim=1)    # (B, hidden)
        denom = visit_mask_unsq.sum(dim=1).clamp_min(1.0)        # (B, 1)
        avg_cls = summed / denom                                # (B, hidden)
        avg_cls = self.dropout(avg_cls)

        # heads -> list of logits per task
        logits = [head(avg_cls) for head in self.classifier]    # each element (B, 1)
        total_loss, card_loss, dig_loss = self.compute_multitask_loss(logits, card_labels, dig_labels)
        
        return (SequenceClassifierOutput(
            loss=total_loss,
            logits=logits
        ), card_loss, dig_loss)

    def compute_multitask_loss(self, task_logits, card_labels, dig_labels, weights=(1.0, 1.0)):
        """
        task_logits: list [logits_card, logits_dig] each shape (B, 2)
        card_labels, dig_labels: (B,) long (0/1)
        weights: tuple of floats to weight task losses
        """
        card_loss = self.loss_fct(task_logits[0], card_labels.unsqueeze(1))
        dig_loss = self.loss_fct(task_logits[1], dig_labels.unsqueeze(1))
        total_loss = weights[0] * card_loss + weights[1] * dig_loss
        return total_loss, card_loss, dig_loss


class LSTMBERT(RobertaForSequenceClassification):
    def __init__(self, config: RobertaConfig, pretrained_model: str=None, **kwargs):
        super().__init__(config)
        self.config = config
        if pretrained_model:
            base_model = RobertaModel.from_pretrained(
                pretrained_model,
                config=config,
                local_files_only=kwargs.pop("local_files_only", False),
            )

            self.roberta.load_state_dict(base_model.state_dict(), strict=False)

        if len(self.config.freeze_bert) != 0:
            self._freeze_bert_layers(self.config.freeze_bert)

        self.config.problem_type = "multi_label_classification"
        self.loss_fct = getattr(nn, self.config.loss_fct)()

        self.dropout = nn.Dropout(
            config.classifier_dropout if hasattr(config, 'classifier_dropout') else config.hidden_dropout_prob
        )

        # LSTM: bidirectional -> hidden dim doubles in outputs
        self.lstm = nn.LSTM(self.roberta.config.hidden_size, self.config.lstm_hidden, batch_first=True, bidirectional=True)

        # **Important**: classifier must accept 2 * lstm_hidden because LSTM is bidirectional
        self.classifier = nn.ModuleList([
            nn.Linear(self.config.lstm_hidden * 2, self.config.output_dim)
            for _ in range(self.config.num_tasks)
        ])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, card_labels: torch.Tensor, dig_labels: torch.Tensor, visit_chunk_size: int = 128):
        """
        input_ids: (B, M, S)
        attention_mask: (B, M, S)
        card_labels, dig_labels: (B,)
        visit_chunk_size: number of visits to process at once
        """
        B, M, S = input_ids.shape

        # flatten visits => (B*M, S)
        flat_input = input_ids.view(B * M, S)
        flat_attn  = attention_mask.view(B * M, S)

        # mask which flattened visits are real (sum attn > 0)
        visit_exists = (flat_attn.sum(dim=-1) > 0).to(torch.float32)  # (B*M,)

        # process visits in chunks to control memory; collect pooled outputs (CLS)
        pooled_chunks = []
        for start in range(0, B * M, visit_chunk_size):
            end = min(B * M, start + visit_chunk_size)
            out = self.roberta(input_ids=flat_input[start:end], attention_mask=flat_attn[start:end], return_dict=True)
            last_hidden = out.last_hidden_state          # (chunk, seq_len, hidden)
            cls_vec = last_hidden[:, 0, :]               # (chunk, hidden)
            pooled_chunks.append(cls_vec)

        pooled_flat = torch.cat(pooled_chunks, dim=0)      # (B*M, hidden)
        hidden_size = pooled_flat.size(-1)
        pooled_visits = pooled_flat.view(B, M, hidden_size)  # (B, M, hidden)

        # mask padded visits (B, M)
        visit_mask = visit_exists.view(B, M).to(pooled_visits.dtype)  # 1.0 for real visits, 0.0 for padded visits

        # compute lengths (number of real visits per sample)
        lengths = visit_mask.sum(dim=1).to(torch.long)  # (B,)

        # to avoid zero-length sequences for pack_padded_sequence, clamp min 1
        lengths_clamped = lengths.clamp_min(1)  # (B,)

        # Pack / run LSTM
        ## Packing tells the LSTM where each patient's history actually ends. We prevent it from reading fake visits created by padding.
        ## After processing, we pad back to a full batch shape so we can index and work normally.
        # Note: pack_padded_sequence expects CPU lengths
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        packed = pack_padded_sequence(pooled_visits, lengths_clamped.cpu(), batch_first=True, enforce_sorted=False) # remove padding in order to avoid LSTM processing the padded visits: "After visit X, stop reading, remaining visits are padding"
        packed_out, (h_n, c_n) = self.lstm(packed)
        out_unpacked, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=M)  # convert back to (B, M, 2*lstm_hidden)

        # gather last valid timestep for each sequence (lengths_clamped - 1)
        last_idxs = (lengths_clamped - 1).to(torch.long)  # (B,)
        batch_idx = torch.arange(B, device=pooled_visits.device)
        lstm_last = out_unpacked[batch_idx, last_idxs, :]  # (B, 2*lstm_hidden)

        x = self.dropout(lstm_last)  # (B, 2*lstm_hidden)

        logits = [head(x) for head in self.classifier]    # each element (B, output_dim)

        total_loss, card_loss, dig_loss = self.compute_multitask_loss(logits, card_labels, dig_labels)

        return (SequenceClassifierOutput(
            loss=total_loss,
            logits=logits
        ), card_loss, dig_loss)

    def compute_multitask_loss(self, task_logits, card_labels, dig_labels, weights=(1.0, 1.0)):
        """
        task_logits: list [logits_card, logits_dig] each shape (B, 2)
        card_labels, dig_labels: (B,) long (0/1)
        weights: tuple of floats to weight task losses
        """
        card_loss = self.loss_fct(task_logits[0], card_labels.unsqueeze(1))
        dig_loss = self.loss_fct(task_logits[1], dig_labels.unsqueeze(1))
        total_loss = weights[0] * card_loss + weights[1] * dig_loss
        return total_loss, card_loss, dig_loss
