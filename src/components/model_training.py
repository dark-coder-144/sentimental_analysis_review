from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
import torch.nn as nn
import math
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
import torch 
import sys 
import os 
import pandas as pd  
from src.logger import logging 
from src.exception import CustomException
import pytorch_lightning as pl


class UCC_Comment_Classifier(pl.LightningModule):

    def __init__(self, config: dict):
        try:
            super().__init__()
            self.config = config
            self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True)
            self.hidden = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
            self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
            torch.nn.init.xavier_uniform_(self.classifier.weight)
            self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
            self.dropout = nn.Dropout()
            logging.info("Model Initialization Done!")
        except Exception as e:
            raise CustomException(e,sys) 
        
    def forward(self, input_ids, attention_mask, labels=None):
        try:
            # roberta layer
            output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = torch.mean(output.last_hidden_state, 1)
            # final logits
            pooled_output = self.dropout(pooled_output)
            pooled_output = self.hidden(pooled_output)
            pooled_output = F.relu(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            # calculate loss
            loss = 0
            if labels is not None:
                loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
            logging.info("Forward Layer Done!")
            return loss, logits
        except Exception as e:
            raise CustomException(e,sys)

    def training_step(self, batch, batch_index):
        try:
            loss, outputs = self(**batch)
            self.log("train loss ", loss, prog_bar = True, logger=True)
            logging.info("Training Steps Done!")
            return {"loss":loss, "predictions":outputs, "labels": batch["labels"]}
        except Exception as e:
            raise CustomException(e,sys)
        
    def validation_step(self, batch, batch_index):
        try:
            loss, outputs = self(**batch)
            self.log("validation loss ", loss, prog_bar = True, logger=True)
            logging.info("Validation step done!")
            return {"val_loss": loss, "predictions":outputs, "labels": batch["labels"]}
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict_step(self, batch, batch_index):
        try:
            loss, outputs = self(**batch)
            logging.info("Predict Step done!")
            return outputs
        except Exception as e:
            raise CustomException(e,sys)
        
    def configure_optimizers(self):
        try:
            optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
            total_steps = self.config['train_size']/self.config['batch_size']
            warmup_steps = math.floor(total_steps * self.config['warmup'])
            warmup_steps = math.floor(total_steps * self.config['warmup'])
            logging.info("Optimization Done!")
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            return [optimizer],[scheduler]
        except Exception as e:
            raise CustomException(e,sys)
        

  # def validation_epoch_end(self, outputs):
  #   losses = []
  #   for output in outputs:
  #     loss = output['val_loss'].detach().cpu()
  #     losses.append(loss)
  #   avg_loss = torch.mean(torch.stack(losses))
  #   self.log("avg_val_loss", avg_loss)


