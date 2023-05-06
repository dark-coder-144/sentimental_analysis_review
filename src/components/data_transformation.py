import sys 
from dataclasses import dataclass  
import pandas as pd 
from src.logger import logging 
from src.exception import CustomException 
from torch.utils.data import Dataset 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer

class UCC_Dataset(Dataset):

    def __init__(self, data_path, tokenizer, attributes, max_token_len: int = 128, sample = 3500):
        try:                
            self.data_path = data_path
            self.tokenizer = tokenizer
            self.attributes = attributes
            self.max_token_len = max_token_len
            self.sample = sample
            self._prepare_data()
            logging.info("Data Preparation is done!")
        except Exception as e:
            raise CustomException(e,sys) 


    def _prepare_data(self):
        try:                
            data = pd.read_csv(self.data_path)
    #         data['unhealthy'] = np.where(data['healthy'] == 1, 0, 1)
            if self.sample is not None:
                sadness = data.loc[data['sadness'] == 1]
                joy = data.loc[data['joy'] == 1]
                love = data.loc[data['love'] == 1]
                anger = data.loc[data['anger'] == 1]
                fear = data.loc[data['fear'] == 1]
                surprise = data.loc[data['surprise'] == 1]
                self.data = pd.concat([sadness.sample(self.sample, random_state=7), joy.sample(self.sample, random_state=7), love, anger, fear, surprise])
            else:
                self.data = data
        except Exception as e:
            raise CustomException(e,sys) 
        
    def __len__(self):
        try:
            return len(self.data)
        except Exception as e:
            raise CustomException(e, sys)
        
    def __getitem__(self, index):
        try:
            item = self.data.iloc[index]
            comment = str(item.text)
            attributes = torch.FloatTensor(item[self.attributes])
            tokens = self.tokenizer.encode_plus(comment,
                                                add_special_tokens=True,
                                                return_tensors='pt',
                                                truncation=True,
                                                padding='max_length',
                                                max_length=self.max_token_len,
                                                return_attention_mask = True)
            return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': attributes}
        except Exception as e:
            raise CustomException(e,sys)

class DataTransformation(pl.LightningDataModule):

    def __init__(self, train_path, val_path, test_path, attributes, batch_size: int = 16, max_token_length: int = 128,  model_name='roberta-base'):
        try:
            super().__init__()
            self.train_path = train_path
            self.val_path = val_path
            self.test_path = test_path
            self.attributes = attributes
            self.batch_size = batch_size
            self.max_token_length = max_token_length
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise CustomException(e,sys)

    def setup(self, stage = None):
        try:
            if stage in (None, "fit"):
                self.train_dataset = UCC_Dataset(self.train_path, attributes=self.attributes, tokenizer=self.tokenizer)
                self.val_dataset = UCC_Dataset(self.val_path, attributes=self.attributes, tokenizer=self.tokenizer, sample=None)
            if stage == 'predict':
                self.test_dataset = UCC_Dataset(self.test_path, attributes=self.attributes, tokenizer=self.tokenizer, sample=None)
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_dataloader(self):
        try:
            return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=2, shuffle=True)
        except Exception as e:
            raise CustomException(e,sys)
        
    def val_dataloader(self):
        try:
            return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=2, shuffle=False)
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict_dataloader(self):
        try:
            return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=2, shuffle=False)
        except Exception as e:
            raise CustomException(e, sys)

