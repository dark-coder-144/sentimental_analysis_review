import sys 
from src.exception import CustomException 
from src.logger import logging 
import pytorch_lightning as pl
from dataclasses import dataclass 
from src.components.data_transformation import UCC_Data_Module
from src.components.model_training import UCC_Comment_Classifier
import numpy as np 
import torch

trainer = pl.Trainer(max_epochs=1, gpus=1, num_sanity_val_steps=50)

class PredictPipeline:
    def __init__(self):
        pass 
    def classify_raw_comments(model, dm):
        try:
            predictions = trainer.predict(model, datamodule=dm)
            flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
            return flattened_predictions
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict(self, features):
        try:
            test_data = features
            test_data.to_csv("notebook\data\encoded_emotions_test_dataset.csv")
            attributes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
            ucc_data_module = UCC_Data_Module(_, _, 'notebook\data\encoded_emotions_test_dataset.csv"', attributes=attributes)
            model1 = torch.load("/content/drive/MyDrive/MLProjects/EmotionDetectionText/RoBERTa_Fine_tuned.pkl")
            predictions = self.classify_raw_comments(model1, ucc_data_module)
            return predictions
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self, text: str):
        self.text=text 
    
    def get_data_as_data_frame(self):
        try: 
            custom_data_input_dict = {
                "text":self.text
            }
            return custom_data_input_dict
        except Exception as e:
            raise CustomException(e,sys)