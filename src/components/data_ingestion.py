import os 
import sys 
from src.exception import CustomException 
from src.logger import logging 
import pandas as pd
import pytorch_lightning as pl
from dataclasses import dataclass 
from src.components.data_transformation import DataTransformation
from src.components.model_training import UCC_Comment_Classifier
import torch 

@dataclass 
class DataIngestionConfig:
    train_data_path = str=os.path.join('artifacts', "train_split.csv")
    test_data_path = str=os.path.join('artifacts', "test_split.csv")
    validation_data_path = str=os.path.join('artifacts', "validation_split.csv") 
    raw_data_path = str=os.path.join('artifacts', "raw_data.csv")

class DataIngestion: 
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\encoded_emotion_dataset.csv') 
            logging.info('Read the dataset as DataFrame') 

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) 

            logging.info("Train, Test and Validation data is initiated")

            train_set = pd.read_csv('notebook\data\encoded_emotion_dataset_train.csv')
            test_set = pd.read_csv('notebook\data\encoded_emotions_test_dataset.csv')
            validation_set = pd.read_csv('notebook\data\encoded_emotions_valid_dataset.csv') 

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) 
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) 
            validation_set.to_csv(self.ingestion_config.validation_data_path, index=False, header=True) 

            logging.info("Data Ingestion is completed!") 

            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path, 
                self.ingestion_config.validation_data_path
            )
        except Exception as e: 
            raise CustomException(e,sys) 

if __name__=="__main__":
    
    obj=DataIngestion()
    train_path, test_path, validation_path = obj.initiate_data_ingestion() 
    logging.info("Data ingestion is done!")

    attributes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    ucc_data_module = DataTransformation(train_path, validation_path, test_path, attributes=attributes)
    ucc_data_module.setup()
    ucc_data_module.train_dataloader()
    logging.info("Data Transformation is done!")
    
    config = {
    'model_name': 'distilroberta-base',
    'n_labels': len(attributes),
    'batch_size': 128,
    'lr': 1.5e-6,
    'warmup': 0.2, 
    'train_size': len(ucc_data_module.train_dataloader()),
    'weight_decay': 0.001,
    'n_epochs': 100
    }
    
    model = UCC_Comment_Classifier(config)
    trainer = pl.Trainer(max_epochs=config['n_epochs'], gpus=1, num_sanity_val_steps=50)
    trainer.fit(model, ucc_data_module)
    torch.save(model, "RoBERTa_Fine_tuned.pkl")
    logging.info(f"Model Training Completed")
