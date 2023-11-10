#Data ingestion refers to the tools & processes used to collect data from various sources 
#and move it to a target site

import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass # used to create class variables in short


# Whenever we are performing data ingestion components there should be some inputs that will be required by the data ingestion component
# The input can be where to save train data, test data or where to save raw data and so on

# inside a class we generally have to define __init__. but using the dataclass decorator we will be able to directly define our class variables
# generally use decorators only when we are defining varialbes in the class. otherwise go with __init__ when working with funcitons in a class
@dataclass # this is a decorator
class DataIngestionConfig():
    
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')


class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self): # if the data is stored in some database or something, we will read data from source over here
        logging.info("Entered the Data ingestion method or component")
        try:
            data = pd.read_csv("notebook/dataset/studentPerformanceDataset.csv")
            logging.info("Read the dataset as dataframe sucessfully")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(data,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of Data Completed")

            # we will be requiring this information in our data transformation. 
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()