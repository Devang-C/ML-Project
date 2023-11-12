import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
import dill

from sklearn.metrics import r2_score

def save_obj(filepath,obj):
    try:
        dir_path = os.path.dirname(filepath)

        os.makedirs(dir_path,exist_ok=True)

        with open (filepath,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except:
        pass



def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(x_train,y_train)

            y_predicted = model.predict(x_test)

            model_score = r2_score(y_test,y_predicted)

            report[list(models.keys())[i]] = model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)