import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
import dill

from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV

def save_obj(filepath,obj):
    try:
        dir_path = os.path.dirname(filepath)

        os.makedirs(dir_path,exist_ok=True)

        with open (filepath,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except:
        pass



def evaluate_model(x_train,y_train,x_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            # adding the hyperparameter tuining
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_predicted = model.predict(x_test)

            model_score = r2_score(y_test,y_predicted)

            report[list(models.keys())[i]] = model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)