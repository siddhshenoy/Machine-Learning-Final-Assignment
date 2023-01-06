import pandas as pd
import custom.DataProcessor as dp
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

from yellowbrick.classifier import ROCAUC

class Model:
    def __init__(self):
        self.data = {
            "x_train": None,
            "y_train": None,
            "x_test": None,
            "y_test": None
        }
        self.dataframe = None
        self.featureData = None
        self.targetData = None
        self.model_name = ""
        self.model = None
        self.strategy = None
        self.total_kfolds = 0
        self.total_crossvals = 0
        self.C_value = None
        self.K_value = None
        self.alpha_value = None
        self.dummy_classifier_strategy = None
        self.sorted_coef_list_indices = []
        self.sorted_coef_list = []
        self.cross_val_scores = None
        self.plot_figures = []
        self.pred_y = None
    def create(self, model, C_value=None, K_value=None, alpha=None, dummy_classifier_strategy=None, initialized=False):
        self.C_value = C_value
        self.K_value = K_value
        self.alpha_value = alpha
        self.dummy_classifier_strategy = dummy_classifier_strategy
        if initialized == False:
            if self.K_value is not None:
                self.model = model(n_neighbors=self.K_value)
            elif self.alpha_value is not None:
                self.model = model(alpha=self.alpha_value)
            elif self.C_value is not None:
                self.model = model(C=self.C_value)
            elif self.dummy_classifier_strategy is not None:
                self.model = model(strategy=self.dummy_classifier_strategy)
            else:
                self.model = model()
        else:
            self.model = model
        self.model_name = type(self.model).__name__
        print(f"Created a '{self.model_name}' model")

    def load_from_csvfile(self, csv_path):
        try:
            self.dataframe = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"File {csv_path} does not exist!")

    def use_hold_out_strategy(self):
        self.strategy = "HOLD_OUT"

    def use_kfold(self, folds):
        self.strategy = "KFOLD"
        self.total_kfolds = folds
    def use_cross_validation(self, total_cross_validations):
        self.strategy = "CROSS_VAL"
        self.total_crossvals = total_cross_validations

    def load_from_dataprocessor(self, dataprocessor, alias):
        pass

    def load_data(self, feature_data, target_data):
        self.featureData = feature_data
        self.targetData = target_data

    def split_data(self, split_size=0.3):
        if self.strategy == "HOLD_OUT" or self.strategy == "CROSS_VAL":
            self.data["x_train"], self.data["x_test"], self.data["y_train"], self.data["y_test"] = train_test_split(self.featureData, self.targetData, random_state=0, test_size = split_size)
#			print(self.data["x_train"].shape)
#			print(self.data["x_test"].shape)
#			print(self.data["y_train"].shape)
#			print(self.data["y_test"].shape)

    def fit_model(self):
        if self.strategy == "HOLD_OUT":
            self.pred_y = self.model.fit(self.data["x_train"], self.data["y_train"])
        elif self.strategy == "KFOLD":
            kfold = KFold(n_splits=self.total_kfolds)
            for train, test in kfold.split(self.featureData):
                self.model.fit(self.featureData[train], self.targetData[train])
                #self.print_model_stats()
        elif self.strategy == "CROSS_VAL":
            clf = self.model.fit(self.data["x_train"], self.data["y_train"])
            
            self.cross_val_scores = cross_val_score(clf, self.data["x_test"], self.data["y_test"], cv=self.total_crossvals)
            print(self.cross_val_scores)
            print("Accuracy: %0.2f (+/− %0.2f)" % (self.cross_val_scores .mean(), self.cross_val_scores .std()))
            
            

    def get_dataframe(self):
        return self.dataframe

    def get_all_data(self):
        return self.data

    def predict(self, x):
        result = self.model.predict(x)

        return result

    def get_model(self):
        return self.model

    def print_model_stats(self):
        result = self.model.predict(self.data["x_test"])
        print(result.shape)
        print("=" * 50)
        print("Coefficients")
        if self.model_name != "KNeighborsClassifier" and self.model_name != "DummyClassifier" and self.model_name != "SVC":
            np.set_printoptions(suppress=True)
            # print(self.model.coef_)
            for arr in self.model.coef_:
                #print(np.sort(arr))
                self.sorted_coef_list.append(np.sort(arr))
                self.sorted_coef_list_indices.append(np.argsort(self.model.coef_))
            np.set_printoptions(suppress=False)
            
        #if self.model_name == "SVC":
        print("MSE: {}".format(mean_squared_error(self.data["y_test"], result)))
        print(confusion_matrix(self.data["y_test"], result))
        print(classification_report(self.data["y_test"], result))
        score = self.model.score(self.data["x_test"], self.data["y_test"])
        print(f"Accuracy: {score}")
        print("=" * 50)
        return score, mean_squared_error(self.data["y_test"], result)
    def get_sorted_coef_list(self):
        return self.sorted_coef_list
    def get_sorted_coef_list_idx(self):
        return self.sorted_coef_list_indices
    def get_cross_val_scores(self):
        return self.cross_val_scores
    def plot_ROC_AUC_curve(self, title=None,outpath=None):
        visualizer = ROCAUC(self.model, macro=False, micro=False)
        if title is not None:
            visualizer.title = title
        # Fitting to the training data first then scoring with the test data
        visualizer.fit(self.data["x_train"], self.data["y_train"])
        visualizer.score(self.data["x_test"], self.data["y_test"])
        if outpath is None:
            visualizer.show()
        else:
            visualizer.show(outpath=outpath)
        return visualizer