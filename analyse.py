import time

import numpy as np
import pyod
import pandas as pd
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.auto_encoder_torch import AutoEncoder
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from detectors.conv_autoencoder import ConvAutoencoder
import matplotlib.pyplot as plt

np.random.seed(0)


class Analyse:
    def __init__(
            self,
            path_training_data,
            path_testing_data_normal,
            path_testing_data_anomaly
    ):
        training_data = np.load(path_training_data)
        print("Training data shape: {}".format(training_data['features'].shape))

        testing_data_normal = np.load(path_testing_data_normal)
        print("Testing data normal shape: {}".format(training_data['features'].shape))

        testing_data_anomaly = np.load(path_testing_data_anomaly)
        print("Testing data anomaly shape: {}".format(training_data['features'].shape))

        _, self.training_dataset = self.unpack_shuffle(training_data)

        self.anomaly_cases = np.unique(testing_data_anomaly['case_id'])

        self.testing_case = np.concatenate([testing_data_normal['case_id'], testing_data_anomaly['case_id']])
        self.testing_dataset = np.concatenate([testing_data_normal['features'], testing_data_anomaly['features']])

        self.all_testing_cases = np.unique(self.testing_case)

    def unpack_shuffle(self, data):
        case_ids, dataset = self.unpack(data)

        # shuffle the dataset
        length = dataset.shape[0]
        indices = np.random.permutation(length)

        dataset = dataset[indices]
        case_ids = case_ids[indices]

        return case_ids, dataset

    def unpack(self, data):
        case_ids = data['case_id']
        dataset = data['features']

        return case_ids, dataset

    def get_results(self, pyod_method):
        method = pyod_method()
        method.fit(self.training_dataset)

        method_name = type(method).__name__

        now = time.time()
        y_test_scores = method.decision_function(self.testing_dataset)
        print("Time taken for inference on {} samples: {}".format(self.testing_dataset.shape[0], time.time() - now))

        max_score = np.array([
            max(y_test_scores[np.where(self.testing_case == case)])
            for case in self.all_testing_cases
        ], dtype=np.float32)

        label = np.array([
            case in self.anomaly_cases
            for case in self.all_testing_cases
        ], dtype=bool)

        fpr, tpr, thresholds = metrics.roc_curve(label, max_score, pos_label=True)

        return method_name, fpr, tpr, thresholds


if __name__ == '__main__':
    list_methods = [ConvAutoencoder]
    a = Analyse(
        path_training_data="dataset_norm_tr_hippo_tn.npz",
        path_testing_data_normal="dataset_norm_te_hippo_tn.npz",
        path_testing_data_anomaly="dataset_anom_te_hippo_tn.npz"
    )

    plt.figure()

    for method in list_methods:
        name, fpr, tpr, thresholds = a.get_results(method)

        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=name + " AUC = %0.2f" % auc(fpr, tpr)
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig('roc.png', dpi=500)
