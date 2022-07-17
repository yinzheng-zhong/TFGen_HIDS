import pandas as pd
import numpy as np


class EventLog:
    def __init__(self, path_normal_even_log,
                 path_anomalous_even_log,
                 num_normal_cases_for_training=40,
                 num_normal_cases_for_testing=20,
                 num_anomalous_cases_for_testing=20):
        """
        Load data from the process event logs. This class makes data for anomaly-based detection, so only normal data
        are used for training. A part of normal cases will be mixed with anomalous cases for testing. Define the
        proportion of anomalous cases in parameter. The rest of data will be not be used anywhere. 50 normal cases
        contain around 5 million events.

        :param path_normal_even_log: path to normal event log
        :param path_anomalous_even_log: path to anomalous event log
        :param num_normal_cases_for_testing: percentage of normal cases to reserve for testing
        :param num_anomalous_cases_for_testing: percentage of anomalous cases to use for testing
        """
        self.path_normal_even_log = path_normal_even_log
        self.path_anomalous_even_log = path_anomalous_even_log

        self.num_normal_cases_for_training = num_normal_cases_for_training
        self.num_normal_cases_for_testing = num_normal_cases_for_testing
        self.num_anomalous_cases_for_testing = num_anomalous_cases_for_testing

        self.size_event_log_normal = 0
        self.size_event_log_anomalous = 0

        self.training_data = pd.DataFrame()
        self.testing_data_mixed = pd.DataFrame()
        self.testing_data_mixed_label = pd.DataFrame(columns=['case_id', 'label'], dtype=np.int)

        self._load_data()

    def _load_data(self):
        event_log_normal = pd.read_csv(self.path_normal_even_log, compression="zip", dtype="string[pyarrow]")
        event_log_anomalous = pd.read_csv(self.path_anomalous_even_log, compression="zip", dtype="string[pyarrow]")

        unique_normal_cases = event_log_normal.case_id.unique()
        unique_anomalous_cases = event_log_anomalous.case_id.unique()

        self.size_event_log_normal = len(unique_normal_cases)
        self.size_event_log_anomalous = len(unique_anomalous_cases)

        set_unique_cases_normal = set(unique_normal_cases)

        np.random.seed(0)

        # divide normal cases into training and testing
        normal_training_cases = set(
            np.random.choice(
                list(set_unique_cases_normal),
                int(self.num_normal_cases_for_training),
                replace=False)
        )

        normal_cases_remaining = set(set_unique_cases_normal) - normal_training_cases
        normal_testing_cases = np.random.choice(
            list(normal_cases_remaining),
            int(self.num_normal_cases_for_testing),
            replace=False
        )

        # get anomalous cases for testing
        anomalous_testing_cases = np.random.choice(
            list(unique_anomalous_cases),
            int(self.num_anomalous_cases_for_testing),
            replace=False
        )

        self.training_data = event_log_normal[event_log_normal.case_id.isin(normal_training_cases)]

        # get mixed cases for testing
        self.testing_data_normal = event_log_normal[event_log_normal.case_id.isin(normal_testing_cases)]
        self.testing_data_anomalous = event_log_anomalous[
            event_log_anomalous.case_id.isin(anomalous_testing_cases)
        ]

        # add label for testing data
        for case_id in normal_testing_cases:
            self.testing_data_mixed_label.loc[len(self.testing_data_mixed_label)] = [case_id, 0]

        for case_id in anomalous_testing_cases:
            self.testing_data_mixed_label.loc[len(self.testing_data_mixed_label)] = [case_id, 1]

        self.testing_data_mixed = pd.concat([self.testing_data_normal, self.testing_data_anomalous])

    def get_training_data(self):
        pass
