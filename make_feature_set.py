import numpy as np
from tfgen import TFGen
from tfgen import get_observable_ec_top_n, get_observable_ec
from tfgen import InitialisingException
from data.eventlog import EventLog
import pandas as pd

from sklearn.decomposition import IncrementalPCA

IPCA_BATCH_SIZE = 2000
IPCA_COMPONENTS = 100
TFGEN_WINDOW_SIZE = 1000


class MakeDatasetML:
    def __init__(self, case_id_col, attributes_cols):
        self.case_id_col = case_id_col
        self.attributes_cols = attributes_cols

        self.ipca = None
        self.tfgen = None

    def print_eot_location(self, event_log_data):
        """
        Print the location of the EOT event. This was to check if the timestamp was universal
        :param event_log_data:
        :return:
        """
        # get index of EOT from proc_name
        eot_search = event_log_data[self.attributes_cols[0]].str.find('EOT').values
        len_eot_search = len(eot_search)

        for i in range(len_eot_search):
            if eot_search[i] == 0:
                print("Case end at", str(i / len_eot_search))

    def convert_data(self, event_log_data):
        self.ipca = IncrementalPCA(n_components=IPCA_COMPONENTS)
        event_log_data = event_log_data.astype(str)

        #self.print_eot_location(event_log_data)

        self.tfgen.load_from_dataframe(
            event_log_data, case_id_col=self.case_id_col,
            attributes_cols=self.attributes_cols
        )

        est_num_matrices = len(event_log_data) - TFGEN_WINDOW_SIZE + 1

        output_all_id = []
        output_all_feature = []

        done = False

        while True:
            if done:
                break

            batch_case_id = []
            batch_transition_matrices = []

            i = 0
            while i < IPCA_BATCH_SIZE:
                try:
                    data = self.tfgen.get_output_next()

                    batch_case_id.append(data[0])
                    batch_transition_matrices.append(np.ndarray.flatten(data[1]))
                    i += 1
                except InitialisingException:
                    continue
                except StopIteration:
                    done = True
                    break

            # n_components=100 must be less or equal to the batch number of samples 1.
            if i < IPCA_BATCH_SIZE:
                break

            self.ipca.partial_fit(batch_transition_matrices)
            reduced = self.ipca.transform(batch_transition_matrices).astype(np.float32)

            output_all_id.extend(batch_case_id)
            output_all_feature.extend(reduced)

            print("processed matrices: {}/{}".format(len(output_all_id), est_num_matrices))

        return np.asarray(output_all_id, dtype=str), np.asarray(output_all_feature, dtype=np.float32)

    def gen_dataset(self,
                    normal_set_file,
                    abnormal_set_file,
                    out_put_suffix,
                    event_limit_training=200000,
                    event_limit_testing=400000):
        """

        :param normal_set_file: path to normal event log file
        :param abnormal_set_file: path to abnormal event log file
        :param out_put_suffix: the suffix to add to the output file name
        :param event_limit_training: the number of events to use for training
        :param event_limit_testing: the number of events to use for testing
        :return:
        """

        columns = [self.case_id_col] + self.attributes_cols
        norm_event_log = pd.read_csv(normal_set_file)[columns]
        anom_data_event_log = pd.read_csv(abnormal_set_file)[columns]

        if event_limit_training is not None and event_limit_training is not None:

            if len(norm_event_log) < event_limit_training + event_limit_testing / 2:
                event_limit_normal_testing = len(norm_event_log) - event_limit_training
            else:
                event_limit_normal_testing = event_limit_testing // 2

            if len(anom_data_event_log) < event_limit_testing / 2:
                event_limit_anomalous_testing = len(anom_data_event_log)
            else:
                event_limit_anomalous_testing = event_limit_testing // 2
        else:
            event_limit_training = len(norm_event_log) // 2

            event_limit_normal_testing = len(norm_event_log) - event_limit_training
            event_limit_anomalous_testing = len(anom_data_event_log)

        oec = get_observable_ec_top_n(norm_event_log[self.attributes_cols], n=50)
        print('Number of events in normal data:', norm_event_log.shape)
        print('Number of observed event classes:', len(oec))

        self.tfgen = TFGen(oec, window_size=TFGEN_WINDOW_SIZE, method=TFGen.METHOD_CLASSIC)

        norm_tr = norm_event_log.iloc[:event_limit_training]
        print('Number of events in normal training data:', norm_tr.shape)
        norm_tr_cases, norm_tr_features = self.convert_data(norm_tr)

        np.savez_compressed(
            f'dataset_norm_tr{out_put_suffix}.npz',
            case_id=norm_tr_cases,
            features=norm_tr_features
        )
        print('Saved normal training data')

        # create normal testing
        norm_te = norm_event_log.iloc[event_limit_training:event_limit_training + event_limit_normal_testing]
        print('Number of events in normal testing data:', norm_te.shape)
        norm_te_cases, norm_te_features = self.convert_data(norm_te)

        np.savez_compressed(
            f'dataset_norm_te{out_put_suffix}.npz',
            case_id=norm_te_cases,
            features=norm_te_features
        )
        print('Saved normal testing data')

        anom_te = anom_data_event_log.iloc[:event_limit_anomalous_testing]
        print('Number of events in anomalous testing data:', anom_te.shape)
        anom_te_cases, anom_te_features = self.convert_data(anom_te)

        np.savez_compressed(
            f'dataset_anom_te{out_put_suffix}.npz',
            case_id=anom_te_cases,
            features=anom_te_features
        )
        print('Saved anomalous testing data')

        print('Done')


if __name__ == '__main__':
    mdl = MakeDatasetML('case_id', ['status', 'api'])
    mdl.gen_dataset(
        'event_log_cuckoo_hippo.csv.zip',
        'event_log_cuckoo_virus.csv.zip',
        '_hippo_tn',
        event_limit_training=200000,
        event_limit_testing=400000
    )

