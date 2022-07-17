import numpy as np
from tfgen import get_observable_ec_top_n, get_observable_ec

import warnings
from make_feature_set import MakeDatasetML
import pandas as pd

TFGEN_WINDOW_SIZE = 1000


class MakeDatasetMLNEC(MakeDatasetML):
    def __init__(self, case_id_col, attributes_cols):
        """
        Class that use top n event classes instead of using PCA
        :param observable_event_classes:
        :param case_id_col:
        :param attributes_cols:
        """
        super().__init__(case_id_col, attributes_cols)

    def convert_data(self, event_log_data):
        event_log_data = event_log_data.astype(str)

        #self.print_eot_location(event_log_data)

        print('before load to tfgen')
        self.tfgen.load_from_dataframe(
            event_log_data, case_id_col=self.case_id_col,
            attributes_cols=self.attributes_cols
        )

        print('after load to tfgen')

        out_tuple = self.tfgen.get_output_list()

        list_case_id = [i[0] for i in out_tuple]
        features = np.array([np.ndarray.flatten(i[1]) for i in out_tuple])

        return np.asarray(list_case_id, dtype=str), np.asarray(features, dtype=np.float32)


if __name__ == '__main__':
    mdl = MakeDatasetMLNEC('case_id', ['status', 'api'])
    mdl.gen_dataset(
        'event_log_cuckoo_hippo.csv.zip',
        'event_log_cuckoo_virus.csv.zip',
        '_hippo_tn',
        event_limit_training=200000,
        event_limit_testing=1000000
    )
