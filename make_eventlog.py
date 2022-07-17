import time

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool


class MakeEventLog:
    TOKEN_END_OF_TRACE = "EOT"

    def __init__(self, workers=8):
        """
        This class was supposed to be used for creating event logs from the KernelDriver dataset. However, it is not
        used anymore. Use MakeEventLogCuckoo from make_eventlog_cuckoo.py instead.

        Load data from kernel driver dataset
        :param paths: list of path to normal data
        """
        self.paths = []
        self.case_id_prefix = ''

        self.workers = workers

        self.folders = []

        self.total_num_cases = 0

        self.dir_id_mapping_normal = {}

        self._scan_folders()
        self._map_dir_to_id()

    def _scan_folders(self):
        """
        Scan folders and return list of folders
        :return: list of folders
        """
        self.folders = [
            os.path.join(path, subfolder).replace("\\", "/")
            for path in self.paths for subfolder in os.listdir(path)
        ]

        self.total_num_cases = len(self.folders)

    def _map_dir_to_id(self):
        """
        Map folder to id
        :return:
        """

        i = 0
        for folder in self.folders:
            self.dir_id_mapping_normal[folder] = i
            i += 1

    @staticmethod
    def _load_one_call(path):
        df = pd.read_csv(path)

        return df

    def _load_case(self, args):
        """
        Load all calls from a process. A process folder is like: 0, 1, ...
        :return:
        """

        case_id, process_folder = args
        case_id = self.case_id_prefix + str(case_id)

        all_calls = pd.DataFrame(columns=['time', 'pid', 'method', 'proc_name', 'args'])

        for file in os.listdir(process_folder):
            if file != 'name.txt':
                path = os.path.join(process_folder, file).replace("\\", "/")
                if os.stat(path).st_size != 0:
                    try:
                        calls = pd.read_csv(path, header=None).dropna()
                    except (pd.errors.ParserError, UnicodeDecodeError):
                        continue

                    try:
                        num_cols = len(calls.iloc[0])
                    except IndexError:
                        continue

                    # we need to make sure all columns are correct. As some files only have 2 cols and some have 4 cols
                    if num_cols == 4:
                        calls.columns = ['time', 'pid', 'method', 'proc_name']
                        calls['args'] = ''
                    elif num_cols == 5:
                        calls.columns = ['time', 'pid', 'method', 'proc_name', 'args']
                    else:
                        continue

                    all_calls = pd.concat([all_calls, calls], ignore_index=True)

        # add rows from calls to all_calls
        all_calls['case_id'] = case_id

        all_calls.drop(columns=['pid'], inplace=True)

        # remove unnecessary strings to save memory
        all_calls['time'] = all_calls['time'].str.replace('Time=', '')
        all_calls['method'] = all_calls['method'].str.replace('MethodName=', '')
        # just keep the system processes are kept, other names are kept as user
        all_calls['proc_name'] = all_calls['proc_name'].apply(lambda x: x.split("\\")[-1] if "system32" in x else "user")

        try:
            max_time = str(int(all_calls['time'].max()) + 1)
        except ValueError:
            # deal with empty folders
            return None

        # finally and the TOKEN_END_OF_TRACE
        final_row = pd.DataFrame(
            [[case_id] + [max_time] + [MakeEventLog.TOKEN_END_OF_TRACE] * 3],
            columns=['case_id', 'time', 'method', 'proc_name', 'args']
        )

        all_calls = pd.concat([all_calls, final_row], ignore_index=True)

        # convert df to "string[pyarrow]" type to save memory
        # (will save around 75% of memory usage compared to str type)
        # all_calls = all_calls.astype("string[pyarrow]")

        # sort by timestamp
        all_calls = all_calls.sort_values(by=['time'], kind='stable', ignore_index=True)

        # Drop consecutive duplicates from DataFrame with multiple columns
        all_calls = all_calls.loc[~(all_calls.shift() == all_calls).all(axis=1)]
        return all_calls

    @staticmethod
    def _collect_data(mapping, size):
        list_cases = []
        for case in tqdm(mapping, total=size):
            if case is not None:
                list_cases.append(case)

        print("Concatenating data... This may take a while.")
        event_log_normal = pd.concat(list_cases, ignore_index=True)
        return event_log_normal

    def load_data(self, paths, case_id_prefix=''):
        print("Loading data...")
        self.paths = paths
        self.case_id_prefix = case_id_prefix

        if len(self.paths) == 0:
            raise Exception("No paths provided")

        self._scan_folders()
        self._map_dir_to_id()

        pool = Pool(self.workers)
        m = pool.imap(self._load_case, zip(self.dir_id_mapping_normal.values(), self.folders))

        event_log_normal = self._collect_data(m, len(self.folders))

        return self._post_processing(event_log_normal)

    @staticmethod
    def _calc_size(df):
        size = sum(df.memory_usage(deep=True)) / 1024 / 1024 / 1024
        return size

    @staticmethod
    def _post_processing(event_log):
        print('Mem usage (GiB) after concat :', MakeEventLog._calc_size(event_log))

        # sort by timestamp
        print("Sorting by timestamp...")
        event_log = event_log.sort_values(by=['time'], kind='stable', ignore_index=True)
        print(event_log.head())

        return event_log


if __name__ == '__main__':
    # put datasets into the list if you want to concatenate them, assuming files contain continuous data
    paths_nor = [
        "KernelDriver/ProcessId/ProcessIdClean"
    ]

    paths_anom = [
        "KernelDriver/ProcessId/ProcessIdVirusShare500"
    ]

    dl = MakeEventLog()
    normal = dl.load_data(paths_nor)
    normal.to_csv("event_log_normal.csv.zip", index=False, compression='zip')
    malicious = dl.load_data(paths_anom)
    malicious.to_csv("event_log_anomalous.csv.zip", index=False, compression='zip')
