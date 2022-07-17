import os
import json

from make_eventlog import MakeEventLog
import pandas as pd


class MakeEventLogCuckoo(MakeEventLog):
    def __init__(self, workers=8):
        super().__init__(workers)

    @staticmethod
    def _extract_call(call_dict):
        time = call_dict['time']
        category = call_dict['category']
        status = call_dict['status']
        api = call_dict['api']
        return time, category, status, api

    def _load_case(self, args):
        """
        Load all calls from a process. A process folder is like: 0, 1, ...
        :return:
        """

        case_id, process_folder = args
        case_id = self.case_id_prefix + str(case_id)

        report_json = os.path.join(process_folder, 'reports/report.json').replace("\\", "/")

        try:
            with open(report_json, 'r') as f:
                report = json.load(f)
        except (FileNotFoundError, NotADirectoryError):
            return None

        try:
            calls = report['behavior']['processes'][1]['calls']
        except (IndexError, KeyError):
            return None

        # empty calls
        if len(calls) == 0:
            return None

        all_calls = map(self._extract_call, calls)

        # convert to pandas dataframe
        all_calls = pd.DataFrame(all_calls, columns=['time', 'category', 'status', 'api'])

        # add rows from calls to all_calls
        all_calls['case_id'] = case_id

        # sort by timestamp
        all_calls = all_calls.sort_values(by=['time'], kind='stable', ignore_index=True)

        max_time = all_calls['time'].iloc[-1]

        # finally and the TOKEN_END_OF_TRACE
        final_row = pd.DataFrame(
            [[case_id] + [max_time] + [MakeEventLog.TOKEN_END_OF_TRACE] * 3],
            columns=['case_id', 'time', 'category', 'status', 'api']
        )

        all_calls = pd.concat([all_calls, final_row], ignore_index=True)

        # convert df to "string[pyarrow]" type to save memory
        # (will save around 75% of memory usage compared to str type)
        # all_calls = all_calls.astype("string[pyarrow]")

        # # Drop consecutive duplicates from DataFrame with multiple columns
        # all_calls = all_calls.loc[~(all_calls.shift() == all_calls).all(axis=1)]
        return all_calls


if __name__ == '__main__':
    # put datasets into the list if you want to concatenate them, assuming files contain continuous data
    paths_norm_train_1 = [
        "Cuckoo/CuckooClean"
    ]

    paths_norm_train_2 = [
        "Cuckoo/CuckooCleanHippo"
    ]

    paths_norm_ec = [
        "Cuckoo/CuckooCleanPippo"
    ]

    paths_anom = [
        "Cuckoo/CuckooVirusShare"
    ]

    dl = MakeEventLogCuckoo()

    normal = dl.load_data(paths_norm_train_1, case_id_prefix="clean_")
    normal.to_csv("event_log_cuckoo_clean.csv.zip", index=False, compression='zip')

    normal = dl.load_data(paths_norm_train_2, case_id_prefix="hippo_")
    normal.to_csv("event_log_cuckoo_hippo.csv.zip", index=False, compression='zip')

    normal = dl.load_data(paths_norm_ec, case_id_prefix="pippo_")
    normal.to_csv("event_log_cuckoo_pippo.zip", index=False, compression='zip')

    malicious = dl.load_data(paths_anom, case_id_prefix="virus_")
    malicious.to_csv("event_log_cuckoo_virus.csv.zip", index=False, compression='zip')
