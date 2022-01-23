import pandas as pd
import scipy.sparse as sp


class Dataset(object):
    def __init__(self, config):
        self.name = None
        self.train_file = None
        self.test_file = None

        self.num_users = None
        self.num_items = None
        self.num_ratings = None

        self.train_matrix = None
        self.test_matrix = None
        self.time_matrix = None

        self._load_data(config)

    def _load_data(self, config):
        self.name = config["data_name"]
        self.train_file = config["train_file"]
        self.test_file = config["test_file"]

        self.file_format = config["format"]
        sep = config["separator"]

        if self.file_format == "UIRT":
            columns = ["user", "item", "rating", "time"]
        elif self.file_format == "UIR":
            columns = ["user", "item", "rating"]
        else:
            raise ValueError("There is not data format '%s'" % self.file_format)

        self.train_data = pd.read_csv(self.train_file, sep=sep, header=None, names=columns)
        self.test_data = pd.read_csv(self.test_file, sep=sep, header=None, names=columns)

        all_data = pd.concat([self.train_data, self.test_data])

        self.num_users = len(all_data["user"].unique())
        self.num_items = len(all_data["item"].unique())
        self.num_ratings = len(all_data)

    def _df2dict(self, data, by_time=False):
        if by_time and "time" not in data:
            raise ValueError("This dataset do not have timestamp!")

        idx_value_dict = {}
        user_grouped = data.groupby("user")
        for user, rows in user_grouped:
            if by_time:
                rows = rows.sort_values(by=["time"])
            idx_value_dict[user] = rows["item"].values

        return idx_value_dict

    def get_user_train_dict(self, by_time=False):
        train_dict = self._df2dict(self.train_data, by_time)
        return train_dict

    def get_user_test_dict(self, by_time=False):
        test_dict = self._df2dict(self.test_data, by_time)
        return test_dict

    def get_train_interactions(self):
        return self.train_data["user"].tolist(), self.train_data["item"].tolist()

    def get_train_csr_mat(self):
        users, items = self.train_data["user"].tolist(), self.train_data["item"].tolist()
        ratings = self.train_data["rating"].tolist() if "R" in self.file_format else [1.0]*len(users)
        train_data_csr = sp.csr_matrix((ratings, (users, items)), shape=(self.num_users, self.num_items))
        return train_data_csr
