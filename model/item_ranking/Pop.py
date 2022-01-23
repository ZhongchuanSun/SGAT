from model.base import AbstractRecommender
import numpy as np
import pandas as pd


class Pop(AbstractRecommender):
    def __init__(self, config):
        super(Pop, self).__init__(config)
        self.users_num, self.items_num = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.get_user_train_dict()

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        items = self.dataset.train_data["item"]
        items_count = pd.value_counts(items, sort=False)
        items = items_count.index.values
        count = items_count.values
        self.ranking_score = np.zeros([self.items_num], dtype=np.float32)
        self.ranking_score[items] = count

        result = self.evaluate_model()
        self.logger.info("result:\t%s" % result)

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict_for_eval(self, users):
        ratings = np.tile(self.ranking_score, len(users))
        ratings = np.reshape(ratings, newshape=[len(users), self.items_num])
        return ratings
