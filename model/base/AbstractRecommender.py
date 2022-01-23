import os
from utils import Logger
from data.dataset import Dataset
import tensorflow as tf
from utils import get_run_id, get_params_id
from reckit import Evaluator
# reckit==0.2.4


def _create_logger(config):
    # create logger
    data_name = config["data_name"]
    model_name = config["model"]
    log_dir = os.path.join("log", data_name, model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_id = get_run_id()

    logger_name = os.path.join(log_dir, run_id + ".log")
    logger = Logger(logger_name)

    logger.info(f"my pid: {os.getpid()}")
    # write configuration into log file
    logger.info(config)

    return logger


class AbstractRecommender(object):
    def __init__(self, config):
        self.config = config
        self.logger = _create_logger(config)

        self.dataset = Dataset(config)
        self.logger.info("\nuser number=%d\nitem number=%d" % (self.dataset.num_users, self.dataset.num_items))
        user_train_dict = self.dataset.get_user_train_dict()
        user_test_dict = self.dataset.get_user_test_dict()
        # self.evaluator = ProxyEvaluator(user_train_dict, user_test_dict, config)
        self.evaluator = Evaluator(user_train_dict, user_test_dict,
                                   metric=config["metric"], top_k=config["top_k"],
                                   batch_size=64, num_thread=config["num_thread"])

        tf_config = tf.ConfigProto()#allow_soft_placement=False, log_device_placement=True
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = config["gpu_mem"]
        self.sess = tf.Session(config=tf_config)
        assert tf.test.is_gpu_available()

    def train_model(self):
        raise NotImplementedError

    def evaluate_model(self):
        raise NotImplementedError

    def predict_for_eval(self, users):
        raise NotImplementedError

    def predict(self, users):
        return self.predict_for_eval(users)

    def _get_saver(self):
        if not hasattr(self, "_saver"):
            max_to_keep = self.config["max_model_to_keep"]
            self._saver = tf.train.Saver(max_to_keep=max_to_keep)
        return self._saver

    def _get_save_path(self):
        params_id = get_params_id()
        return os.path.join("saved_model", self.__class__.__name__, self.dataset.name, params_id)

    def save_model(self, global_step=None):
        _saver = self._get_saver()
        save_path = self._get_save_path()
        save_path = os.path.join(save_path, self.__class__.__name__)
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        global_step = 0 if global_step is None else global_step
        _saver.save(self.sess, save_path, global_step=global_step)

    def restore_model(self):
        _saver = self._get_saver()
        save_path = self._get_save_path()
        ckpt = tf.train.get_checkpoint_state(save_path)
        if ckpt and ckpt.model_checkpoint_path:
            _saver.restore(self.sess, ckpt.model_checkpoint_path)
            global_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        else:
            raise ValueError("Restore model from %s failed!" % save_path)
        self.logger.info("Restore model from %s successfully!" % os.path.join(ckpt.model_checkpoint_path))
        return global_step
