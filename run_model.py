import os
from utils import lib_config
os.environ['CUDA_VISIBLE_DEVICES'] = str(lib_config["gpu_id"])
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import random
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from model.ModelFactory import ModelFactory


# fix random seed
np.random.seed(lib_config["seed"])
random.seed(lib_config["seed"])
tf.set_random_seed(lib_config["seed"])

# change the directory of "TFRec/"
pwd = os.getcwd()
os.chdir(pwd)


if __name__ == "__main__":
    model_factory = ModelFactory()
    model = model_factory.get_model(lib_config)
    model.logger.info(f"random_seed: {lib_config['seed']}")
    model.train_model()
