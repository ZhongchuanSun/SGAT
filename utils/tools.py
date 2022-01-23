import numpy as np
from scipy.sparse import csr_matrix
from utils import typeassert
import tensorflow as tf
import functools

from reckit import randint_choice, batch_randint_choice
# reckit==0.2.4

batch_random_choice = functools.partial(batch_randint_choice, thread_num=4)


@typeassert(matrix=csr_matrix)
def csr_to_user_dict(matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    idx_value_dict = {}
    for idx, value in enumerate(matrix):
        if any(value.indices):
            idx_value_dict[idx] = value.indices.copy()
    return idx_value_dict


@typeassert(data_matrix=csr_matrix, time_matrix=csr_matrix)
def csr_to_user_dict_sorted(data_matrix, time_matrix):
    user_pos_train = {}
    user_pos_items = csr_to_user_dict(data_matrix)
    for u, items in user_pos_items.items():
        sorted_items = sorted(items, key=lambda x: time_matrix[u, x])
        user_pos_train[u] = np.array(sorted_items, dtype=np.int32)

    return user_pos_train


def pad_sequences(array, value=0, max_len=None, padding='post', truncating='post'):
    """padding: String, 'pre' or 'post':
            pad either before or after each sequence.
       truncating: String, 'pre' or 'post':
            remove values from sequences larger than `maxlen`,
            either at the beginning or at the end of the sequences.
    """
    array = tf.keras.preprocessing.sequence.pad_sequences(array, maxlen=max_len, value=value, dtype='int32',
                                                          padding=padding, truncating=truncating)

    return array
