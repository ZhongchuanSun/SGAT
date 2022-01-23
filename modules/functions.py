import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix


def inner_product(a, b, name="inner_product"):
    with tf.name_scope(name=name):
        return tf.reduce_sum(tf.multiply(a, b), axis=-1)


def scaled_inner_product(a, b, name="scaled_inner_product"):
    raise NotImplementedError("NotImplementedError")


dot_product = inner_product
scaled_dot_product = scaled_inner_product


def euclidean_distance(a, b, name="euclidean_distance"):
    return tf.norm(a - b, ord='euclidean', axis=-1, name=name)


l2_distance = euclidean_distance


def l2_loss(*params):
    return tf.add_n([tf.nn.l2_loss(w) for w in params])


def sp_mat_to_sp_tensor(sp_mat):
    if not isinstance(sp_mat, coo_matrix):
        sp_mat = sp_mat.tocoo().astype(np.float32)
    indices = np.asarray([sp_mat.row, sp_mat.col]).transpose()
    return tf.SparseTensor(indices, sp_mat.data, sp_mat.shape)


def normalize_adj_matrix(sp_mat, norm_method="left"):
    """Normalize adjacent matrix

    Args:
        sp_mat: A sparse adjacent matrix
        norm_method (str): The normalization method, can be 'symmetric'
            or 'left'.

    Returns:
        sp.spmatrix: The normalized adjacent matrix.

    """

    d_in = np.asarray(sp_mat.sum(axis=1))  # indegree
    if norm_method == "left":
        rec_d_in = np.power(d_in, -1).flatten()  # reciprocal
        rec_d_in[np.isinf(rec_d_in)] = 0.  # replace inf
        rec_d_in = sp.diags(rec_d_in)  # to diagonal matrix
        norm_sp_mat = rec_d_in.dot(sp_mat)  # left matmul
    elif norm_method == "symmetric":
        rec_sqrt_d_in = np.power(d_in, -0.5).flatten()
        rec_sqrt_d_in[np.isinf(rec_sqrt_d_in)] = 0.
        rec_sqrt_d_in = sp.diags(rec_sqrt_d_in)

        mid_sp_mat = rec_sqrt_d_in.dot(sp_mat)  # left matmul
        norm_sp_mat = mid_sp_mat.dot(rec_sqrt_d_in)  # right matmul
    else:
        raise ValueError(f"'{norm_method}' is an invalid normalization method.")

    return norm_sp_mat
