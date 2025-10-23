import os
import socket
from datetime import datetime
import numpy as np
import tqdm


class CustomizedError(Exception):
    """
    Custom error type for outputting error information of this type
    """

    def __init__(self, error_str):
        self.error_str = error_str

    def __str__(self):
        return self.error_str


class Utils(object):
    @staticmethod
    def polynomial_ring_multiply(a, b):
        """
        Dilithium polynomial ring multiplication, polynomial multiplication needs to be modulo x^256+1,
        need to ensure the absolute value after multiplication is less than q. Since in the attack part,
        the result of cs in z=y+cs is already very small and cannot exceed q, so q is not needed as input
        :param a: Input polynomial 1, 1D array representing a polynomial
        :param b: Input polynomial 2, 1D array representing a polynomial
        :return: 1D array representing a polynomial
        """
        res = np.zeros((a.shape[0]), dtype=int)
        for i in range(res.shape[0]):
            for j in range(256):
                if j <= i:
                    res[i] += a[j] * b[(i - j) % 256]
                else:
                    res[i] -= a[j] * b[(i - j) % 256]
        return res

    @staticmethod
    def y2a_value(y):
        """
        Convert y values to corresponding a values
        :param y: 2D array, each row is a y
        :return: 3D array, first dimension represents samples, second dimension represents which a, third dimension represents which round in 64 rounds of sampling
        """
        a = np.zeros((y.shape[0], 9, 64), dtype=int)
        GAMMA1 = 1 << 17
        for i in tqdm.trange(y.shape[0], desc="Extracting a from y", leave=False):
            for j in range(64):
                r0 = GAMMA1 - y[i, 4 * j]
                r1 = GAMMA1 - y[i, 4 * j + 1]
                r2 = GAMMA1 - y[i, 4 * j + 2]
                r3 = GAMMA1 - y[i, 4 * j + 3]
                a[i, 0, j] = r0 & 0xFF
                a[i, 1, j] = (r0 >> 8) & 0xFF
                a[i, 2, j] = (r0 >> 16) | ((r1 & 0x3F) << 2)
                a[i, 3, j] = (r1 >> 6) & 0xFF
                a[i, 4, j] = (r1 >> 14) | ((r2 & 0xF) << 4)
                a[i, 5, j] = (r2 >> 4) & 0xFF
                a[i, 6, j] = (r2 >> 12) | ((r3 & 0x3) << 6)
                a[i, 7, j] = (r3 >> 2) & 0xFF
                a[i, 8, j] = (r3 >> 10) & 0xFF
        return a

    @staticmethod
    def s2y(z, c, s, idx):
        """
        Calculate y from the Dilithium ciphertext equation z=y+cs, given z, c and s to get y.
        Since the absolute value of cs is small, there is no modulo q issue after adding y
        :param z: 2D array, each row represents a sample, note that the dataset gives only one element of vector z per signature
        :param c: 2D array, each row represents data used for one signature
        :param s: 2D array, 4 rows, which component of the key vector is used for each signature is determined by idx
        :param idx: 1D array, indicating which component of the key vector is used for each signature
        :return: 2D array y
        """
        y = np.zeros((idx.shape[0], 256), dtype=int)
        for i in tqdm.trange(y.shape[0], desc="Extracting y", leave=False):
            y[i] = z[i] - Utils.polynomial_ring_multiply(c[i], s[idx[i]])
        return y

    @staticmethod
    def softmax(x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x

    @staticmethod
    def more_stable_softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    @staticmethod
    def SNR(data, label, class_num):
        """Calculate signal-to-noise ratio, using variance of means divided by mean of variances, only counting SNR for existing classes"""
        data_len = data.shape[1]
        non_zero_class = []
        for class_idx in range(class_num):
            if np.sum(label == class_idx) > 0:
                non_zero_class.append(class_idx)
        ave = np.zeros((len(non_zero_class), data_len), dtype=float)
        var = np.zeros((len(non_zero_class), data_len), dtype=float)
        for class_idx in tqdm.trange(
            len(non_zero_class), desc="Calculating SNR", leave=False
        ):
            ave[class_idx] = np.average(
                data[label == non_zero_class[class_idx]], axis=0
            )
            var[class_idx] = np.var(data[label == non_zero_class[class_idx]], axis=0)
        snr = np.zeros((data_len,), dtype=float)
        snr = np.var(ave, axis=0) / np.average(var, axis=0)
        return snr
