import random
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
import serial
import serial.tools.list_ports
import ctypes
import os
import tqdm
from picosdk.ps3000a import ps3000a as ps
from picosdk.functions import adc2mV, assert_pico_ok
from multiprocessing import Pool, Queue, Process, Manager
from multiprocessing import shared_memory, RLock, resource_tracker
from Template_Attack import Template


def c_value2label(c1, c2):
    return (c1 + 1) * 3 + c2 + 1


def c_label2value(label):
    return int(label / 3) - 1, int(label % 3) - 1


def more_stable_softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def id_equation_gen(zk, zk1, zk2, yk, yk1, yk2, c, k):
    C = np.zeros((256,), dtype=int)
    for i in range(256):
        if i <= k:
            C[i] = c[(k - i) % 256]
        else:
            C[i] = -c[(k - i) % 256]
    z_minus_y = None

    if zk1 == yk1 and zk2 == yk2:
        z_minus_y = zk - yk
    elif zk1 == yk1 and zk2 != yk2:
        pass
    elif zk1 != yk1 and zk2 == yk2:
        if yk1 == 0:
            z_minus_y = zk - yk + 2**6
        elif yk1 == 1:
            z_minus_y = zk - yk - 2**6
    elif zk1 != yk1 and zk2 != yk2:
        if yk1 == 0:
            z_minus_y = zk - yk - 2**6
        elif yk1 == 1:
            z_minus_y = zk - yk + 2**6
    return z_minus_y, C


def TA_callback(default_return):
    global pbar_for_TA_id
    pbar_for_TA_id.update(1)


def calc_snr_single_coeff(args):
    """Calculate SNR for a single coefficient (global function for multiprocessing)"""
    j, train_trs, train_label_y, trs_len, min_class_count = args

    # Get labels for this coefficient
    labels = train_label_y[:, j]
    unique_class = np.unique(labels)

    if len(unique_class) <= 1:
        # If only one class, return zero SNR
        return j, np.zeros(trs_len, dtype=float)

    # Count occurrences of each class, filter out classes with too few occurrences
    valid_classes = []
    for cls in unique_class:
        count = np.sum(labels == cls)
        if count > min_class_count:
            valid_classes.append(cls)

    valid_classes = np.array(valid_classes)

    if len(valid_classes) <= 1:
        # If only one or no valid classes after filtering, return zero SNR
        return j, np.zeros(trs_len, dtype=float)

    # Calculate mean and variance for each valid class
    mean = np.zeros((len(valid_classes), trs_len), dtype=float)
    var = np.zeros((len(valid_classes), trs_len), dtype=float)

    for i, cls in enumerate(valid_classes):
        mask = labels == cls
        if np.sum(mask) > 0:
            class_data = train_trs[mask]
            mean[i] = np.mean(class_data, axis=0)
            var[i] = np.var(class_data, axis=0)
        else:
            mean[i] = 0
            var[i] = 1  # Avoid division by zero

    # Calculate SNR
    signal_power = np.var(mean, axis=0)
    noise_power = np.mean(var, axis=0)

    # Avoid division by zero
    snr = np.divide(
        signal_power,
        noise_power,
        out=np.zeros_like(signal_power),
        where=noise_power != 0,
    )

    return j, snr


def calc_snr_single_coeff_shared(args):
    """Calculate SNR for a single coefficient (using shared memory for multiprocessing)"""
    (
        j,
        train_trs_shm_name,
        train_label_shm_name,
        train_trs_shape,
        train_label_shape,
        trs_len,
        min_class_count,
    ) = args

    # Connect to shared memory
    train_trs_shm = shared_memory.SharedMemory(name=train_trs_shm_name)
    train_label_shm = shared_memory.SharedMemory(name=train_label_shm_name)

    try:
        # Reconstruct numpy arrays from shared memory
        train_trs = np.ndarray(train_trs_shape, dtype=int, buffer=train_trs_shm.buf)
        train_label_y = np.ndarray(
            train_label_shape, dtype=int, buffer=train_label_shm.buf
        )

        # Get labels for this coefficient
        labels = train_label_y[:, j]
        unique_class = np.unique(labels)

        if len(unique_class) <= 1:
            # If only one class, return zero SNR
            return j, np.zeros(trs_len, dtype=float)

        # Count occurrences of each class, filter out classes with too few occurrences
        valid_classes = []
        for cls in unique_class:
            count = np.sum(labels == cls)
            if count > min_class_count:
                valid_classes.append(cls)

        valid_classes = np.array(valid_classes)

        if len(valid_classes) <= 1:
            # If only one or no valid classes after filtering, return zero SNR
            return j, np.zeros(trs_len, dtype=float)

        # Calculate mean and variance for each valid class
        mean = np.zeros((len(valid_classes), trs_len), dtype=float)
        var = np.zeros((len(valid_classes), trs_len), dtype=float)

        for i, cls in enumerate(valid_classes):
            mask = labels == cls
            if np.sum(mask) > 0:
                class_data = train_trs[mask]
                mean[i] = np.mean(class_data, axis=0)
                var[i] = np.var(class_data, axis=0)
            else:
                mean[i] = 0
                var[i] = 1  # Avoid division by zero

        # Calculate SNR
        signal_power = np.var(mean, axis=0)
        noise_power = np.mean(var, axis=0)

        # Avoid division by zero
        snr = np.divide(
            signal_power,
            noise_power,
            out=np.zeros_like(signal_power),
            where=noise_power != 0,
        )

        return j, snr

    finally:
        # Close shared memory connections
        train_trs_shm.close()
        train_label_shm.close()


def template_attack_single_coeff(args):
    """Template attack for a single coefficient (global function for multiprocessing)"""
    (
        i,
        train_trs,
        test_trs,
        train_label_y,
        exp_save_path,
    ) = args

    # Get unique classes for this coefficient
    unique_labels = np.unique(train_label_y)
    num_classes = len(unique_labels)

    ta = Template()
    ta.build_model(
        train_trs,
        train_label_y,
        num_classes,
        "lda",
        min(8, num_classes - 1),
    )
    prob_test = ta.apply_model(test_trs)
    prob_train = ta.apply_model(train_trs)

    # Save results
    np.save(exp_save_path + "idx%03dProbTest" % (i,), prob_test)
    np.save(exp_save_path + "idx%03dProbTrain" % (i,), prob_train)
    return i


def gen_parallel_optimized(test_prob, c, z, use_parallel=True, process_num=4):
    """
    Parallelized optimized gen function
    :param test_prob: Expected input dimension signature_count*polynomial_coeff_count*probability_distribution, i.e., signature_count*256*256
    :param c: Expected input dimension signature_count*256
    :param z: Expected input dimension signature_count*256
    :param use_parallel: Whether to use parallelization
    :param process_num: Number of processes
    :return:
    """
    # Get test dataset size
    tau = 39
    eta = 2
    z_minus_y_max_abs = tau * eta
    test_trace_num = test_prob.shape[0]
    equation_num = test_trace_num * 256
    trace_num = test_trace_num

    id_TA_res = test_prob
    C = c
    Z = z
    Z &= 0xFF

    Z_MINUS_Y_ID = np.zeros((equation_num, 2 * z_minus_y_max_abs + 1), dtype=float)
    CS_ID = np.zeros((equation_num, 256), dtype=int)
    Z = Z.flatten()

    if (
        use_parallel and equation_num > 1000
    ):  # Only use parallelization when data volume is large enough
        # Process equations in batches
        batch_size = max(1, equation_num // process_num)
        batches = []
        for i in range(0, equation_num, batch_size):
            end_idx = min(i + batch_size, equation_num)
            batches.append((i, end_idx, Z, C, id_TA_res, z_minus_y_max_abs))

        with Pool(process_num) as pool:
            with tqdm.tqdm(
                total=len(batches), desc="Gen processing", leave=False, position=2
            ) as pbar:
                for start_idx, Z_MINUS_Y_ID_batch, CS_ID_batch in pool.imap(
                    process_equation_batch_gen, batches
                ):
                    Z_MINUS_Y_ID[
                        start_idx : start_idx + Z_MINUS_Y_ID_batch.shape[0]
                    ] = Z_MINUS_Y_ID_batch
                    CS_ID[start_idx : start_idx + CS_ID_batch.shape[0]] = CS_ID_batch
                    pbar.update(1)
    else:
        # Serial version
        for equ_idx in tqdm.trange(
            equation_num, desc="Gen processing", leave=False, position=2
        ):
            k = equ_idx % 256
            Z_tmp = Z[equ_idx]
            C_tmp = C[int(equ_idx / 256)]
            id_attack_res_tmp = id_TA_res[int(equ_idx / 256), k]

            zk = Z_tmp & 0x3F
            zk1 = (Z_tmp >> 6) & 0x1
            zk2 = (Z_tmp >> 7) & 0x1

            for y_prob in range(256):
                yk = y_prob & 0x3F
                yk1 = (y_prob >> 6) & 0x1
                yk2 = (y_prob >> 7) & 0x1

                z_minus_y, c = id_equation_gen(zk, zk1, zk2, yk, yk1, yk2, C_tmp, k)

                if z_minus_y is None:
                    continue
                if np.abs(z_minus_y) > z_minus_y_max_abs:
                    continue

                Z_MINUS_Y_ID[
                    equ_idx, z_minus_y + z_minus_y_max_abs
                ] += id_attack_res_tmp[y_prob]
                CS_ID[equ_idx, :] = c

    return Z_MINUS_Y_ID, CS_ID


def sumproduct(
    cs=None,
    z_minus_y=None,
    iter_num=10,
    sk_start=None,
    sk_true=None,
    progress_callback=None,
):
    tau = 39
    eta = 2
    equ_num = cs.shape[0]

    sk_nodes = np.empty((equ_num, tau, 2 * eta + 1), dtype=float)

    sk_idx = np.zeros((equ_num, tau), dtype=int)
    tmp = np.zeros((equ_num, tau), dtype=int)
    # for equ_idx in tqdm.trange(equ_num, desc="Extract non-zero elements from cs", leave=False):
    for equ_idx in range(equ_num):
        sk_idx[equ_idx] = np.nonzero(cs[equ_idx])[0]
        tmp[equ_idx] = cs[equ_idx, sk_idx[equ_idx]]
    cs = tmp

    if sk_start is None:
        sk_nodes[:] = 1 / (2 * eta + 1)  # Initial distribution is uniform
    else:
        for i in range(equ_num):
            for j in range(tau):
                sk_nodes[i, j] = sk_start[sk_idx[i, j]]

    x_nodes = z_minus_y
    bp_res = np.zeros((iter_num, 256, 5), dtype=float)
    # for bp_iter in tqdm.trange(iter_num, desc="BP iteration", leave=False):
    for bp_iter in range(iter_num):
        # for equ_idx in tqdm.trange(equ_num, desc="Iterate through each equation", leave=False):
        for equ_idx in range(equ_num):
            tmp = [[1]] * tau
            acc = [1]

            # Calculate cumulative distribution of each sk except itself
            for i in range(tau):
                tmp[i] = acc
                if cs[equ_idx, i] == 1:
                    acc = np.convolve(acc, sk_nodes[equ_idx, i], mode="full")
                else:
                    acc = np.convolve(acc, np.flip(sk_nodes[equ_idx, i]), mode="full")
            acc = [1]
            for i in range(tau - 1, -1, -1):
                tmp[i] = np.convolve(tmp[i], acc, mode="full")
                if cs[equ_idx, i] == 1:
                    acc = np.convolve(acc, sk_nodes[equ_idx, i], mode="full")
                else:
                    acc = np.convolve(acc, np.flip(sk_nodes[equ_idx, i]), mode="full")

            # Subtract cumulative distribution of each sk except itself from z-y distribution
            for i in range(tau):
                if cs[equ_idx, i] == 1:
                    tmp[i] = np.convolve(x_nodes[equ_idx], np.flip(tmp[i]), mode="full")
                else:
                    tmp[i] = np.convolve(tmp[i], np.flip(x_nodes[equ_idx]), mode="full")

            # Update sk distribution, only take probabilities in [-eta,eta]
            for i in range(tau):
                start_idx = int(tmp[i].shape[0] / 2) - eta
                sk_nodes[equ_idx, i] = tmp[i][start_idx : start_idx + 2 * eta + 1]

        # Determine final sk distribution using maximum likelihood
        sk_mle = np.zeros(
            (256, 5), dtype=float
        )  # Key distribution after maximum likelihood
        tmp = np.log(sk_nodes + 1e-100)
        # for equ_idx in tqdm.trange(equ_num, desc="Calculate sk distribution using maximum likelihood", leave=False):
        for equ_idx in range(equ_num):
            for j in range(tau):
                sk_mle[sk_idx[equ_idx, j]] += tmp[equ_idx, j]

        # for equ_idx in tqdm.trange(equ_num, desc="BP update sk distribution", leave=False):
        for equ_idx in range(equ_num):
            for j in range(tau):
                sk_nodes[equ_idx, j] = more_stable_softmax(
                    sk_mle[sk_idx[equ_idx, j]] - tmp[equ_idx, j]
                )

        sk_hat = np.argmax(sk_mle, axis=1) - 2
        if sk_true is not None:
            # print(sk_hat)
            sk_hat_num = np.zeros((5,), dtype=int)
            for i in range(-2, 3, 1):
                # print("Number of key coefficients equal to %d is %d" % (i, np.sum(sk_hat == i)))
                sk_hat_num[i + 2] = np.sum(sk_hat == i)
            print(
                "Number of correct key coefficients is %d" % (np.sum(sk_hat == sk_true))
            )

        for i in range(256):
            sk_mle[i] = more_stable_softmax(sk_mle[i])

        bp_res[bp_iter] = sk_mle

        # Progress callback
        if progress_callback is not None:
            progress_callback(bp_iter + 1, iter_num)

    return bp_res


def sumproduct_soft(cs=None, z_minus_y=None, iter_num=10, max_cs_abs=32):
    """
    The difference from regular sumproduct is that c here has a distribution
    :param cs: Dimension is equation_count*256*3, since it's a distribution there's no positive/negative concept
    :param z_minus_y: Dimension is equation_count*coefficient_range
    :param iter_num: Number of iterations, default 10
    :param max_cs_abs: Maximum cs absolute value, default 32
    :return:
    """
    eta = 2
    equ_num = cs.shape[0]

    sk_nodes = np.empty((equ_num, 256, 2 * eta + 1), dtype=float)

    sk_nodes[:] = 1 / (2 * eta + 1)  # Initial distribution is uniform

    x_nodes = z_minus_y

    bp_res = np.zeros((iter_num, 256, 5), dtype=float)

    for bp_iter in range(iter_num):
        for equ_idx in range(equ_num):
            tmp = [[1]] * 256
            acc = [1]

            # Calculate cumulative distribution of each sk except itself
            for i in range(256):
                tmp[i] = acc
                acc = np.convolve(
                    acc,
                    cs[equ_idx, i, 0] * np.flip(sk_nodes[equ_idx, i])
                    + cs[equ_idx, i, 2] * sk_nodes[equ_idx, i]
                    + [0, 0, cs[equ_idx, i, 1], 0, 0],
                    mode="full",
                )
                if (
                    acc.shape[0] > 2 * max_cs_abs + 1
                ):  # Convolution result cannot exceed certain range
                    overflow_num = acc.shape[0] - (2 * max_cs_abs + 1)
                    acc = acc[int(overflow_num / 2) : -int(overflow_num / 2)]
            acc = [1]
            for i in range(256 - 1, -1, -1):
                tmp[i] = np.convolve(tmp[i], acc, mode="full")
                if (
                    len(tmp[i]) > 2 * max_cs_abs + 1
                ):  # Convolution result cannot exceed certain range
                    overflow_num = len(tmp[i]) - (2 * max_cs_abs + 1)
                    tmp[i] = tmp[i][int(overflow_num / 2) : -int(overflow_num / 2)]
                acc = np.convolve(
                    acc,
                    cs[equ_idx, i, 0] * np.flip(sk_nodes[equ_idx, i])
                    + cs[equ_idx, i, 2] * sk_nodes[equ_idx, i]
                    + [0, 0, cs[equ_idx, i, 1], 0, 0],
                    mode="full",
                )
                if (
                    acc.shape[0] > 2 * max_cs_abs + 1
                ):  # Convolution result cannot exceed certain range
                    overflow_num = acc.shape[0] - (2 * max_cs_abs + 1)
                    acc = acc[int(overflow_num / 2) : -int(overflow_num / 2)]

            # Subtract cumulative distribution of each sk except itself from z-y distribution
            for i in range(256):
                _tmp = np.convolve(x_nodes[equ_idx], np.flip(tmp[i]), mode="full")
                tmp[i] = cs[equ_idx, i, 0] * np.flip(_tmp) + cs[equ_idx, i, 2] * _tmp

            # Update sk distribution, only take probabilities in [-eta,eta]
            for i in range(256):
                start_idx = int(tmp[i].shape[0] / 2) - eta
                sk_nodes[equ_idx, i] = tmp[i][start_idx : start_idx + 2 * eta + 1]

        # Determine final sk distribution using maximum likelihood
        sk_mle = np.zeros(
            (256, 5), dtype=float
        )  # Key distribution after maximum likelihood
        tmp = np.log(sk_nodes + 1e-100)
        for equ_idx in range(equ_num):
            for j in range(256):
                sk_mle[j] += tmp[equ_idx, j]

        for equ_idx in range(equ_num):
            for j in range(256):
                sk_nodes[equ_idx, j] = more_stable_softmax(sk_mle[j] - tmp[equ_idx, j])

        for i in range(256):
            sk_mle[i] = more_stable_softmax(sk_mle[i])

        bp_res[bp_iter] = sk_mle

    return bp_res


def sumproduct_parallel_optimized(
    cs=None,
    z_minus_y=None,
    iter_num=10,
    sk_start=None,
    sk_true=None,
    use_parallel=True,
    process_num=4,
):
    """
    Parallelized optimized sumproduct function
    """
    tau = 39
    eta = 2
    equ_num = cs.shape[0]

    sk_nodes = np.empty((equ_num, tau, 2 * eta + 1), dtype=float)
    sk_idx = np.zeros((equ_num, tau), dtype=int)
    tmp = np.zeros((equ_num, tau), dtype=int)

    # Preprocessing: extract non-zero elements from cs
    for equ_idx in tqdm.trange(
        equ_num, desc="Preprocessing cs", leave=False, position=2
    ):
        sk_idx[equ_idx] = np.nonzero(cs[equ_idx])[0]
        tmp[equ_idx] = cs[equ_idx, sk_idx[equ_idx]]
    cs = tmp

    if sk_start is None:
        sk_nodes[:] = 1 / (2 * eta + 1)
    else:
        for i in range(equ_num):
            for j in range(tau):
                sk_nodes[i, j] = sk_start[sk_idx[i, j]]

    x_nodes = z_minus_y
    bp_res = np.zeros((iter_num, 256, 5), dtype=float)

    if (
        use_parallel and equ_num > 1000
    ):  # Only use parallelization when data volume is large enough
        # Process equations in batches
        batch_size = max(1, equ_num // process_num)
        batches = []
        for i in range(0, equ_num, batch_size):
            end_idx = min(i + batch_size, equ_num)
            batches.append((i, end_idx, sk_nodes, sk_idx, cs, x_nodes, tau, eta))

        for bp_iter in tqdm.trange(
            iter_num, desc="BP iteration", leave=False, position=2
        ):
            with Pool(process_num) as pool:
                for start_idx, sk_nodes_batch in pool.imap(
                    process_equation_batch_sumproduct, batches
                ):
                    sk_nodes[start_idx : start_idx + sk_nodes_batch.shape[0]] = (
                        sk_nodes_batch
                    )

            # Determine final sk distribution using maximum likelihood
            sk_mle = np.zeros((256, 5), dtype=float)
            tmp = np.log(sk_nodes + 1e-100)
            for equ_idx in range(equ_num):
                for j in range(tau):
                    sk_mle[sk_idx[equ_idx, j]] += tmp[equ_idx, j]

            for equ_idx in range(equ_num):
                for j in range(tau):
                    sk_nodes[equ_idx, j] = more_stable_softmax(
                        sk_mle[sk_idx[equ_idx, j]] - tmp[equ_idx, j]
                    )

            sk_hat = np.argmax(sk_mle, axis=1) - 2
            if sk_true is not None:
                sk_hat_num = np.zeros((5,), dtype=int)
                for i in range(-2, 3, 1):
                    sk_hat_num[i + 2] = np.sum(sk_hat == i)
                print(
                    "Number of correct key coefficients is %d"
                    % (np.sum(sk_hat == sk_true))
                )

            for i in range(256):
                sk_mle[i] = more_stable_softmax(sk_mle[i])

            bp_res[bp_iter] = sk_mle
    else:
        # Serial version
        for bp_iter in tqdm.trange(
            iter_num, desc="BP iteration", leave=False, position=2
        ):
            for equ_idx in tqdm.trange(
                equ_num, desc="Iterate through each equation", leave=False, position=3
            ):
                tmp = [[1]] * tau
                acc = [1]

                # Calculate cumulative distribution of each sk except itself
                for i in range(tau):
                    tmp[i] = acc
                    if cs[equ_idx, i] == 1:
                        acc = np.convolve(acc, sk_nodes[equ_idx, i], mode="full")
                    else:
                        acc = np.convolve(
                            acc, np.flip(sk_nodes[equ_idx, i]), mode="full"
                        )
                acc = [1]
                for i in range(tau - 1, -1, -1):
                    tmp[i] = np.convolve(tmp[i], acc, mode="full")
                    if cs[equ_idx, i] == 1:
                        acc = np.convolve(acc, sk_nodes[equ_idx, i], mode="full")
                    else:
                        acc = np.convolve(
                            acc, np.flip(sk_nodes[equ_idx, i]), mode="full"
                        )

                # Subtract cumulative distribution of each sk except itself from z-y distribution
                for i in range(tau):
                    if cs[equ_idx, i] == 1:
                        tmp[i] = np.convolve(
                            x_nodes[equ_idx], np.flip(tmp[i]), mode="full"
                        )
                    else:
                        tmp[i] = np.convolve(
                            tmp[i], np.flip(x_nodes[equ_idx]), mode="full"
                        )

                # Update sk distribution, only take probabilities in [-eta,eta]
                for i in range(tau):
                    start_idx_eta = int(tmp[i].shape[0] / 2) - eta
                    sk_nodes[equ_idx, i] = tmp[i][
                        start_idx_eta : start_idx_eta + 2 * eta + 1
                    ]

            # Determine final sk distribution using maximum likelihood
            sk_mle = np.zeros((256, 5), dtype=float)
            tmp = np.log(sk_nodes + 1e-100)
            for equ_idx in range(equ_num):
                for j in range(tau):
                    sk_mle[sk_idx[equ_idx, j]] += tmp[equ_idx, j]

            for equ_idx in range(equ_num):
                for j in range(tau):
                    sk_nodes[equ_idx, j] = more_stable_softmax(
                        sk_mle[sk_idx[equ_idx, j]] - tmp[equ_idx, j]
                    )

            sk_hat = np.argmax(sk_mle, axis=1) - 2
            if sk_true is not None:
                sk_hat_num = np.zeros((5,), dtype=int)
                for i in range(-2, 3, 1):
                    sk_hat_num[i + 2] = np.sum(sk_hat == i)
                print(
                    "Number of correct key coefficients is %d"
                    % (np.sum(sk_hat == sk_true))
                )

            for i in range(256):
                sk_mle[i] = more_stable_softmax(sk_mle[i])

            bp_res[bp_iter] = sk_mle

    return bp_res


def y_valid_bp_experiment(
    exp_min_sign_num=1,
    exp_max_sign_num=20,
    rerun_snr=True,
    rerun_ta=True,
    exp_repeat_num=10,
    bp_iter_num=15,
    snr_threshold=0.5,
    use_parallel_gen_sp=False,
    bp_process_num=5,
):
    """
    BP experiment using y_valid data
    1. Use template attack to model y
    2. Use SNR to select points of interest
    3. Training set 9000, test set 1000
    4. Repeat BP experiment 10 times

    Parameters:
    - exp_min_sign_num: Minimum number of signatures for BP experiment
    - exp_max_sign_num: Maximum number of signatures for BP experiment
    - rerun_snr: Whether to rerun SNR calculation
    - rerun_ta: Whether to rerun template attack
    - exp_repeat_num: Number of BP experiment repetitions
    - bp_iter_num: Number of BP iterations
    - snr_threshold: SNR threshold
    - use_parallel_gen_sp: Whether to use parallelization for gen and sumproduct functions
    - bp_process_num: Number of processes for BP experiment
    """
    # Experiment parameters
    exp_save_path = "../exp_res/y_valid_bp/"
    y_cut_trs_path = "../traces/y_cut/"
    y_data_path = "../data/y_valid/"
    c_data_path = "../data/c_all/"  # Modified to c_all directory
    z_data_path = "../data/z_valid/"

    train_sign_num = 9000
    test_sign_num = 1000
    total_sign_num = 10000
    train_test_split_seed = 123456

    # Create experiment save directory
    if not os.path.exists(exp_save_path):
        os.makedirs(exp_save_path)

    print("Starting y_valid BP experiment...")
    print(
        f"BP experiment parameters: min_sign={exp_min_sign_num}, max_sign={exp_max_sign_num}"
    )
    print(f"Rerun SNR calculation: {rerun_snr}")
    print(f"Rerun template attack: {rerun_ta}")

    # Step 1: Load cropped waveform data
    print("Loading cropped waveform data...")
    trs_len = np.load(y_cut_trs_path + "trs000000.npy").shape[0]
    print(f"Waveform length: {trs_len}")

    # Step 2: Split training and test sets
    print("Splitting training and test sets...")
    np.random.seed(train_test_split_seed)
    sign_idx_list = np.arange(total_sign_num, dtype=int)
    np.random.shuffle(sign_idx_list)

    train_sign_idx_list = sign_idx_list[:train_sign_num]
    test_sign_idx_list = sign_idx_list[train_sign_num : train_sign_num + test_sign_num]

    np.save(exp_save_path + "train_sign_idx", train_sign_idx_list)
    np.save(exp_save_path + "test_sign_idx", test_sign_idx_list)

    print(f"Training set size: {train_sign_num}, test set size: {test_sign_num}")

    # Step 3: Load training and test data
    print("Loading training and test data...")
    train_trs = np.zeros((train_sign_num, trs_len), dtype=int)
    test_trs = np.zeros((test_sign_num, trs_len), dtype=int)
    train_y = np.zeros((train_sign_num, 256), dtype=int)
    test_y = np.zeros((test_sign_num, 256), dtype=int)
    test_c = np.zeros((test_sign_num, 256), dtype=int)
    test_z = np.zeros((test_sign_num, 256), dtype=int)

    # Read rejection count information
    print("Reading rejection count information...")
    rej_num_list = []
    with open("seed and rej num.txt", "r") as f:
        for i in range(total_sign_num):
            line = f.readline().strip().split(" ")
            rej_num = int(line[1])
            rej_num_list.append(rej_num)
    rej_num_array = np.array(rej_num_list)

    # Load training data
    for i in tqdm.trange(train_sign_num, desc="Loading training data"):
        train_trs[i] = np.load(
            y_cut_trs_path + "trs%06d.npy" % (train_sign_idx_list[i],)
        )
        train_y[i] = np.load(y_data_path + "y%06d.npy" % (train_sign_idx_list[i],))

    # Load test data
    for i in tqdm.trange(test_sign_num, desc="Loading test data"):
        sign_idx = test_sign_idx_list[i]
        test_trs[i] = np.load(y_cut_trs_path + "trs%06d.npy" % (sign_idx,))
        test_y[i] = np.load(y_data_path + "y%06d.npy" % (sign_idx,))

        # Read the last rejection sampling c data from c_all directory
        rej_num = rej_num_array[sign_idx]
        last_rej_idx = rej_num - 1  # rej starts counting from 0
        test_c[i] = np.load(c_data_path + "c%06drej%06d.npy" % (sign_idx, last_rej_idx))

        test_z[i] = np.load(z_data_path + "z%06d.npy" % (sign_idx,))

    # Step 4: Calculate SNR and select points of interest
    # Whether or not to recalculate SNR, need to prepare train_label_y for template attack
    print("Preparing training label data...")
    train_label_y = train_y.copy()
    for i in range(train_label_y.shape[0]):
        for j in range(train_label_y.shape[1]):
            train_label_y[i, j] = train_label_y[i, j] & 0xFF

    # Prepare test label data (for template attack accuracy calculation)
    print("Preparing test label data...")
    test_label_y = test_y.copy()
    for i in range(test_label_y.shape[0]):
        for j in range(test_label_y.shape[1]):
            test_label_y[i, j] = test_label_y[i, j] & 0xFF

    if rerun_snr:
        print("Calculating SNR and selecting points of interest...")
        # Calculate SNR (multiprocessing)
        print("Calculating SNR (multiprocessing)...")
        train_snr = np.zeros((256, trs_len), dtype=float)

        pool = Pool(8)  # Use 8 processes
        snr_args = [(j, train_trs, train_label_y, trs_len, 0) for j in range(256)]
        snr_results = []
        with tqdm.tqdm(total=256, desc="Calculating SNR") as pbar:
            for result in pool.imap(calc_snr_single_coeff, snr_args, chunksize=4):
                snr_results.append(result)
                pbar.update(1)
        pool.close()
        pool.join()

        for j, snr in snr_results:
            train_snr[j] = snr

        # Save SNR file
        np.save(exp_save_path + "y_valid_snr", train_snr)

    else:
        print("Skipping SNR calculation, using existing results...")
        # Load existing SNR file
        train_snr = np.load(exp_save_path + "y_valid_snr.npy")

    # Plot SNR for all coefficients on the same graph
    plt.figure(figsize=(12, 8))
    for i in range(256):
        plt.plot(
            train_snr[i], linewidth=0.5, alpha=0.7, label=f"Coeff {i}" if i < 10 else ""
        )
    plt.title("SNR for All Coefficients")
    plt.xlabel("Sample Index")
    plt.ylabel("SNR")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(exp_save_path + "snr_all_coeffs.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Select points of interest based on SNR
    poi_num = 200
    POI = []
    for i in range(256):
        poi_indices = np.argsort(-train_snr[i])[:poi_num]
        POI.append(poi_indices.flatten())

    print(f"SNR threshold: {snr_threshold}")
    for i in range(256):  # Show points of interest count for first 10 coefficients
        print(f"Coefficient {i} points of interest count: {len(POI[i])}")

    # Step 5: Template attack modeling (multiprocessing)
    if rerun_ta:
        print("Performing template attack modeling (multiprocessing)...")

        pool = Pool(8)  # Use 8 processes
        ta_args = [
            (
                i,
                train_trs[:, POI[i]],
                test_trs[:, POI[i]],
                train_label_y[:, i],
                exp_save_path,
            )
            for i in range(256)
        ]
        with tqdm.tqdm(total=256, desc="Template attack modeling") as pbar:
            for result in pool.imap(template_attack_single_coeff, ta_args, chunksize=4):
                pbar.update(1)
        pool.close()
        pool.join()
    else:
        print("Skipping template attack, using existing results...")

    # Calculate and plot template attack accuracy
    print("Calculating template attack accuracy...")
    test_ta_acc = np.zeros((256,), dtype=float)
    for i in range(256):
        test_ta_prob = np.load(exp_save_path + "idx%03dProbTest.npy" % (i,))
        test_ta_predict = np.argmax(test_ta_prob, axis=1)
        test_ta_acc[i] = np.sum(test_ta_predict == test_label_y[:, i]) / test_sign_num

    # Plot accuracy graph
    plt.figure(figsize=(12, 8))
    plt.plot(test_ta_acc, linewidth=1.0, alpha=0.8, marker="o", markersize=2)
    plt.title("Template Attack Accuracy for All Coefficients")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(exp_save_path + "ta_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print accuracy statistics
    print(f"Average accuracy: {np.mean(test_ta_acc):.4f}")
    print(
        f"Highest accuracy: {np.max(test_ta_acc):.4f} (coefficient {np.argmax(test_ta_acc)})"
    )
    print(
        f"Lowest accuracy: {np.min(test_ta_acc):.4f} (coefficient {np.argmin(test_ta_acc)})"
    )
    print(f"Number of coefficients with accuracy > 0.5: {np.sum(test_ta_acc > 0.5)}")
    print(f"Number of coefficients with accuracy > 0.7: {np.sum(test_ta_acc > 0.7)}")

    exit()

    # Step 6: BP experiment (serial execution, internal parallelization optimization)
    print("Starting BP experiment...")
    test_z &= 0xFF  # Only use lower 8 bits

    # Load test probabilities
    test_prob = np.zeros((test_sign_num, 256, 256), dtype=float)
    for i in range(256):
        test_prob[:, i, :] = np.load(exp_save_path + "idx%03dProbTest.npy" % (i,))

    # Serial execution of BP experiments, each experiment uses internal parallelization optimization
    exp_repeat_seed = 234567  # Fixed seed

    for exp_idx in tqdm.trange(
        exp_repeat_num, desc="BP experiment progress", position=0
    ):
        np.random.seed(
            exp_repeat_seed + exp_idx
        )  # Ensure randomness for each experiment
        test_trs_idx = np.arange(test_sign_num)
        np.random.shuffle(test_trs_idx)

        for bp_sign_num in tqdm.trange(
            exp_min_sign_num,
            exp_max_sign_num + 1,
            desc=f"Exp{exp_idx:02d} signatures",
            leave=False,
            position=1,
        ):
            # Use parallelized optimized gen function
            zmy, cs = gen_parallel_optimized(
                test_prob[test_trs_idx[:bp_sign_num]],
                test_c[test_trs_idx[:bp_sign_num]],
                test_z[test_trs_idx[:bp_sign_num]],
                use_parallel=use_parallel_gen_sp,
                process_num=bp_process_num,
            )

            # Use parallelized optimized sumproduct function
            bp_res = sumproduct_parallel_optimized(
                cs,
                zmy,
                bp_iter_num,
                None,
                None,
                use_parallel=use_parallel_gen_sp,
                process_num=bp_process_num,
            )

            # Save results
            np.save(
                exp_save_path + "exp%02dSignNum%02dBpRes" % (exp_idx, bp_sign_num),
                bp_res,
            )

    print("y_valid BP experiment completed!")
    print(f"Results saved in: {exp_save_path}")


def run_single_bp_experiment(args):
    """单个BP实验的全局函数，用于多进程，包含完整的BP实现"""
    (
        exp_idx,
        bp_sign_num,
        test_sign_num,
        test_cs1_prob,
        test_c,
        bp_iter_num,
        exp_save_path,
        exp_repeat_seed,
        test_ta_acc,
    ) = args

    # 设置随机种子
    np.random.seed(exp_repeat_seed + exp_idx)
    test_trs_idx = np.arange(test_sign_num)
    np.random.shuffle(test_trs_idx)

    max_cs_abs = 32  # cs1的最大绝对值
    used_sign_idx = test_trs_idx[:bp_sign_num]
    # used_coeff_idx = np.concatenate(
    #     (np.arange(16, 128, dtype=int), np.arange(144, 256, dtype=int))
    # )
    used_coeff_idx = np.arange(256, dtype=int)
    # for coeff_idx in used_coeff_idx:
    #     print("coeff_idx: %d, acc: %f" % (coeff_idx, test_ta_acc[coeff_idx]))

    # 构建约束矩阵
    BP_C = np.zeros((bp_sign_num * len(used_coeff_idx), 256, 3), dtype=float)
    BP_CS = np.zeros(
        (bp_sign_num * len(used_coeff_idx), 2 * max_cs_abs + 1), dtype=float
    )

    equ_idx = 0
    for i in range(bp_sign_num):
        sign_idx = used_sign_idx[i]
        for coeff_idx in used_coeff_idx:
            # 构建cs1的分布 - 现在test_cs1_prob的索引就是sign_idx
            BP_CS[equ_idx, :max_cs_abs] = test_cs1_prob[
                sign_idx, coeff_idx, -max_cs_abs:
            ]
            BP_CS[equ_idx, max_cs_abs:] = test_cs1_prob[
                sign_idx, coeff_idx, : max_cs_abs + 1
            ]

            # 构建c的分布作为约束
            for s_idx in range(256):
                if s_idx <= coeff_idx:
                    c_val = test_c[sign_idx, coeff_idx - s_idx]
                else:
                    c_val = -test_c[sign_idx, (coeff_idx - s_idx) % 256]

                # 将c值转换为概率分布（3维：-1, 0, 1）
                if c_val == -1:
                    BP_C[equ_idx, s_idx] = [1, 0, 0]  # -1
                elif c_val == 0:
                    BP_C[equ_idx, s_idx] = [0, 1, 0]  # 0
                elif c_val == 1:
                    BP_C[equ_idx, s_idx] = [0, 0, 1]  # 1
                else:
                    BP_C[equ_idx, s_idx] = [0, 0, 1]  # 默认

            equ_idx += 1

    bp_res = sumproduct_soft(BP_C, BP_CS, bp_iter_num, max_cs_abs)

    # Save results
    np.save(
        exp_save_path + "exp%02dSignNum%02dBpRes" % (exp_idx, bp_sign_num),
        bp_res,
    )

    return exp_idx, bp_sign_num


def run_single_bp_experiment_more_simple(args):
    """单个BP实验的全局函数，用于多进程，包含完整的BP实现"""
    (
        exp_idx,
        BP_C,
        BP_CS,
        bp_sign_num,
        bp_iter_num,
        exp_save_path,
    ) = args

    max_cs_abs = 32  # cs1的最大绝对值

    bp_res = sumproduct_soft(BP_C, BP_CS, bp_iter_num, max_cs_abs)

    # Save results
    np.save(
        exp_save_path + "exp%02dSignNum%02dBpRes" % (exp_idx, bp_sign_num),
        bp_res,
    )

    return exp_idx, bp_sign_num


def run_single_bp_experiment_more_simple_vary_c_acc(args):
    """单个BP实验的全局函数，用于多进程，包含完整的BP实现"""
    (
        exp_idx,
        BP_C,
        BP_CS,
        bp_sign_num,
        bp_iter_num,
        exp_save_path,
        c_acc,
    ) = args

    max_cs_abs = 32  # cs1的最大绝对值

    bp_res = sumproduct_soft(BP_C, BP_CS, bp_iter_num, max_cs_abs)

    # Save results
    np.save(
        exp_save_path
        + "cAcc%.2fexp%02dSignNum%02dBpRes" % (c_acc, exp_idx, bp_sign_num),
        bp_res,
    )

    return exp_idx, bp_sign_num


def template_attack_single_coeff_cs1(args):
    """单个系数的cs1模板攻击（全局函数，用于多进程）"""
    (
        i,
        train_trs,
        test_trs,
        train_label_cs1,
        exp_save_path,
    ) = args
    # 获取该系数的所有类别
    unique_class = np.unique(train_label_cs1)
    unique_class_num = np.zeros((unique_class.shape[0],), dtype=int)
    for j in range(unique_class.shape[0]):
        unique_class_num[j] = np.sum(train_label_cs1 == unique_class[j])

    # 保留数量足够的类别
    reserve_class = unique_class[unique_class_num > 10]
    drop_class = unique_class[unique_class_num <= 10]

    # 转换标签
    label_transformed = train_label_cs1.copy()
    for j in range(len(reserve_class)):
        label_transformed[train_label_cs1 == reserve_class[j]] = j
    for j in range(len(drop_class)):
        label_transformed[train_label_cs1 == drop_class[j]] = len(reserve_class)

    # 进行模板攻击
    ta = Template()
    ta.build_model(
        train_trs,
        label_transformed,
        len(reserve_class) + 1,
        "lda",
        8,
    )
    prob_test = ta.apply_model(test_trs)
    prob_train = ta.apply_model(train_trs)
    reserved_class = reserve_class

    # Save results
    np.save(exp_save_path + "cs1_idx%03dReserveClass" % (i,), reserved_class)
    np.save(exp_save_path + "cs1_idx%03dProbTest" % (i,), prob_test)
    np.save(exp_save_path + "cs1_idx%03dProbTrain" % (i,), prob_train)
    return i


def cs1_valid_bp_experiment(
    exp_min_sign_num=1,
    exp_max_sign_num=20,
    rerun_snr=True,
    rerun_ta=True,
    exp_repeat_num=10,
    bp_iter_num=15,
    poi_num=100,
    class_reserve_threshold=100,
):
    """
    使用cs1_valid数据进行BP实验
    1. 使用模板攻击对cs1进行建模
    2. 使用信噪比选择兴趣点
    3. 训练集9000，测试集1000
    4. 重复10次BP实验
    5. 使用c数据作为约束

    参数:
    - exp_min_sign_num: BP实验最小签名数
    - exp_max_sign_num: BP实验最大签名数
    - rerun_snr: 是否重新运行信噪比计算
    - rerun_ta: 是否重新运行模板攻击
    - exp_repeat_num: BP实验重复次数
    - bp_iter_num: BP迭代次数
    - snr_threshold: 信噪比阈值
    - class_reserve_threshold: 类别保留阈值（数量少于此值的类别会被丢弃）
    """
    # Experiment parameters
    exp_save_path = "../exp_res/cs1_valid_bp/"
    cs1_cut_trs_path = "../traces/cs1_cut/"
    cs1_data_path = "../data/cs1_all/"
    c_data_path = "../data/c_all/"  # 用于约束的c数据

    train_sign_num = 9000
    test_sign_num = 1000
    total_sign_num = 10000
    train_test_split_seed = 123456

    # Create experiment save directory
    if not os.path.exists(exp_save_path):
        os.makedirs(exp_save_path)

    print("开始cs1_valid BP实验...")
    print(f"BP实验参数: min_sign={exp_min_sign_num}, max_sign={exp_max_sign_num}")
    print(f"重新运行信噪比计算: {rerun_snr}")
    print(f"重新运行模板攻击: {rerun_ta}")
    print(f"类别保留阈值: {class_reserve_threshold}")

    # 第一步：加载截取后的波形数据
    print("加载截取后的波形数据...")
    trs_len = np.load(cs1_cut_trs_path + "trs000000.npy").shape[1]
    print(f"波形长度: {trs_len}")

    # 第二步：划分训练集和测试集
    print("划分训练集和测试集...")
    np.random.seed(train_test_split_seed)
    sign_idx_list = np.arange(total_sign_num, dtype=int)
    np.random.shuffle(sign_idx_list)

    train_sign_idx_list = sign_idx_list[:train_sign_num]
    test_sign_idx_list = sign_idx_list[train_sign_num : train_sign_num + test_sign_num]

    np.save(exp_save_path + "train_sign_idx", train_sign_idx_list)
    np.save(exp_save_path + "test_sign_idx", test_sign_idx_list)

    print(f"Training set size: {train_sign_num}, test set size: {test_sign_num}")

    # 第三步：读取拒绝次数信息
    print("Reading rejection count information...")
    rej_num_list = []
    with open("seed and rej num.txt", "r") as f:
        for i in range(total_sign_num):
            line = f.readline().strip().split(" ")
            rej_num = int(line[1])
            rej_num_list.append(rej_num)
    rej_num_array = np.array(rej_num_list)

    # 第四步：加载训练和测试数据
    print("Loading training and test data...")
    train_rej_num = rej_num_array[train_sign_idx_list]
    test_rej_num = rej_num_array[test_sign_idx_list]

    # 训练和测试数据都只使用每个签名的最后一次拒绝采样
    train_trs_num = train_sign_num
    test_trs_num = test_sign_num

    print(f"训练波形总数: {train_trs_num}, 测试波形总数: {test_trs_num}")

    train_trs = np.zeros((train_trs_num, trs_len), dtype=int)
    test_trs = np.zeros((test_trs_num, trs_len), dtype=int)
    train_cs1 = np.zeros((train_trs_num, 256), dtype=int)
    test_cs1 = np.zeros((test_trs_num, 256), dtype=int)
    test_c = np.zeros((test_sign_num, 256), dtype=int)

    # Load training data - 只使用每个签名的最后一次拒绝采样
    for i in tqdm.trange(train_sign_num, desc="Loading training data"):
        trs = np.load(cs1_cut_trs_path + "trs%06d.npy" % (train_sign_idx_list[i],))
        if trs.ndim == 1:
            trs = trs.reshape(1, -1)

        # 只使用最后一次拒绝采样的波形
        last_rej_idx = train_rej_num[i] - 1  # rej从0开始计数
        train_trs[i] = trs[last_rej_idx]

        # 从cs1_all目录读取对应的cs1数据（最后一次拒绝采样）
        cs1 = np.load(
            cs1_data_path + "cs%06drej%06d.npy" % (train_sign_idx_list[i], last_rej_idx)
        )
        train_cs1[i] = cs1 & 0x7F  # 只保留低7位

    # Load test data - 只使用每个签名的最后一次拒绝采样
    for i in tqdm.trange(test_sign_num, desc="Loading test data"):
        sign_idx = test_sign_idx_list[i]
        trs = np.load(cs1_cut_trs_path + "trs%06d.npy" % (sign_idx,))
        if trs.ndim == 1:
            trs = trs.reshape(1, -1)
        # 只使用最后一次拒绝采样的波形
        last_rej_idx = test_rej_num[i] - 1  # rej从0开始计数
        test_trs[i] = trs[last_rej_idx]

        # 从cs1_all目录读取对应的cs1数据（最后一次拒绝采样）
        cs1 = np.load(cs1_data_path + "cs%06drej%06d.npy" % (sign_idx, last_rej_idx))
        test_cs1[i] = cs1 & 0x7F  # 只保留低7位

        # Read the last rejection sampling c data from c_all directory
        test_c[i] = np.load(c_data_path + "c%06drej%06d.npy" % (sign_idx, last_rej_idx))

    # 第五步：计算信噪比并选择兴趣点
    print("Preparing training label data...")
    train_label_cs1 = train_cs1.copy()

    if rerun_snr:
        print("Calculating SNR and selecting points of interest...")
        # Calculate SNR (multiprocessing)
        print("Calculating SNR (multiprocessing)...")
        train_snr = np.zeros((256, trs_len), dtype=float)

        pool = Pool(8)  # Use 8 processes，避免过度竞争
        min_class_count = class_reserve_threshold  # 最小类别数量阈值
        snr_args = [
            (j, train_trs, train_label_cs1, trs_len, min_class_count)
            for j in range(256)
        ]
        snr_results = []
        # 计算chunk_size：任务数量/(进程数*4)
        snr_chunk_size = max(1, len(snr_args) // (8 * 4))
        with tqdm.tqdm(total=256, desc="Calculating SNR") as pbar:
            for result in pool.imap(
                calc_snr_single_coeff, snr_args, chunksize=snr_chunk_size
            ):
                snr_results.append(result)
                pbar.update(1)
        pool.close()
        pool.join()

        for j, snr in snr_results:
            train_snr[j] = snr

        # Save SNR file
        np.save(exp_save_path + "cs1_valid_snr", train_snr)

    else:
        print("Skipping SNR calculation, using existing results...")
        # Load existing SNR file
        try:
            train_snr = np.load(exp_save_path + "cs1_valid_snr.npy")
        except FileNotFoundError:
            print("未找到信噪比文件，创建默认值...")
            # 创建默认的信噪比值（全零）
            train_snr = np.zeros((256, trs_len), dtype=float)

    # Plot SNR for all coefficients on the same graph
    plt.figure(figsize=(12, 8))
    for i in range(0, 256):
        plt.plot(
            train_snr[i], linewidth=0.5, alpha=0.7, label=f"Coeff {i}" if i < 10 else ""
        )
    plt.title("SNR for All CS1 Coefficients")
    plt.xlabel("Sample Index")
    plt.ylabel("SNR")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(exp_save_path + "snr_all_coeffs.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Select points of interest based on SNR
    POI = []
    for i in range(256):
        poi_indices = np.argsort(-train_snr[i])[:poi_num]
        POI.append(poi_indices.flatten())

    # 第六步：模板攻击建模（多进程）
    if rerun_ta:
        print("Performing template attack modeling (multiprocessing)...")

        pool = Pool(4)  # 使用8个进程
        ta_args = []
        for i in range(256):
            ta_args.append(
                (
                    i,
                    train_trs[:, POI[i]],
                    test_trs[:, POI[i]],
                    train_label_cs1[:, i],
                    exp_save_path,
                )
            )
        with tqdm.tqdm(total=256, desc="Template attack modeling") as pbar:
            for result in pool.imap(
                template_attack_single_coeff_cs1, ta_args, chunksize=4
            ):
                pbar.update(1)
        pool.close()
        pool.join()
    else:
        print("Skipping template attack, using existing results...")

    # Calculate and plot template attack accuracy
    print("Calculating template attack accuracy...")
    test_ta_acc = np.zeros((256,), dtype=float)
    test_ta_drop_acc = np.zeros((256,), dtype=float)  # drop类别准确率
    test_ta_drop_recall = np.zeros((256,), dtype=float)  # drop类别召回率

    for i in range(256):
        try:
            reserved_class = np.load(
                exp_save_path + "cs1_idx%03dReserveClass.npy" % (i,)
            )
            test_ta_prob = np.load(exp_save_path + "cs1_idx%03dProbTest.npy" % (i,))
            test_ta_predict = np.argmax(test_ta_prob, axis=1)

            # 计算保留类别的准确率
            correct = 0
            total = 0
            print(f"Reserved class: {len(reserved_class)}")
            for j in range(len(reserved_class)):
                mask = test_ta_predict == j
                if np.sum(mask) > 0:
                    correct += np.sum(test_cs1[mask, i] == reserved_class[j])
                    total += np.sum(mask)

            if total > 0:
                test_ta_acc[i] = correct / total
            else:
                test_ta_acc[i] = 0.0

            # 计算drop类别的准确率和召回率
            if len(reserved_class) > 0:
                # drop类别被映射到最后一个类别（索引为len(reserved_class)）
                drop_class_idx = len(reserved_class)

                # 计算drop类别的准确率：预测为drop类别的样本中，真正属于drop类别的比例
                drop_predict_mask = test_ta_predict == drop_class_idx
                if np.sum(drop_predict_mask) > 0:
                    # 获取所有drop类别的真实值
                    all_classes = np.unique(test_cs1[:, i])
                    drop_classes = all_classes[
                        np.isin(all_classes, reserved_class, invert=True)
                    ]

                    if len(drop_classes) > 0:
                        # 计算预测为drop类别的样本中，真正属于drop类别的数量
                        drop_correct = 0
                        for drop_val in drop_classes:
                            drop_correct += np.sum(
                                (test_cs1[drop_predict_mask, i] == drop_val)
                            )
                        test_ta_drop_acc[i] = drop_correct / np.sum(drop_predict_mask)
                    else:
                        test_ta_drop_acc[i] = 0.0
                else:
                    test_ta_drop_acc[i] = 0.0

                # 计算drop类别的召回率：真正属于drop类别的样本中，被正确预测为drop类别的比例
                all_classes = np.unique(test_cs1[:, i])
                drop_classes = all_classes[
                    np.isin(all_classes, reserved_class, invert=True)
                ]

                if len(drop_classes) > 0:
                    # 计算真正属于drop类别的样本数量
                    drop_true_mask = np.isin(test_cs1[:, i], drop_classes)
                    if np.sum(drop_true_mask) > 0:
                        # 计算真正属于drop类别的样本中，被预测为drop类别的数量
                        drop_recall_correct = np.sum(
                            test_ta_predict[drop_true_mask] == drop_class_idx
                        )
                        test_ta_drop_recall[i] = drop_recall_correct / np.sum(
                            drop_true_mask
                        )
                    else:
                        test_ta_drop_recall[i] = 0.0
                else:
                    test_ta_drop_recall[i] = 0.0
            else:
                test_ta_drop_acc[i] = 0.0
                test_ta_drop_recall[i] = 0.0

        except FileNotFoundError:
            test_ta_acc[i] = 0.0
            test_ta_drop_acc[i] = 0.0
            test_ta_drop_recall[i] = 0.0

    # Plot accuracy graph
    plt.figure(figsize=(15, 5))

    # 子图1: 保留类别准确率
    plt.subplot(1, 3, 1)
    plt.plot(
        test_ta_acc, linewidth=1.0, alpha=0.8, marker="o", markersize=2, color="blue"
    )
    plt.title("Reserved Classes Accuracy")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # 子图2: drop类别准确率
    plt.subplot(1, 3, 2)
    plt.plot(
        test_ta_drop_acc,
        linewidth=1.0,
        alpha=0.8,
        marker="s",
        markersize=2,
        color="red",
    )
    plt.title("Dropped Classes Accuracy")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # 子图3: drop类别召回率
    plt.subplot(1, 3, 3)
    plt.plot(
        test_ta_drop_recall,
        linewidth=1.0,
        alpha=0.8,
        marker="^",
        markersize=2,
        color="green",
    )
    plt.title("Dropped Classes Recall")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Recall")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        exp_save_path + "ta_accuracy_drop_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # Print accuracy statistics
    print(f"\n=== 保留类别准确率统计 ===")
    print(f"Average accuracy: {np.mean(test_ta_acc):.4f}")
    print(
        f"Highest accuracy: {np.max(test_ta_acc):.4f} (coefficient {np.argmax(test_ta_acc)})"
    )
    print(
        f"Lowest accuracy: {np.min(test_ta_acc):.4f} (coefficient {np.argmin(test_ta_acc)})"
    )

    print(f"\n=== Drop类别准确率统计 ===")
    print(f"平均准确率: {np.mean(test_ta_drop_acc):.4f}")
    print(
        f"最高准确率: {np.max(test_ta_drop_acc):.4f} (系数 {np.argmax(test_ta_drop_acc)})"
    )
    print(
        f"最低准确率: {np.min(test_ta_drop_acc):.4f} (系数 {np.argmin(test_ta_drop_acc)})"
    )

    print(f"\n=== Drop类别召回率统计 ===")
    print(f"平均召回率: {np.mean(test_ta_drop_recall):.4f}")
    print(
        f"最高召回率: {np.max(test_ta_drop_recall):.4f} (系数 {np.argmax(test_ta_drop_recall)})"
    )
    print(
        f"最低召回率: {np.min(test_ta_drop_recall):.4f} (系数 {np.argmin(test_ta_drop_recall)})"
    )

    # 计算F1分数
    f1_scores = np.zeros((256,), dtype=float)
    for i in range(256):
        if test_ta_drop_acc[i] + test_ta_drop_recall[i] > 0:
            f1_scores[i] = (
                2
                * test_ta_drop_acc[i]
                * test_ta_drop_recall[i]
                / (test_ta_drop_acc[i] + test_ta_drop_recall[i])
            )
        else:
            f1_scores[i] = 0.0

    print(f"\n=== Drop类别F1分数统计 ===")
    print(f"平均F1分数: {np.mean(f1_scores):.4f}")
    print(f"最高F1分数: {np.max(f1_scores):.4f} (系数 {np.argmax(f1_scores)})")
    print(f"最低F1分数: {np.min(f1_scores):.4f} (系数 {np.argmin(f1_scores)})")

    # 绘制F1分数图
    plt.figure(figsize=(10, 6))
    plt.plot(
        f1_scores, linewidth=1.0, alpha=0.8, marker="d", markersize=2, color="purple"
    )
    plt.title("Dropped Classes F1 Score")
    plt.xlabel("Coefficient Index")
    plt.ylabel("F1 Score")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(exp_save_path + "ta_drop_f1_score.png", dpi=300, bbox_inches="tight")
    plt.show()

    max_snr = np.max(train_snr, axis=1)
    plt.plot(max_snr, linewidth=1.0, alpha=0.8, marker="o", markersize=2)
    plt.show()

    # 第七步：BP实验（串行执行，内部使用并行化优化）
    print("Starting BP experiment...")

    # Load test probabilities
    test_cs1_prob = np.zeros((test_trs_num, 256, 128), dtype=float)
    for i in range(256):
        try:
            reserved_class = np.load(
                exp_save_path + "cs1_idx%03dReserveClass.npy" % (i,)
            )
            test_ta_prob = np.load(exp_save_path + "cs1_idx%03dProbTest.npy" % (i,))
            for j in range(len(reserved_class)):
                for k in range(test_trs_num):
                    test_cs1_prob[k, i, reserved_class[j]] = test_ta_prob[k, j]
        except FileNotFoundError:
            continue
    # 将test_cs1_prob置为正确分布测试测试后续代码的正确性
    # for i in range(256):
    #     for j in range(test_trs_num):
    #         test_cs1_prob[j, i, test_cs1[j, i]] = 1

    # 并行执行BP实验，每个bp_sign_num用多个进程并行跑多次实验
    exp_repeat_seed = 234567  # Fixed seed
    bp_args = []
    # 外层循环：bp_sign_num，内层循环：exp_idx（并行）
    for bp_sign_num in range(exp_min_sign_num, exp_max_sign_num + 1):
        # 为当前bp_sign_num准备所有实验的参数
        for exp_idx in range(exp_repeat_num):
            bp_args.append(
                (
                    exp_idx,
                    bp_sign_num,
                    test_sign_num,
                    test_cs1_prob,
                    test_c,
                    bp_iter_num,
                    exp_save_path,
                    exp_repeat_seed,
                    test_ta_acc,
                )
            )

    # 单进程写法
    # for args in bp_args:
    #     run_single_bp_experiment(args)

    # 使用多进程并行执行所有实验
    with Pool(10) as pool:
        with tqdm.tqdm(
            total=exp_repeat_num * (exp_max_sign_num - exp_min_sign_num + 1),
            desc=f"总实验进度",
        ) as pbar:
            for _ in pool.imap_unordered(
                run_single_bp_experiment, bp_args, chunksize=1
            ):
                pbar.update(1)

    print("cs1_valid BP实验完成！")
    print(f"Results saved in: {exp_save_path}")


def c_all_cs1_all_bp_experiment(
    rerun_c_snr: bool = True,
    snr_c_plot: bool = True,
    rerun_c_ta: bool = True,
    rerun_cs1_snr: bool = True,
    snr_cs1_plot: bool = True,
    rerun_cs1_ta: bool = True,
    process_num: int = 8,
    poi_num: int = 100,
    exp_repeat_num: int = 10,
    exp_min_sign_num: int = 1,
    exp_max_sign_num: int = 20,
    bp_iter_num: int = 15,
):
    """
    使用全部的c,以及cs1进行bp实验,不包括y是因为y的信噪比较低,先看看纯cs1的效果。
    """

    # 路径与参数
    exp_save_path = "../exp_res/c_all_cs1_all_bp/"
    # 创建保存目录
    if not os.path.exists(exp_save_path):
        os.makedirs(exp_save_path)
    train_sign_num = 9000
    test_sign_num = 1000
    total_sign_num = 10000
    train_test_split_seed = 123456

    # 划分训练集和测试集
    print("划分训练集和测试集...")
    np.random.seed(train_test_split_seed)
    sign_idx_list = np.arange(total_sign_num, dtype=int)
    np.random.shuffle(sign_idx_list)
    train_sign_idx_list = sign_idx_list[:train_sign_num]
    test_sign_idx_list = sign_idx_list[train_sign_num : train_sign_num + test_sign_num]
    np.save(exp_save_path + "train_sign_idx", train_sign_idx_list)
    np.save(exp_save_path + "test_sign_idx", test_sign_idx_list)
    print(f"Training set size: {train_sign_num}, test set size: {test_sign_num}")

    # 读取拒绝次数，每签名最后一次拒绝采样为合法签名,其余为被拒绝的签名
    print("Reading rejection count information...")
    rej_num_list = []
    with open("seed and rej num.txt", "r") as f:
        for _ in range(total_sign_num):
            line = f.readline().strip().split(" ")
            rej_num_list.append(int(line[1]))
    rej_num_array = np.array(rej_num_list)

    # 先处理c的数据,计算信噪比,根据信噪比选择兴趣点然后做模板
    c_cut_trs_path = "../traces/c_cut/"
    c_data_path = "../data/c_all/"

    print("开始 c 的SNR测试和模板攻击...")
    print(f"重新运行信噪比计算: {rerun_c_snr}")
    print(f"重新运行模板攻击: {rerun_c_ta}")

    # 读取一个样本确定波形长度（c_cut 可能为 [rej, len] 或 [len]）
    print("加载截取后的 c 波形长度...")
    sample_trs = np.load(c_cut_trs_path + "trs000000.npy")
    if sample_trs.ndim == 1:
        c_trs_len = sample_trs.shape[0]
    else:
        c_trs_len = sample_trs.shape[1]
    print(f"c波形长度: {c_trs_len}")

    # 加载训练和测试数据
    print("加载c训练和测试数据...")
    c_train_trs = np.zeros(
        (np.sum(rej_num_array[train_sign_idx_list]), c_trs_len), dtype=int
    )
    c_test_trs = np.zeros(
        (np.sum(rej_num_array[test_sign_idx_list]), c_trs_len), dtype=int
    )
    c_train_label = np.zeros(
        (np.sum(rej_num_array[train_sign_idx_list]), 256), dtype=int
    )
    c_test_label = np.zeros((np.sum(rej_num_array[test_sign_idx_list]), 256), dtype=int)

    # 加载c训练数据
    for i in tqdm.trange(train_sign_num, desc="加载c训练数据"):
        sign_idx = train_sign_idx_list[i]
        trs = np.load(c_cut_trs_path + "trs%06d.npy" % (sign_idx,))
        if trs.ndim == 1:
            raise ValueError("c的波形数组不应该为1维")
        else:
            for j in range(rej_num_array[sign_idx]):
                c_train_trs[np.sum(rej_num_array[train_sign_idx_list[:i]]) + j] = trs[j]
                c_train_label[np.sum(rej_num_array[train_sign_idx_list[:i]]) + j] = (
                    np.load(c_data_path + "c%06drej%06d.npy" % (sign_idx, j))
                )
    # 加载c测试数据
    for i in tqdm.trange(test_sign_num, desc="加载c测试数据"):
        sign_idx = test_sign_idx_list[i]

        trs = np.load(c_cut_trs_path + "trs%06d.npy" % (sign_idx,))
        if trs.ndim == 1:
            raise ValueError("c的波形数组不应该为1维")
        else:
            # 使用最后一次拒绝采样的波形（与标签一致）
            for j in range(rej_num_array[sign_idx]):
                c_test_trs[np.sum(rej_num_array[test_sign_idx_list[:i]]) + j] = trs[j]
                c_test_label[np.sum(rej_num_array[test_sign_idx_list[:i]]) + j] = (
                    np.load(c_data_path + "c%06drej%06d.npy" % (sign_idx, j))
                )

    print(
        f"c训练数据长度: {c_train_trs.shape[0]}, c测试数据长度: {c_test_trs.shape[0]}"
    )
    print(
        f"c训练标签长度: {c_train_label.shape[0]}, c测试标签长度: {c_test_label.shape[0]}"
    )

    c_train_label += 1  # -1的标签不适合作模板，因此将-1,0,1分别映射为0,1,2
    c_test_label += 1

    # 仅做 SNR
    if rerun_c_snr:
        print("计算 c 的信噪比（多进程，使用共享内存）...")
        c_train_snr = np.zeros((256, c_trs_len), dtype=float)
        # for i in tqdm.trange(256, desc="计算c的信噪比"):
        #     _, c_train_snr[i] = calc_snr_single_coeff(
        #         (i, c_train_trs, c_train_label, c_trs_len, 0)
        #     )

        # 创建共享内存
        train_trs_shm = shared_memory.SharedMemory(create=True, size=c_train_trs.nbytes)
        train_label_shm = shared_memory.SharedMemory(
            create=True, size=c_train_label.nbytes
        )

        # 将数据复制到共享内存
        train_trs_shared = np.ndarray(
            c_train_trs.shape, dtype=c_train_trs.dtype, buffer=train_trs_shm.buf
        )
        train_label_shared = np.ndarray(
            c_train_label.shape, dtype=c_train_label.dtype, buffer=train_label_shm.buf
        )
        train_trs_shared[:] = c_train_trs[:]
        train_label_shared[:] = c_train_label[:]

        pool = Pool(processes=2)
        snr_args = [
            (
                j,
                train_trs_shm.name,
                train_label_shm.name,
                c_train_trs.shape,
                c_train_label.shape,
                c_trs_len,
                0,
            )  # 0表示不考虑最小类别数量阈值
            for j in range(256)
        ]
        snr_results = []
        with tqdm.tqdm(total=256, desc="计算SNR") as pbar:
            for result in pool.imap(
                calc_snr_single_coeff_shared,
                snr_args,
                chunksize=1,
            ):
                snr_results.append(result)
                pbar.update(1)
        pool.close()
        pool.join()

        for j, snr in snr_results:
            c_train_snr[j] = snr

        # 清理共享内存
        train_trs_shm.close()
        train_label_shm.close()
        train_trs_shm.unlink()
        train_label_shm.unlink()

        # 保存 SNR
        np.save(exp_save_path + "c_all_snr", c_train_snr)
    else:
        print("Skipping SNR calculation, using existing results...")
        c_train_snr = np.load(exp_save_path + "c_all_snr.npy")

    # 绘图
    if snr_c_plot:
        plt.figure(figsize=(12, 8))
        for i in range(256):
            plt.plot(
                c_train_snr[i],
                linewidth=0.5,
                alpha=0.7,
                label=f"Coeff {i}" if i < 10 else "",
            )
            print(np.sum(c_train_snr[i] > 0.003))
        plt.title("SNR for All Coefficients (c)")
        plt.xlabel("Sample Index")
        plt.ylabel("SNR")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            exp_save_path + "snr_all_coeffs_c.png", dpi=300, bbox_inches="tight"
        )
        plt.show()
        plt.close()

    # Select points of interest based on SNR
    POI = []
    for i in range(256):
        poi_indices = np.argsort(-c_train_snr[i])[:poi_num]
        POI.append(poi_indices.flatten())

    # 模板攻击建模（多进程）
    if rerun_c_ta:
        print("Performing template attack modeling (multiprocessing)...")

        pool = Pool(processes=4)
        ta_args = [
            (
                i,
                c_train_trs[:, POI[i]],
                c_test_trs[:, POI[i]],
                c_train_label[:, i],
                exp_save_path,
            )
            for i in range(256)
        ]
        with tqdm.tqdm(total=256, desc="Template attack modeling") as pbar:
            for result in pool.imap(template_attack_single_coeff, ta_args, chunksize=1):
                pbar.update(1)
        pool.close()
        pool.join()
    else:
        print("Skipping template attack, using existing results...")

    # 内存管理，从这个位置开始就不用c的trace和c的train label了
    del c_train_trs
    del c_train_label
    del c_test_trs

    # Calculate and plot template attack accuracy
    print("Calculating template attack accuracy...")
    c_test_ta_acc = np.zeros((256,), dtype=float)
    for i in range(256):
        try:
            c_test_ta_prob = np.load(exp_save_path + "idx%03dProbTest.npy" % (i,))
            # 将合法c的概率修改为正确分布
            for j in range(len(test_sign_idx_list)):
                c_test_ta_prob[
                    np.sum(rej_num_array[test_sign_idx_list[:j]])
                    + rej_num_array[test_sign_idx_list[j]]
                    - 1
                ] = 0
                c_test_ta_prob[
                    np.sum(rej_num_array[test_sign_idx_list[:j]])
                    + rej_num_array[test_sign_idx_list[j]]
                    - 1,
                    c_test_label[
                        np.sum(rej_num_array[test_sign_idx_list[:j]])
                        + rej_num_array[test_sign_idx_list[j]]
                        - 1,
                        i,
                    ],
                ] = 1
            c_test_ta_predict = np.argmax(c_test_ta_prob, axis=1)
            c_test_ta_acc[i] = (
                np.sum(c_test_ta_predict == c_test_label[:, i]) / c_test_label.shape[0]
            )
        except FileNotFoundError:
            c_test_ta_acc[i] = 0.0

    # Plot accuracy graph
    plt.figure(figsize=(12, 8))
    plt.plot(c_test_ta_acc, linewidth=1.0, alpha=0.8, marker="o", markersize=2)
    plt.title("Template Attack Accuracy for All C Coefficients")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(exp_save_path + "c_ta_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print accuracy statistics
    print(f"\n=== C 系数模板攻击准确率统计 ===")
    print(f"平均准确率: {np.mean(c_test_ta_acc):.4f}")
    print(f"最高准确率: {np.max(c_test_ta_acc):.4f} (系数 {np.argmax(c_test_ta_acc)})")
    print(f"最低准确率: {np.min(c_test_ta_acc):.4f} (系数 {np.argmin(c_test_ta_acc)})")
    print(f"准确率 > 0.5 的系数数量: {np.sum(c_test_ta_acc > 0.5)}")
    print(f"准确率 > 0.7 的系数数量: {np.sum(c_test_ta_acc > 0.7)}")

    # 分析每个系数三类标签的准确率和召回率
    print("\n=== C 系数三类标签准确率和召回率分析 ===")

    # 初始化存储数组
    class_accuracy = np.zeros((256, 3), dtype=float)  # [系数, 类别] 的准确率
    class_recall = np.zeros((256, 3), dtype=float)  # [系数, 类别] 的召回率
    class_f1 = np.zeros((256, 3), dtype=float)  # [系数, 类别] 的F1分数

    for i in range(256):
        try:
            c_test_ta_prob = np.load(exp_save_path + "idx%03dProbTest.npy" % (i,))
            # 将合法c的概率修改为正确分布
            for j in range(len(test_sign_idx_list)):
                c_test_ta_prob[
                    np.sum(rej_num_array[test_sign_idx_list[:j]])
                    + rej_num_array[test_sign_idx_list[j]]
                    - 1
                ] = 0
                c_test_ta_prob[
                    np.sum(rej_num_array[test_sign_idx_list[:j]])
                    + rej_num_array[test_sign_idx_list[j]]
                    - 1,
                    c_test_label[
                        np.sum(rej_num_array[test_sign_idx_list[:j]])
                        + rej_num_array[test_sign_idx_list[j]]
                        - 1,
                        i,
                    ],
                ] = 1

            c_test_ta_predict = np.argmax(c_test_ta_prob, axis=1)

            # 计算每个类别的准确率和召回率
            for class_label in range(3):  # 0, 1, 2 (对应原来的 -1, 0, 1)
                # 准确率：预测为该类别的样本中，真正属于该类别的比例
                pred_mask = c_test_ta_predict == class_label
                if np.sum(pred_mask) > 0:
                    class_accuracy[i, class_label] = np.sum(
                        c_test_label[pred_mask, i] == class_label
                    ) / np.sum(pred_mask)
                else:
                    class_accuracy[i, class_label] = 0.0

                # 召回率：真正属于该类别的样本中，被正确预测为该类别的比例
                true_mask = c_test_label[:, i] == class_label
                if np.sum(true_mask) > 0:
                    class_recall[i, class_label] = np.sum(
                        c_test_ta_predict[true_mask] == class_label
                    ) / np.sum(true_mask)
                else:
                    class_recall[i, class_label] = 0.0

                # F1分数
                if class_accuracy[i, class_label] + class_recall[i, class_label] > 0:
                    class_f1[i, class_label] = (
                        2
                        * class_accuracy[i, class_label]
                        * class_recall[i, class_label]
                        / (
                            class_accuracy[i, class_label]
                            + class_recall[i, class_label]
                        )
                    )
                else:
                    class_f1[i, class_label] = 0.0

        except FileNotFoundError:
            class_accuracy[i] = 0.0
            class_recall[i] = 0.0
            class_f1[i] = 0.0

    # 绘制三类标签的准确率和召回率
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1: 各类别准确率
    ax1 = axes[0]
    for class_label in range(3):
        ax1.plot(
            class_accuracy[:, class_label],
            linewidth=1.0,
            alpha=0.8,
            marker="o",
            markersize=2,
            label=f"Class {class_label} ({class_label-1})",
        )
    ax1.set_title("Per-Class Accuracy for C Coefficients")
    ax1.set_xlabel("Coefficient Index")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)

    # 子图2: 各类别召回率
    ax2 = axes[1]
    for class_label in range(3):
        ax2.plot(
            class_recall[:, class_label],
            linewidth=1.0,
            alpha=0.8,
            marker="s",
            markersize=2,
            label=f"Class {class_label} ({class_label-1})",
        )
    ax2.set_title("Per-Class Recall for C Coefficients")
    ax2.set_xlabel("Coefficient Index")
    ax2.set_ylabel("Recall")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        exp_save_path + "c_class_wise_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # 打印详细统计信息
    print("\n=== 各类别平均性能统计 ===")
    for class_label in range(3):
        print(f"Class {class_label} (原{class_label-1}):")
        print(f"  平均准确率: {np.mean(class_accuracy[:, class_label]):.4f}")
        print(f"  平均召回率: {np.mean(class_recall[:, class_label]):.4f}")
        print(f"  平均F1分数: {np.mean(class_f1[:, class_label]):.4f}")
        print(
            f"  最高准确率: {np.max(class_accuracy[:, class_label]):.4f} (系数 {np.argmax(class_accuracy[:, class_label])})"
        )
        print(
            f"  最高召回率: {np.max(class_recall[:, class_label]):.4f} (系数 {np.argmax(class_recall[:, class_label])})"
        )
        print()

    print("c 的SNR测试和模板攻击完成，结果保存在:", exp_save_path)

    # 开始cs1的snr测试和模板攻击
    print("开始计算cs1的snr测试和模板攻击...")
    print(f"重新运行信噪比计算: {rerun_cs1_snr}")
    print(f"重新运行模板攻击: {rerun_cs1_ta}")

    cs1_cut_trs_path = "../traces/cs1_cut/"
    cs1_data_path = "../data/cs1_all/"

    # 第一步：加载截取后的波形数据
    print("加载截取后的波形数据...")
    cs1_trs_len = np.load(cs1_cut_trs_path + "trs000000.npy").shape[1]
    print(f"波形长度: {cs1_trs_len}")

    # 第四步：加载训练和测试数据
    print("Loading training and test data...")

    cs1_train_trs = np.zeros(
        (np.sum(rej_num_array[train_sign_idx_list]), cs1_trs_len), dtype=int
    )
    cs1_test_trs = np.zeros(
        (np.sum(rej_num_array[test_sign_idx_list]), cs1_trs_len), dtype=int
    )
    cs1_train_label = np.zeros(
        (np.sum(rej_num_array[train_sign_idx_list]), 256), dtype=int
    )
    cs1_test_label = np.zeros(
        (np.sum(rej_num_array[test_sign_idx_list]), 256), dtype=int
    )

    # Load training data - 使用全部数据，被拒绝的以及合法的
    for i in tqdm.trange(train_sign_num, desc="加载cs1训练数据"):
        trs = np.load(cs1_cut_trs_path + "trs%06d.npy" % (train_sign_idx_list[i],))
        if trs.ndim == 1:
            raise ValueError("cs1的波形数组不应该为1维")

        # 使用全部数据，被拒绝的以及合法的
        for j in range(rej_num_array[train_sign_idx_list[i]]):
            cs1_train_trs[np.sum(rej_num_array[train_sign_idx_list[:i]]) + j] = trs[j]
            cs1_train_label[np.sum(rej_num_array[train_sign_idx_list[:i]]) + j] = (
                np.load(
                    cs1_data_path + "cs%06drej%06d.npy" % (train_sign_idx_list[i], j)
                )
            )
    cs1_train_label &= 0x7F

    # Load test data - 使用全部数据，被拒绝的以及合法的
    for i in tqdm.trange(test_sign_num, desc="加载cs1测试数据"):
        sign_idx = test_sign_idx_list[i]
        trs = np.load(cs1_cut_trs_path + "trs%06d.npy" % (sign_idx,))
        if trs.ndim == 1:
            raise ValueError("cs1的波形数组不应该为1维")
        # 使用全部数据，被拒绝的以及合法的
        for j in range(rej_num_array[test_sign_idx_list[i]]):
            cs1_test_trs[np.sum(rej_num_array[test_sign_idx_list[:i]]) + j] = trs[j]
            cs1_test_label[np.sum(rej_num_array[test_sign_idx_list[:i]]) + j] = np.load(
                cs1_data_path + "cs%06drej%06d.npy" % (test_sign_idx_list[i], j)
            )
    cs1_test_label &= 0x7F

    # 第五步：计算信噪比并选择兴趣点
    print("Preparing training label data...")

    if rerun_cs1_snr:
        print("计算cs1的信噪比（多进程，使用共享内存）...")
        cs1_train_snr = np.zeros((256, cs1_trs_len), dtype=float)
        # for i in tqdm.trange(256, desc="计算cs1的信噪比"):
        #     _, cs1_train_snr[i] = calc_snr_single_coeff(
        #         (i, cs1_train_trs, cs1_train_label, cs1_trs_len, 10)
        #     )

        # 创建共享内存
        cs1_train_trs_shm = shared_memory.SharedMemory(
            create=True, size=cs1_train_trs.nbytes
        )
        cs1_train_label_shm = shared_memory.SharedMemory(
            create=True, size=cs1_train_label.nbytes
        )

        # 将数据复制到共享内存
        cs1_train_trs_shared = np.ndarray(
            cs1_train_trs.shape, dtype=cs1_train_trs.dtype, buffer=cs1_train_trs_shm.buf
        )
        cs1_train_label_shared = np.ndarray(
            cs1_train_label.shape,
            dtype=cs1_train_label.dtype,
            buffer=cs1_train_label_shm.buf,
        )
        cs1_train_trs_shared[:] = cs1_train_trs[:]
        cs1_train_label_shared[:] = cs1_train_label[:]

        pool = Pool(processes=2)
        snr_args = [
            (
                j,
                cs1_train_trs_shm.name,
                cs1_train_label_shm.name,
                cs1_train_trs.shape,
                cs1_train_label.shape,
                cs1_trs_len,
                10,  # min_class_count
            )
            for j in range(256)
        ]
        snr_results = []

        with tqdm.tqdm(total=256, desc="计算cs1信噪比") as pbar:
            for result in pool.imap(
                calc_snr_single_coeff_shared,
                snr_args,
                chunksize=1,
            ):
                snr_results.append(result)
                pbar.update(1)
        pool.close()
        pool.join()

        for j, snr in snr_results:
            cs1_train_snr[j] = snr

        # 清理共享内存
        cs1_train_trs_shm.close()
        cs1_train_label_shm.close()
        cs1_train_trs_shm.unlink()
        cs1_train_label_shm.unlink()

        # Save SNR file
        np.save(exp_save_path + "cs1_all_snr", cs1_train_snr)

    else:
        print("Skipping SNR calculation, using existing results...")
        # Load existing SNR file
        try:
            cs1_train_snr = np.load(exp_save_path + "cs1_all_snr.npy")
        except FileNotFoundError:
            raise FileNotFoundError("未找到信噪比文件")

    if snr_cs1_plot:
        # Plot SNR for all coefficients on the same graph
        plt.figure(figsize=(12, 8))
        for i in range(0, 256):
            plt.plot(
                cs1_train_snr[i],
                linewidth=0.5,
                alpha=0.7,
                label=f"Coeff {i}" if i < 10 else "",
            )
        plt.title("SNR for All CS1 Coefficients")
        plt.xlabel("Sample Index")
        plt.ylabel("SNR")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            exp_save_path + "snr_all_coeffs_cs1.png", dpi=300, bbox_inches="tight"
        )
        plt.show()
        pass

    # Select points of interest based on SNR
    POI = []
    for i in range(256):
        poi_indices = np.argsort(-cs1_train_snr[i])[:poi_num]
        POI.append(poi_indices.flatten())

    # 第六步：模板攻击建模（多进程）
    if rerun_cs1_ta:
        print("Performing template attack modeling (multiprocessing)...")

        pool = Pool(4)  # 使用8个进程
        ta_args = []
        for i in range(256):
            ta_args.append(
                (
                    i,
                    cs1_train_trs[:, POI[i]],
                    cs1_test_trs[:, POI[i]],
                    cs1_train_label[:, i],
                    exp_save_path,
                )
            )
        with tqdm.tqdm(total=256, desc="Template attack modeling") as pbar:
            for result in pool.imap_unordered(
                template_attack_single_coeff_cs1, ta_args, chunksize=1
            ):
                pbar.update(1)
        pool.close()
        pool.join()
    else:
        print("Skipping template attack, using existing results...")

    # Calculate and plot template attack accuracy
    print("Calculating template attack accuracy...")
    test_ta_acc = np.zeros((256,), dtype=float)
    test_ta_drop_acc = np.zeros((256,), dtype=float)  # drop类别准确率
    test_ta_drop_recall = np.zeros((256,), dtype=float)  # drop类别召回率

    for i in range(256):
        try:
            reserved_class = np.load(
                exp_save_path + "cs1_idx%03dReserveClass.npy" % (i,)
            )
            test_ta_prob = np.load(exp_save_path + "cs1_idx%03dProbTest.npy" % (i,))
            test_ta_predict = np.argmax(test_ta_prob, axis=1)

            # 计算保留类别的准确率
            correct = 0
            total = 0
            # print(f"Reserved class: {len(reserved_class)}")
            for j in range(len(reserved_class)):
                mask = test_ta_predict == j
                if np.sum(mask) > 0:
                    correct += np.sum(cs1_test_label[mask, i] == reserved_class[j])
                    total += np.sum(mask)

            if total > 0:
                test_ta_acc[i] = correct / total
            else:
                test_ta_acc[i] = 0.0

            # 计算drop类别的准确率和召回率
            if len(reserved_class) > 0:
                # drop类别被映射到最后一个类别（索引为len(reserved_class)）
                drop_class_idx = len(reserved_class)

                # 计算drop类别的准确率：预测为drop类别的样本中，真正属于drop类别的比例
                drop_predict_mask = test_ta_predict == drop_class_idx
                if np.sum(drop_predict_mask) > 0:
                    # 获取所有drop类别的真实值
                    all_classes = np.unique(cs1_test_label[:, i])
                    drop_classes = all_classes[
                        np.isin(all_classes, reserved_class, invert=True)
                    ]

                    if len(drop_classes) > 0:
                        # 计算预测为drop类别的样本中，真正属于drop类别的数量
                        drop_correct = 0
                        for drop_val in drop_classes:
                            drop_correct += np.sum(
                                (cs1_test_label[drop_predict_mask, i] == drop_val)
                            )
                        test_ta_drop_acc[i] = drop_correct / np.sum(drop_predict_mask)
                    else:
                        test_ta_drop_acc[i] = 0.0
                else:
                    test_ta_drop_acc[i] = 0.0

                # 计算drop类别的召回率：真正属于drop类别的样本中，被正确预测为drop类别的比例
                all_classes = np.unique(cs1_test_label[:, i])
                drop_classes = all_classes[
                    np.isin(all_classes, reserved_class, invert=True)
                ]

                if len(drop_classes) > 0:
                    # 计算真正属于drop类别的样本数量
                    drop_true_mask = np.isin(cs1_test_label[:, i], drop_classes)
                    if np.sum(drop_true_mask) > 0:
                        # 计算真正属于drop类别的样本中，被预测为drop类别的数量
                        drop_recall_correct = np.sum(
                            test_ta_predict[drop_true_mask] == drop_class_idx
                        )
                        test_ta_drop_recall[i] = drop_recall_correct / np.sum(
                            drop_true_mask
                        )
                    else:
                        test_ta_drop_recall[i] = 0.0
                else:
                    test_ta_drop_recall[i] = 0.0
            else:
                test_ta_drop_acc[i] = 0.0
                test_ta_drop_recall[i] = 0.0

        except FileNotFoundError:
            test_ta_acc[i] = 0.0
            test_ta_drop_acc[i] = 0.0
            test_ta_drop_recall[i] = 0.0

    # Plot accuracy graph
    plt.figure(figsize=(15, 5))

    # 子图1: 保留类别准确率
    plt.subplot(1, 3, 1)
    plt.plot(
        test_ta_acc, linewidth=1.0, alpha=0.8, marker="o", markersize=2, color="blue"
    )
    plt.title("Reserved Classes Accuracy")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # 子图2: drop类别准确率
    plt.subplot(1, 3, 2)
    plt.plot(
        test_ta_drop_acc,
        linewidth=1.0,
        alpha=0.8,
        marker="s",
        markersize=2,
        color="red",
    )
    plt.title("Dropped Classes Accuracy")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # 子图3: drop类别召回率
    plt.subplot(1, 3, 3)
    plt.plot(
        test_ta_drop_recall,
        linewidth=1.0,
        alpha=0.8,
        marker="^",
        markersize=2,
        color="green",
    )
    plt.title("Dropped Classes Recall")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Recall")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        exp_save_path + "ta_accuracy_drop_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # Print accuracy statistics
    print(f"\n=== 保留类别准确率统计 ===")
    print(f"Average accuracy: {np.mean(test_ta_acc):.4f}")
    print(
        f"Highest accuracy: {np.max(test_ta_acc):.4f} (coefficient {np.argmax(test_ta_acc)})"
    )
    print(
        f"Lowest accuracy: {np.min(test_ta_acc):.4f} (coefficient {np.argmin(test_ta_acc)})"
    )

    print(f"\n=== Drop类别准确率统计 ===")
    print(f"平均准确率: {np.mean(test_ta_drop_acc):.4f}")
    print(
        f"最高准确率: {np.max(test_ta_drop_acc):.4f} (系数 {np.argmax(test_ta_drop_acc)})"
    )
    print(
        f"最低准确率: {np.min(test_ta_drop_acc):.4f} (系数 {np.argmin(test_ta_drop_acc)})"
    )

    print(f"\n=== Drop类别召回率统计 ===")
    print(f"平均召回率: {np.mean(test_ta_drop_recall):.4f}")
    print(
        f"最高召回率: {np.max(test_ta_drop_recall):.4f} (系数 {np.argmax(test_ta_drop_recall)})"
    )
    print(
        f"最低召回率: {np.min(test_ta_drop_recall):.4f} (系数 {np.argmin(test_ta_drop_recall)})"
    )

    # 内存管理
    del cs1_train_trs
    del cs1_train_label
    del cs1_test_trs

    # 该做BP了
    test_cs1_prob = np.zeros(
        (np.sum(rej_num_array[test_sign_idx_list]), 256, 128), dtype=float
    )
    for i in range(256):
        try:
            reserved_class = np.load(
                exp_save_path + "cs1_idx%03dReserveClass.npy" % (i,)
            )
            test_ta_prob = np.load(exp_save_path + "cs1_idx%03dProbTest.npy" % (i,))
            for j in range(len(reserved_class)):
                for k in range(test_cs1_prob.shape[0]):
                    test_cs1_prob[k, i, reserved_class[j]] = test_ta_prob[k, j]
        except FileNotFoundError:
            continue

    # 将test_cs1_prob置为正确分布测试测试后续代码的正确性
    # for i in range(256):
    #     for j in range(test_trs_num):
    #         test_cs1_prob[j, i, test_cs1[j, i]] = 1

    # 对cs1的分布进行切分，仅保留绝对值小于32的分布
    max_cs_abs = 32  # cs1的最大绝对值
    test_cs1_prob_cut = np.zeros(
        (np.sum(rej_num_array[test_sign_idx_list]), 256, 2 * max_cs_abs + 1),
        dtype=float,
    )
    for i in range(256):
        for j in range(test_cs1_prob.shape[0]):
            test_cs1_prob_cut[j, i, :max_cs_abs] = test_cs1_prob[j, i, -max_cs_abs:]
            test_cs1_prob_cut[j, i, max_cs_abs:] = test_cs1_prob[j, i, : max_cs_abs + 1]
    del test_cs1_prob

    test_c_prob = np.zeros(
        (np.sum(rej_num_array[test_sign_idx_list]), 256, 3), dtype=float
    )
    for i in tqdm.trange(256, desc="load c prob"):
        test_c_prob[:, i, :] = np.load(exp_save_path + "idx%03dProbTest.npy" % (i,))

    """注释掉下面的代码,就几乎等效于仅使用被拒绝签名"""
    # 将合法c的概率修改为正确分布
    for i in range(len(test_sign_idx_list)):
        trs_idx = (
            np.sum(rej_num_array[test_sign_idx_list[:i]])
            + rej_num_array[test_sign_idx_list[i]]
            - 1
        )
        for j in range(256):
            if c_test_label[trs_idx, j] == 0:
                test_c_prob[trs_idx, j, :] = [1, 0, 0]
            elif c_test_label[trs_idx, j] == 1:
                test_c_prob[trs_idx, j, :] = [0, 1, 0]
            elif c_test_label[trs_idx, j] == 2:
                test_c_prob[trs_idx, j, :] = [0, 0, 1]

    # 先将c设置为完全正确的，验证后续代码的正确性
    # for i in range(test_c_prob.shape[0]):
    #     for j in range(256):
    #         if c_test_label[i, j] == 0:
    #             test_c_prob[i, j] = [1, 0, 0]
    #         elif c_test_label[i, j] == 1:
    #             test_c_prob[i, j] = [0, 1, 0]
    #         elif c_test_label[i, j] == 2:
    #             test_c_prob[i, j] = [0, 0, 1]
    #         else:
    #             raise ValueError("c_test_label中存在非法值")

    # 并行执行BP实验，每个bp_sign_num用多个进程并行跑多次实验
    exp_repeat_seed = 234567  # Fixed seed
    bp_args = []
    # 提前将bp所需的CS和C矩阵在这里构造后，后面就只需要直接sumprouct就行了，减少多进程内存开销
    for bp_sign_num in tqdm.trange(
        exp_min_sign_num, exp_max_sign_num + 1, 1, desc="构建bp参数列表"
    ):
        # 为当前bp_sign_num准备所有实验的参数
        for exp_idx in tqdm.trange(exp_repeat_num, desc="exp_idx", leave=False):
            # 设置随机种子
            np.random.seed(exp_repeat_seed + exp_idx)
            test_sign_idx = np.arange(test_sign_num)
            np.random.shuffle(test_sign_idx)
            used_sign_idx = test_sign_idx[:bp_sign_num]
            used_coeff_idx = np.arange(256, dtype=int)  # 构建约束矩阵
            # used_coeff_idx = np.arange(64, 128, dtype=int)  # 构建约束矩阵
            # used_coeff_idx = np.concatenate(
            #     (np.arange(16, 128, dtype=int), np.arange(144, 256, dtype=int))
            # )
            # BP_C = np.zeros(
            #     (
            #         np.sum(rej_num_array[test_sign_idx_list[used_sign_idx]])
            #         * len(used_coeff_idx),
            #         256,
            #         3,
            #     ),
            #     dtype=float,
            # )
            # BP_CS = np.zeros(
            #     (
            #         np.sum(rej_num_array[test_sign_idx_list[used_sign_idx]])
            #         * len(used_coeff_idx),
            #         2 * max_cs_abs + 1,
            #     ),
            #     dtype=float,
            # )
            # 考虑到需要筛选约束，因此约束矩阵为可变大小，这里先空着
            BP_C = []
            BP_CS = []

            used_trs_num = 0
            for sign_idx in used_sign_idx:
                for rej_idx in range(rej_num_array[test_sign_idx_list[sign_idx]]):
                    trs_idx = (
                        np.sum(rej_num_array[test_sign_idx_list[:sign_idx]]) + rej_idx
                    )
                    c_predict = np.argmax(test_c_prob[trs_idx], axis=1)
                    # print(
                    #     "c取-1和+1的数量为：",
                    #     np.sum(c_predict == 0) + np.sum(c_predict == 2),
                    # )

                    if (
                        np.abs(np.sum(c_predict == 0) + np.sum(c_predict == 2) - 39)
                        >= 3
                    ):  # 仅选择那些c取-1和+1的数量为39左右的签名，即tau
                        continue

                    # print(
                    #     "wrong num is %d"
                    #     % (np.sum(c_predict != c_test_label[trs_idx]),)
                    # )
                    # print(np.where(c_predict != c_test_label[trs_idx]))

                    # if (
                    #     rej_idx != rej_num_array[test_sign_idx_list[sign_idx]] - 1
                    # ):  # 只使用合法签名作为基准效果
                    #     continue

                    tmp_test_c_prob = test_c_prob[trs_idx].copy()
                    if (
                        rej_idx != rej_num_array[test_sign_idx_list[sign_idx]] - 1
                    ):  # 只有非法签名需要考虑这些分布，合法签名直接使用正确分布
                        for i in range(0, 256):
                            if c_predict[i] == 1:
                                tmp_test_c_prob[i, :] = [
                                    0,
                                    1,
                                    0,
                                ]
                        for i in range(0, 256):
                            if c_predict[i] == 0:
                                tmp_test_c_prob[i, :] = [1, 0, 0]
                            elif c_predict[i] == 2:
                                tmp_test_c_prob[i, :] = [0, 0, 1]
                        for i in range(16):
                            if c_predict[i] == 2:
                                tmp_test_c_prob[i, :] = [
                                    0,
                                    0.5,
                                    0.5,
                                ]
                        for i in range(16, 32):
                            if c_predict[i] == 2:
                                tmp_test_c_prob[i, :] = [
                                    0,
                                    0.4,
                                    0.6,
                                ]
                        for i in range(32, 64):
                            if c_predict[i] == 2:
                                tmp_test_c_prob[i, :] = [
                                    0,
                                    0.2,
                                    0.8,
                                ]
                    for coeff_idx in used_coeff_idx:
                        BP_CS.append(test_cs1_prob_cut[trs_idx, coeff_idx])
                        BP_C.append(np.zeros((256, 3), dtype=float))
                        # 构建c的分布作为约束
                        for s_idx in range(256):
                            if s_idx <= coeff_idx:
                                BP_C[-1][s_idx] = tmp_test_c_prob[coeff_idx - s_idx]
                            else:
                                BP_C[-1][s_idx] = np.flip(
                                    tmp_test_c_prob[(coeff_idx - s_idx) % 256]
                                )
                    used_trs_num += 1
            print("used trs num is %d" % (used_trs_num,))

            bp_args.append(
                (
                    exp_idx,
                    np.array(BP_C, dtype=float),
                    np.array(BP_CS, dtype=float),
                    bp_sign_num,
                    bp_iter_num,
                    exp_save_path,
                )
            )

    # 单进程写法
    # for args in bp_args:
    #     run_single_bp_experiment_more_simple(args)

    # 使用多进程并行执行所有实验
    with Pool(12) as pool:
        with tqdm.tqdm(
            total=len(bp_args),
            desc=f"总实验进度",
        ) as pbar:
            for _ in pool.imap_unordered(
                run_single_bp_experiment_more_simple, bp_args, chunksize=1
            ):
                pbar.update(1)

    print("cs1_valid BP实验完成！")
    print(f"Results saved in: {exp_save_path}")


def c_all_cs1_all_bp_vary_c_acc_experiment(
    exp_repeat_num: int = 10,
    exp_min_sign_num: int = 1,
    exp_max_sign_num: int = 20,
    bp_iter_num: int = 15,
):
    """
    使用全部的c,以及cs1进行bp实验,不包括y是因为y的信噪比较低,先看看纯cs1的效果。
    """

    # 路径与参数
    exp_save_path = "../exp_res/c_all_cs1_all_bp_vary_c_acc/"
    # 创建保存目录
    if not os.path.exists(exp_save_path):
        os.makedirs(exp_save_path)
    train_sign_num = 9000
    test_sign_num = 1000
    total_sign_num = 10000
    train_test_split_seed = 123456

    # 划分训练集和测试集
    print("划分训练集和测试集...")
    np.random.seed(train_test_split_seed)
    sign_idx_list = np.arange(total_sign_num, dtype=int)
    np.random.shuffle(sign_idx_list)
    train_sign_idx_list = sign_idx_list[:train_sign_num]
    test_sign_idx_list = sign_idx_list[train_sign_num : train_sign_num + test_sign_num]
    np.save(exp_save_path + "train_sign_idx", train_sign_idx_list)
    np.save(exp_save_path + "test_sign_idx", test_sign_idx_list)
    print(f"Training set size: {train_sign_num}, test set size: {test_sign_num}")

    # 读取拒绝次数，每签名最后一次拒绝采样为合法签名,其余为被拒绝的签名
    print("Reading rejection count information...")
    rej_num_list = []
    with open("seed and rej num.txt", "r") as f:
        for _ in range(total_sign_num):
            line = f.readline().strip().split(" ")
            rej_num_list.append(int(line[1]))
    rej_num_array = np.array(rej_num_list)

    c_data_path = "../data/c_all/"

    c_test_label = np.zeros((np.sum(rej_num_array[test_sign_idx_list]), 256), dtype=int)
    # 加载c测试数据
    for i in tqdm.trange(test_sign_num, desc="加载c测试数据"):
        sign_idx = test_sign_idx_list[i]
        for j in range(rej_num_array[sign_idx]):
            c_test_label[np.sum(rej_num_array[test_sign_idx_list[:i]]) + j] = np.load(
                c_data_path + "c%06drej%06d.npy" % (sign_idx, j)
            )
    c_test_label += 1

    # 该做BP了
    test_cs1_prob = np.zeros(
        (np.sum(rej_num_array[test_sign_idx_list]), 256, 128), dtype=float
    )
    for i in range(256):
        try:
            reserved_class = np.load(
                exp_save_path + "cs1_idx%03dReserveClass.npy" % (i,)
            )
            test_ta_prob = np.load(exp_save_path + "cs1_idx%03dProbTest.npy" % (i,))
            for j in range(len(reserved_class)):
                for k in range(test_cs1_prob.shape[0]):
                    test_cs1_prob[k, i, reserved_class[j]] = test_ta_prob[k, j]
        except FileNotFoundError:
            continue

    # 将test_cs1_prob置为正确分布测试测试后续代码的正确性
    # for i in range(256):
    #     for j in range(test_trs_num):
    #         test_cs1_prob[j, i, test_cs1[j, i]] = 1

    # 对cs1的分布进行切分，仅保留绝对值小于32的分布
    max_cs_abs = 32  # cs1的最大绝对值
    test_cs1_prob_cut = np.zeros(
        (np.sum(rej_num_array[test_sign_idx_list]), 256, 2 * max_cs_abs + 1),
        dtype=float,
    )
    for i in range(256):
        for j in range(test_cs1_prob.shape[0]):
            test_cs1_prob_cut[j, i, :max_cs_abs] = test_cs1_prob[j, i, -max_cs_abs:]
            test_cs1_prob_cut[j, i, max_cs_abs:] = test_cs1_prob[j, i, : max_cs_abs + 1]
    del test_cs1_prob

    # 并行执行BP实验，每个bp_sign_num用多个进程并行跑多次实验
    exp_repeat_seed = 234567  # Fixed seed
    bp_args = []
    # 提前将bp所需的CS和C矩阵在这里构造后，后面就只需要直接sumprouct就行了，减少多进程内存开销
    for c_acc in [0.96]:
        for bp_sign_num in tqdm.trange(
            exp_min_sign_num,
            exp_max_sign_num + 1,
            1,
            desc="cACC%.2f构建bp参数列表" % (c_acc,),
        ):
            # 为当前bp_sign_num准备所有实验的参数
            for exp_idx in tqdm.trange(exp_repeat_num, desc="exp_idx", leave=False):
                # 设置随机种子
                np.random.seed(exp_repeat_seed + exp_idx)
                test_sign_idx = np.arange(test_sign_num)
                np.random.shuffle(test_sign_idx)
                used_sign_idx = test_sign_idx[:bp_sign_num]
                used_coeff_idx = np.arange(256, dtype=int)  # 构建约束矩阵
                # 考虑到需要筛选约束，因此约束矩阵为可变大小，这里先空着
                BP_C = []
                BP_CS = []

                used_trs_num = 0
                for sign_idx in used_sign_idx:
                    for rej_idx in range(rej_num_array[test_sign_idx_list[sign_idx]]):
                        trs_idx = (
                            np.sum(rej_num_array[test_sign_idx_list[:sign_idx]])
                            + rej_idx
                        )

                        c_predict = c_test_label[trs_idx].copy()

                        if (
                            rej_idx != rej_num_array[test_sign_idx_list[sign_idx]] - 1
                        ):  # 只有非法签名需要考虑这些分布，合法签名直接使用正确分布
                            for i in range(256):
                                if np.random.rand() > c_acc:
                                    if c_predict[i] == 0:
                                        c_predict[i] = np.random.choice([1, 2])
                                    elif c_predict[i] == 1:
                                        c_predict[i] = np.random.choice([0, 2])
                                    elif c_predict[i] == 2:
                                        c_predict[i] = np.random.choice([0, 1])

                        """不进行筛选,直接研究准确率变化造成的影响"""
                        # if (
                        #     np.abs(np.sum(c_predict == 0) + np.sum(c_predict == 2) - 39)
                        #     >= 3
                        # ):  # 仅选择那些c取-1和+1的数量为39左右的签名，即tau
                        #     continue

                        tmp_test_c_prob = np.zeros((256, 3), dtype=float)
                        for i in range(0, 256):
                            if c_predict[i] == 1:
                                tmp_test_c_prob[i, :] = [
                                    0,
                                    1,
                                    0,
                                ]
                            elif c_predict[i] == 0:
                                tmp_test_c_prob[i, :] = [1, 0, 0]
                            elif c_predict[i] == 2:
                                tmp_test_c_prob[i, :] = [
                                    0,
                                    0,
                                    1,
                                ]

                        for coeff_idx in used_coeff_idx:
                            BP_CS.append(test_cs1_prob_cut[trs_idx, coeff_idx])
                            BP_C.append(np.zeros((256, 3), dtype=float))
                            # 构建c的分布作为约束
                            for s_idx in range(256):
                                if s_idx <= coeff_idx:
                                    BP_C[-1][s_idx] = tmp_test_c_prob[coeff_idx - s_idx]
                                else:
                                    BP_C[-1][s_idx] = np.flip(
                                        tmp_test_c_prob[(coeff_idx - s_idx) % 256]
                                    )
                        used_trs_num += 1
                print("used trs num is %d" % (used_trs_num,))

                bp_args.append(
                    (
                        exp_idx,
                        np.array(BP_C, dtype=float),
                        np.array(BP_CS, dtype=float),
                        bp_sign_num,
                        bp_iter_num,
                        exp_save_path,
                        c_acc,
                    )
                )

        # 使用多进程并行执行所有实验
        with Pool(14) as pool:
            with tqdm.tqdm(
                total=len(bp_args),
                desc=f"总实验进度",
            ) as pbar:
                for _ in pool.imap_unordered(
                    run_single_bp_experiment_more_simple_vary_c_acc,
                    bp_args,
                    chunksize=2,
                ):
                    pbar.update(1)
        bp_args.clear()

    print("BP实验完成！")
    print(f"Results saved in: {exp_save_path}")


def c_rej_cs1_rej_bp_vary_c_acc_experiment(
    exp_repeat_num: int = 10,
    exp_min_sign_num: int = 1,
    exp_max_sign_num: int = 20,
    bp_iter_num: int = 15,
):
    """
    使用全部的c,以及cs1进行bp实验,不包括y是因为y的信噪比较低,先看看纯cs1的效果。
    """

    # 路径与参数
    exp_save_path = "../exp_res/c_rej_cs1_rej_bp_vary_c_acc/"
    # 创建保存目录
    if not os.path.exists(exp_save_path):
        os.makedirs(exp_save_path)
    train_sign_num = 9000
    test_sign_num = 1000
    total_sign_num = 10000
    train_test_split_seed = 123456

    # 划分训练集和测试集
    print("划分训练集和测试集...")
    np.random.seed(train_test_split_seed)
    sign_idx_list = np.arange(total_sign_num, dtype=int)
    np.random.shuffle(sign_idx_list)
    train_sign_idx_list = sign_idx_list[:train_sign_num]
    test_sign_idx_list = sign_idx_list[train_sign_num : train_sign_num + test_sign_num]
    np.save(exp_save_path + "train_sign_idx", train_sign_idx_list)
    np.save(exp_save_path + "test_sign_idx", test_sign_idx_list)
    print(f"Training set size: {train_sign_num}, test set size: {test_sign_num}")

    # 读取拒绝次数，每签名最后一次拒绝采样为合法签名,其余为被拒绝的签名
    print("Reading rejection count information...")
    rej_num_list = []
    with open("seed and rej num.txt", "r") as f:
        for _ in range(total_sign_num):
            line = f.readline().strip().split(" ")
            rej_num_list.append(int(line[1]))
    rej_num_array = np.array(rej_num_list)

    c_data_path = "../data/c_all/"

    c_test_label = np.zeros((np.sum(rej_num_array[test_sign_idx_list]), 256), dtype=int)
    # 加载c测试数据
    for i in tqdm.trange(test_sign_num, desc="加载c测试数据"):
        sign_idx = test_sign_idx_list[i]
        for j in range(rej_num_array[sign_idx]):
            c_test_label[np.sum(rej_num_array[test_sign_idx_list[:i]]) + j] = np.load(
                c_data_path + "c%06drej%06d.npy" % (sign_idx, j)
            )
    c_test_label += 1

    # 该做BP了
    test_cs1_prob = np.zeros(
        (np.sum(rej_num_array[test_sign_idx_list]), 256, 128), dtype=float
    )
    for i in range(256):
        try:
            reserved_class = np.load(
                exp_save_path + "cs1_idx%03dReserveClass.npy" % (i,)
            )
            test_ta_prob = np.load(exp_save_path + "cs1_idx%03dProbTest.npy" % (i,))
            for j in range(len(reserved_class)):
                for k in range(test_cs1_prob.shape[0]):
                    test_cs1_prob[k, i, reserved_class[j]] = test_ta_prob[k, j]
        except FileNotFoundError:
            continue

    # 将test_cs1_prob置为正确分布测试测试后续代码的正确性
    # for i in range(256):
    #     for j in range(test_trs_num):
    #         test_cs1_prob[j, i, test_cs1[j, i]] = 1

    # 对cs1的分布进行切分，仅保留绝对值小于32的分布
    max_cs_abs = 32  # cs1的最大绝对值
    test_cs1_prob_cut = np.zeros(
        (np.sum(rej_num_array[test_sign_idx_list]), 256, 2 * max_cs_abs + 1),
        dtype=float,
    )
    for i in range(256):
        for j in range(test_cs1_prob.shape[0]):
            test_cs1_prob_cut[j, i, :max_cs_abs] = test_cs1_prob[j, i, -max_cs_abs:]
            test_cs1_prob_cut[j, i, max_cs_abs:] = test_cs1_prob[j, i, : max_cs_abs + 1]
    del test_cs1_prob

    # 并行执行BP实验，每个bp_sign_num用多个进程并行跑多次实验
    exp_repeat_seed = 234567  # Fixed seed
    bp_args = []
    # 提前将bp所需的CS和C矩阵在这里构造后，后面就只需要直接sumprouct就行了，减少多进程内存开销
    for c_acc in [0.97, 0.98, 0.99, 1.00]:
        for bp_sign_num in tqdm.trange(
            exp_min_sign_num,
            exp_max_sign_num + 1,
            1,
            desc="cACC%.2f构建bp参数列表" % (c_acc,),
        ):
            # 为当前bp_sign_num准备所有实验的参数
            for exp_idx in tqdm.trange(exp_repeat_num, desc="exp_idx", leave=False):
                # 设置随机种子
                np.random.seed(exp_repeat_seed + exp_idx)
                test_sign_idx = np.arange(test_sign_num)
                np.random.shuffle(test_sign_idx)
                used_sign_idx = test_sign_idx[:bp_sign_num]
                used_coeff_idx = np.arange(256, dtype=int)  # 构建约束矩阵
                # 考虑到需要筛选约束，因此约束矩阵为可变大小，这里先空着
                BP_C = []
                BP_CS = []

                used_trs_num = 0
                for sign_idx in used_sign_idx:
                    for rej_idx in range(rej_num_array[test_sign_idx_list[sign_idx]]):

                        if (
                            rej_idx == rej_num_array[test_sign_idx_list[sign_idx]] - 1
                        ):  # 跳过合法签名,仅使用非法签名
                            continue

                        trs_idx = (
                            np.sum(rej_num_array[test_sign_idx_list[:sign_idx]])
                            + rej_idx
                        )

                        c_predict = c_test_label[trs_idx].copy()

                        if (
                            rej_idx != rej_num_array[test_sign_idx_list[sign_idx]] - 1
                        ):  # 只有非法签名需要考虑这些分布，合法签名直接使用正确分布
                            for i in range(256):
                                if np.random.rand() > c_acc:
                                    if c_predict[i] == 0:
                                        c_predict[i] = np.random.choice([1, 2])
                                    elif c_predict[i] == 1:
                                        c_predict[i] = np.random.choice([0, 2])
                                    elif c_predict[i] == 2:
                                        c_predict[i] = np.random.choice([0, 1])

                        """不进行筛选,直接研究准确率变化造成的影响"""
                        # if (
                        #     np.abs(np.sum(c_predict == 0) + np.sum(c_predict == 2) - 39)
                        #     >= 3
                        # ):  # 仅选择那些c取-1和+1的数量为39左右的签名，即tau
                        #     continue

                        tmp_test_c_prob = np.zeros((256, 3), dtype=float)
                        for i in range(0, 256):
                            if c_predict[i] == 1:
                                tmp_test_c_prob[i, :] = [
                                    0,
                                    1,
                                    0,
                                ]
                            elif c_predict[i] == 0:
                                tmp_test_c_prob[i, :] = [1, 0, 0]
                            elif c_predict[i] == 2:
                                tmp_test_c_prob[i, :] = [
                                    0,
                                    0,
                                    1,
                                ]

                        for coeff_idx in used_coeff_idx:
                            BP_CS.append(test_cs1_prob_cut[trs_idx, coeff_idx])
                            BP_C.append(np.zeros((256, 3), dtype=float))
                            # 构建c的分布作为约束
                            for s_idx in range(256):
                                if s_idx <= coeff_idx:
                                    BP_C[-1][s_idx] = tmp_test_c_prob[coeff_idx - s_idx]
                                else:
                                    BP_C[-1][s_idx] = np.flip(
                                        tmp_test_c_prob[(coeff_idx - s_idx) % 256]
                                    )
                        used_trs_num += 1
                print("used trs num is %d" % (used_trs_num,))

                bp_args.append(
                    (
                        exp_idx,
                        np.array(BP_C, dtype=float),
                        np.array(BP_CS, dtype=float),
                        bp_sign_num,
                        bp_iter_num,
                        exp_save_path,
                        c_acc,
                    )
                )

    # 使用多进程并行执行所有实验
    with Pool(30) as pool:
        with tqdm.tqdm(
            total=len(bp_args),
            desc=f"总实验进度",
        ) as pbar:
            for _ in pool.imap_unordered(
                run_single_bp_experiment_more_simple_vary_c_acc,
                bp_args,
                chunksize=1,
            ):
                pbar.update(1)
    bp_args.clear()

    print("BP实验完成！")
    print(f"Results saved in: {exp_save_path}")


def view_bp_res_vary_c_acc(
    exp_save_path="../exp_res/c_all_cs1_all_bp_vary_c_acc/",
    exp_min_sign_num=1,
    exp_max_sign_num=30,
    exp_repeat_num=10,
    c_acc_list=[0.99, 0.98, 0.97, 0.96, 0.95],
):
    """
    查看c_all_cs1_all_bp_vary_c_acc实验的BP结果
    绘制不同c准确率下BP成功率和密钥系数恢复率随签名次数变化的图

    参数:
    - exp_save_path: 实验结果保存路径
    - exp_min_sign_num: 最小签名数
    - exp_max_sign_num: 最大签名数
    - exp_repeat_num: 实验重复次数
    - c_acc_list: c准确率列表
    """
    # 加载真实密钥（如果存在）
    try:
        s1 = np.load("s1.npy")
        has_true_key = True
        print("找到真实密钥文件，将计算准确率")
    except FileNotFoundError:
        has_true_key = False
        print("未找到真实密钥文件，将只显示BP迭代过程")

    # 为每个c准确率创建数据结构
    success_rates = {}
    coeff_recovery_ave_nums = {}

    for c_acc in c_acc_list:
        success_rates[c_acc] = np.zeros(
            (exp_max_sign_num - exp_min_sign_num + 1), dtype=float
        )
        coeff_recovery_ave_nums[c_acc] = np.zeros(
            (exp_max_sign_num - exp_min_sign_num + 1), dtype=float
        )

    # 处理每个c准确率
    for c_acc in tqdm.trange(len(c_acc_list), desc="处理c准确率"):
        c_acc_value = c_acc_list[c_acc]

        for bp_sign_num in tqdm.trange(
            exp_min_sign_num,
            exp_max_sign_num + 1,
            desc=f"c准确率{c_acc_value}",
            leave=False,
        ):
            for exp_idx in tqdm.trange(
                exp_repeat_num, leave=False, desc=f"签名数{bp_sign_num}"
            ):
                try:
                    bp_res = np.load(
                        exp_save_path
                        + "cAcc%.2fexp%02dSignNum%02dBpRes.npy"
                        % (c_acc_value, exp_idx, bp_sign_num)
                    )

                    max_right_coeff_num = 0

                    for bp_iter in range(bp_res.shape[0]):
                        sk_hat = np.argmax(bp_res[bp_iter], axis=1) - 2
                        right_coeff_num = np.sum(sk_hat == s1[0])
                        max_right_coeff_num = max(max_right_coeff_num, right_coeff_num)

                    if max_right_coeff_num == 256:
                        success_rates[c_acc_value][bp_sign_num - exp_min_sign_num] += 1

                    coeff_recovery_ave_nums[c_acc_value][
                        bp_sign_num - exp_min_sign_num
                    ] += max_right_coeff_num

                except FileNotFoundError:
                    print(
                        f"警告: 未找到文件 cAcc{c_acc_value:.2f}exp{exp_idx:02d}SignNum{bp_sign_num:02d}BpRes.npy"
                    )
                    continue

            # 计算平均值
            success_rates[c_acc_value][bp_sign_num - exp_min_sign_num] /= exp_repeat_num
            coeff_recovery_ave_nums[c_acc_value][
                bp_sign_num - exp_min_sign_num
            ] /= exp_repeat_num

    # 绘制结果
    fig = plt.figure(figsize=(12, 5))

    # 子图1: 成功率
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.grid(linestyle="--", alpha=0.7)

    colors = ["blue", "green", "red", "orange", "purple"]
    markers = ["o", "s", "^", "v", "D"]

    for i, c_acc in enumerate(c_acc_list):
        ax1.plot(
            np.arange(exp_min_sign_num, exp_max_sign_num + 1, dtype=int),
            success_rates[c_acc],
            linewidth=1.5,
            alpha=0.9,
            marker=markers[i],
            markersize=4,
            color=colors[i],
            label=f"c_acc={c_acc}",
        )

    ax1.set_xlabel("Number of Signatures")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("BP Success Rate vs Number of Signatures")
    ax1.set_xticks(np.arange(exp_min_sign_num, exp_max_sign_num + 1, 2, dtype=int))
    ax1.set_ylim(0, 1.1)
    ax1.legend()

    # 子图2: 平均恢复系数数量
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.grid(linestyle="--", alpha=0.7)

    for i, c_acc in enumerate(c_acc_list):
        ax2.plot(
            np.arange(exp_min_sign_num, exp_max_sign_num + 1, dtype=int),
            coeff_recovery_ave_nums[c_acc],
            linewidth=1.5,
            alpha=0.9,
            marker=markers[i],
            markersize=4,
            color=colors[i],
            label=f"c_acc={c_acc}",
        )

    ax2.set_xlabel("Number of Signatures")
    ax2.set_ylabel("Average Recovered Coefficients")
    ax2.set_title("Average Recovered Coefficients vs Number of Signatures")
    ax2.set_xticks(np.arange(exp_min_sign_num, exp_max_sign_num + 1, 2, dtype=int))
    if has_true_key:
        ax2.set_yticks(np.arange(0, 257, 32, dtype=int))
        ax2.set_ylim(0, 256)
    else:
        ax2.set_ylim(0, 256)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        exp_save_path + "bp_results_vary_c_acc.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # 打印统计信息
    print("\n=== BP Experiment Results (Vary c Accuracy) ===")
    print(f"Experiment range: {exp_min_sign_num}-{exp_max_sign_num} signatures")
    print(f"Repeat times: {exp_repeat_num}")
    print(f"c accuracy values: {c_acc_list}")
    print(f"Results saved to: {exp_save_path}")

    if has_true_key:
        for c_acc in c_acc_list:
            print(f"\n--- c_acc = {c_acc} ---")
            print(
                f"Highest success rate: {np.max(success_rates[c_acc]):.3f} (at {np.argmax(success_rates[c_acc]) + exp_min_sign_num} signatures)"
            )
            print(
                f"Highest average recovered coefficients: {np.max(coeff_recovery_ave_nums[c_acc]):.1f} (at {np.argmax(coeff_recovery_ave_nums[c_acc]) + exp_min_sign_num} signatures)"
            )

            # 找到成功率超过50%的最小签名数
            success_50_idx = np.where(success_rates[c_acc] >= 0.5)[0]
            if len(success_50_idx) > 0:
                print(
                    f"Minimum signatures for 50% success rate: {success_50_idx[0] + exp_min_sign_num}"
                )
            else:
                print("Did not reach 50% success rate")
    else:
        print(
            "Note: No true key found, results based on probability quality assessment"
        )


def gen_seed_and_rej_num(data_num):
    """
    随机生成签名种子发送给下位机，下位机统计拒绝采样次数，返回给上位机并保存，注意小端
    :param data_num:
    :return:
    """
    file_handle = open("seed and rej num.txt", "w")
    ser = serial.Serial("/dev/ttyUSB0", 256000, timeout=None)
    for i in tqdm.trange(data_num):
        seed_bytes = os.urandom(4)
        file_handle.write(seed_bytes.hex())
        ser.write(bytes.fromhex("01") + seed_bytes)
        rej_num = int.from_bytes(ser.read(4), byteorder="little")
        file_handle.write(" " + str(rej_num) + "\n")
        file_handle.flush()
    ser.close()
    file_handle.close()


def gen_valid_data():
    ser = serial.Serial(
        "/dev/ttyUSB0",
        256000,
        timeout=None,
    )

    seed_handle = open("seed and rej num.txt", "r")

    y_data_path = "../data/y_valid/"
    # z_data_path = "../data/z_valid/"
    if not os.path.exists(y_data_path):
        os.makedirs(y_data_path)
    # if not os.path.exists(z_data_path):
    #     os.makedirs(z_data_path)

    start_idx = 0
    for i in range(0, start_idx):  # 跳过前面的数据
        seed_handle.readline()

    for i in tqdm.trange(start_idx, 10000):
        input = seed_handle.readline().split(" ")
        seed = bytes.fromhex(input[0])
        rej_num = int(input[1])
        ser.write(bytes.fromhex("02"))
        ser.write(seed)
        ser.write(rej_num.to_bytes(4, byteorder="little"))
        ser.write(int(0).to_bytes(4, byteorder="little"))  # poly idx

        y = np.zeros((256,), dtype=int)
        for j in range(256):
            y[j] = int.from_bytes(ser.read(4), byteorder="little", signed=True)
        # z = np.zeros((256,), dtype=int)
        # for j in range(256):
        #     z[j] = int.from_bytes(ser.read(4), byteorder="little", signed=True)

        ser.read(5)  # 读取结束标志代表板子运行完成

        np.save(y_data_path + "y%06d" % (i,), y)
        # np.save(z_data_path + "z%06d" % (i,), z)


def gen_all_data():
    ser = serial.Serial("/dev/ttyUSB0", 256000, timeout=None)

    seed_handle = open("seed and rej num.txt", "r")

    c_data_path = "../data/c_all/"
    cs1_data_path = "../data/cs1_all/"
    if not os.path.exists(c_data_path):
        os.makedirs(c_data_path)
    if not os.path.exists(cs1_data_path):
        os.makedirs(cs1_data_path)

    start_idx = 0
    for i in range(0, start_idx):  # 跳过前面的数据
        seed_handle.readline()

    for i in tqdm.trange(start_idx, 10000):
        input = seed_handle.readline().split(" ")
        seed = bytes.fromhex(input[0])
        rej_num = int(input[1])
        for rej_idx in tqdm.trange(0, rej_num, desc="rej loop", leave=False):
            ser.write(bytes.fromhex("02"))
            ser.write(seed)
            ser.write((rej_idx + 1).to_bytes(4, byteorder="little"))
            ser.write(int(0).to_bytes(4, byteorder="little"))  # poly idx

            c = np.zeros((256,), dtype=int)
            for j in range(256):
                c[j] = int.from_bytes(ser.read(4), byteorder="little", signed=True)
            # cs1 = np.zeros((256,), dtype=int)
            # for j in range(256):
            #     cs1[j] = int.from_bytes(ser.read(4), byteorder="little", signed=True)
            ser.read(5)  # 读取结束标志代表板子运行完成

            np.save(c_data_path + "c%06drej%06d" % (i, rej_idx), c)
            # np.save(cs1_data_path + "cs%06drej%06d" % (i, rej_idx), cs1)


def view_cs1_valid_bp_res(
    exp_save_path="../exp_res/cs1_valid_bp/",
    exp_min_sign_num=1,
    exp_max_sign_num=30,
    exp_repeat_num=10,
):
    """
    查看cs1_valid_bp实验的BP结果
    绘制BP成功率和密钥系数恢复率随签名次数变化的图

    参数:
    - exp_save_path: 实验结果保存路径
    - exp_min_sign_num: 最小签名数
    - exp_max_sign_num: 最大签名数
    - exp_repeat_num: 实验重复次数
    """
    # 加载真实密钥（如果存在）
    try:
        s1 = np.load("s1.npy")
        has_true_key = True
        print("找到真实密钥文件，将计算准确率")
    except FileNotFoundError:
        has_true_key = False
        print("未找到真实密钥文件，将只显示BP迭代过程")

    success_rate = np.zeros((exp_max_sign_num - exp_min_sign_num + 1), dtype=float)
    coeff_recovery_ave_num = np.zeros(
        (exp_max_sign_num - exp_min_sign_num + 1), dtype=float
    )

    # 存储每次迭代的恢复系数数量（用于绘制迭代过程）
    iter_recovery_data = []

    for bp_sign_num in tqdm.trange(
        exp_min_sign_num,
        exp_max_sign_num + 1,
        desc="处理签名数量",
    ):
        sign_recovery_data = []

        for exp_idx in tqdm.trange(
            exp_repeat_num, leave=False, desc=f"签名数{bp_sign_num}"
        ):
            try:
                bp_res = np.load(
                    exp_save_path
                    + "exp%02dSignNum%02dBpRes.npy" % (exp_idx, bp_sign_num)
                )

                max_right_coeff_num = 0
                iter_recovery = []

                for bp_iter in range(bp_res.shape[0]):
                    if has_true_key:
                        sk_hat = np.argmax(bp_res[bp_iter], axis=1) - 2
                        right_coeff_num = np.sum(sk_hat == s1[0])
                        max_right_coeff_num = max(max_right_coeff_num, right_coeff_num)
                        iter_recovery.append(right_coeff_num)
                    else:
                        # 如果没有真实密钥，计算最大概率的分布
                        max_probs = np.max(bp_res[bp_iter], axis=1)
                        # 使用平均最大概率作为"恢复质量"的指标
                        quality_score = np.mean(max_probs)
                        max_right_coeff_num = max(
                            max_right_coeff_num, quality_score * 256
                        )
                        iter_recovery.append(quality_score * 256)

                sign_recovery_data.append(iter_recovery)

                if has_true_key:
                    if max_right_coeff_num == 256:
                        success_rate[bp_sign_num - exp_min_sign_num] += 1
                else:
                    # 使用质量阈值判断"成功"
                    if max_right_coeff_num > 200:  # 可调整的阈值
                        success_rate[bp_sign_num - exp_min_sign_num] += 1

                coeff_recovery_ave_num[
                    bp_sign_num - exp_min_sign_num
                ] += max_right_coeff_num

            except FileNotFoundError:
                print(
                    f"警告: 未找到文件 exp%02dSignNum%02dUsedCoeffNum256BpRes.npy"
                    % (exp_idx, bp_sign_num)
                )
                continue

        if sign_recovery_data:
            iter_recovery_data.append(
                {"sign_num": bp_sign_num, "data": np.array(sign_recovery_data)}
            )

        success_rate[bp_sign_num - exp_min_sign_num] /= exp_repeat_num
        coeff_recovery_ave_num[bp_sign_num - exp_min_sign_num] /= exp_repeat_num

    # 绘制结果
    fig = plt.figure(figsize=(15, 5))

    # 子图1: 成功率
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.grid(linestyle="--", alpha=0.7)
    ax1.plot(
        np.arange(exp_min_sign_num, exp_max_sign_num + 1, dtype=int),
        success_rate,
        linewidth=1.5,
        alpha=0.9,
        marker="o",
        markersize=4,
        color="blue",
    )
    ax1.set_xlabel("Number of Signatures")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("BP Success Rate vs Number of Signatures")
    ax1.set_xticks(np.arange(exp_min_sign_num, exp_max_sign_num + 1, 2, dtype=int))
    ax1.set_ylim(0, 1.1)

    # 子图2: 平均恢复系数数量
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.grid(linestyle="--", alpha=0.7)
    ax2.plot(
        np.arange(exp_min_sign_num, exp_max_sign_num + 1, dtype=int),
        coeff_recovery_ave_num,
        linewidth=1.5,
        alpha=0.9,
        marker="s",
        markersize=4,
        color="red",
    )
    ax2.set_xlabel("Number of Signatures")
    ax2.set_ylabel("Average Recovered Coefficients")
    ax2.set_title("Average Recovered Coefficients vs Number of Signatures")
    ax2.set_xticks(np.arange(exp_min_sign_num, exp_max_sign_num + 1, 2, dtype=int))
    if has_true_key:
        ax2.set_yticks(np.arange(0, 257, 32, dtype=int))
        ax2.set_ylim(0, 256)
    else:
        ax2.set_ylim(0, 256)

    # 子图3: BP迭代过程（选择几个代表性的签名数量）
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.grid(linestyle="--", alpha=0.7)

    # 选择几个代表性的签名数量进行展示
    selected_sign_nums = [
        exp_min_sign_num,
        (exp_min_sign_num + exp_max_sign_num) // 2,
        exp_max_sign_num,
    ]

    colors = ["blue", "green", "red"]
    for i, sign_num in enumerate(selected_sign_nums):
        if sign_num in [data["sign_num"] for data in iter_recovery_data]:
            data_idx = next(
                j
                for j, data in enumerate(iter_recovery_data)
                if data["sign_num"] == sign_num
            )
            mean_recovery = np.mean(iter_recovery_data[data_idx]["data"], axis=0)
            std_recovery = np.std(iter_recovery_data[data_idx]["data"], axis=0)

            iterations = np.arange(len(mean_recovery))
            ax3.plot(
                iterations,
                mean_recovery,
                linewidth=1.5,
                alpha=0.8,
                marker="o",
                markersize=3,
                color=colors[i],
                label=f"{sign_num} signatures",
            )
            ax3.fill_between(
                iterations,
                mean_recovery - std_recovery,
                mean_recovery + std_recovery,
                alpha=0.2,
                color=colors[i],
            )

    ax3.set_xlabel("BP Iterations")
    ax3.set_ylabel("Recovered Coefficients")
    ax3.set_title("BP Iteration Process")
    ax3.legend()
    if has_true_key:
        ax3.set_ylim(0, 256)

    plt.tight_layout()
    plt.savefig(exp_save_path + "bp_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印统计信息
    print("\n=== BP Experiment Results ===")
    print(f"Experiment range: {exp_min_sign_num}-{exp_max_sign_num} signatures")
    print(f"Repeat times: {exp_repeat_num}")
    print(f"Results saved to: {exp_save_path}")

    if has_true_key:
        print(
            f"Highest success rate: {np.max(success_rate):.3f} (at {np.argmax(success_rate) + exp_min_sign_num} signatures)"
        )
        print(
            f"Highest average recovered coefficients: {np.max(coeff_recovery_ave_num):.1f} (at {np.argmax(coeff_recovery_ave_num) + exp_min_sign_num} signatures)"
        )

        # 找到成功率超过50%的最小签名数
        success_50_idx = np.where(success_rate >= 0.5)[0]
        if len(success_50_idx) > 0:
            print(
                f"Minimum signatures for 50% success rate: {success_50_idx[0] + exp_min_sign_num}"
            )
        else:
            print("Did not reach 50% success rate")
    else:
        print(
            "Note: No true key found, results based on probability quality assessment"
        )


def view_y_valid_bp_res(
    exp_save_path="../exp_res/y_valid_bp/",
    exp_min_sign_num=1,
    exp_max_sign_num=30,
    exp_repeat_num=10,
):
    """
    查看y_valid_bp实验的BP结果
    绘制BP成功率和密钥系数恢复率随签名次数变化的图

    参数:
    - exp_save_path: 实验结果保存路径
    - exp_min_sign_num: 最小签名数
    - exp_max_sign_num: 最大签名数
    - exp_repeat_num: 实验重复次数
    """
    # 加载真实密钥（如果存在）
    try:
        s1 = np.load("s1.npy")
        has_true_key = True
        print("找到真实密钥文件，将计算准确率")
    except FileNotFoundError:
        has_true_key = False
        print("未找到真实密钥文件，将只显示BP迭代过程")

    success_rate = np.zeros((exp_max_sign_num - exp_min_sign_num + 1), dtype=float)
    coeff_recovery_ave_num = np.zeros(
        (exp_max_sign_num - exp_min_sign_num + 1), dtype=float
    )

    # 存储每次迭代的恢复系数数量（用于绘制迭代过程）
    iter_recovery_data = []

    for bp_sign_num in tqdm.trange(
        exp_min_sign_num,
        exp_max_sign_num + 1,
        desc="处理签名数量",
    ):
        sign_recovery_data = []

        for exp_idx in tqdm.trange(
            exp_repeat_num, leave=False, desc=f"签名数{bp_sign_num}"
        ):
            try:
                bp_res = np.load(
                    exp_save_path
                    + "exp%02dSignNum%02dBpRes.npy" % (exp_idx, bp_sign_num)
                )

                max_right_coeff_num = 0
                iter_recovery = []

                for bp_iter in range(bp_res.shape[0]):
                    if has_true_key:
                        sk_hat = np.argmax(bp_res[bp_iter], axis=1) - 2
                        right_coeff_num = np.sum(sk_hat == s1[0])
                        max_right_coeff_num = max(max_right_coeff_num, right_coeff_num)
                        iter_recovery.append(right_coeff_num)
                    else:
                        # 如果没有真实密钥，计算最大概率的分布
                        max_probs = np.max(bp_res[bp_iter], axis=1)
                        # 使用平均最大概率作为"恢复质量"的指标
                        quality_score = np.mean(max_probs)
                        max_right_coeff_num = max(
                            max_right_coeff_num, quality_score * 256
                        )
                        iter_recovery.append(quality_score * 256)

                sign_recovery_data.append(iter_recovery)

                if has_true_key:
                    if max_right_coeff_num == 256:
                        success_rate[bp_sign_num - exp_min_sign_num] += 1
                else:
                    # 使用质量阈值判断"成功"
                    if max_right_coeff_num > 200:  # 可调整的阈值
                        success_rate[bp_sign_num - exp_min_sign_num] += 1

                coeff_recovery_ave_num[
                    bp_sign_num - exp_min_sign_num
                ] += max_right_coeff_num

            except FileNotFoundError:
                print(
                    f"警告: 未找到文件 exp%02dSignNum%02dBpRes.npy"
                    % (exp_idx, bp_sign_num)
                )
                continue

        if sign_recovery_data:
            iter_recovery_data.append(
                {"sign_num": bp_sign_num, "data": np.array(sign_recovery_data)}
            )

        success_rate[bp_sign_num - exp_min_sign_num] /= exp_repeat_num
        coeff_recovery_ave_num[bp_sign_num - exp_min_sign_num] /= exp_repeat_num

    # 绘制结果
    fig = plt.figure(figsize=(15, 5))

    # 子图1: 成功率
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.grid(linestyle="--", alpha=0.7)
    ax1.plot(
        np.arange(exp_min_sign_num, exp_max_sign_num + 1, dtype=int),
        success_rate,
        linewidth=1.5,
        alpha=0.9,
        marker="o",
        markersize=4,
        color="blue",
    )
    ax1.set_xlabel("Number of Signatures")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("BP Success Rate vs Number of Signatures")
    ax1.set_xticks(np.arange(exp_min_sign_num, exp_max_sign_num + 1, 2, dtype=int))
    ax1.set_ylim(0, 1.1)

    # 子图2: 平均恢复系数数量
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.grid(linestyle="--", alpha=0.7)
    ax2.plot(
        np.arange(exp_min_sign_num, exp_max_sign_num + 1, dtype=int),
        coeff_recovery_ave_num,
        linewidth=1.5,
        alpha=0.9,
        marker="s",
        markersize=4,
        color="red",
    )
    ax2.set_xlabel("Number of Signatures")
    ax2.set_ylabel("Average Recovered Coefficients")
    ax2.set_title("Average Recovered Coefficients vs Number of Signatures")
    ax2.set_xticks(np.arange(exp_min_sign_num, exp_max_sign_num + 1, 2, dtype=int))
    if has_true_key:
        ax2.set_yticks(np.arange(0, 257, 32, dtype=int))
        ax2.set_ylim(0, 256)
    else:
        ax2.set_ylim(0, 256)

    # 子图3: BP迭代过程（选择几个代表性的签名数量）
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.grid(linestyle="--", alpha=0.7)

    # 选择几个代表性的签名数量进行展示
    selected_sign_nums = [
        exp_min_sign_num,
        (exp_min_sign_num + exp_max_sign_num) // 2,
        exp_max_sign_num,
    ]

    colors = ["blue", "green", "red"]
    for i, sign_num in enumerate(selected_sign_nums):
        if sign_num in [data["sign_num"] for data in iter_recovery_data]:
            data_idx = next(
                j
                for j, data in enumerate(iter_recovery_data)
                if data["sign_num"] == sign_num
            )
            mean_recovery = np.mean(iter_recovery_data[data_idx]["data"], axis=0)
            std_recovery = np.std(iter_recovery_data[data_idx]["data"], axis=0)

            iterations = np.arange(len(mean_recovery))
            ax3.plot(
                iterations,
                mean_recovery,
                linewidth=1.5,
                alpha=0.8,
                marker="o",
                markersize=3,
                color=colors[i],
                label=f"{sign_num} signatures",
            )
            ax3.fill_between(
                iterations,
                mean_recovery - std_recovery,
                mean_recovery + std_recovery,
                alpha=0.2,
                color=colors[i],
            )

    ax3.set_xlabel("BP Iterations")
    ax3.set_ylabel("Recovered Coefficients")
    ax3.set_title("BP Iteration Process")
    ax3.legend()
    if has_true_key:
        ax3.set_ylim(0, 256)

    plt.tight_layout()
    plt.savefig(exp_save_path + "bp_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印统计信息
    print("\n=== BP Experiment Results ===")
    print(f"Experiment range: {exp_min_sign_num}-{exp_max_sign_num} signatures")
    print(f"Repeat times: {exp_repeat_num}")
    print(f"Results saved to: {exp_save_path}")

    if has_true_key:
        print(
            f"Highest success rate: {np.max(success_rate):.3f} (at {np.argmax(success_rate) + exp_min_sign_num} signatures)"
        )
        print(
            f"Highest average recovered coefficients: {np.max(coeff_recovery_ave_num):.1f} (at {np.argmax(coeff_recovery_ave_num) + exp_min_sign_num} signatures)"
        )

        # 找到成功率超过50%的最小签名数
        success_50_idx = np.where(success_rate >= 0.5)[0]
        if len(success_50_idx) > 0:
            print(
                f"Minimum signatures for 50% success rate: {success_50_idx[0] + exp_min_sign_num}"
            )
        else:
            print("Did not reach 50% success rate")
    else:
        print(
            "Note: No true key found, results based on probability quality assessment"
        )


def get_s1():
    # COM = find_my_device()
    # if COM is None:
    #     print("Can't find the serial port")
    #     exit(1)
    ser = serial.Serial("/dev/ttyUSB0", 256000, timeout=None)
    s1 = np.zeros((4, 256), dtype=int)
    ser.write(bytes.fromhex("03"))
    for i in range(4):
        for j in range(256):
            s1[i, j] = int.from_bytes(ser.read(4), byteorder="little", signed=True)
    print("s1", s1)
    np.save("s1", s1)


def y_trace_capture(save_path, start_idx=0, end_idx=10000):
    """
    采波函数，支持断点续采
    :param save_path: 波形保存的地址，如果地址不存在自动创建
    :param start_idx: 保存波形的起始索引，从0开始
    :param end_idx: 保存波形的结束索引
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    status = {}
    """标识示波器设备的句柄号"""
    chandle = ctypes.c_int16()

    """打开示波器，返回句柄号"""
    status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(chandle), None)

    """一些检测是否成功打开的判断"""
    try:
        assert_pico_ok(status["openunit"])
    except:

        # powerstate becomes the status number of openunit
        powerstate = status["openunit"]

        # If powerstate is the same as 282 then it will run this if statement
        if powerstate == 282:
            # Changes the power input to "PICO_POWER_SUPPLY_NOT_CONNECTED"
            status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 282)
        # If the powerstate is the same as 286 then it will run this if statement
        elif powerstate == 286:
            # Changes the power input to "PICO_USB3_0_DEVICE_NON_USB3_0_PORT"
            status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 286)
        else:
            raise

        assert_pico_ok(status["ChangePowerSource"])

    """PICO3203D实际上只有20mV,50mV,100mV,200mV,500mV,1V,2V,5V,10V和20V
    这段注释的意思是，range这个参数是枚举型，从10mV开始，但是不是每个参数都能用，因为pico的型号关系，所以这个参数0不能用，
    例如range=0意味着量程是5V，但是实际上0没法用，因为PICO3203D不支持
    A，触发通道常用Range为8，即5V
    B，能量通道常用Range为1-4，即20mV到200mV"""

    """实测8是5V"""
    chARange = 8

    """设置通道参数"""
    handle = chandle
    channel = PS3000A_CHANNEL_A = 0
    enabled = 1
    coupling_type = PS3000A_DC = 1
    analogue_offset = 0
    status["setChA"] = ps.ps3000aSetChannel(
        handle, channel, enabled, coupling_type, chARange, analogue_offset
    )
    assert_pico_ok(status["setChA"])

    chBRange = 4
    """开启通道B"""
    handle = chandle
    channel = PS3000A_CHANNEL_B = 1
    enabled = 1
    coupling_type = PS3000A_DC = 1
    analogue_offset = 0
    status["setChA"] = ps.ps3000aSetChannel(
        handle, channel, enabled, coupling_type, chBRange, analogue_offset
    )
    assert_pico_ok(status["setChA"])

    # chBRange = 2
    # """关闭通道C"""
    # handle = chandle
    # channel = PS3000A_CHANNEL_C = 2
    # enabled = 0
    # coupling_type = PS3000A_DC = 1
    # analogue_offset = 0
    # status["setChA"] = ps.ps3000aSetChannel(handle, channel, enabled, coupling_type, chBRange, analogue_offset)
    # assert_pico_ok(status["setChA"])

    # chBRange = 2
    # """关闭通道D"""
    # handle = chandle
    # channel = PS3000A_CHANNEL_D = 3
    # enabled = 0
    # coupling_type = PS3000A_DC = 1
    # analogue_offset = 0
    # status["setChA"] = ps.ps3000aSetChannel(handle, channel, enabled, coupling_type, chBRange, analogue_offset)
    # assert_pico_ok(status["setChA"])

    """设置触发参数"""
    handle = chandle
    enable = 1
    source = ps3000A_channel_A = 0
    threshold = 10000
    direction = 3
    delay = 0
    autoTrigger_ms = 1000  # 如果1s没有检测到触发，则自动触发
    status["trigger"] = ps.ps3000aSetSimpleTrigger(
        handle, enable, source, threshold, direction, delay, autoTrigger_ms
    )
    assert_pico_ok(status["trigger"])

    """配置触发前采波数，触发后采波数"""
    preTriggerSamples = 1000
    postTriggerSamples = 13000
    maxsamples = preTriggerSamples + postTriggerSamples

    """这个函数根据设置的参数查询采样率和最大采样的点数"""
    handle = chandle
    timebase = 1  # 采样率相关参数，见编程文档，其中0为1GHz，1为500Mhz
    no_sample = maxsamples
    timeIntervalns = ctypes.c_float()
    returnedMaxSamples = ctypes.c_int16()
    TimeIntervalNanoseconds = ctypes.byref(timeIntervalns)
    MaxSamples = ctypes.byref(returnedMaxSamples)
    Segement_index = 0

    status["GetTimebase"] = ps.ps3000aGetTimebase2(
        handle,
        timebase,
        no_sample,
        TimeIntervalNanoseconds,
        1,
        MaxSamples,
        Segement_index,
    )
    assert_pico_ok(status["GetTimebase"])

    # Creates converted types maxsamples
    cmaxSamples = ctypes.c_int32(maxsamples)

    ser = serial.Serial("/dev/ttyUSB0", 256000, timeout=None)

    file_handle = open("seed and rej num.txt", "r")
    for i in range(0, start_idx):  # 跳过前面的数据
        file_handle.readline()

    # if start_idx == 0:  # 说明是在重新采波
    #     file_list = os.listdir(save_path)
    #     for file in file_list:
    #         os.remove(os.path.join(save_path, file))

    for trace_idx in tqdm.trange(start_idx, end_idx, desc="trace capturing"):
        """配置开始采波"""
        status["runblock"] = ps.ps3000aRunBlock(
            chandle,
            preTriggerSamples,
            postTriggerSamples,
            timebase,
            1,
            None,
            0,
            None,
            None,
        )
        assert_pico_ok(status["runblock"])

        input = file_handle.readline().split(" ")
        seed = bytes.fromhex(input[0])
        rej_num = int(input[1])
        ser.write(bytes.fromhex("02"))
        ser.write(seed)
        ser.write(rej_num.to_bytes(4, byteorder="little"))
        ser.write(int(0).to_bytes(4, byteorder="little"))  # poly idx

        ser.read(5)  # 读取结束标志代表板子运行完成

        """创建buffer用于保存返回数据"""
        bufferAMax = np.empty(maxsamples, dtype=np.dtype("int16"))
        bufferBMax = np.empty(maxsamples, dtype=np.dtype("int16"))

        """把A通道的数据返回到bufferA里面"""
        handle = chandle
        source = ps3000A_channel_A = 0
        buffer_length = maxsamples
        segment_index = 0
        ratio_mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffer(
            handle,
            source,
            bufferAMax.ctypes.data,
            buffer_length,
            segment_index,
            ratio_mode,
        )
        assert_pico_ok(status["SetDataBuffers"])

        """把B通道的数据返回到bufferB里面"""
        handle = chandle
        source = ps3000A_channel_B = 1
        buffer_length = maxsamples
        segment_index = 0
        ratio_mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffer(
            handle,
            source,
            bufferBMax.ctypes.data,
            buffer_length,
            segment_index,
            ratio_mode,
        )
        assert_pico_ok(status["SetDataBuffers"])

        # Creates a overlow location for data
        overflow = ctypes.c_int16()
        # Creates converted types maxsamples
        cmaxSamples = ctypes.c_int32(maxsamples)

        """等待采波结束"""
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            status["isReady"] = ps.ps3000aIsReady(chandle, ctypes.byref(ready))

        """从示波器取回采波数据"""
        status["GetValuesBulk"] = ps.ps3000aGetValues(
            chandle, 0, ctypes.byref(cmaxSamples), 1, 0, 0, ctypes.byref(overflow)
        )
        assert_pico_ok(status["GetValuesBulk"])

        """获取波形水平偏移"""
        Times = ctypes.c_int64()
        TimeUnits = ctypes.c_char()
        status["GetTriggerTimeOffset"] = ps.ps3000aGetTriggerTimeOffset64(
            chandle, ctypes.byref(Times), ctypes.byref(TimeUnits), 0
        )

        time = (
            np.linspace(
                0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value
            )
            / 10**6
        )

        # plt.plot(time, bufferAMax, linewidth=0.9, alpha=0.9)
        # plt.plot(time, bufferBMax, linewidth=0.9, alpha=0.9)
        # plt.xlabel("Time (ms)")
        # plt.ylabel("Voltage (mV)")
        # plt.show()
        np.savez_compressed(
            save_path + "trace%06d.npz" % (trace_idx,),
            trace=bufferBMax,
            trigger=bufferAMax,
        )
        # if trace_idx == start_idx:
        #     np.savez_compressed(save_path + "time.npz", time=time)
    # plt.show()
    ser.close()
    file_handle.close()

    # Stops the scope
    # Handle = chandle
    status["stop"] = ps.ps3000aStop(chandle)
    assert_pico_ok(status["stop"])

    # Closes the unit
    # Handle = chandle
    status["close"] = ps.ps3000aCloseUnit(chandle)
    assert_pico_ok(status["close"])


def c_trace_capture(save_path, start_idx=0, end_idx=10000):
    """
    采波函数，支持断点续采
    :param save_path: 波形保存的地址，如果地址不存在自动创建
    :param start_idx: 保存波形的起始索引，从0开始
    :param end_idx: 保存波形的结束索引
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    status = {}
    """标识示波器设备的句柄号"""
    chandle = ctypes.c_int16()

    """打开示波器，返回句柄号"""
    status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(chandle), None)

    """一些检测是否成功打开的判断"""
    try:
        assert_pico_ok(status["openunit"])
    except:

        # powerstate becomes the status number of openunit
        powerstate = status["openunit"]

        # If powerstate is the same as 282 then it will run this if statement
        if powerstate == 282:
            # Changes the power input to "PICO_POWER_SUPPLY_NOT_CONNECTED"
            status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 282)
        # If the powerstate is the same as 286 then it will run this if statement
        elif powerstate == 286:
            # Changes the power input to "PICO_USB3_0_DEVICE_NON_USB3_0_PORT"
            status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 286)
        else:
            raise

        assert_pico_ok(status["ChangePowerSource"])

    """PICO3203D实际上只有20mV,50mV,100mV,200mV,500mV,1V,2V,5V,10V和20V
    这段注释的意思是，range这个参数是枚举型，从10mV开始，但是不是每个参数都能用，因为pico的型号关系，所以这个参数0不能用，
    例如range=0意味着量程是5V，但是实际上0没法用，因为PICO3203D不支持
    A，触发通道常用Range为8，即5V
    B，能量通道常用Range为1-4，即20mV到200mV"""

    """实测8是5V"""
    chARange = 8

    """设置通道参数"""
    handle = chandle
    channel = PS3000A_CHANNEL_A = 0
    enabled = 1
    coupling_type = PS3000A_DC = 1
    analogue_offset = 0
    status["setChA"] = ps.ps3000aSetChannel(
        handle, channel, enabled, coupling_type, chARange, analogue_offset
    )
    assert_pico_ok(status["setChA"])

    chBRange = 5
    """开启通道B"""
    handle = chandle
    channel = PS3000A_CHANNEL_B = 1
    enabled = 1
    coupling_type = PS3000A_DC = 1
    analogue_offset = 0
    status["setChA"] = ps.ps3000aSetChannel(
        handle, channel, enabled, coupling_type, chBRange, analogue_offset
    )
    assert_pico_ok(status["setChA"])

    # chBRange = 2
    # """关闭通道C"""
    # handle = chandle
    # channel = PS3000A_CHANNEL_C = 2
    # enabled = 0
    # coupling_type = PS3000A_DC = 1
    # analogue_offset = 0
    # status["setChA"] = ps.ps3000aSetChannel(handle, channel, enabled, coupling_type, chBRange, analogue_offset)
    # assert_pico_ok(status["setChA"])

    # chBRange = 2
    # """关闭通道D"""
    # handle = chandle
    # channel = PS3000A_CHANNEL_D = 3
    # enabled = 0
    # coupling_type = PS3000A_DC = 1
    # analogue_offset = 0
    # status["setChA"] = ps.ps3000aSetChannel(handle, channel, enabled, coupling_type, chBRange, analogue_offset)
    # assert_pico_ok(status["setChA"])

    """设置触发参数"""
    handle = chandle
    enable = 1
    source = ps3000A_channel_A = 0
    threshold = 10000
    direction = 3
    delay = 0
    autoTrigger_ms = 1000  # 如果1s没有检测到触发，则自动触发
    status["trigger"] = ps.ps3000aSetSimpleTrigger(
        handle, enable, source, threshold, direction, delay, autoTrigger_ms
    )
    assert_pico_ok(status["trigger"])

    """配置触发前采波数，触发后采波数"""
    preTriggerSamples = 1000
    postTriggerSamples = 29000
    maxsamples = preTriggerSamples + postTriggerSamples

    """这个函数根据设置的参数查询采样率和最大采样的点数"""
    handle = chandle
    timebase = 1  # 采样率相关参数，见编程文档，其中0为1GHz，1为500Mhz
    no_sample = maxsamples
    timeIntervalns = ctypes.c_float()
    returnedMaxSamples = ctypes.c_int16()
    TimeIntervalNanoseconds = ctypes.byref(timeIntervalns)
    MaxSamples = ctypes.byref(returnedMaxSamples)
    Segement_index = 0

    status["GetTimebase"] = ps.ps3000aGetTimebase2(
        handle,
        timebase,
        no_sample,
        TimeIntervalNanoseconds,
        1,
        MaxSamples,
        Segement_index,
    )
    assert_pico_ok(status["GetTimebase"])

    # Creates converted types maxsamples
    cmaxSamples = ctypes.c_int32(maxsamples)

    ser = serial.Serial("/dev/ttyUSB0", 256000, timeout=None)

    file_handle = open("seed and rej num.txt", "r")
    for i in range(0, start_idx):  # 跳过前面的数据
        file_handle.readline()

    if start_idx == 0:  # 说明是在重新采波
        file_list = os.listdir(save_path)
        for file in file_list:
            os.remove(os.path.join(save_path, file))

    for trace_idx in tqdm.trange(start_idx, end_idx, desc="trace capturing"):
        input = file_handle.readline().split(" ")
        seed = bytes.fromhex(input[0])
        rej_num = int(input[1])
        for rej_idx in tqdm.trange(0, rej_num, desc="rej capturing", leave=False):
            """配置开始采波"""
            status["runblock"] = ps.ps3000aRunBlock(
                chandle,
                preTriggerSamples,
                postTriggerSamples,
                timebase,
                1,
                None,
                0,
                None,
                None,
            )
            assert_pico_ok(status["runblock"])

            ser.write(bytes.fromhex("02"))
            ser.write(seed)
            ser.write((rej_idx + 1).to_bytes(4, byteorder="little"))
            ser.write(int(0).to_bytes(4, byteorder="little"))  # poly idx
            ser.read(5)  # 读取结束标志代表板子运行完成

            """创建buffer用于保存返回数据"""
            bufferAMax = np.empty(maxsamples, dtype=np.dtype("int16"))
            bufferBMax = np.empty(maxsamples, dtype=np.dtype("int16"))

            """把A通道的数据返回到bufferA里面"""
            handle = chandle
            source = ps3000A_channel_A = 0
            buffer_length = maxsamples
            segment_index = 0
            ratio_mode = ps3000A_Ratio_Mode_None = 0
            status["SetDataBuffers"] = ps.ps3000aSetDataBuffer(
                handle,
                source,
                bufferAMax.ctypes.data,
                buffer_length,
                segment_index,
                ratio_mode,
            )
            assert_pico_ok(status["SetDataBuffers"])

            """把B通道的数据返回到bufferB里面"""
            handle = chandle
            source = ps3000A_channel_B = 1
            buffer_length = maxsamples
            segment_index = 0
            ratio_mode = ps3000A_Ratio_Mode_None = 0
            status["SetDataBuffers"] = ps.ps3000aSetDataBuffer(
                handle,
                source,
                bufferBMax.ctypes.data,
                buffer_length,
                segment_index,
                ratio_mode,
            )
            assert_pico_ok(status["SetDataBuffers"])

            # Creates a overlow location for data
            overflow = ctypes.c_int16()
            # Creates converted types maxsamples
            cmaxSamples = ctypes.c_int32(maxsamples)

            """等待采波结束"""
            ready = ctypes.c_int16(0)
            check = ctypes.c_int16(0)
            while ready.value == check.value:
                status["isReady"] = ps.ps3000aIsReady(chandle, ctypes.byref(ready))

            """从示波器取回采波数据"""
            status["GetValuesBulk"] = ps.ps3000aGetValues(
                chandle, 0, ctypes.byref(cmaxSamples), 1, 0, 0, ctypes.byref(overflow)
            )
            assert_pico_ok(status["GetValuesBulk"])

            """获取波形水平偏移"""
            Times = ctypes.c_int64()
            TimeUnits = ctypes.c_char()
            status["GetTriggerTimeOffset"] = ps.ps3000aGetTriggerTimeOffset64(
                chandle, ctypes.byref(Times), ctypes.byref(TimeUnits), 0
            )

            time = (
                np.linspace(
                    0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value
                )
                / 10**6
            )

            # plt.plot(time, bufferAMax, linewidth=0.9, alpha=0.9)
            # plt.plot(time, bufferBMax, linewidth=0.9, alpha=0.9)
            # plt.xlabel("Time (ms)")
            # plt.ylabel("Voltage (mV)")
            # plt.show()
            np.savez_compressed(
                save_path + "trace%06drej%06d.npz" % (trace_idx, rej_idx),
                trace=bufferBMax,
                trigger=bufferAMax,
            )
            # if trace_idx == start_idx:
            #     np.savez_compressed(save_path + "time.npz", time=time)
        # plt.show()
    ser.close()
    file_handle.close()

    # Stops the scope
    # Handle = chandle
    status["stop"] = ps.ps3000aStop(chandle)
    assert_pico_ok(status["stop"])

    # Closes the unit
    # Handle = chandle
    status["close"] = ps.ps3000aCloseUnit(chandle)
    assert_pico_ok(status["close"])


def cs1_trace_capture(save_path, start_idx=0, end_idx=10000):
    """
    采波函数，支持断点续采
    :param save_path: 波形保存的地址，如果地址不存在自动创建
    :param start_idx: 保存波形的起始索引，从0开始
    :param end_idx: 保存波形的结束索引
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    status = {}
    """标识示波器设备的句柄号"""
    chandle = ctypes.c_int16()

    """打开示波器，返回句柄号"""
    status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(chandle), None)

    """一些检测是否成功打开的判断"""
    try:
        assert_pico_ok(status["openunit"])
    except:

        # powerstate becomes the status number of openunit
        powerstate = status["openunit"]

        # If powerstate is the same as 282 then it will run this if statement
        if powerstate == 282:
            # Changes the power input to "PICO_POWER_SUPPLY_NOT_CONNECTED"
            status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 282)
        # If the powerstate is the same as 286 then it will run this if statement
        elif powerstate == 286:
            # Changes the power input to "PICO_USB3_0_DEVICE_NON_USB3_0_PORT"
            status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 286)
        else:
            raise

        assert_pico_ok(status["ChangePowerSource"])

    """PICO3203D实际上只有20mV,50mV,100mV,200mV,500mV,1V,2V,5V,10V和20V
    这段注释的意思是，range这个参数是枚举型，从10mV开始，但是不是每个参数都能用，因为pico的型号关系，所以这个参数0不能用，
    例如range=0意味着量程是5V，但是实际上0没法用，因为PICO3203D不支持
    A，触发通道常用Range为8，即5V
    B，能量通道常用Range为1-4，即20mV到200mV"""

    """实测8是5V"""
    chARange = 8

    """设置通道参数"""
    handle = chandle
    channel = PS3000A_CHANNEL_A = 0
    enabled = 1
    coupling_type = PS3000A_DC = 1
    analogue_offset = 0
    status["setChA"] = ps.ps3000aSetChannel(
        handle, channel, enabled, coupling_type, chARange, analogue_offset
    )
    assert_pico_ok(status["setChA"])

    chBRange = 5
    """开启通道B"""
    handle = chandle
    channel = PS3000A_CHANNEL_B = 1
    enabled = 1
    coupling_type = PS3000A_DC = 1
    analogue_offset = 0
    status["setChA"] = ps.ps3000aSetChannel(
        handle, channel, enabled, coupling_type, chBRange, analogue_offset
    )
    assert_pico_ok(status["setChA"])

    # chBRange = 2
    # """关闭通道C"""
    # handle = chandle
    # channel = PS3000A_CHANNEL_C = 2
    # enabled = 0
    # coupling_type = PS3000A_DC = 1
    # analogue_offset = 0
    # status["setChA"] = ps.ps3000aSetChannel(handle, channel, enabled, coupling_type, chBRange, analogue_offset)
    # assert_pico_ok(status["setChA"])

    # chBRange = 2
    # """关闭通道D"""
    # handle = chandle
    # channel = PS3000A_CHANNEL_D = 3
    # enabled = 0
    # coupling_type = PS3000A_DC = 1
    # analogue_offset = 0
    # status["setChA"] = ps.ps3000aSetChannel(handle, channel, enabled, coupling_type, chBRange, analogue_offset)
    # assert_pico_ok(status["setChA"])

    """设置触发参数"""
    handle = chandle
    enable = 1
    source = ps3000A_channel_A = 0
    threshold = 10000
    direction = 3
    delay = 0
    autoTrigger_ms = 1000  # 如果1s没有检测到触发，则自动触发
    status["trigger"] = ps.ps3000aSetSimpleTrigger(
        handle, enable, source, threshold, direction, delay, autoTrigger_ms
    )
    assert_pico_ok(status["trigger"])

    """配置触发前采波数，触发后采波数"""
    preTriggerSamples = 1000
    postTriggerSamples = 31000
    maxsamples = preTriggerSamples + postTriggerSamples

    """这个函数根据设置的参数查询采样率和最大采样的点数"""
    handle = chandle
    timebase = 1  # 采样率相关参数，见编程文档，其中0为1GHz，1为500Mhz
    no_sample = maxsamples
    timeIntervalns = ctypes.c_float()
    returnedMaxSamples = ctypes.c_int16()
    TimeIntervalNanoseconds = ctypes.byref(timeIntervalns)
    MaxSamples = ctypes.byref(returnedMaxSamples)
    Segement_index = 0

    status["GetTimebase"] = ps.ps3000aGetTimebase2(
        handle,
        timebase,
        no_sample,
        TimeIntervalNanoseconds,
        1,
        MaxSamples,
        Segement_index,
    )
    assert_pico_ok(status["GetTimebase"])

    # Creates converted types maxsamples
    cmaxSamples = ctypes.c_int32(maxsamples)

    ser = serial.Serial("/dev/ttyUSB0", 256000, timeout=None)

    file_handle = open("seed and rej num.txt", "r")
    for i in range(0, start_idx):  # 跳过前面的数据
        file_handle.readline()

    # if start_idx == 0:  # 说明是在重新采波
    #     file_list = os.listdir(save_path)
    #     for file in file_list:
    #         os.remove(os.path.join(save_path, file))

    for trace_idx in tqdm.trange(start_idx, end_idx, desc="trace capturing"):
        input = file_handle.readline().split(" ")
        seed = bytes.fromhex(input[0])
        rej_num = int(input[1])
        for rej_idx in tqdm.trange(0, rej_num, desc="rej capturing", leave=False):
            """配置开始采波"""
            status["runblock"] = ps.ps3000aRunBlock(
                chandle,
                preTriggerSamples,
                postTriggerSamples,
                timebase,
                1,
                None,
                0,
                None,
                None,
            )
            assert_pico_ok(status["runblock"])

            ser.write(bytes.fromhex("02"))
            ser.write(seed)
            ser.write((rej_idx + 1).to_bytes(4, byteorder="little"))
            ser.write(int(0).to_bytes(4, byteorder="little"))  # poly idx
            ser.read(5)  # 读取结束标志代表板子运行完成

            """创建buffer用于保存返回数据"""
            bufferAMax = np.empty(maxsamples, dtype=np.dtype("int16"))
            bufferBMax = np.empty(maxsamples, dtype=np.dtype("int16"))

            """把A通道的数据返回到bufferA里面"""
            handle = chandle
            source = ps3000A_channel_A = 0
            buffer_length = maxsamples
            segment_index = 0
            ratio_mode = ps3000A_Ratio_Mode_None = 0
            status["SetDataBuffers"] = ps.ps3000aSetDataBuffer(
                handle,
                source,
                bufferAMax.ctypes.data,
                buffer_length,
                segment_index,
                ratio_mode,
            )
            assert_pico_ok(status["SetDataBuffers"])

            """把B通道的数据返回到bufferB里面"""
            handle = chandle
            source = ps3000A_channel_B = 1
            buffer_length = maxsamples
            segment_index = 0
            ratio_mode = ps3000A_Ratio_Mode_None = 0
            status["SetDataBuffers"] = ps.ps3000aSetDataBuffer(
                handle,
                source,
                bufferBMax.ctypes.data,
                buffer_length,
                segment_index,
                ratio_mode,
            )
            assert_pico_ok(status["SetDataBuffers"])

            # Creates a overlow location for data
            overflow = ctypes.c_int16()
            # Creates converted types maxsamples
            cmaxSamples = ctypes.c_int32(maxsamples)

            """等待采波结束"""
            ready = ctypes.c_int16(0)
            check = ctypes.c_int16(0)
            while ready.value == check.value:
                status["isReady"] = ps.ps3000aIsReady(chandle, ctypes.byref(ready))

            """从示波器取回采波数据"""
            status["GetValuesBulk"] = ps.ps3000aGetValues(
                chandle, 0, ctypes.byref(cmaxSamples), 1, 0, 0, ctypes.byref(overflow)
            )
            assert_pico_ok(status["GetValuesBulk"])

            """获取波形水平偏移"""
            Times = ctypes.c_int64()
            TimeUnits = ctypes.c_char()
            status["GetTriggerTimeOffset"] = ps.ps3000aGetTriggerTimeOffset64(
                chandle, ctypes.byref(Times), ctypes.byref(TimeUnits), 0
            )

            time = (
                np.linspace(
                    0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value
                )
                / 10**6
            )

            # plt.plot(time, bufferAMax, linewidth=0.9, alpha=0.9)
            # plt.plot(time, bufferBMax, linewidth=0.9, alpha=0.9)
            # plt.xlabel("Time (ms)")
            # plt.ylabel("Voltage (mV)")
            # plt.show()
            np.savez_compressed(
                save_path + "trace%06drej%06d.npz" % (trace_idx, rej_idx),
                trace=bufferBMax,
                trigger=bufferAMax,
            )
            # if trace_idx == start_idx:
            # np.savez_compressed(save_path + "time.npz", time=time)
        # plt.show()
    ser.close()
    file_handle.close()

    # Stops the scope
    # Handle = chandle
    status["stop"] = ps.ps3000aStop(chandle)
    assert_pico_ok(status["stop"])

    # Closes the unit
    # Handle = chandle
    status["close"] = ps.ps3000aCloseUnit(chandle)
    assert_pico_ok(status["close"])


def process_equation_batch_gen(args):
    """全局函数，用于gen的并行化处理"""
    start_idx, end_idx, Z, C, id_TA_res, z_minus_y_max_abs = args
    batch_size = end_idx - start_idx
    Z_MINUS_Y_ID_batch = np.zeros((batch_size, 2 * z_minus_y_max_abs + 1), dtype=float)
    CS_ID_batch = np.zeros((batch_size, 256), dtype=int)

    for local_idx in range(batch_size):
        equ_idx = start_idx + local_idx
        k = equ_idx % 256
        Z_tmp = Z[equ_idx]
        C_tmp = C[int(equ_idx / 256)]
        id_attack_res_tmp = id_TA_res[int(equ_idx / 256), k]

        zk = Z_tmp & 0x3F
        zk1 = (Z_tmp >> 6) & 0x1
        zk2 = (Z_tmp >> 7) & 0x1

        for y_prob in range(256):
            yk = y_prob & 0x3F
            yk1 = (y_prob >> 6) & 0x1
            yk2 = (y_prob >> 7) & 0x1

            z_minus_y, c = id_equation_gen(zk, zk1, zk2, yk, yk1, yk2, C_tmp, k)

            if z_minus_y is None:
                continue
            if np.abs(z_minus_y) > z_minus_y_max_abs:
                continue

            Z_MINUS_Y_ID_batch[
                local_idx, z_minus_y + z_minus_y_max_abs
            ] += id_attack_res_tmp[y_prob]
            CS_ID_batch[local_idx, :] = c

    return start_idx, Z_MINUS_Y_ID_batch, CS_ID_batch


def process_equation_batch_sumproduct(args):
    """全局函数，用于sumproduct的并行化处理"""
    start_idx, end_idx, sk_nodes, sk_idx, cs, x_nodes, tau, eta = args
    batch_size = end_idx - start_idx
    sk_nodes_batch = sk_nodes[start_idx:end_idx].copy()

    for local_idx in range(batch_size):
        equ_idx = start_idx + local_idx
        tmp = [[1]] * tau
        acc = [1]

        # 求每个sk除自己之外的累加分布
        for i in range(tau):
            tmp[i] = acc
            if cs[equ_idx, i] == 1:
                acc = np.convolve(acc, sk_nodes_batch[local_idx, i], mode="full")
            else:
                acc = np.convolve(
                    acc, np.flip(sk_nodes_batch[local_idx, i]), mode="full"
                )
        acc = [1]
        for i in range(tau - 1, -1, -1):
            tmp[i] = np.convolve(tmp[i], acc, mode="full")
            if cs[equ_idx, i] == 1:
                acc = np.convolve(acc, sk_nodes_batch[local_idx, i], mode="full")
            else:
                acc = np.convolve(
                    acc, np.flip(sk_nodes_batch[local_idx, i]), mode="full"
                )

        # 用z-y的分布减去每个sk除自己之外的累加分布
        for i in range(tau):
            if cs[equ_idx, i] == 1:
                tmp[i] = np.convolve(x_nodes[equ_idx], np.flip(tmp[i]), mode="full")
            else:
                tmp[i] = np.convolve(tmp[i], np.flip(x_nodes[equ_idx]), mode="full")

        # 更新sk的分布，只取[-eta,eta]之间的概率
        for i in range(tau):
            start_idx_eta = int(tmp[i].shape[0] / 2) - eta
            sk_nodes_batch[local_idx, i] = tmp[i][
                start_idx_eta : start_idx_eta + 2 * eta + 1
            ]

    return start_idx, sk_nodes_batch


def batch_cut_traces():
    """
    批量处理波形截取函数
    读取 ../traces/ 目录下的 c, cs1, y 三个目录的波形文件
    根据触发信号截取波形，保存到对应的 cut 目录中
    """
    base_traces_dir = "../traces/"
    trigger_threshold = 15000
    min_trs_len = 5000  # 最小波形长度阈值
    total_sign_num = 10000

    # 定义源目录和目标目录
    # trace_dirs = ["c", "cs1", "y"]
    trace_dirs = ["y"]

    for trace_type in trace_dirs:
        raw_trs_dir = base_traces_dir + trace_type + "/"
        cut_trs_dir = base_traces_dir + trace_type + "_cut/"

        print(f"处理 {trace_type} 目录...")

        # 创建目标目录
        if not os.path.exists(cut_trs_dir):
            os.makedirs(cut_trs_dir)

        # 获取所有波形文件
        if not os.path.exists(raw_trs_dir):
            print(f"警告: 源目录 {raw_trs_dir} 不存在，跳过")
            continue

        trace_files = [f for f in os.listdir(raw_trs_dir) if f.endswith(".npz")]
        if not trace_files:
            print(f"警告: 目录 {raw_trs_dir} 中没有找到 .npz 文件，跳过")
            continue

        print(f"找到 {len(trace_files)} 个波形文件")

        # 第一步：找到最短的波形长度
        trs_len = 0x3F3F3F3F
        print(f"正在计算 {trace_type} 的最短波形长度...")

        for trace_file in tqdm.tqdm(trace_files, desc=f"计算 {trace_type} 最短长度"):
            try:
                data = np.load(raw_trs_dir + trace_file)
                if "trigger" in data:
                    trigger = data["trigger"]
                    current_len = np.sum(trigger < trigger_threshold)
                    trs_len = min(trs_len, current_len)
                    if current_len < min_trs_len:
                        print(f"警告: 文件 {trace_file} 的波形长度异常: {current_len}")
                data.close()
            except Exception as e:
                print(f"错误: 读取文件 {trace_file} 时出错: {e}")
                continue

        if trs_len == 0x3F3F3F3F:
            print(f"错误: 无法确定 {trace_type} 的波形长度，跳过")
            continue

        print(f"{trace_type} 的最短波形长度: {trs_len}")

        # 第二步：截取并保存波形
        print(f"正在截取并保存 {trace_type} 波形...")

        if trace_type == "y":
            # y 目录处理：直接截取每个文件
            for trace_file in tqdm.tqdm(trace_files, desc=f"截取 {trace_type} 波形"):
                try:
                    data = np.load(raw_trs_dir + trace_file)
                    trigger = data["trigger"]
                    trace = data["trace"]

                    # 根据触发信号截取波形
                    cut_trace = trace[trigger < trigger_threshold][:trs_len]

                    # 生成输出文件名: trace000000.npz -> trs000000.npy
                    output_name = trace_file.replace("trace", "trs").replace(
                        ".npz", ".npy"
                    )

                    # 保存截取后的波形
                    np.save(cut_trs_dir + output_name, cut_trace)

                    data.close()

                except Exception as e:
                    print(f"错误: 处理文件 {trace_file} 时出错: {e}")
                    continue
        else:
            # c 和 cs1 目录处理：需要按签名分组，合并拒绝采样的波形
            print("Reading rejection count information...")
            rej_num_list = []
            with open("seed and rej num.txt", "r") as f:
                for i in range(total_sign_num):
                    line = f.readline().strip().split(" ")
                    rej_num = int(line[1])
                    rej_num_list.append(rej_num)
            rej_num_array = np.array(rej_num_list)

            for sign_idx in tqdm.trange(total_sign_num, desc="处理签名"):
                sign_traces = []
                for rej_idx in tqdm.trange(
                    rej_num_array[sign_idx], desc="处理拒绝采样", leave=False
                ):
                    try:
                        data = np.load(
                            raw_trs_dir + f"trace{sign_idx:06d}rej{rej_idx:06d}.npz"
                        )
                        trigger = data["trigger"]
                        trace = data["trace"]
                        cut_trace = trace[trigger < trigger_threshold][:trs_len]
                        sign_traces.append(cut_trace)
                    except Exception as e:
                        print(f"错误: 处理文件 {f} 时出错: {e}")
                        continue
                if len(sign_traces) > 0:
                    sign_traces_array = np.array(sign_traces)
                    output_name = f"trs{sign_idx:06d}.npy"
                    np.save(cut_trs_dir + output_name, sign_traces_array)

        print(f"{trace_type} 处理完成，截取后的文件保存在: {cut_trs_dir}")

    print("所有波形截取处理完成！")


if __name__ == "__main__":
    # gen_seed_and_rej_num(10000)
    # gen_valid_data()
    # get_s1()
    # gen_all_data()
    # cs1_trace_capture("../traces/cs1/", 0, 10000)
    # y_trace_capture("../traces/y/", 0, 10000)
    # c_trace_capture("../traces/c/", 0, 10000)

    # 批量处理波形截取
    # batch_cut_traces()

    # 补充实验：31-40个签名，使用简化版本测试
    # y_valid_bp_experiment(
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=20,  # 先测试小范围
    #     rerun_snr=False,
    #     rerun_ta=True,
    #     exp_repeat_num=1,  # 先测试少量重复
    #     bp_iter_num=15,
    #     snr_threshold=0.05,
    #     use_parallel_gen_sp=True,  # 先关闭并行化测试
    #     bp_process_num=8,  # 减少进程数
    # )

    # 查看BP实验结果
    # view_y_valid_bp_res(
    #     exp_save_path="../exp_res/y_valid_bp/",
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=20,
    #     exp_repeat_num=10,
    # )

    # cs1_valid_bp_experiment(
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=20,
    #     rerun_snr=False,
    #     rerun_ta=False,
    #     exp_repeat_num=10,
    #     bp_iter_num=10,
    #     poi_num=200,
    #     class_reserve_threshold=10,
    # )

    # view_cs1_valid_bp_res(
    #     exp_save_path="../exp_res/cs1_valid_bp/",
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=20,
    #     exp_repeat_num=10,
    # )

    # c_all_cs1_all_bp_experiment(
    #     rerun_c_snr=False,
    #     snr_c_plot=False,
    #     rerun_c_ta=False,
    #     rerun_cs1_snr=False,
    #     snr_cs1_plot=False,
    #     rerun_cs1_ta=False,
    #     process_num=4,
    #     poi_num=200,
    #     exp_repeat_num=1,
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=51,
    #     bp_iter_num=10,
    # )

    # view_cs1_valid_bp_res(
    #     exp_save_path="../exp_res/c_all_cs1_all_bp/",
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=20,
    #     exp_repeat_num=10,
    # )    #     exp_save_path="../exp_res/cs1_valid_bp/",
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=20,
    #     exp_repeat_num=10,
    # )

    # c_all_cs1_all_bp_experiment(
    #     rerun_c_snr=False,
    #     snr_c_plot=False,
    #     rerun_c_ta=False,
    #     rerun_cs1_snr=False,
    #     snr_cs1_plot=False,
    #     rerun_cs1_ta=False,
    #     process_num=4,
    #     poi_num=200,
    #     exp_repeat_num=10,
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=20,
    #     bp_iter_num=10,
    # )

    # view_cs1_valid_bp_res(
    #     exp_save_path="../exp_res/c_rej_cs1_rej_bp/",
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=20,
    #     exp_repeat_num=10,
    # )

    # c_all_cs1_all_bp_vary_c_acc_experiment(
    #     exp_repeat_num=3,
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=20,
    #     bp_iter_num=5,
    # )

    # view_bp_res_vary_c_acc(
    #     exp_save_path="../exp_res/c_all_cs1_all_bp_vary_c_acc/",
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=20,
    #     exp_repeat_num=3,
    #     c_acc_list=[0.96, 0.97, 0.98, 0.99],
    # )

    # c_rej_cs1_rej_bp_vary_c_acc_experiment(
    #     exp_repeat_num=10,
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=30,
    #     bp_iter_num=5,
    # )

    # view_bp_res_vary_c_acc(
    #     exp_save_path="../exp_res/c_rej_cs1_rej_bp_vary_c_acc/",
    #     exp_min_sign_num=1,
    #     exp_max_sign_num=30,
    #     exp_repeat_num=10,
    #     c_acc_list=[0.97, 0.98, 0.99, 1.00],
    # )

    pass
