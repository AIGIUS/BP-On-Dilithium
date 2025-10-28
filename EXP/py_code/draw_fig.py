import math
import multiprocessing
import os
import sys
from multiprocessing import Pool
from multiprocessing import shared_memory, RLock, resource_tracker

import matplotlib.gridspec
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
import tqdm

plt.rc("font", family="Times New Roman")


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def more_stable_softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def draw_bp_fig(implementation_type):
    s1 = np.load(f"{implementation_type}-s1.npy")
    exp_min_sign_num = 1
    exp_max_sign_num = 20
    exp_repeat_num = 10
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.grid(linestyle="--")
    ax1.set_xlabel("Number of Traces", fontsize=30)
    ax1.set_ylabel("Success Rate", fontsize=30)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.grid(linestyle="--")
    ax2.set_xlabel("Number of Traces", fontsize=30)
    ax2.set_ylabel("Coefficient Recovery Num", fontsize=30)
    marker_list = ["^", "v"]
    plot_label_list = ["All Signatures", "Valid Signatures"]
    for idx, res_path in enumerate(["c_all_cs1_all_bp", "cs1_valid_bp"]):
        success_rate = np.zeros((exp_max_sign_num - exp_min_sign_num + 1), dtype=float)
        coeff_recovery_ave_num = np.zeros(
            (exp_max_sign_num - exp_min_sign_num + 1), dtype=float
        )
        for bp_sign_num in tqdm.trange(
            exp_min_sign_num, exp_max_sign_num + 1, desc="bp sign num loop"
        ):
            for exp_idx in tqdm.trange(exp_repeat_num, leave=False):
                bp_res = np.load(
                    f"{implementation_type}-exp-res/{res_path}/exp%02dSignNum%02dBpRes.npy"
                    % (exp_idx, bp_sign_num)
                )
                max_right_coeff_num = 0
                for bp_iter in range(bp_res.shape[0]):
                    sk_hat = np.argmax(bp_res[bp_iter], axis=1) - 2
                    max_right_coeff_num = max(
                        max_right_coeff_num, np.sum(sk_hat == s1[0])
                    )
                if max_right_coeff_num == 256:
                    success_rate[bp_sign_num - 1] += 1
                coeff_recovery_ave_num[bp_sign_num - 1] += max_right_coeff_num
            success_rate[bp_sign_num - 1] /= exp_repeat_num
            coeff_recovery_ave_num[bp_sign_num - 1] /= exp_repeat_num

        ax1.plot(
            np.arange(exp_min_sign_num, exp_max_sign_num + 1, dtype=int),
            success_rate,
            linewidth=2,
            alpha=0.8,
            marker=marker_list[idx],
            markersize=8,
            label=plot_label_list[idx],
        )
        ax1.set_xticks(
            np.arange(exp_min_sign_num - 1, exp_max_sign_num + 1, 2, dtype=int)
        )
        ax1.set_xticklabels(
            np.arange(exp_min_sign_num - 1, exp_max_sign_num + 1, 2, dtype=int),
            fontsize=30,
        )
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=30)
        ax2.plot(
            np.arange(exp_min_sign_num, exp_max_sign_num + 1, dtype=int),
            coeff_recovery_ave_num,
            linewidth=2,
            alpha=0.8,
            marker=marker_list[idx],
            markersize=8,
            label=plot_label_list[idx],
        )
        ax2.set_xticks(
            np.arange(exp_min_sign_num - 1, exp_max_sign_num + 1, 2, dtype=int)
        )
        ax2.set_xticklabels(
            np.arange(exp_min_sign_num - 1, exp_max_sign_num + 1, 2, dtype=int),
            fontsize=30,
        )
        ax2.set_yticks(np.arange(64, 257, 32, dtype=int))
        ax2.set_yticklabels(np.arange(64, 257, 32, dtype=int), fontsize=30)
    ax1.legend(fontsize=30, loc="lower right")
    ax2.legend(fontsize=30, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"./img/{implementation_type}-BP结果.png", dpi=600, format="png")
    plt.show()


def draw_c_ta_res():
    # 绘制ref实现和ASM实现下c的模板攻击结果
    # 先画ref的
    ref_exp_save_path = "./ml-dsa-ref-exp-res/c_all_cs1_all_bp/"
    test_sign_num = 1000
    total_sign_num = 10000

    test_sign_idx_list = np.load(ref_exp_save_path + "test_sign_idx.npy")

    rej_num_list = []
    with open(ref_exp_save_path + "seed and rej num.txt", "r") as f:
        for _ in range(total_sign_num):
            line = f.readline().strip().split(" ")
            rej_num_list.append(int(line[1]))
    rej_num_array = np.array(rej_num_list)

    c_data_path = "./ml-dsa-ref-data/c_all/"

    c_test_label = np.zeros((np.sum(rej_num_array[test_sign_idx_list]), 256), dtype=int)
    # 加载c测试数据
    for i in tqdm.trange(test_sign_num, desc="加载c测试数据"):
        sign_idx = test_sign_idx_list[i]

        for j in range(rej_num_array[sign_idx]):
            c_test_label[np.sum(rej_num_array[test_sign_idx_list[:i]]) + j] = np.load(
                c_data_path + "c%06drej%06d.npy" % (sign_idx, j)
            )

    c_test_label += 1

    c_accuracy = np.zeros(256, dtype=float)
    # 计算c的平均准确率
    for i in range(256):
        c_test_ta_prob = np.load(ref_exp_save_path + "idx%03dProbTest.npy" % (i,))
        c_test_ta_predict = np.argmax(c_test_ta_prob, axis=1)
        c_accuracy[i] = np.sum(c_test_ta_predict == c_test_label[:, i]) / len(
            c_test_label[:, i]
        )
    print("参考实现c的平均准确率为：", np.mean(c_accuracy))

    # 初始化存储数组
    class_accuracy = np.zeros((256, 3), dtype=float)  # [系数, 类别] 的准确率

    for i in range(256):
        try:
            c_test_ta_prob = np.load(ref_exp_save_path + "idx%03dProbTest.npy" % (i,))
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

        except FileNotFoundError:
            class_accuracy[i] = 0.0

    # 绘制最终结果图
    plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(
        np.arange(0, 256, 1),
        class_accuracy[:, 0],
        c="#1D2747",
        linewidth=0.8,
        marker="^",
        alpha=0.8,
        label="-1",
        markersize=8,
    )
    ax1.plot(
        np.arange(0, 256, 1),
        class_accuracy[:, 1],
        c="#489693",
        linewidth=0.8,
        marker="v",
        alpha=0.8,
        label="0",
        markersize=8,
    )
    ax1.plot(
        np.arange(0, 256, 1),
        class_accuracy[:, 2],
        c="#984EA3",
        linewidth=0.8,
        marker="s",
        alpha=0.8,
        label="1",
        markersize=8,
    )
    ax1.set_xticks(np.arange(0, 257, 64))
    ax1.set_xticklabels(np.arange(0, 257, 64), fontsize=30)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=30)
    ax1.set_xlim(left=-1, right=257)
    ax1.set_ylim(0.1, 1.01)
    ax1.set_xlabel("Index of Coefficients", fontsize=30)
    ax1.set_ylabel("Accuracy", fontsize=30)
    ax1.legend(fontsize=30, loc="lower right")
    ax1.grid(linestyle="--")
    # ax1.set_title("Train Set", fontsize=30)

    # 先画ref的
    ref_exp_save_path = "./ml-dsa-m4-exp-res/c_all_cs1_all_bp/"
    test_sign_num = 1000
    total_sign_num = 10000

    test_sign_idx_list = np.load(ref_exp_save_path + "test_sign_idx.npy")

    rej_num_list = []
    with open(ref_exp_save_path + "seed and rej num.txt", "r") as f:
        for _ in range(total_sign_num):
            line = f.readline().strip().split(" ")
            rej_num_list.append(int(line[1]))
    rej_num_array = np.array(rej_num_list)

    c_data_path = "./ml-dsa-m4-data/c_all/"

    c_test_label = np.zeros((np.sum(rej_num_array[test_sign_idx_list]), 256), dtype=int)
    # 加载c测试数据
    for i in tqdm.trange(test_sign_num, desc="加载c测试数据"):
        sign_idx = test_sign_idx_list[i]

        for j in range(rej_num_array[sign_idx]):
            c_test_label[np.sum(rej_num_array[test_sign_idx_list[:i]]) + j] = np.load(
                c_data_path + "c%06drej%06d.npy" % (sign_idx, j)
            )

    c_test_label += 1

    c_accuracy = np.zeros(256, dtype=float)
    # 计算c的平均准确率
    for i in range(256):
        c_test_ta_prob = np.load(ref_exp_save_path + "idx%03dProbTest.npy" % (i,))
        c_test_ta_predict = np.argmax(c_test_ta_prob, axis=1)
        c_accuracy[i] = np.sum(c_test_ta_predict == c_test_label[:, i]) / len(
            c_test_label[:, i]
        )
    print("m4实现c的平均准确率为：", np.mean(c_accuracy))

    # 初始化存储数组
    class_accuracy = np.zeros((256, 3), dtype=float)  # [系数, 类别] 的准确率

    for i in range(256):
        try:
            c_test_ta_prob = np.load(ref_exp_save_path + "idx%03dProbTest.npy" % (i,))
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

        except FileNotFoundError:
            class_accuracy[i] = 0.0

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(
        np.arange(0, 256, 1),
        class_accuracy[:, 0],
        c="#1D2747",
        linewidth=0.8,
        marker="^",
        alpha=0.8,
        label="-1",
        markersize=8,
    )
    ax2.plot(
        np.arange(0, 256, 1),
        class_accuracy[:, 1],
        c="#489693",
        linewidth=0.8,
        marker="v",
        alpha=0.8,
        label="0",
        markersize=8,
    )
    ax2.plot(
        np.arange(0, 256, 1),
        class_accuracy[:, 2],
        c="#984EA3",
        linewidth=0.8,
        marker="s",
        alpha=0.8,
        label="1",
        markersize=8,
    )
    ax2.set_xticks(np.arange(0, 257, 64))
    ax2.set_xticklabels(np.arange(0, 257, 64), fontsize=30)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=30)
    ax2.set_xlim(left=-1, right=257)
    ax2.set_ylim(0.1, 1.01)
    ax2.set_xlabel("Index of Coefficients", fontsize=30)
    # ax2.ylabel('Accuracy', fontsize=22)
    ax2.legend(fontsize=30, loc="lower right")
    ax2.grid(linestyle="--")
    # # ax2.set_title("Test Set", fontsize=30)

    plt.tight_layout()
    plt.savefig("./img/c模板攻击准确率结果.png", dpi=600, format="png")
    plt.show()


def draw_cs1_ta_res():
    # 先画ref实现的cs1模板攻击结果
    ref_exp_save_path = "./ml-dsa-ref-exp-res/c_all_cs1_all_bp/"
    test_sign_num = 1000
    total_sign_num = 10000
    test_sign_idx_list = np.load(ref_exp_save_path + "test_sign_idx.npy")

    rej_num_list = []
    with open(ref_exp_save_path + "seed and rej num.txt", "r") as f:
        for _ in range(total_sign_num):
            line = f.readline().strip().split(" ")
            rej_num_list.append(int(line[1]))
    rej_num_array = np.array(rej_num_list)

    cs1_data_path = "./ml-dsa-ref-data/cs1_all/"

    cs1_test_label = np.zeros(
        (np.sum(rej_num_array[test_sign_idx_list]), 256), dtype=int
    )

    # 加载测试数据 - 使用全部数据，被拒绝的以及合法的
    for i in tqdm.trange(test_sign_num, desc="加载cs1测试数据"):
        for j in range(rej_num_array[test_sign_idx_list[i]]):
            cs1_test_label[np.sum(rej_num_array[test_sign_idx_list[:i]]) + j] = np.load(
                cs1_data_path + "cs%06drej%06d.npy" % (test_sign_idx_list[i], j)
            )
    cs1_test_label &= 0x7F

    test_ta_acc = np.zeros((256,), dtype=float)

    for i in range(256):
        reserved_class = np.load(
            ref_exp_save_path + "cs1_idx%03dReserveClass.npy" % (i,)
        )
        test_ta_prob = np.load(ref_exp_save_path + "cs1_idx%03dProbTest.npy" % (i,))
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

    plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(
        np.arange(0, 256, 1),
        test_ta_acc,
        c="#489693",
        linewidth=0.8,
        marker="^",
        alpha=0.8,
        markersize=8,
    )
    ax1.set_xticks(np.arange(0, 257, 64))
    ax1.set_xticklabels(np.arange(0, 257, 64), fontsize=30)
    ax1.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax1.set_yticklabels([0.1, 0.3, 0.5, 0.7, 0.9], fontsize=30)
    ax1.set_xlim(left=-1, right=257)
    ax1.set_ylim(0, 0.91)
    ax1.set_xlabel("Index of Coefficients", fontsize=30)
    ax1.set_ylabel("Accuracy", fontsize=30)
    # ax1.legend(fontsize=30, loc='lower right')
    ax1.grid(linestyle="--")
    # ax1.set_title("Train Set", fontsize=30)

    # 再画m4实现的cs1模板攻击结果
    ref_exp_save_path = "./ml-dsa-m4-exp-res/c_all_cs1_all_bp/"
    test_sign_num = 1000
    total_sign_num = 10000
    test_sign_idx_list = np.load(ref_exp_save_path + "test_sign_idx.npy")

    rej_num_list = []
    with open(ref_exp_save_path + "seed and rej num.txt", "r") as f:
        for _ in range(total_sign_num):
            line = f.readline().strip().split(" ")
            rej_num_list.append(int(line[1]))
    rej_num_array = np.array(rej_num_list)

    cs1_data_path = "./ml-dsa-m4-data/cs1_all/"

    cs1_test_label = np.zeros(
        (np.sum(rej_num_array[test_sign_idx_list]), 256), dtype=int
    )

    # 加载测试数据 - 使用全部数据，被拒绝的以及合法的
    for i in tqdm.trange(test_sign_num, desc="加载cs1测试数据"):
        for j in range(rej_num_array[test_sign_idx_list[i]]):
            cs1_test_label[np.sum(rej_num_array[test_sign_idx_list[:i]]) + j] = np.load(
                cs1_data_path + "cs%06drej%06d.npy" % (test_sign_idx_list[i], j)
            )
    cs1_test_label &= 0x7F

    test_ta_acc = np.zeros((256,), dtype=float)

    for i in range(256):
        reserved_class = np.load(
            ref_exp_save_path + "cs1_idx%03dReserveClass.npy" % (i,)
        )
        test_ta_prob = np.load(ref_exp_save_path + "cs1_idx%03dProbTest.npy" % (i,))
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

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(
        np.arange(0, 256, 1),
        test_ta_acc,
        c="#489693",
        linewidth=0.8,
        marker="v",
        alpha=0.8,
        markersize=8,
    )
    ax2.set_xticks(np.arange(0, 257, 64))
    ax2.set_xticklabels(np.arange(0, 257, 64), fontsize=30)
    ax2.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax2.set_yticklabels([0.1, 0.3, 0.5, 0.7, 0.9], fontsize=30)
    ax2.set_xlim(left=-1, right=257)
    ax2.set_ylim(0, 0.91)
    ax2.set_xlabel("Index of Coefficients", fontsize=30)
    # ax2.ylabel('Accuracy', fontsize=22)
    # ax2.legend(fontsize=30, loc='lower right')
    ax2.grid(linestyle="--")
    # ax2.set_title("Test Set", fontsize=30)

    plt.tight_layout()
    plt.savefig("./img/cs1模板攻击准确率结果.png", dpi=600, format="png")
    plt.show()


def draw_y_ta_res():
    # 先画ref实现的y的模板准确率
    ref_exp_save_path = "./ml-dsa-ref-exp-res/y_valid_bp/"
    y_data_path = "./ml-dsa-ref-data/y_valid/"

    test_sign_num = 1000

    test_sign_idx_list = np.load(ref_exp_save_path + "test_sign_idx.npy")

    test_y = np.zeros((test_sign_num, 256), dtype=int)

    # 加载测试数据
    for i in tqdm.trange(test_sign_num, desc="加载测试数据"):
        sign_idx = test_sign_idx_list[i]
        test_y[i] = np.load(y_data_path + "y%06d.npy" % (sign_idx,))

    # 准备测试标签数据（用于模板攻击准确率计算）
    print("准备测试标签数据...")
    test_label_y = test_y.copy()
    for i in range(test_label_y.shape[0]):
        for j in range(test_label_y.shape[1]):
            test_label_y[i, j] = test_label_y[i, j] & 0xFF

    test_ta_acc = np.zeros((256,), dtype=float)
    for i in range(256):
        test_ta_prob = np.load(ref_exp_save_path + "idx%03dProbTest.npy" % (i,))
        test_ta_predict = np.argmax(test_ta_prob, axis=1)
        test_ta_acc[i] = np.sum(test_ta_predict == test_label_y[:, i]) / test_sign_num

    # 绘制一个训练集加测试集的双栏图试一下
    plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(
        np.arange(0, 256, 1),
        test_ta_acc,
        c="#489693",
        linewidth=0.8,
        marker="v",
        alpha=0.8,
        markersize=8,
    )
    ax1.set_xticks(np.arange(0, 257, 64))
    ax1.set_xticklabels(np.arange(0, 257, 64), fontsize=30)
    ax1.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    ax1.set_yticklabels([0, 0.02, 0.04, 0.06, 0.08, 0.1], fontsize=30)
    ax1.set_xlim(left=-0.5, right=256.5)
    ax1.set_ylim(0.0, 0.105)
    ax1.set_xlabel("Index of Coefficients", fontsize=30)
    ax1.set_ylabel("Accuracy", fontsize=30)
    # ax1.legend(fontsize=30, loc='lower right')
    ax1.grid(linestyle="--")
    # ax1.set_title("Train Set", fontsize=30)

    # 再画m4实现的y的模板准确率
    ref_exp_save_path = "./ml-dsa-m4-exp-res/y_valid_bp/"
    y_data_path = "./ml-dsa-m4-data/y_valid/"

    test_sign_num = 1000

    test_sign_idx_list = np.load(ref_exp_save_path + "test_sign_idx.npy")

    test_y = np.zeros((test_sign_num, 256), dtype=int)

    # 加载测试数据
    for i in tqdm.trange(test_sign_num, desc="加载测试数据"):
        sign_idx = test_sign_idx_list[i]
        test_y[i] = np.load(y_data_path + "y%06d.npy" % (sign_idx,))

    # 准备测试标签数据（用于模板攻击准确率计算）
    print("准备测试标签数据...")
    test_label_y = test_y.copy()
    for i in range(test_label_y.shape[0]):
        for j in range(test_label_y.shape[1]):
            test_label_y[i, j] = test_label_y[i, j] & 0xFF

    test_ta_acc = np.zeros((256,), dtype=float)
    for i in range(256):
        test_ta_prob = np.load(ref_exp_save_path + "idx%03dProbTest.npy" % (i,))
        test_ta_predict = np.argmax(test_ta_prob, axis=1)
        test_ta_acc[i] = np.sum(test_ta_predict == test_label_y[:, i]) / test_sign_num

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(
        np.arange(0, 256, 1),
        test_ta_acc,
        c="#489693",
        linewidth=0.8,
        marker="v",
        alpha=0.8,
        markersize=8,
    )
    ax2.set_xticks(np.arange(0, 257, 64))
    ax2.set_xticklabels(np.arange(0, 257, 64), fontsize=30)
    ax2.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    ax2.set_yticklabels([0, 0.02, 0.04, 0.06, 0.08, 0.1], fontsize=30)
    ax2.set_xlim(left=-0.5, right=256.5)
    ax2.set_ylim(0.0, 0.105)
    ax2.set_xlabel("Index of Coefficients", fontsize=30)
    # ax2.ylabel('Accuracy', fontsize=22)
    # ax2.legend(fontsize=30, loc='lower right')
    ax2.grid(linestyle="--")
    # ax2.set_title("Test Set", fontsize=30)

    plt.tight_layout()
    plt.savefig("./img/y模板攻击准确率结果.png", dpi=600, format="png")
    plt.show()


def draw_convergence_speed(implementation_type):
    s1 = np.load(f"{implementation_type}-exp-res/s1.npy")
    exp_min_sign_num = 1
    exp_max_sign_num = 20
    exp_repeat_num = 10

    all_sig_res_path = f"{implementation_type}-exp-res/c_all_cs1_all_bp/"

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
            bp_res = np.load(
                all_sig_res_path
                + "exp%02dSignNum%02dBpRes.npy" % (exp_idx, bp_sign_num)
            )

            max_right_coeff_num = 0
            iter_recovery = []

            for bp_iter in range(bp_res.shape[0]):
                sk_hat = np.argmax(bp_res[bp_iter], axis=1) - 2
                right_coeff_num = np.sum(sk_hat == s1[0])
                max_right_coeff_num = max(max_right_coeff_num, right_coeff_num)
                iter_recovery.append(right_coeff_num)

            sign_recovery_data.append(iter_recovery)

            if max_right_coeff_num == 256:
                success_rate[bp_sign_num - exp_min_sign_num] += 1

            coeff_recovery_ave_num[
                bp_sign_num - exp_min_sign_num
            ] += max_right_coeff_num

        if sign_recovery_data:
            iter_recovery_data.append(
                {"sign_num": bp_sign_num, "data": np.array(sign_recovery_data)}
            )

        success_rate[bp_sign_num - exp_min_sign_num] /= exp_repeat_num
        coeff_recovery_ave_num[bp_sign_num - exp_min_sign_num] /= exp_repeat_num

    # 绘制结果
    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.grid(linestyle="--", alpha=0.7)

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
            ax1.plot(
                iterations + 1,
                mean_recovery,
                linewidth=1.5,
                alpha=0.8,
                marker="o",
                markersize=5,
                color=colors[i],
                label=f"{sign_num} traces",
            )
            ax1.fill_between(
                iterations + 1,
                mean_recovery - std_recovery,
                mean_recovery + std_recovery,
                alpha=0.2,
                color=colors[i],
            )
    ax1.set_xticks(np.arange(1, 11, 1))
    ax1.set_xticklabels(np.arange(1, 11, 1), fontsize=30)
    ax1.set_yticks(np.arange(0, 257, 64))
    ax1.set_yticklabels(np.arange(0, 257, 64), fontsize=30)
    ax1.set_xlim(left=0.5, right=10.5)
    ax1.set_ylim(0, 260)
    ax1.set_xlabel("BP Iterations", fontsize=30)
    ax1.set_ylabel("Recovered Coefficients", fontsize=30)
    ax1.legend(fontsize=30, loc="lower right")
    ax1.grid(linestyle="--")

    all_sig_res_path = f"{implementation_type}-exp-res/cs1_valid_bp/"

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
                    all_sig_res_path
                    + "exp%02dSignNum%02dBpRes.npy" % (exp_idx, bp_sign_num)
                )

                max_right_coeff_num = 0
                iter_recovery = []

                for bp_iter in range(bp_res.shape[0]):
                    sk_hat = np.argmax(bp_res[bp_iter], axis=1) - 2
                    right_coeff_num = np.sum(sk_hat == s1[0])
                    max_right_coeff_num = max(max_right_coeff_num, right_coeff_num)
                    iter_recovery.append(right_coeff_num)

                sign_recovery_data.append(iter_recovery)

                if max_right_coeff_num == 256:
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

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.grid(linestyle="--", alpha=0.7)

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
            ax2.plot(
                iterations + 1,
                mean_recovery,
                linewidth=1.5,
                alpha=0.8,
                marker="o",
                markersize=3,
                color=colors[i],
                label=f"{sign_num} signatures",
            )
            ax2.fill_between(
                iterations + 1,
                mean_recovery - std_recovery,
                mean_recovery + std_recovery,
                alpha=0.2,
                color=colors[i],
            )

    ax2.set_xticks(np.arange(1, 11, 1))
    ax2.set_xticklabels(np.arange(1, 11, 1), fontsize=30)
    ax2.set_yticks(np.arange(0, 257, 64))
    ax2.set_yticklabels(np.arange(0, 257, 64), fontsize=30)
    ax2.set_xlim(left=0.5, right=10.5)
    ax2.set_ylim(0, 260)
    ax2.set_xlabel("BP Iterations", fontsize=30)
    # ax2.set_ylabel('Recovered Coefficients', fontsize=30)
    ax2.legend(fontsize=30, loc="lower right")
    ax2.grid(linestyle="--")

    plt.tight_layout()
    plt.savefig(f"./img/{implementation_type}实现bp收敛速度.png", dpi=600, format="png")
    plt.show()


def view_bp_res_vary_c_acc(
    implementation_type,
    exp_min_sign_num=1,
    exp_max_sign_num=30,
    exp_repeat_num=10,
    c_acc_list=[0.99, 0.98, 0.97, 0.96, 0.95],
    sig_used="all",
):
    # 加载真实密钥
    s1 = np.load(f"{implementation_type}-s1.npy")

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
                        f"{implementation_type}-exp-res/c_{sig_used}_cs1_{sig_used}_bp_vary_c_acc/"
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

    # 绘制结果 - 参照draw_bp_fig的样式
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.grid(linestyle="--")
    ax1.set_xlabel("Number of Traces", fontsize=30)
    ax1.set_ylabel("Success Rate", fontsize=30)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.grid(linestyle="--")
    ax2.set_xlabel("Number of Traces", fontsize=30)
    ax2.set_ylabel("Coefficient Recovery Num", fontsize=30)

    # 颜色和标记样式
    colors = ["blue", "green", "red", "orange", "purple"]
    markers = ["^", "v", "o", "s", "D"]

    for i, c_acc in enumerate(c_acc_list):
        ax1.plot(
            np.arange(exp_min_sign_num, exp_max_sign_num + 1, dtype=int),
            success_rates[c_acc],
            linewidth=2,
            alpha=0.8,
            marker=markers[i % len(markers)],
            markersize=8,
            color=colors[i % len(colors)],
            label=f"c_acc={c_acc}",
        )
        ax1.set_xticks(
            np.arange(
                exp_min_sign_num - 1,
                exp_max_sign_num + 1,
                int(exp_max_sign_num - exp_min_sign_num + 1) / 10,
                dtype=int,
            )
        )
        ax1.set_xticklabels(
            np.arange(
                exp_min_sign_num - 1,
                exp_max_sign_num + 1,
                int(exp_max_sign_num - exp_min_sign_num + 1) / 10,
                dtype=int,
            ),
            fontsize=30,
        )
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=30)

        ax2.plot(
            np.arange(exp_min_sign_num, exp_max_sign_num + 1, dtype=int),
            coeff_recovery_ave_nums[c_acc],
            linewidth=2,
            alpha=0.8,
            marker=markers[i % len(markers)],
            markersize=8,
            color=colors[i % len(colors)],
            label=f"c_acc={c_acc}",
        )
        ax2.set_xticks(
            np.arange(
                exp_min_sign_num - 1,
                exp_max_sign_num + 1,
                int(exp_max_sign_num - exp_min_sign_num + 1) / 10,
                dtype=int,
            )
        )
        ax2.set_xticklabels(
            np.arange(
                exp_min_sign_num - 1,
                exp_max_sign_num + 1,
                int(exp_max_sign_num - exp_min_sign_num + 1) / 10,
                dtype=int,
            ),
            fontsize=30,
        )
        ax2.set_yticks(np.arange(64, 257, 32, dtype=int))
        ax2.set_yticklabels(np.arange(64, 257, 32, dtype=int), fontsize=30)

    ax1.legend(fontsize=25, loc="lower right")
    ax2.legend(fontsize=25, loc="lower right")
    plt.tight_layout()
    plt.savefig(
        f"./img/{implementation_type}实现{sig_used}签名c准确率对BP结果的影响.png",
        dpi=600,
        format="png",
    )
    plt.show()


if __name__ == "__main__":
    # draw_bp_fig("ml-dsa-m4")
    # draw_c_ta_res()
    # draw_cs1_ta_res()
    # draw_y_ta_res()
    # draw_convergence_speed("ml-dsa-m4")
    view_bp_res_vary_c_acc(
        "ml-dsa-m4",
        exp_min_sign_num=1,
        exp_max_sign_num=30,
        exp_repeat_num=10,
        c_acc_list=[1.00, 0.99, 0.98, 0.97],
        sig_used="rej",
    )
    # view_bp_res_vary_c_acc("ml-dsa-ref", exp_min_sign_num=1, exp_max_sign_num=30, exp_repeat_num=10, c_acc_list=[1.00,0.99,0.98,0.97,0.96], sig_used="rej")
    # view_bp_res_vary_c_acc("ml-dsa-ref", exp_min_sign_num=1, exp_max_sign_num=20, exp_repeat_num=10, c_acc_list=[0.99,0.98,0.97,0.96,0.95], sig_used="all")
    # view_bp_res_vary_c_acc("ml-dsa-m4", exp_min_sign_num=1, exp_max_sign_num=20, exp_repeat_num=10, c_acc_list=[0.99,0.98,0.97,0.96], sig_used="all")
    pass
