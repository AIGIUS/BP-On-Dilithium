import numpy as np
import tqdm

from utils import Utils, CustomizedError
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Template:
    def __init__(self):
        self.ave = None
        self.cor = None
        self.dimen_reduce_method = None
        self.dimen_reduce_num = None
        self.dimen_reduce_model = None
        self.class_num = None

    def build_model(
        self, traces, label, class_num, dimen_reduce_method, dimen_reduce_num=10
    ):
        """
        Template attack modeling process
        :param traces: Modeling waveforms, 2D array, each row is a sample
        :param label: Modeling labels, 1D array
        :param class_num: Number of classification classes
        :param dimen_reduce_method: Dimensionality reduction method, currently only lda and native, native means no dimensionality reduction
        :param dimen_reduce_num: Dimensionality reduction dimension, native default is 10, native does not reduce dimensions
        :return: None, the established mean and covariance matrices and dimensionality reduction models are stored in class instance
        """
        if dimen_reduce_method == "lda":
            self.dimen_reduce_method = "lda"
            self.dimen_reduce_num = dimen_reduce_num
            self.dimen_reduce_model = LinearDiscriminantAnalysis(
                n_components=self.dimen_reduce_num
            )
            self.dimen_reduce_model.fit(traces, label)
            traces = self.dimen_reduce_model.transform(traces)
        elif dimen_reduce_method == "native":
            self.dimen_reduce_method = "native"
            self.dimen_reduce_num = traces.shape[1]
            self.dimen_reduce_model = None
        self.class_num = class_num
        self.ave = np.zeros((self.class_num, self.dimen_reduce_num), dtype=float)
        self.cor = np.zeros(
            (self.class_num, self.dimen_reduce_num, self.dimen_reduce_num), dtype=float
        )
        for i in range(self.class_num):
            self.ave[i] = np.average(traces[label == i], axis=0)
            self.cor[i] = np.cov(traces[label == i].transpose())

    def apply_model(self, traces):
        """
        Apply the established template for classification, return classification probabilities
        :param traces: Data to be classified
        :return: Classification probabilities, normalized
        """
        if (
            self.ave is None
            or self.cor is None
            or self.dimen_reduce_method is None
            or self.class_num is None
        ):
            raise Exception("Template not yet established")
        prob = np.zeros((traces.shape[0], self.class_num), dtype=float)
        if self.dimen_reduce_method == "lda":
            traces = self.dimen_reduce_model.transform(traces)
        # for i in tqdm.trange(traces.shape[0], desc="Template classification (traces)", leave=False):
        for i in range(traces.shape[0]):
            for j in range(self.class_num):
                noise_valid = traces[i] - self.ave[j]
                P1 = np.exp(
                    -0.5 * (noise_valid @ np.linalg.inv(self.cor[j]) @ noise_valid.T)
                )
                P2 = np.sqrt(
                    (2 * np.pi) ** self.dimen_reduce_num * np.linalg.det(self.cor[j])
                )
                prob[i, j] = P1 / P2
            prob[i] = Utils.more_stable_softmax(np.log(prob[i] + 1e-50))
        return prob
