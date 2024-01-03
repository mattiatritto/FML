import numpy as np

class ClassificationMetrics:

    def __init__(self, model):
        self._model = model



    def compute_performance(self, X, y):
        """
        This method compute some parameters to evaluate the perfomance of the model
        :param X: test sample with shape (m, n_features)
        :param y: ground truth (correct) target values shape(m)
        :return: a dictionary with name of specific metric as key and specific performance as value
        """

        preds = np.round(self._model.predict(X))

        self.TP = np.sum(np.logical_and(preds, y))
        self.TN = np.sum(np.logical_and(np.logical_not(preds), np.logical_not(y)))
        self.FP = np.sum((preds - y) == 1)
        self.FN = np.sum((preds-y) == -1)

        accuracy = self._accuracy()
        precision = self._precision()
        recall = self._recall()
        f1_score = self._f1_score()
        return {'true_positive': self.TP, 'true_negative': self.TN, 'false_positive': self.FP, 'false_negative': self.FN, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}



    def _accuracy(self):
        """
        The accuracy score is the ratio between the correct prediction and the total observation we have:
        :return: accuracy score (non-negative, floating point). The best value is 1.0.
        """
        return (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)



    def _precision(self):
        """
        The precision score (or the positive predicited values) is the ratio between the TP and the sum of TP and FP.
        :return: precision score (non-negative, floating point). The best value is 1.0.
        """
        return self.TP / (self.TP + self.FP)



    def _recall(self):
        """
        The recall score (also known as sensitivity) is the ratio between the TP and the sum of TP and FN.
        :return: recall score (non-negative, floating point). The best value is 1.0.
        """
        return self.TP / (self.TP + self.FN)



    def _f1_score(self):
        """
        The F1 score is a measure of a test's accuracy, and takes into account precision and recall.
        :return: F1 score (non-negative, floating point). The best value is 1.0.
        """
        return 2*self.TP / (2*self.TP + self.FP + self.FN)