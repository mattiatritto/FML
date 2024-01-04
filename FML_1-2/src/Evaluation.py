import numpy as np

class Evaluation:

    def __init__(self, model):
        self._model = model



    def compute_performance(self, X, y):
        """
        This method compute some parameters to evaluate the perfomance of the model
        :param X: test sample with shape (m, n_features)
        :param y: ground truth (correct) target values shape(m)
        :return: a dictionary with name of specific metric as key and specific performance as value
        """
        preds = self._model.predict(X)

        mae = self._mean_absolute_error(preds, y)
        mape = self._mean_absolute_percentage_error(preds, y)
        mpe = self._mean_percentage_error(preds, y)
        mse = self._mean_squared_error(preds, y)
        rmse = self._root_mean_squared_error(preds, y)
        r2 = self._r_2(preds, y)
        return {'mae': mae, 'mape': mape, 'mpe': mpe, 'mse': mse, 'rmse': rmse, 'r2': r2}



    def _mean_absolute_error(self, preds, y):
        """
        Compute the MAE (Mean Absolute Error)
        :param preds: prediction values with shape (m)
        :param y: ground truth (correct) target values with shape (m)
        :return: MAE value (non-negative, floating point). The best value is 0.0
        """
        output_errors = np.abs(preds-y)
        return np.average(output_errors)



    def _mean_squared_error(self, preds, y):
        """
        Compute the MSE (Mean Squared Error)
        :param preds: predictiom values with shape (m)
        :param y: ground truth (correct) target values with shape (m)
        :return: MSE value (non-negative, floating point). The best value is 0.0
        """
        output_errors = (preds-y)**2
        return np.average(output_errors)



    def _root_mean_squared_error(self, preds, y):
        """
        Compute the RMSE (Root Mean Squared Error)
        :param preds: predictiom values with shape (m)
        :param y: ground truth (correct) target values with shape (m)
        :return: RMSE value (non-negative, floating point). The best value is 0.0
        """
        return np.sqrt(self._mean_squared_error(preds, y))



    def _mean_absolute_percentage_error(self, preds, y):
        """
        Compute the MAPE (Mean Absolute Percentage Error)
        :param preds: predictiom values with shape (m)
        :param y: ground truth (correct) target values with shape (m)
        :return: MAPE value (non-negative, floating point). The best value is 0.0

        But note the fact that bad predictions can lead to arbitarily large
        MAPE values, especially if some y_true values are very close to zero.
        Note that we return a large value instead of `inf` when y_true is zero.
        """
        output_errors = np.abs((preds-y)/y)
        return np.average(output_errors)



    def _mean_percentage_error(self, preds, y):
        """
        Compute the MPE (Mean Percentage Error)
        :param preds: predictiom values with shape (m)
        :param y: ground truth (correct) target values with shape (m)
        :return: MPE value (floating point). The best value is 0.0
        """
        output_errors = (preds-y)/y
        return np.average(output_errors)*100



    def _r_2(self, preds, y):
        """
        Compute the R2 score
        :param preds: predictiom values with shape (m)
        :param y: ground truth (correct) target values with shape (m)
        :return: R2 score (0 <= R2 <= 1, floating point). The best value is 1.0
        """
        sst = np.sum((y - y.mean()) ** 2)
        ssr = np.sum((preds - y) ** 2)
        return 1 - (ssr/sst)