# Code used in the course _Foundation of Machine Learning_

## Reviewed Codes

- __FML_4__: Sklearn basics and pipelines definition
- __FML_5__: Decision Trees and Random Forests
- __FML_6__: Neural Networks from scratch (to review again)
- __FML_7__: Neural Networks with PyTorch (there is a summary version with just the essentials to set up a NN)
- __FML_8__: KMeans and KMedoids alghoritms for clustering data

## Homeworks

- Linear Regression with normal equations
- Classification metrics from scratch
- Pipeline in the contest of regression
- Include the grid search in the pipeline (done)
- SVM classifier using the toy dataset diabetes.csv

## Optimization found

__FML_4__
- When makes the rows shuffle, we have to add the _random_state_ parameter in order to make the results reproducible
- In _Sklearn_Pipeline_Grid.py_, in Pipeline definition, it is unnecessary to define _C_ and _penalty_ as parameter in the classifier definition
- Instead of rewriting the fourth file, I've changed it to include the grid search (homework)

__FML_6__
- _loss_val_, _X_val_ and _y_val_ are not used. I've deleted them.

__FML_8__
- In the constructor, instead of defining rstate, I've defined self.randomInteger = np.random.RandomState(random_state).randint, so that in fit() method I haven't to declare rint.
- In KMedoids, I haven't done attributes of the class self.indices and self.y_pred in the constructor (because they're used only in the fit() method, it is unnecessary to store them as attributes)