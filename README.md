# Linear and Logistic Regression from scratch

Linear regression and logistic regression implemented with NumPy and trained with mini-batch gradient descent. Both classes have `fit` and `predict` like scikit-learn.

## Linear Regression

`Linear Regression.ipynb`

- Loss: mean squared error.
- Update: `w = w - lr * (-2/b) * X_b^T (y_b - X_b w)` for each mini-batch of size `b`.
- Stops early when the epoch loss changes less than `tol`.
- Tested on synthetic data (100 samples, 10 features, small noise), with a plot of the loss curve and true against predicted values.

## Logistic Regression

`LogisticRegression.ipynb`

- Sigmoid activation and binary cross entropy loss.
- Optional regularization with the `penalty` argument: `"l1"`, `"l2"`, or a mix of both with `l1_ratio`. The strength is `alpha`. The intercept is not penalized.
- `decision_boundary` sets the probability threshold for class 1 (default 0.5).
- Tested on 2D synthetic data where the class is 1 when `x1 + x2 > 0`, with an 80/20 train/test split. The notebook plots the decision boundary on the test points.

Example:

```python
model = LogisticRegression(max_iter=2000, learning_rate=0.01, penalty="l2", alpha=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Requirements

numpy, matplotlib, scikit-learn (only for `train_test_split`)
