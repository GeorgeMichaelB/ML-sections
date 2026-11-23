# ML — Full Guide: Rules + Code Examples (line by line)

Use this **after** the 1-hour cheat sheet. Every code block has a comment on (almost) every line saying what it does.
Read the **RULES** boxes first in each part: those are what exams love to ask.

---

## 0. The 12 golden rules

1. **Split first.** Put the test set away before looking at the data. Touch it ONCE at the very end.
2. **`fit` on TRAIN only.** Scalers, imputers, encoders and models learn from training data. For test data only call `transform`/`predict`.
3. **Scale features** (`StandardScaler`) for gradient descent, KNN, K-Means, SVM, Ridge/Lasso/ElasticNet.
4. **`X` must be 2-D** (rows = samples, columns = features): shape `(m, n)`. A single column is `(m, 1)`, NOT `(m,)`.
5. **`y` is 1-D** for most sklearn models: `(m,)`. Use `y.ravel()` if you have `(m, 1)`.
6. **Add a bias column** (a column of ones) when you do the maths by hand: `np.c_[np.ones((m,1)), X]`. sklearn does it for you.
7. **Accuracy lies on unbalanced data.** Check precision, recall, F1, confusion matrix.
8. **Use cross-validation**, not one lucky split, to compare models.
9. **Overfit** = train good / validation bad. **Underfit** = both bad. Fix overfit: more data, simpler model, regularize. Fix underfit: more features / complex model / less regularization.
10. **Names ending with `_`** (`coef_`, `intercept_`, `labels_`, `inertia_`) are things the model **learned** — they exist only after `.fit()`.
11. **sklearn scores are "higher = better"**, so error metrics come out **negative** (`neg_root_mean_squared_error`). Use `-scores`.
12. **Set `random_state` / `np.random.seed`** so results are repeatable.

---

## 1. NumPy — line by line

```python
import numpy as np                      # load NumPy and call it np

a = np.array([1, 2, 3])                 # 1-D array, shape (3,)
M = np.array([[1, 2, 3],
              [4, 5, 6]])               # 2-D array (matrix), shape (2, 3): 2 rows, 3 columns

np.zeros((2, 3))                        # 2x3 matrix full of 0
np.ones((2, 3))                         # 2x3 matrix full of 1
np.eye(3)                               # 3x3 identity matrix (1 on diagonal, 0 elsewhere)
np.full((2, 2), 7)                      # 2x2 matrix full of 7
np.arange(0, 10, 2)                     # [0 2 4 6 8]  (start, stop NOT included, step)
np.linspace(0, 1, 5)                    # [0. .25 .5 .75 1.]  5 evenly spaced points, stop INCLUDED

M.shape                                 # (2, 3)
M.T                                     # transpose: rows <-> columns, shape (3, 2)
M.reshape(3, 2)                         # same numbers, new shape (total count must match)
a.reshape(-1, 1)                        # -1 = "work it out": turns (3,) into a column (3, 1)
M.ravel()                               # flatten to 1-D: [1 2 3 4 5 6]

M[0, 1]                                 # row 0, column 1 -> 2
M[:, 0]                                 # all rows, column 0 -> [1 4]
M[1, :]                                 # row 1, all columns -> [4 5 6]
M[M > 2]                                # boolean mask: elements bigger than 2

M + 10                                  # adds 10 to every element (broadcasting)
M * M                                   # ELEMENT-WISE multiply (NOT matrix product)
M @ M.T                                 # MATRIX product (2x3 @ 3x2 -> 2x2); same as np.dot(M, M.T)

M.sum()                                 # sum of everything
M.sum(axis=0)                           # sum DOWN each column -> [5 7 9]
M.mean(axis=1)                          # mean ACROSS each row -> [2. 5.]
M.std()                                 # standard deviation

np.c_[a, a]                             # glue 1-D arrays as COLUMNS next to each other
np.exp(a), np.log(a), np.sqrt(a)        # element-wise exponent, natural log, square root
np.argmax([3, 9, 2])                    # index of the maximum -> 1
np.unique([1, 1, 2, 3])                 # [1 2 3]
np.allclose(0.1 + 0.2, 0.3)             # True: "equal" ignoring tiny rounding errors
np.linalg.inv(np.eye(2))                # inverse of a matrix
np.linalg.norm([3, 4])                  # vector length -> 5.0
np.meshgrid(np.arange(3), np.arange(2)) # all (x, y) grid coordinates (used to draw decision boundaries)

np.random.seed(42)                      # fix the random generator -> same numbers each run
np.random.rand(3)                       # 3 uniform random numbers in [0, 1)
np.random.randn(3)                      # 3 numbers from normal distribution (mean 0, std 1)
np.random.randint(0, 10, 3)             # 3 random integers from 0 to 9
np.random.permutation(5)                # shuffled [0..4]
```

**RULES:** `*` is element-wise, `@` is matrix product · `axis=0` goes down columns, `axis=1` goes across rows · slicing `a[start:stop]` excludes `stop` · broadcasting stretches the smaller array.

---

## 2. Matplotlib — the standard template

```python
import matplotlib.pyplot as plt          # plotting library

x = np.linspace(0, 10, 100)              # 100 x-values
plt.figure(figsize=(8, 4))               # new figure, 8x4 inches
plt.plot(x, np.sin(x), "b-", label="sin")  # line plot; "b-" = blue line; label used in legend
plt.scatter(x[::10], np.sin(x[::10]))    # dots (every 10th point)
plt.xlabel("x")                          # x-axis name
plt.ylabel("sin(x)")                     # y-axis name
plt.title("Example")                     # title
plt.legend()                             # show the labels
plt.grid(True)                           # grid lines
plt.axis([0, 10, -1.5, 1.5])             # [xmin, xmax, ymin, ymax]
plt.show()                               # draw it

plt.subplot(1, 2, 1)                     # grid of 1 row, 2 columns, use plot #1 (then call plt.plot...)
plt.imshow(image, cmap="binary")         # show an image (MNIST digit: image = row.reshape(28, 28))
plt.contourf(xx, yy, zz)                 # filled regions (decision boundary background)
```

---

## 3. Pandas — line by line

```python
import pandas as pd

df = pd.DataFrame({"age": [20, 30, None],
                   "city": ["Cairo", "Giza", "Cairo"]})  # table from a dictionary
df = pd.read_csv("file.csv")             # load a CSV file into a table

df.head()                                # first 5 rows
df.info()                                # columns, types, NON-NULL counts (spot missing values)
df.describe()                            # count, mean, std, min, quartiles, max (numbers only)
df["city"].value_counts()                # how many of each category
df.corr(numeric_only=True)               # correlation between numeric columns (-1 to +1)
df["age"].fillna(df["age"].median())     # fill missing values with the median
pd.get_dummies(df["city"])               # one-hot: one 0/1 column per category
df["age_group"] = pd.cut(df["age"], bins=[0, 25, 100], labels=["young", "old"])  # bin numbers into categories
pd.crosstab(df["city"], df["age_group"]) # count table of two columns
```

---

## 4. The scikit-learn pattern (know this cold)

| Kind | Methods | Examples |
|---|---|---|
| **Estimator** (learns from data) | `fit(X, y)` | everything |
| **Transformer** (changes data) | `fit`, `transform`, `fit_transform` | `StandardScaler`, `SimpleImputer`, `OneHotEncoder`, `PolynomialFeatures` |
| **Predictor** (makes answers) | `fit`, `predict`, `score` | `LinearRegression`, `LogisticRegression`, `KNeighborsClassifier`, `KMeans` |

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,                     # features and labels
    test_size=0.2,            # 20% goes to the test set
    random_state=42)          # fixed shuffle -> same split every time

scaler = StandardScaler()                       # object that will compute (x - mean) / std
X_train_s = scaler.fit_transform(X_train)       # LEARN mean/std from train AND apply it
X_test_s  = scaler.transform(X_test)            # apply the SAME mean/std to test (no fit!)

model.fit(X_train_s, y_train)                   # learn
model.predict(X_test_s)                         # answers
model.score(X_test_s, y_test)                   # accuracy (classifiers) or R² (regressors)
```

---

## 5. Linear Regression

**Idea:** predict a number with a straight line. **ŷ = θ₀ + θ₁x₁ + …** · Cost **MSE = (1/m)·Σ(ŷ − y)²**.

### 5.1 Make data
```python
np.random.seed(42)
m = 100                                  # number of samples
X = 2 * np.random.rand(m, 1)             # (100,1) random x between 0 and 2
y = 4 + 3 * X + np.random.randn(m, 1)    # true line: y = 4 + 3x, plus random noise
```

### 5.2 Normal Equation  θ = (XᵀX)⁻¹ Xᵀ y
```python
X_b = np.c_[np.ones((m, 1)), X]                         # add the bias column of ones -> shape (100, 2)
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y          # the formula; theta ≈ [[4],[3]]
X_new = np.array([[0], [2]])                            # two new x values
X_new_b = np.c_[np.ones((2, 1)), X_new]                 # they also need the bias column
y_pred = X_new_b @ theta                                # prediction = X · theta
```
**Rules:** direct formula, no learning rate, no loop · slow when features are very many · fails if XᵀX isn't invertible (sklearn uses SVD/`pinv` to avoid this).

### 5.3 Same thing with Scikit-Learn
```python
from sklearn.linear_model import LinearRegression
lin = LinearRegression()          # create the model (adds the bias for you)
lin.fit(X, y)                     # learn
lin.intercept_, lin.coef_         # θ₀ ≈ 4, θ₁ ≈ 3
lin.predict(X_new)                # predictions
```

### 5.4 Batch Gradient Descent (BGD)
```python
eta = 0.1                          # learning rate (step size)
n_epochs = 1000                    # how many full passes
theta = np.random.randn(2, 1)      # start with random weights

for epoch in range(n_epochs):                       # repeat many times
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)   # slope of the cost; uses ALL m samples
    theta = theta - eta * gradients                 # step DOWNHILL (minus the gradient)
```
**Learning rate:** too small → very slow · too big → jumps around and **diverges** · good → converges to the Normal Equation answer.
**Always scale features** or the "bowl" is stretched and GD is slow.

### 5.5 Stochastic GD (one random sample per step)
```python
theta = np.random.randn(2, 1)
for epoch in range(50):
    for i in range(m):
        idx = np.random.randint(m)             # pick ONE random sample
        xi, yi = X_b[idx:idx+1], y[idx:idx+1]  # keep it 2-D
        gradients = 2 * xi.T @ (xi @ theta - yi)   # gradient from just that sample
        eta = 0.1 / (1 + 0.01 * (epoch * m + i))   # learning schedule: shrink eta over time
        theta = theta - eta * gradients
```
**SGD:** fast and noisy, can escape local minima, never settles exactly (so shrink η). **Mini-batch GD:** small groups (e.g. 32 samples): smoother than SGD, fast on GPUs. Both **BGD, SGD, Mini-batch** end near the same θ.

### 5.6 SGD with Scikit-Learn
```python
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(max_iter=1000,   # max passes
                   tol=1e-5,        # stop when improvement < this
                   penalty=None,    # no regularization
                   eta0=0.01,       # starting learning rate
                   random_state=42)
sgd.fit(X, y.ravel())               # y must be 1-D here
sgd.intercept_, sgd.coef_
```

---

## 6. Logistic Regression & Softmax

### 6.1 The maths in 4 lines
- **Sigmoid** σ(t) = 1/(1+e⁻ᵗ) → output between 0 and 1.
- **Probability** p = σ(θᵀx). **Predict** 1 if p ≥ 0.5, else 0.
- **Decision boundary:** θᵀx = 0.
- **Log loss:** J = −(1/m) Σ [ y·log(p) + (1−y)·log(1−p) ] (huge penalty for confident wrong answers).

```python
def sigmoid(t):
    return 1 / (1 + np.exp(-t))     # squashes any real number into (0, 1)
sigmoid(0)    # 0.5
sigmoid(10)   # ≈ 1
```

### 6.2 Detect "Iris virginica" from petal width (1 feature)
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)                    # iris as pandas tables
X = iris.data[["petal width (cm)"]].values         # (150,1) one feature, 2-D
y = (iris.target == 2)                              # True where species is virginica (class 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
log = LogisticRegression(random_state=42)           # create model
log.fit(X_train, y_train)                           # learn θ

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)      # 1000 petal widths from 0 to 3
proba = log.predict_proba(X_new)                    # shape (1000,2): [P(not virginica), P(virginica)]
boundary = X_new[proba[:, 1] >= 0.5][0]             # first width where P(virginica) reaches 0.5
log.predict([[1.7], [1.5]])                         # class labels -> [ True False ]
```

### 6.3 Changing the threshold
```python
p = log.predict_proba(X_test)[:, 1]    # probability of the positive class
y_pred_strict = p >= 0.9               # high threshold: only very sure -> HIGHER precision, LOWER recall
y_pred_loose  = p >= 0.1               # low threshold: say yes easily -> HIGHER recall, LOWER precision
```

### 6.4 Two features → a line boundary
```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values   # (150,2)
log = LogisticRegression(C=2, random_state=42).fit(X, y)          # C = 1/regularization strength
log.coef_, log.intercept_                                         # boundary: w1*x1 + w2*x2 + b = 0
```
**C:** large C = weak regularization (can overfit) · small C = strong regularization.

### 6.5 Softmax Regression (3 classes)
```python
y = iris.target                                      # values 0, 1, 2
softmax_reg = LogisticRegression(C=30, random_state=42)   # handles >2 classes automatically (softmax)
softmax_reg.fit(X, y)
softmax_reg.predict([[5, 2]])                        # predicted class
softmax_reg.predict_proba([[5, 2]]).round(2)         # probabilities for 3 classes; they add up to 1
```
**Softmax:** score sₖ = θₖᵀx → pₖ = exp(sₖ) / Σⱼ exp(sⱼ) → pick the biggest. **Cost: cross-entropy.** Use when classes are **mutually exclusive**.

---

## 7. K-Means

**Unsupervised** (no labels). Goal: put similar points in the same group.

### 7.1 Algorithm in NumPy
```python
k = 3
centroids = X[np.random.choice(len(X), k, replace=False)]   # 1) pick k random points as centres
for _ in range(10):                                         # repeat until stable
    dist = np.linalg.norm(X[:, None] - centroids[None], axis=2)   # distance from every point to every centre
    labels = dist.argmin(axis=1)                            # 2) each point -> nearest centre
    centroids = np.array([X[labels == j].mean(axis=0)       # 3) move each centre to the mean of its points
                          for j in range(k)])
```

### 7.2 Scikit-Learn
```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,        # k
            n_init=10,           # run 10 times with different starts, keep the best
            random_state=42)
km.fit(X)                        # run the algorithm
km.labels_                       # cluster number for each training point
km.cluster_centers_              # centre coordinates, shape (k, n_features)
km.inertia_                      # sum of squared distances to the closest centre (lower = tighter)
km.predict([[0, 2], [3, 2]])     # which cluster for NEW points
km.transform(X[:2])              # distance from each point to every centre
```

### 7.3 Choosing k
```python
from sklearn.metrics import silhouette_score

inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_
            for k in range(1, 10)]               # inertia for k = 1..9 -> plot it, pick the ELBOW
sil = [silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).labels_)
       for k in range(2, 10)]                    # silhouette needs k >= 2; pick the HIGHEST
```
**Rules:** inertia always decreases as k grows (don't pick the minimum) · silhouette ∈ [−1, +1], **higher = better**.
**Limits:** must choose k · depends on initial centres · bad on non-round or very different-sized clusters · needs scaled features.

### 7.4 Colour segmentation (reduce image colours)
```python
from sklearn.datasets import load_sample_image
image = load_sample_image("flower.jpg")                   # (height, width, 3) RGB image
X = (image / 255).reshape(-1, 3)                          # every pixel becomes one row of 3 numbers (0-1)
km = KMeans(n_clusters=8, n_init=10, random_state=42).fit(X)   # find 8 main colours
segmented = km.cluster_centers_[km.labels_]               # replace each pixel with its centre colour
segmented = segmented.reshape(image.shape)                # back to image shape
```

---

## 8. Classification, KNN & metrics

### 8.1 KNN
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)    # look at the 5 nearest points
knn.fit(X_train_s, y_train)                  # "learning" = just storing the data
knn.predict(X_test_s)                        # majority vote of the 5 neighbours
knn.score(X_test_s, y_test)                  # accuracy
```
**Rules:** small k → overfit (jagged boundary) · large k → underfit (too smooth) · **scale features** (distance-based) · slow prediction on big data.

### 8.2 MNIST: "is it a 5?" (binary classifier)
```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", as_frame=False)     # 70,000 digit images, 784 pixels each
X, y = mnist.data, mnist.target                       # y holds STRINGS "0".."9"
X_train, X_test = X[:60000], X[60000:]                # first 60k train, last 10k test (already shuffled)
y_train, y_test = y[:60000], y[60000:]
y_train_5 = (y_train == "5")                          # True only for the digit 5

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state=42)                  # fast linear classifier trained with SGD
sgd.fit(X_train, y_train_5)                           # learn "5 vs not 5"
sgd.predict([X[0]])                                   # prediction for one image
```

### 8.3 Evaluate (never trust one number)
```python
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             precision_recall_curve, roc_curve, roc_auc_score)
from sklearn.dummy import DummyClassifier

cross_val_score(sgd, X_train, y_train_5, cv=3, scoring="accuracy")   # accuracy on 3 folds
cross_val_score(DummyClassifier(), X_train, y_train_5, cv=3)         # baseline that always says "not 5": ~90%! (accuracy misleads)

y_pred = cross_val_predict(sgd, X_train, y_train_5, cv=3)    # each sample predicted by a model that never saw it
confusion_matrix(y_train_5, y_pred)                          # [[TN, FP], [FN, TP]]
precision_score(y_train_5, y_pred)                           # TP / (TP + FP)
recall_score(y_train_5, y_pred)                              # TP / (TP + FN)
f1_score(y_train_5, y_pred)                                  # harmonic mean of both

y_scores = cross_val_predict(sgd, X_train, y_train_5, cv=3, method="decision_function")  # raw scores, not 0/1
prec, rec, thr = precision_recall_curve(y_train_5, y_scores)  # P and R for EVERY threshold
fpr, tpr, thr = roc_curve(y_train_5, y_scores)                # false/true positive rate per threshold
roc_auc_score(y_train_5, y_scores)                            # area under ROC: 1 perfect, 0.5 random
y_pred_90 = (y_scores >= 3000)                                # your own threshold: raise it -> precision up, recall down
```

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | (TP+TN)/all | % right (bad for unbalanced) |
| Precision | TP/(TP+FP) | of my "yes", how many correct |
| Recall / TPR | TP/(TP+FN) | of real "yes", how many found |
| FPR | FP/(FP+TN) | of real "no", how many wrongly flagged |
| F1 | 2PR/(P+R) | balance of P and R |

**Use PR curve** when positives are rare; **ROC** when classes are balanced.

---

## 9. Polynomial, Learning Curves, Regularization

### 9.1 Polynomial regression
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m, 1) - 3                       # x between -3 and 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)         # curved data (quadratic + noise)

poly = PolynomialFeatures(degree=2, include_bias=False)   # create columns [x, x²]
X_poly = poly.fit_transform(X)                            # (100,2)
lin = LinearRegression().fit(X_poly, y)                   # a LINEAR model on the new features
lin.intercept_, lin.coef_                                 # ≈ 2, [1, 0.5]
```
**Rules:** degree too low → underfit · too high → overfit (wiggles through the noise) · features count explodes with degree · scale after expanding.

### 9.2 Learning curves
```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y.ravel(),                    # model + data
    train_sizes=np.linspace(0.01, 1.0, 40),              # use 1% ... 100% of the training data
    cv=5, scoring="neg_root_mean_squared_error")         # 5-fold CV, negative RMSE
train_errors = -train_scores.mean(axis=1)                # make positive, average over folds
valid_errors = -valid_scores.mean(axis=1)
plt.plot(train_sizes, train_errors, "r-+", label="train")
plt.plot(train_sizes, valid_errors, "b-", label="validation")
plt.legend(); plt.show()
```
**Read the plot:** both curves high and close together → **underfit** · train low, validation high with a gap → **overfit** (gap closes with more data).

### 9.3 Regularization (penalize big weights; never penalize θ₀)
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

Ridge(alpha=0.1, solver="cholesky").fit(X_poly, y)       # L2: adds alpha * ½ * Σθ² to the cost
Lasso(alpha=0.1).fit(X_poly, y.ravel())                  # L1: adds alpha * Σ|θ| -> some weights become EXACTLY 0
ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_poly, y.ravel())   # mix: l1_ratio 0 = Ridge, 1 = Lasso
```
| | Penalty | Effect |
|---|---|---|
| Ridge | L2 (squares) | shrinks all weights, keeps all features |
| Lasso | L1 (absolute) | feature selection (zeros) |
| Elastic Net | both | use when features are correlated / more than samples |

Bigger `alpha` = stronger regularization = simpler model. **Scale first.** In `LogisticRegression` the knob is `C = 1/alpha`.

### 9.4 Early stopping
```python
from copy import deepcopy
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error

sgd = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
best_rmse, best_model = float("inf"), None
for epoch in range(500):
    sgd.partial_fit(X_train_s, y_train.ravel())               # ONE more epoch of training
    val_rmse = root_mean_squared_error(y_valid, sgd.predict(X_valid_s))   # error on validation set
    if val_rmse < best_rmse:                                  # new best?
        best_rmse, best_model = val_rmse, deepcopy(sgd)       # save a COPY of the model
```
**Idea:** training error keeps falling but validation error starts to rise (overfitting) → stop at the minimum.

---

## 10. End-to-end project (the pipeline notebook)

**Steps:** frame the problem → get data → explore → **split** → prepare (impute, encode, scale) → select & train models → **cross-validate** → tune (grid/random search) → evaluate on test **once** → launch & save.

```python
# Stratified split: keep the same proportion of each income level in train and test
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
train_set, test_set = train_test_split(housing, test_size=0.2,
                                       stratify=housing["income_cat"], random_state=42)
y = train_set["median_house_value"]                    # label
X = train_set.drop("median_house_value", axis=1)       # features
```

```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

num_pipe = make_pipeline(SimpleImputer(strategy="median"),    # fill missing numbers with the median
                         StandardScaler())                    # then scale
cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),         # fill missing categories
                         OneHotEncoder(handle_unknown="ignore"))          # text -> 0/1 columns

preprocessing = ColumnTransformer([
    ("num", num_pipe, make_column_selector(dtype_include=np.number)),     # apply num_pipe to numeric columns
    ("cat", cat_pipe, make_column_selector(dtype_include=object)),        # apply cat_pipe to text columns
])

model = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))  # preprocess THEN model
model.fit(X, y)                                           # fits every step in order
model.predict(X.iloc[:5])                                 # raw data in, predictions out
```

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
scores = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=10)  # minus makes RMSE positive
scores.mean(), scores.std()                               # average error and how stable it is

param_grid = {"randomforestregressor__n_estimators": [50, 100],       # step name + "__" + parameter
              "randomforestregressor__max_features": [4, 6, 8]}
grid = GridSearchCV(model, param_grid, cv=3, scoring="neg_root_mean_squared_error")
grid.fit(X, y)                                            # tries EVERY combination with cross-validation
grid.best_params_                                         # winner
final_model = grid.best_estimator_                        # best model, already refit on all training data
# final evaluation, ONCE:  root_mean_squared_error(y_test, final_model.predict(X_test))

import joblib
joblib.dump(final_model, "model.pkl")                     # save
model = joblib.load("model.pkl")                          # load later
```
**GridSearch** = all combos (small spaces) · **RandomizedSearch** = random combos with `n_iter` (large spaces, give distributions like `randint`).

### Custom transformer (class)
```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class MyScaler(BaseEstimator, TransformerMixin):      # Mixin gives you fit_transform for free
    def __init__(self, with_mean=True):                # only store hyperparameters here
        self.with_mean = with_mean
    def fit(self, X, y=None):                          # LEARN something from the data
        X = check_array(X)                             # validate it's a 2-D numeric array
        self.mean_ = X.mean(axis=0)                    # learned value ends with "_"
        self.scale_ = X.std(axis=0)
        return self                                    # fit must return self
    def transform(self, X):                            # APPLY what was learned
        check_is_fitted(self)                          # error if fit() wasn't called
        return (check_array(X) - self.mean_) / self.scale_
```

---

## 11. Common errors & traps

| Problem | Cause / fix |
|---|---|
| `Expected 2D array, got 1D array` | `X` must be 2-D: `X.reshape(-1, 1)` |
| `y` shape warning / wrong broadcasting | `(m,1)` vs `(m,)` — use `y.ravel()` |
| `ValueError: Input contains NaN` | impute first (`SimpleImputer`) |
| `could not convert string to float` | encode text (`OneHotEncoder`) |
| scores are negative | sklearn flips errors so "higher is better" → `-scores` |
| great train score, bad test score | overfitting / leakage (did you `fit` the scaler on all data?) |
| `NotFittedError` | call `.fit()` before `.predict()` / `.transform()` |
| MNIST labels are strings | compare with `"5"`, not `5` |
| loss turns to `nan`/`inf` in GD | learning rate too big or features not scaled |
| `*` gives wrong shape | you wanted `@` (matrix product) |

---

## 12. Read-the-code practice (cover the answers!)

```python
X = np.random.rand(100, 3)
y = (X[:, 0] > 0.5)
```
1. Shape of `X`? **(100, 3)** · `y` type? **boolean array**, shape (100,).

```python
scaler = StandardScaler().fit(X_train)
X_test_s = scaler.transform(X_test)
```
2. Why not `fit` on `X_test`? **It would leak test information into preprocessing.**

```python
km = KMeans(n_clusters=4, n_init=10).fit(X)
print(km.inertia_)
```
3. Will inertia go up or down if `n_clusters=5`? **Down (or equal).** So how to choose k? **Elbow / silhouette.**

```python
cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
```
4. Scores are `[-3.1, -2.9, ...]`. Which is the **better** model? **The one closer to 0** (e.g. −2.9).

```python
LogisticRegression(C=0.01)
```
5. Stronger or weaker regularization than `C=100`? **Stronger** (C = 1/alpha) → simpler model, risk of underfit.

*Good luck — and run each block at least once in your notebooks so the syntax feels familiar.*
