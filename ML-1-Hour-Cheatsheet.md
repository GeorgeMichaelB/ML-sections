# ML — 1-Hour Cheat Sheet (all notebooks)

**Timer plan (60 min):** 1 NumPy/Python 8 · 2 Linear Regression 10 · 3 Logistic/Softmax 8 · 4 K-Means 8 · 5 Classification metrics 10 · 6 Regularization 6 · 7 Pipeline 6 · 8 Quiz 4.
Read each part, say the one-line idea out loud, move on. Do not stop to understand every line.

---

## 1. Python / NumPy / Matplotlib basics (8 min)

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt
```

| Function | What it does |
|---|---|
| `np.array([1,2,3])` | make an array |
| `np.zeros((2,3))` `np.ones` `np.eye(3)` `np.full((2,2),7)` | filled arrays; eye = identity |
| `np.arange(0,10,2)` | like `range`, gives an array |
| `np.linspace(0,1,100)` | 100 evenly spaced points (used for plotting curves) |
| `a.reshape(2,3)` / `a.reshape(-1,1)` | change shape; `-1` = "figure it out" (column vector) |
| `a.T` | transpose |
| `a.dot(b)` / `a @ b` / `np.dot` | matrix multiplication |
| `a * b` | element-wise multiplication (NOT matrix product) |
| `np.sum` `np.mean` `np.std` (`axis=0` columns, `axis=1` rows) | statistics |
| `np.c_[a, b]` | glue columns side by side (add a bias column of ones) |
| `np.random.seed(42)` | make random numbers repeatable |
| `np.random.rand` / `randn` / `randint` / `permutation` | uniform 0–1 / normal (mean 0) / random ints / shuffled order |
| `np.exp` `np.log` `np.sqrt` `np.sin` | math, element-wise |
| `np.linalg.inv` / `norm` | matrix inverse / vector length |
| `np.argmax(a)` | index of the biggest value |
| `np.meshgrid(x, y)` | grid of points (for decision boundaries) |
| `a.ravel()` | flatten to 1-D |
| `np.unique(a)` | distinct values |
| `np.allclose(a, b)` | equal up to tiny rounding |
| `np.newaxis` | add a dimension |

**Broadcasting:** a small array is stretched automatically to match a bigger one, e.g. `A + [1,2,3]` adds to every row.

**Matplotlib:** `plt.plot(x,y)` · `plt.scatter` · `plt.xlabel/ylabel/title/legend/grid` · `plt.figure(figsize=(8,4))` · `plt.subplot(1,2,1)` · `plt.imshow(img)` · `plt.contour/contourf` (boundaries) · `plt.show()`.

**Pandas:** `pd.DataFrame(dict)` · `pd.read_csv` · `df.head() / info() / describe()` · `df["col"].value_counts()` · `df.corr()` · `pd.cut` (bin numbers into categories) · `pd.get_dummies` (one-hot) · `pd.crosstab` (count table of two columns).

---

## 2. Linear Regression (10 min)

Model: **ŷ = θ₀ + θ₁x₁ + …** = `X @ theta` (θ₀ = intercept/bias, so add a column of ones).
Cost: **MSE = (1/m) Σ (ŷ − y)²**.

**Normal Equation (one-shot formula):** θ = (XᵀX)⁻¹ Xᵀ y
```python
X_b = np.c_[np.ones((m,1)), X]
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```
Pros: no learning rate, no loop. Cons: slow with many features (matrix inverse), needs XᵀX invertible.

**Scikit-Learn:**
```python
from sklearn.linear_model import LinearRegression
lin = LinearRegression().fit(X, y)
lin.intercept_, lin.coef_        # θ₀, θ₁...
lin.predict(X_new)
```

**Batch Gradient Descent:** repeat `theta = theta - eta * (2/m) * X_b.T @ (X_b @ theta - y)`. Uses ALL data each step.
**Learning rate η:** too small → very slow; too big → diverges (error grows). Just right → converges.
**Stochastic GD:** 1 random instance per step → fast, noisy, can escape local minima, never settles exactly.
**Mini-batch GD:** small batches → middle ground (best on GPUs).
**Always scale features** (`StandardScaler`) before gradient descent.

```python
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01)
sgd.fit(X, y.ravel())
```

---

## 3. Logistic Regression & Softmax (8 min)

Logistic regression is for **classification** (despite the name): it outputs a probability.

- **Sigmoid:** σ(t) = 1 / (1 + e⁻ᵗ) → squashes any number into 0–1.
- **Prediction:** p = σ(θᵀx). Predict class 1 if p ≥ 0.5 (the **threshold**).
- **Decision boundary:** where θᵀx = 0 (p = 0.5). With 1 feature a point, with 2 features a line.
- **Cost (log loss):** J = −(1/m) Σ [ y·log(p) + (1−y)·log(1−p) ] → punishes confident wrong answers hard.
- **Threshold:** raise it → higher precision, lower recall. Lower it → higher recall, lower precision.

```python
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(C=2).fit(X, y)   # C = 1/regularization (bigger C = weaker regularization)
log.predict(X_new)          # class labels
log.predict_proba(X_new)    # [[p(class0), p(class1)]]
log.coef_, log.intercept_
```

**Softmax Regression** = logistic regression for **more than 2 classes**:
score sₖ = θₖᵀx → pₖ = exp(sₖ) / Σ exp(sⱼ) (probabilities add up to 1) → predict the class with the highest probability (`argmax`).
Cost = **cross-entropy**. Use for exclusive classes (iris: setosa / versicolor / virginica).
Dataset: `load_iris()` → `iris.data`, `iris.target`.

---

## 4. K-Means (8 min)

**Unsupervised clustering** — no labels. Classification = labels known; clustering = find groups.

**Algorithm:** 1) pick k random centroids → 2) assign each point to nearest centroid → 3) move each centroid to the mean of its points → repeat 2–3 until nothing changes.

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
km.labels_              # cluster of each training point
km.cluster_centers_     # the centroids
km.inertia_             # sum of squared distances to nearest centroid
km.predict(X_new)       # cluster for new points
```

**Choosing k:**
- **Inertia / elbow:** plot inertia vs k; pick the "elbow" where it stops dropping fast. (Inertia always falls as k grows, so you can't just take the minimum.)
- **Silhouette score:** `silhouette_score(X, km.labels_)` — range −1 to +1, **higher is better** (+1 = tight, well separated; near 0 = overlapping; negative = wrong cluster).

**Limits:** you must choose k · results depend on random start (use `n_init`) · bad with different sizes/densities and non-round shapes · sensitive to feature scale.
**Colour segmentation:** reshape image to `(-1, 3)`, run KMeans, replace each pixel by its centroid colour.

---

## 5. Classification & metrics (10 min)

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
knn.predict(X_test); knn.score(X_test, y_test)   # score = accuracy
```
**KNN:** a point gets the majority class of its k nearest neighbours. Small k → overfits (noisy); large k → underfits (too smooth). **Scale features first**: `scaler.fit_transform(X_train)` then only `scaler.transform(X_test)`.

**Confusion matrix** (rows = actual, columns = predicted):

| | Pred NEG | Pred POS |
|---|---|---|
| **Actual NEG** | TN | FP |
| **Actual POS** | FN | TP |

- **Accuracy** = (TP+TN) / all — misleading when classes are unbalanced (a model that always says "not 5" gets ~90%).
- **Precision** = TP / (TP+FP) — "of what I called positive, how much was right?"
- **Recall** (sensitivity) = TP / (TP+FN) — "of all real positives, how many did I catch?"
- **F1** = 2·P·R / (P+R) — harmonic mean; low if either is low.
- **Precision/recall trade-off:** raising the threshold ↑ precision ↓ recall.
- **ROC curve:** TPR (recall) vs FPR; **AUC** = area under it (1 = perfect, 0.5 = random).

| Function | Use |
|---|---|
| `train_test_split(X, y, test_size=0.2, random_state=42)` | split data |
| `DummyClassifier()` | baseline to beat |
| `SGDClassifier()` | fast linear classifier (`.fit`, `.predict`) |
| `cross_val_score(model, X, y, cv=3, scoring="accuracy")` | accuracy on each fold |
| `cross_val_predict(model, X, y, cv=3)` | predictions made on unseen folds |
| `confusion_matrix(y, y_pred)` / `ConfusionMatrixDisplay` | matrix / plot it |
| `precision_score` `recall_score` `f1_score` | the metrics |
| `model.decision_function(X)` | raw scores (before threshold) |
| `precision_recall_curve(y, scores)` | P and R at every threshold |
| `roc_curve(y, scores)` / `roc_auc_score(y, scores)` | ROC points / AUC |
| `fetch_openml("mnist_784", as_frame=False)` | MNIST: 70,000 images, 28×28 = 784 pixels (0–255) |
| `make_moons`, `load_wine` | toy datasets |

**Rule: fit scalers/models on the TRAIN set only; touch the test set once at the end.**

---

## 6. Polynomial, Learning Curves, Regularization (6 min)

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
poly = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), LinearRegression())
```
**Polynomial regression:** add x², x³… as features, then fit a *linear* model. Higher degree = more flexible = more overfitting.

**Learning curves** (`learning_curve(model, X, y, train_sizes=..., cv=5, scoring="neg_root_mean_squared_error")`):
- **Underfitting:** train & validation error both HIGH, close together, flat → need a better/more complex model (more features).
- **Overfitting:** train error LOW, validation error HIGH, big gap → more data, simpler model, or regularize.

**Regularization = penalize big weights** (don't penalize θ₀):

| Model | Penalty | Behaviour |
|---|---|---|
| `Ridge(alpha=1)` | L2: α·½·Σθ² | shrinks weights, keeps all features |
| `Lasso(alpha=0.1)` | L1: α·Σ\|θ\| | pushes some weights to exactly 0 (feature selection) |
| `ElasticNet(alpha, l1_ratio=0.5)` | mix of both | `l1_ratio` 0 = Ridge, 1 = Lasso |

Bigger `alpha` = stronger regularization (simpler model). Scale features first.
**Early stopping:** during GD, stop when validation error is lowest (before it climbs again = overfitting); keep the best model (`deepcopy`).
Remember: `RMSE = root_mean_squared_error(y, y_pred)`.

---

## 7. End-to-end project pipeline (6 min)

Steps: **get data → look at it → split → clean → transform → train → evaluate → tune → final test → save.**

| Function | Use |
|---|---|
| `df.info() / describe() / value_counts() / corr()` | explore |
| `scatter_matrix(df[cols])` | scatter plots of every pair |
| `train_test_split` / `StratifiedShuffleSplit` | random split / split that keeps proportions (e.g. income groups via `pd.cut`) |
| `SimpleImputer(strategy="median")` | fill missing values |
| `OneHotEncoder()` / `OrdinalEncoder()` | text categories → numbers (one-hot = no order, ordinal = ordered) |
| `StandardScaler()` | (x − mean)/std |
| `MinMaxScaler()` | squeeze into 0–1 |
| `FunctionTransformer(np.log)` | apply your own function (e.g. log for skewed data) |
| `Pipeline([...])` / `make_pipeline(...)` | chain steps so fit/transform happen in order |
| `ColumnTransformer` / `make_column_transformer` | different steps for numeric vs text columns |
| custom transformer | class inheriting `BaseEstimator, TransformerMixin` with `fit()` and `transform()` |
| `RandomForestRegressor` `DecisionTreeRegressor` `SVR` `KNeighborsRegressor` | other models |
| `mean_squared_error` / `root_mean_squared_error` | error metrics |
| `cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=10)` | scores come out NEGATIVE → use `-scores` |
| `GridSearchCV` | try every combo of hyperparameters |
| `RandomizedSearchCV` | try random combos (faster for big spaces) |
| `joblib.dump(model, "m.pkl")` / `joblib.load` | save / load model |

**Common model fit/transform pattern:** `fit()` learns → `transform()` applies → `fit_transform()` both (training data only) → `predict()` for models.
**Decision tree** with no limit overfits badly (train error ≈ 0, CV error high) → that's why we use cross-validation.

---

## 8. Mix-ups to avoid (read before the exam)

- **Precision vs Recall:** precision = *correct among my positives*; recall = *found among real positives*.
- **Overfit vs underfit:** overfit = great on train, bad on new data; underfit = bad on both.
- **Classification vs clustering:** labels known vs unknown.
- **Logistic regression** is classification, **linear regression** predicts numbers.
- **fit vs transform:** never `fit` on test data.
- **`*` vs `@`:** element-wise vs matrix product.
- **Ridge vs Lasso:** L2 shrinks, L1 zeros out.
- **Inertia ↓ always with k** — use elbow/silhouette, don't just minimize.

## 9. Quick self-test (4 min) — answer out loud

1. Normal Equation formula? When is it bad?
2. What happens if the learning rate is too big?
3. Why is accuracy misleading on unbalanced data?
4. Raise the threshold: what happens to precision and recall?
5. Softmax vs sigmoid — when each?
6. How do you choose k in K-Means (two ways)?
7. Train error low, validation error high → what's wrong, what to do?
8. Ridge vs Lasso difference?
9. Why fit the scaler only on the training set?
10. `cross_val_score` returns negative numbers — why?

*Answers are all in the sections above. Good luck!*
