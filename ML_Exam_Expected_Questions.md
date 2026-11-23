# 🎯 بنك الأسئلة المتوقعة لامتحان الـ Machine Learning (مستوحى من امتحانات السنوات السابقة)

تم إعداد هذا الملف بناءً على النمط الدقيق لأسئلة الامتحانين السابقين، والتركيز على الأسئلة المباشرة: **"الكود ده بيعمل إيه؟"، "اشرح المعاملات (Parameters)"، "قارن بين كذا وكذا"، ومسائل الحساب المباشرة.**

---

## 📑 الفهرس السريع:
1. [القسم الأول: تنظيف وتجهيز البيانات (Data Preprocessing) - الأهم 50%](#1-تنظيف-وتجهيز-البيانات-data-preprocessing)
2. [القسم الثاني: النماذج الخطية والانحدار (Linear Regression & Optimization)](#2-النماذج-الخطية-والانحدار-linear-regression)
3. [القسم الثالث: التصنيف ومقاييس الأداء (Classification & Performance Metrics)](#3-التصنيف-ومقاييس-الأداء-classification--metrics)
4. [القسم الرابع: ضبط النماذج والـ Pipelines (Tuning & Pipelines)](#4-ضبط-النماذج-والـ-pipelines)

---

## 1. تنظيف وتجهيز البيانات (Data Preprocessing)

### س1: What does `df.isnull().sum()` do?
* **الإجابة:** يقوم بفحص الجدول بالكامل وحساب **إجمالي عدد القيم المفقودة (Missing Values / NaN)** في كل عمود على حدة.
* **مثال ناتج:**
  ```text
  age            15
  salary          0
  total_rooms     8
  ```

---

### س2: What are the 3 main methods to handle missing values (NaN)?
* **الإجابة:**
  1. **حذف السطور (Drop Rows):** حذف الصفوف التي تحتوي على قيم مفقودة باستخدام `dropna(subset=[...])`.
  2. **حذف الأعمدة (Drop Columns):** حذف العمود بالكامل إذا كان أغلبه مفقوداً أو غير مهم باستخدام `drop("col", axis=1)`.
  3. **التعويض (Imputation):** ملء الفراغات بقيمة إحصائية بديلة (كالوسيط Median، المتوسط Mean، أو المنوال Mode) باستخدام `fillna()` أو `SimpleImputer`.

---

### س3: What is the difference between `dropna()` and `fillna()`?
* **`dropna()`:** تقوم **بحذف** السطور أو الأعمدة التي تحتوي على قيم مفقودة (تقلل عدد سطور الداتا).
* **`fillna(value)`:** تقوم **بالتعويض وملء** الخانات الفارغة بقيمة محددة دون حذف أي سطر من الداتا.

---

### س4: الكود ده بيعمل إيه واشرح كل جزء فيه؟
```python
df["age"].fillna(df["age"].mean(), inplace=True)
```
* **الشرح:**
  1. `df["age"].mean()`: يحسب المتوسط الحسابي للأعمار الموجودة.
  2. `df["age"].fillna(...)`: يملأ أي خانة مفقودة (NaN) في عمود `age` بهذا المتوسط المحسوب.
  3. `inplace=True`: يُطبق التعديل ويحفظه في الجدول الأصلي مباشرة بدون الحاجة لإنشاء متغير جديد.

---

### س5: متى نفضل استخدام الـ Median على الـ Mean في تعويض القيم المفقودة؟
* **الإجابة:** نفضل الـ **Median (الوسيط)** عندما تحتوي البيانات على **قيم شاذة أو متطرفة (Outliers)** أو عندما يكون توزيع البيانات ملتوياً (Skewed)، لأن الوسيط لا يتأثر بالقيم المتطرفة، بينما الـ Mean (المتوسط) يتأثر بها بشدة وينحرف.

---

### س6: What does Label Encoding / Categorical Encoding do?
* **الإجابة:** يقوم بتحويل النصوص والفئات (Categorical/Text data) إلى **أرقام صحيحة (Integers)**؛ لأن خوارزميات الـ Machine Learning عبارة عن معادلات رياضية ومصفوفات ولا تستطيع معالجة الكلمات النصية مباشرة.

---

### س7: What happens if you didn't encode categorical features?
* **الإجابة:** ستفشل الخوارزمية في التدريب وسيظهر خطأ فوري (`ValueError: could not convert string to float`) ولن تتمكن دالة `model.fit()` من العمل.

---

### س8: Which encoding technique should be used for the following?
1. `["Small", "Medium", "Large"]`
2. `["Poor", "Average", "Good", "Excellent"]`
3. `["Cairo", "Alexandria", "Giza"]`
4. `["Red", "Green", "Blue"]`

* **الإجابة:**
  * للقائمتين (1 و 2): نستخدم **`Ordinal Encoding`** لأن لها **ترتيباً منطقياً ورتبة (Ordered / Ranked)**.
  * للقائمتين (3 و 4): نستخدم **`One-Hot Encoding`** لأنها فئات اسمية **ليس لها ترتيب أو أفضلية (Nominal / Unordered)**.

---

### س9: What is the output of Label Encoding for `["Small", "Medium", "Large", "Small"]`?
* **الإجابة:**
  * `Small` $\rightarrow$ 0
  * `Medium` $\rightarrow$ 1
  * `Large` $\rightarrow$ 2
  * **النتيجة المشفرة:** `[0, 1, 2, 0]`

---

### س10: What is One-Hot Encoding and why do we use it instead of Label Encoding for nominal data?
* **الإجابة:** يقوم بإنشاء **عمود ثنائي منفصل (Binary / Dummy Column)** لكل فئة فريدة، بحيث تأخذ الفئة الحالية القيمة 1 والباقي 0.
* **السبب:** لمنع الموديل من افتراض علاقة رياضية أو ترتيب غير حقيقي (مثل افتراض أن الفئة ذات الرقم 2 أكبر من الفئة ذات الرقم 0).

---

### س11: What is the difference between `StandardScaler` and `MinMaxScaler`?
* **`MinMaxScaler` (Normalization):**
  * القانون: $x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$
  * يحول القيم إلى نطاق محدد بدقة (غالباً بين 0 و 1).
  * **شديد التأثر بالقيم الشاذة (Outliers)**.
* **`StandardScaler` (Standardization):**
  * القانون: $z = \frac{x - \mu}{\sigma}$ (طرح المتوسط $\mu$ والقسمة على الانحراف المعياري $\sigma$).
  * يجعل المتوسط = 0، والتباين والانحراف = 1.
  * **أقل تأثراً بالقيم الشاذة** وهو الأكثر استخداماً في معظم خوارزميات الـ ML.

---

### س12: الكود ده بيعمل إيه وكل Parameter بيعبر عن إيه؟
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
* **الوظيفة:** تقسيم البيانات إلى مجموعتي تدريب (Training set) واختبار (Testing set).
* **المعاملات (Parameters):**
  * `X`: مصفوفة الميزات والمدخلات (Features / Independent variables).
  * `y`: عمود الهدف والنتيجة المراد توقعها (Target / Dependent variable).
  * `test_size=0.2`: نسبة بيانات الاختبار (20% للاختبار و 80% للتدريب).
  * `random_state=42`: قيمة التوليد العشوائي (Seed) لضمان الحصول على **نفس التقسيم في كل مرة يتم تشغيل الكود فيها** (Reproducibility).

---

### س13: Why do we use `StratifiedShuffleSplit` instead of pure random `train_test_split`?
* **الإجابة:** لمنع **تحيز العينة (Sampling Bias)**، وضمان أن نسب وتوزيع الفئات المهمة (مثل فئات الدخل أو الفئات النادرة) في مجموعة الـ Test تكون متطابقة تماماً مع نسبتها في الداتا الأصلية.

---

### س14: Why do we call `fit_transform()` on the training set, but only `transform()` on the test set?
* **الإجابة:** لمنع **تسريب البيانات (Data Leakage)**؛ فدالة `fit` تتعلم المعاملات (المتوسط والانحراف المعياري) من بيانات التدريب فقط، ثم نطبق نفس هذه المعاملات على بيانات الاختبار بـ `transform` دون أن يتعلم الموديل أي معلومة مسبقة عن الـ Test.

---

## 2. النماذج الخطية والانحدار (Linear Regression)

### س15: Linear Regression equation: $y = Wx + b$ (اشرح كل Variable):
* **$y$ (أو $\hat{y}$):** القيمة المتوقعة (Dependent / Output variable).
* **$x$:** الميزة المدخلة (Independent / Input feature).
* **$W$ (Weight / Slope):** ميل الخط المستقيم أو وزن الميزة (يعبر عن مقدار التغير في $y$ عند زيادة $x$ بمقدار وحدة واحدة).
* **$b$ (Bias / Intercept):** نقطة التقاطع مع محور الصادات (قيمة $y$ المتوقعة عندما يكون $x = 0$).

---

### س16: What is the Normal Equation in Linear Regression and what is its main limitation?
* **القانون:** $\hat{\boldsymbol\theta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$
* **الوظيفة:** يعطي الحل الرياضي المباشر والمغلق (Closed-form) لأفضل أوزان تجعل الخطأ أقل ما يمكن.
* **العيب القاتل (Limitation):** التعقيد الحسابي لمقلوب المصفوفة هو **$O(n^{2.4})$ إلى $O(n^3)$**، مما يجعله بطيئاً جداً ومستحيلاً إذا كان عدد الميزات $n$ كبيراً جداً (مثلاً 100,000 ميزة).

---

### س17: قارن بين أنواع الـ Gradient Descent الثلاثة (Batch vs Stochastic vs Mini-batch):
| وجه المقارنة | Batch GD | Stochastic GD (SGD) | Mini-batch GD |
| :--- | :--- | :--- | :--- |
| **عدد العينات في كل خطوة** | كامل الداتا ($m$ عينة) | عينة عشوائية واحدة فقط | حزمة صغيرة (مثلاً 16، 32، 64) |
| **السرعة** | بطيء جداً مع البيانات الضخمة | سريع جداً | سريع ومثالي لكروت الشاشة GPU |
| **استقرار التقارب** | نزول ناعم ومستقر للقاع | متذبذب وعشوائي (Zigzag) | وسط بينهما وأكثر استقراراً |
| **الهروب من الـ Local Minima** | قد يعلق في قاع محلي | **ممتاز في الهروب** بسبب قفزاته | يستطيع الهروب بسلاسة |

---

### س18: What happens if the Learning Rate ($\eta$) in Gradient Descent is too small or too large?
* **إذا كان صغيراً جداً (Too small):** يتطلب آلاف التكرارات ويأخذ وقتاً طويلاً جداً للوصول للحل (Slow convergence).
* **إذا كان كبيراً جداً (Too large):** يقفز فوق القاع الأدنى ويبتعد تماماً ويفشل في الحل (Diverges).

---

### س19: قارن بين أنواع الـ Regularization (Ridge vs Lasso vs Elastic Net):
* **Ridge Regression ($L_2$):**
  * تضيف عقوبة بمربعات الأوزان: $\alpha \sum \theta^2$.
  * تصغر الأوزان وتجعلها قريبة من الصفر **ولكن لا تجعلها صفراً أبداً**.
* **Lasso Regression ($L_1$):**
  * تضيف عقوبة بالقيمة المطلقة للأوزان: $\alpha \sum |\theta|$.
  * **تصفر أوزان الميزات غير المهمة تماماً ($\theta = 0$)** وتقوم بعمل **Feature Selection تلقائي**.
* **Elastic Net:**
  * مزيج يجمع بين Ridge و Lasso بنسبة يحددها المعامل `l1_ratio`.

---

## 3. التصنيف ومقاييس الأداء (Classification & Metrics)

### س20: اشرح مصفوفة الارتباك (Confusion Matrix) وخاناتها الأربعة:
* **True Positive (TP):** توقع إيجابي صحيح (مريض وقال الموديل إنه مريض).
* **True Negative (TN):** توقع سلبي صحيح (سليم وقال الموديل إنه سليم).
* **False Positive (FP):** إنذار كاذب (سليم ولكن الموديل اتهمه بالمرض - خطأ من النوع الأول Type I).
* **False Negative (FN):** تفويت خطير (مريض والموديل قال إنه سليم - خطأ من النوع الثاني Type II).

---

### س21: احفظ القوانين الحسابية الآتية (تأتي كمسألة أرقام):
1. **$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$**
2. **$\text{Precision} = \frac{TP}{TP + FP}$** (من بين كل الحالات التي توقعها الموديل إيجابية، كم حالة صحيحة؟)
3. **$\text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$** (من بين كل الحالات الإيجابية الحقيقية، كم حالة نجح الموديل في صيدها؟)
4. **$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$** (الوسط التوافقي بينهما).

---

### س22: Why is Accuracy misleading for imbalanced / skewed datasets?
* **الإجابة:** لأنه في البيانات غير المتوازنة (مثلاً 95% أصحاء و 5% مصابون بالسرطان)، يمكن لموديل "غبي" أن يتوقع أن الجميع "أصحاء" ويحصل على دقة **95%** دون أن يكتشف مريضاً واحداً! لذلك نحتاج لمقاييس الـ Precision والـ Recall.

---

### س23: متى نفضل Precision عالية ومتى نفضل Recall عالية؟
* **High Precision:** في **فلتر البريد المزعج (Spam Filter)**؛ لأننا لا نريد أن تذهب رسالة عمل مهمة إلى الـ Spam (نتجنب الـ False Positive).
* **High Recall:** في **تشخيص الأمراض الخطيرة (كشف السرطان)**؛ لأن تفويت مريض مصاب واعتباره سليماً يؤدي لوفاته (نتجنب الـ False Negative بأي ثمن).

---

### س24: What is the ROC Curve and what does AUC mean?
* **ROC Curve:** رسم بياني يمثل العلاقة بين:
  * المحور الرأسي (Y-axis): **True Positive Rate (TPR / Recall)**.
  * المحور الأفقي (X-axis): **False Positive Rate (FPR)**.
* **AUC (Area Under the Curve):** المساحة تحت المنحنى:
  * **$AUC = 1.0$:** موديل مثالي.
  * **$AUC = 0.5$:** موديل عشوائي عديم الفائدة (رمية عملة).

---

### س25: ما هي خوارزمية KNN ولماذا تسمى بـ "Lazy Learner"؟
* **KNN (K-Nearest Neighbors):** خوارزمية تصنيف تقيس المسافة بين النقطة الجديدة وأقرب $k$ جيران لها، وتأخذ تصويت الأغلبية.
* **تسمى بـ Lazy Learner لأنه:** لا يوجد تدريب فعلي؛ فدالة `fit()` تقوم فقط بتخزين البيانات في الذاكرة، وكل العمليات الحسابية وحساب المسافات تحدث وقت التوقع `predict()`.
* **تأثير $k$:**
  * $k = 1$ $\rightarrow$ **Overfitting** (حساس للشوشرة).
  * $k$ كبيرة جداً $\rightarrow$ **Underfitting** (تعميم أعمى).

---

### س26: ما هي خوارزمية K-Means وإلى أي نوع من الـ ML تنتمي؟
* تنتمي إلى **التعلم غير الخاضع للإشراف (Unsupervised Learning / Clustering)** لأن البيانات لا تحتوي على تصنيفات صحيحة (No Labels).
* **خطواتها:** اختيار $k$ مراكز عشوائية (Centroids) $\rightarrow$ تنسيب كل نقطة لأقرب مركز $\rightarrow$ إعادة حساب المركز كمتوسط للنقاط $\rightarrow$ التكرار حتى الثبات.
* **تحديد $k$ المثالي:** يتم باستخدام طريقة **الكوع (Elbow Method)** بمراقبة مقياس الـ **Inertia**.

---

## 4. ضبط النماذج والـ Pipelines

### س27: قارن بين `GridSearchCV` و `RandomizedSearchCV`:
* **`GridSearchCV`:** يجرب كل التباديل والتوافيق بين المعاملات المحددة (Exhaustive search)، دقيق ولكنه **بطيء جداً** عند كثرة الاحتمالات.
* **`RandomizedSearchCV`:** يختار عينات عشوائية من المعاملات بعدد تكرارات محدد (`n_iter`)، وهو **أسرع بكثير** ومناسب لمساحات البحث الكبيرة.

---

### س28: كيف نفرق بين Overfitting و Underfitting من خلال الـ Learning Curves؟
* **Underfitting (High Bias):** خطأ الـ Train وخطأ الـ Validation **مرتفعان جداً وقريبان من بعضهما**، وزيادة الداتا لن تحل المشكلة (الحل: استخدام موديل أكثر تعقيداً).
* **Overfitting (High Variance):** خطأ الـ Train منخفض جداً، بينما خطأ الـ Validation مرتفع وبينهما **فجوة واسعة (Gap)**، وزيادة الداتا ستساعد في غلق الفجوة وتحسين الأداء.
