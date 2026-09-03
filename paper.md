# HYBRID META-LABELING AND POLICY OPTIMIZATION WITH REGIME-SWITCHING FOR ADAPTIVE SHORT-TERM TRADING IN THE VIETNAM STOCK EXCHANGE MARKET

**Submitted in partial fulfillment of the requirements for the degree ofBACHELOR OF SCIENCEIN APPLIED MATHEMATICSSPECIALIZATION IN FINANCIAL ENGINEERING AND RISK MANAGEMENT**

**Student's Name:** TONG SON LAM
**Student's ID:** MAMAIU21076

**Ho Chi Minh City, Vietnam Jan 2026**

# Abstract

This thesis proposes an integrated framework that combines meta-labeling, reinforcement learning (PPO), and mixture modeling to address these challenges. Meta-labeling, introduced by López de Prado (2018), is used to filter base classifiers and retain only high-confidence signals. Proximal Policy Optimization (PPO) serves as an adaptive meta-model, learning when to trust filtered signals and how to optimize execution policies through interaction with the market environment. Finally, mixture models, including Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM), are incorporated to capture regime dependence and structural breaks, enabling robustness under shifting volatility, momentum, and correlation structures.

By decoupling signal generation from execution, calibrating model confidence, and embedding regime awareness, the proposed framework improves both predictive reliability and risk-adjusted returns. The results contribute to the development of more resilient systematic trading strategies that can adapt dynamically to the evolving complexity of financial markets.

# Chapter 1: Introduction

Financial markets present unique challenges for predictive modeling due to their non-stationary nature, heavy-tailed distributions, and regime-dependent dynamics [1]. Traditional statistical and machine learning models—such as logistic regression or random forests—can produce directional forecasts but often struggle to distinguish between signals that are actionable and those that are misleading. Direct reliance on these outputs can lead to frequent false positives, unnecessary trading costs, and poor risk-adjusted performance [4].

Reinforcement learning (RL) offers an alternative by framing trading as a sequential decision-making process [58]. Among RL methods, Proximal Policy Optimization (PPO) has gained attention for its stability and policy-based optimization framework [2]. However, RL approaches are still sensitive to noise and may underperform when exposed to structural breaks or heterogeneous market regimes [3].

To address these limitations, this thesis proposes a layered framework that integrates three complementary components:

- **Meta-labeling:** serving as a filter to validate the reliability of base classifier outputs [5].
- **Reinforcement learning (PPO):** functioning as a meta-model to determine when to trust signals and how to act adaptively [2].
- **Mixture modeling:** capturing regime shifts through approaches such as Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM) [59, 60].

The motivation for this integration arises from three key observations:

1. **Not all signals are equally valuable** - meta-labeling reduces false positives by validating trades before execution [5].
2. **Markets are sequential and adaptive** - PPO introduces flexibility by learning optimal responses in real time [58].
3. **Markets evolve through regimes** - mixture models provide robustness by accounting for structural breaks and regime heterogeneity [59].

By combining these components, the thesis aims to decouple prediction from execution, calibrate model confidence, and embed regime awareness. This layered design provides a pathway toward trading systems that are not only more accurate but also more resilient under dynamic market conditions.

## Motivation

The motivation for this research arises from three fundamental observations about financial prediction and trading strategy design.

First, not all predictive signals are equally valuable. Primary classifiers often generate forecasts with varying levels of reliability. A direct application of these signals can result in frequent false positives and unnecessary trading costs. Meta-labeling provides a principled method to separate signal generation from signal validation, ensuring that only those trades with a high likelihood of profitability are executed [4].

Second, financial decision-making is inherently sequential and adaptive. Market conditions evolve continuously, and static classifiers cannot fully capture this dynamic nature. Reinforcement learning, particularly PPO, introduces a policy-based optimization framework where the model adapts its decision-making strategy through interaction with the environment [2]. By using PPO as a meta-model, the system can learn when to trust primary model outputs and how to adjust execution strategies dynamically.

Third, markets are not governed by a single regime. Empirical evidence consistently shows that asset returns exhibit regime dependence, clustering, and structural breaks. Mixture models, such as Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM), provide a natural way to capture these heterogeneities [59, 60]. Incorporating mixture-based extensions allows the proposed framework to remain robust under varying volatility, momentum, and correlation structures, thereby addressing one of the most critical challenges in financial forecasting.

Together, these motivations highlight the need for a layered framework: (i) meta-labeling to filter predictions, (ii) PPO to adaptively act on signals, and (iii) mixture models to account for regime heterogeneity. This integration not only enhances predictive reliability but also improves execution efficiency and robustness, paving the way for more resilient systematic trading strategies.

# Chapter 2: Meta-Labeling

This concept of meta-labeling in quantitative finance is explained in Wikipedia [72], which separates the prediction of trade direction and the decision to act (i.e. filtering or sizing).

Meta-labeling is a second-layer modeling technique introduced by López de Prado [4] in Advances in Financial Machine Learning. Unlike traditional approaches that discard weak or noisy base models, meta-labeling leverages them as feature generators for a secondary predictive stage. The central principle is the separation of prediction from action: the primary model provides directional or probabilistic forecasts, while the secondary, or meta-model, evaluates whether these forecasts should be acted upon. This separation enables the construction of more robust systematic trading strategies by filtering out unprofitable signals and dynamically adjusting position sizing.

Meta-labeling introduces an additional layer of decision-making in systematic trading strategies, enabling practitioners to separate the task of forecasting from that of execution. By combining a primary model for directional prediction with a secondary classifier for profitability assessment, and by translating the latter's output into calibrated position sizes, this framework improves precision, reduces false positives, and enhances risk-adjusted returns [5].

## 2.1 Conceptual Framework

In its simplest form, meta-labeling can be described as follows. Let $f(x)$ denote the prediction from a base model, where $x$ represents the input features. The realized meta-label $y_m$ is defined as:

$$y_m = \begin{cases} 1, & \text{if a trade based on } f(x) \text{ would have been profitable,} \\ 0, & \text{otherwise.} \end{cases}$$

Alternatively, when $f(x)$ corresponds to a binary directional prediction (e.g., Buy = 1, Sell = -1), the meta-label can be expressed as:

$$y_m = \begin{cases} 1, & f(x) \times r_{t+1} > 0, \\ 0, & f(x) \times r_{t+1} < 0, \end{cases}$$

where $r_{t+1}$ is the realized return at time $t+1$. The meta-model $g(x, f(x))$ is then trained to predict $y_m$, learning under which conditions the base model's output is reliable:

$$\hat{y}_m = g(x, f(x))$$

A trading decision is executed only if $\hat{y}_m = 1$, thereby filtering out low-quality forecasts and improving the precision of executed trades [4].

## 2.2 General Architecture

The architecture of meta-labeling can be conceptualized as a three-stage process consisting of: (i) the primary model (M1), (ii) the secondary model (M2), and (iii) the position sizing algorithm (M3) [4].

### 2.2.1 Primary Model (M1)

The first stage involves training a base model to forecast the directional movement of an asset. This model may employ classical machine learning algorithms such as logistic regression, random forests, support vector machines, or ensemble methods such as XGBoost, as well as more sophisticated reinforcement learning agents [58, 7]. The output is typically a side prediction $Y \in \{-1, 0, 1\}$, where -1 denotes a short position, 0 represents neutrality or abstention, and 1 corresponds to a long position.

To enhance the reliability of predictions, practitioners often employ a recall check by introducing a minimum threshold $\tau$, such that forecasts are only issued when the model's conviction surpasses this threshold. Consequently, the primary model produces not only directional signals but also diagnostic metrics such as rolling accuracy, F1-score, recall, precision, and AUC, which can serve as auxiliary features for the subsequent stage.

### 2.2.2 Secondary Model (M2)

The second stage introduces a binary classifier, termed the meta-model, which determines whether a forecast generated by the primary model is likely to result in a profitable trade. The target variable is the meta-label $F \in \{0, 1\}$, as defined earlier. The feature space of M2 may encompass:

1. General input data predictive of false positives, such as rolling volatility.
2. Performance diagnostics from M1 (e.g., precision, recall).
3. Market state or regime indicators, as trading strategies often exhibit regime dependence [59, 60].
4. The raw output of the primary model, reflecting the strength of its conviction.

The secondary model outputs a probability $\hat{p} \in [0, 1]$, which reflects the estimated likelihood that the trade will be profitable. Depending on the activation function used, this may take the form of a sigmoid output (yielding values between 0 and 1) or a hyperbolic tangent (tanh) transformation (yielding values between -1 and 1). This probabilistic assessment provides a filter to discard unprofitable signals and serves as an input for position sizing.

### 2.2.3 Position Sizing Algorithm (M3)

The third stage translates the output probability of the meta-model into a position size. Several approaches exist for this transformation:

- **All-or-nothing:** allocate full capital if $p$ exceeds a threshold (e.g., 0.5), otherwise abstain.
- **Model confidence:** allocate capital proportionally to $p$.
- **Linear scaling:** apply min-max normalization to rescale probabilities.
- **Normal CDF (NCDF):** map the predicted probability to a cumulative distribution value based on a z-statistic.
- **Empirical CDF (ECDF):** rank probabilities according to their percentile in the training set.
- **Sigmoid Optimal Position Sizing (SOPS):** apply a non-linear sigmoid transformation optimized for maximizing risk-adjusted returns, such as the Sharpe ratio [4].

# Chapter 3: The First Layer - Classification Model

## 3.1 Optimal Feature Selection of Technical Indicators for Stock Prediction using Machine Learning Techniques

### 3.1.1 Introduction

In financial forecasting, particularly in stock price prediction, the selection of relevant technical indicators plays a pivotal role in model performance. A large set of indicators often contains redundant, irrelevant, or highly correlated features, which can deteriorate generalization capability, increase computational cost, and complicate interpretation.

To mitigate these issues, this research proposes an **Optimal Feature Selection Framework** designed to systematically identify the most informative subset of technical indicators for stock prediction tasks. The framework adopts a **model-oriented perspective**—meaning feature selection is evaluated and optimized within the context of specific machine learning models such as Random Forest [10], Logistic Regression [11], K-Nearest Neighbors (KNN) [12], and XGBoost [61].

This approach integrates multiple selection techniques and leverages empirical performance evaluation to determine the optimal set of indicators that yield the best predictive accuracy.

### 3.1.2 Theoretical Framework

Let the dataset be denoted as:

$$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N, \quad x_i \in \mathbb{R}^p, \quad y_i \in \{0, 1\}$$

where $x_i$ represents the vector of $p$ technical indicators for observation $i$, and $y_i$ is the binary stock movement label (e.g., up or down).

A feature subset $F_k \subseteq \{1, 2, ..., p\}$ is defined as a specific combination of indicators selected for model evaluation. For a given machine learning model $\mathcal{M}$, the performance function $\mathcal{J}(\mathcal{M}, F_k)$ is defined as the expected predictive performance (e.g., F1-score, accuracy) estimated via cross-validation.

Formally, the objective of optimal feature selection can be expressed as:

$$F^* = \arg\max_{F_k \in \mathcal{F}} \mathbb{E}[\mathcal{J}(\mathcal{M}, F_k)]$$

where:

- $\mathcal{F}$ is the set of all candidate feature subsets generated by the framework,
- $\mathbb{E}[\cdot]$ denotes the cross-validated performance expectation.

This optimization problem seeks the subset $F^*$ that maximizes model performance under a given evaluation metric.

### 3.1.3 Initial Feature Set Construction

The first stage involves constructing a comprehensive pool of technical indicators derived from market price and volume data. A total of **91 indicators** were generated, covering diverse categories such as trend-based, momentum-based, volatility, and volume indicators.

This extensive set ensures a rich feature space that supports diverse selection mechanisms in subsequent stages. It provides redundancy intentionally, so that the framework can later determine which indicators truly contribute to predictive accuracy.

### 3.1.4 Application of Individual Feature Selection Methods

Each feature selection technique is applied independently to the full feature set to produce an initial group of candidate sub-feature sets. The framework employs three categories of methods:

1. **Filter-Based Methods:**
    - Statistical tests such as Variance Threshold, Correlation Analysis, and Mutual Information (MI) [13] are used to measure the individual relevance of each feature with respect to the target variable.
2. **Wrapper-Based Methods:**
    - Techniques like Recursive Feature Elimination (RFE) [14] iteratively train the model, removing the least important features based on validation performance.
3. **Embedded Methods:**
    - Algorithms with built-in feature selection mechanisms—such as Random Forest, XGBoost, or L1-Regularized Logistic Regression—are used to extract feature importance scores during training.

Each method outputs a subset $F_i$ of selected features, contributing to the candidate pool $\mathcal{F} = \{F_1, F_2, ..., F_n\}$.

### 3.1.5 Combinatorial Subset Generation

To explore the interaction between different feature selection techniques, the framework expands the candidate pool using combinatorial operations:

**Union (Combination):**

$$F_u = F_i \cup F_j$$

Combines features from multiple methods to include diverse signals.

**Intersection (Consensus):**

$$F_c = F_i \cap F_j$$

Retains only features commonly selected by multiple methods, forming a conservative subset.

These derived subsets help balance between diversity and robustness, broadening the search space for the optimal configuration.

### 3.1.6 Sub-Feature Packaging and Evaluation Pipeline

All candidate subsets (individual and combined) are compiled into a feature package, each representing a hypothesis about the optimal configuration. For each subset $F_k$, the corresponding model $\mathcal{M}$ is trained and evaluated using K-Fold Cross-Validation (CV) [15].

For a given fold $t \in \{1, ..., k\}$, performance is computed as:

$$\mathcal{J}_t(\mathcal{M}, F_k) = \text{Score}(\mathcal{M}(\mathcal{D}_{train}^{(t)}; F_k), \mathcal{D}_{test}^{(t)}, F_k)$$

and the overall performance estimate is:

$$\mathbb{E}[\mathcal{J}(\mathcal{M}, F_k)] = \frac{1}{k} \sum_{t=1}^k \mathcal{J}_t(\mathcal{M}, F_k)$$

The subset achieving the maximum expected performance across folds is chosen as the optimal set $F^*$.

### 3.1.7 Computational Efficiency and the Cross-Valid Package

Given the large number of subsets and models, exhaustive cross-validation can be computationally expensive. To address this, a lightweight evaluation module termed the **Cross-Valid Package** is introduced.

This alternative performs an approximate validation using simplified or stratified sampling, offering a trade-off between computational cost and evaluation fidelity. It can be used as:

- A pre-screening step to identify promising feature subsets before full K-Fold validation, or
- A substitute in resource-constrained environments, where computational efficiency is prioritized.

### 3.1.8 Optimal Subset Selection and Final Output

The framework concludes by selecting the best-performing feature subset $F^*$, formally defined as:

$$F^* = \arg\max_{F_k \in \mathcal{F}} \mathbb{E}[\mathcal{J}(\mathcal{M}, F_k)]$$

This subset of technical indicators represents the optimal trade-off between dimensionality reduction and predictive accuracy for the given model. It is then used for final training and deployment within the trading or stock prediction system.

### 3.1.9 Conclusion

The optimal feature selection framework offers a structured and performance-driven approach to identifying relevant technical indicators for stock market prediction. By integrating multiple selection methods, combining their outputs, and relying on rigorous model-based evaluation, it ensures that the chosen features are not only statistically significant but also functionally aligned with the model's predictive behavior.

Although computationally demanding, the inclusion of the Cross-Valid Package enhances practicality and scalability, making this framework suitable for high-dimensional, dynamic financial environments. Ultimately, this methodology improves both the interpretability and predictive reliability of machine learning models applied to stock forecasting tasks.

## 3.2 Logistic Regression: A Probabilistic Approach

The explanation of the Expectation-Maximization (EM) algorithm in this section is adapted from the materials published on Machine Learning Cơ Bản [71], which served as a foundational resource in understanding and reconstructing the derivation.

### 3.2.1 Incorporating Explanatory Variables into Binary Outcomes

The Bernoulli distribution studied earlier answers the question of which of two outcomes $y \in \{0, 1\}$ would be selected with probability $p$:

$$\mathbb{P}(y) = p^y(1-p)^{1-y}$$

Traditionally, parameter estimation of $p$ relies on maximum likelihood estimation (MLE) using observed binary data $\{y_i\}_{i=1}^n$. However, in practical scenarios, the probability of success may depend on additional observed features. Suppose each binary outcome $y_i$ is associated with a continuous predictor $x_i$, resulting in a dataset $\{(x_i, y_i)\}_{i=1}^n$. The challenge is to incorporate this explanatory variable into the probability estimate of $p$.

A naive linear model $p = ax + b$ is inappropriate, since $p \in [0, 1]$ while a linear function is unbounded. To resolve this, we introduce a link function that maps real values to probabilities. The logistic (sigmoid) function is particularly suitable:

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \lim_{x \to -\infty} \sigma(x) = 0, \quad \lim_{x \to +\infty} \sigma(x) = 1.$$

Thus, the probability of success is parameterized as

$$\hat{p} = \sigma(ax + b) = \frac{1}{1 + e^{-(ax + b)}}$$

The associated inverse function, the logit transform, extracts the linear predictor:

$$\text{logit}(p) = \log\left(\frac{p}{1-p}\right) = ax + b.$$

More generally, for multiple predictors $x_k$, the logit model becomes

$$\text{logit}(p) = b + \sum_k a_k x_k.$$

This formulation leads to logistic regression, a widely used model for classification problems [11].

### 3.2.2 Likelihood Formulation and Loss Function

For a training dataset $x = [x_1, ..., x_N] \in \mathbb{R}^{d \times N}$ with labels $y = [y_1, ..., y_N]$, the model assumes

$$\mathbb{P}(y_i = 1 \vert{} x_i; w) = f(w^\top x_i), \quad \mathbb{P}(y_i = 0 \vert{} x_i; w) = 1 - f(w^\top x_i),$$

where $f(\cdot)$ is the sigmoid function and $w$ is the parameter vector. For compactness, define $z_i = f(w^\top x_i)$. The probability of observing $y_i$ is then

$$\mathbb{P}(y_i \vert{} x_i; w) = z_i^{y_i}(1 - z_i)^{1-y_i}.$$

Assuming independent and identically distributed (i.i.d.) data, the joint likelihood across the dataset is

$$\mathbb{P}(y \vert{} x; w) = \prod_{i=1}^N z_i^{y_i}(1 - z_i)^{1-y_i}.$$

Direct optimization of this product is numerically unstable for large $N$, so we maximize the log-likelihood instead. Equivalently, we minimize the negative log-likelihood (NLL):

$$\mathcal{L}(w) = -\sum_{i=1}^N [y_i \log z_i + (1 - y_i) \log(1 - z_i)].$$

This is also known as the binary cross-entropy loss, a standard objective in classification tasks [8].

### 3.2.3 Gradient and Optimization

For a single training instance $(x_i, y_i)$, the loss function is

$$\mathcal{L}(w; x_i, y_i) = -[y_i \log z_i + (1 - y_i) \log(1 - z_i)],$$

where $z_i = \sigma(w^\top x_i)$. Differentiating with respect to $w$ yields

$$\frac{\partial\mathcal{L}(w; x_i, y_i)}{\partial w} = (z_i - y_i)x_i.$$

This gradient has a simple interpretation: it is proportional to the prediction error $(z_i - y_i)$ scaled by the input $x_i$.

To optimize the weights, Stochastic Gradient Descent (SGD) is typically employed. The update rule for each sample is

$$w \leftarrow w + \eta (y_i - z_i) x_i$$

where $\eta$ is the learning rate. Despite its simplicity, this iterative procedure effectively learns the model parameters and remains widely adopted in practice [46].

## 3.3 Random Forest (RF) Classifier

### 3.3.1 Model Construction

**Tree-Structured Models**
In classification problems where the output space is a finite set of discrete values, $Y = \{c_1, c_2, ..., c_J\}$, the supervised learning problem can be interpreted as learning a partition of the universe $\Omega$ based on these target classes:

$$\Omega = \Omega_{c_1} \cup \Omega_{c_2} \cup ... \cup \Omega_{c_J}$$

where $\Omega_{c_k}$ represents the set of objects for which the response variable $Y$ takes the value $c_k$.

Similarly, a classifier $\phi$ defines an approximation $\hat{Y}$ of $Y$, which can also be viewed as a partition of the input space $\mathcal{X}$:

$$\mathcal{X} = \mathcal{X}_{c_1}^\phi \cup \mathcal{X}_{c_2}^\phi \cup ... \cup \mathcal{X}_{c_J}^\phi$$

where $\mathcal{X}_{c_k}^\phi$ denotes the subset of feature vectors $(x \in \mathcal{X})$ such that $\phi(x) = c_k$.

The optimal classifier, known as the Bayes model $\phi_B$, defines the ideal partition of $\mathcal{X}$:

$$\mathcal{X} = \mathcal{X}_{c_1}^{\phi_B} \cup \mathcal{X}_{c_2}^{\phi_B} \cup ... \cup \mathcal{X}_{c_J}^{\phi_B}$$

In this context, the learning task involves approximating this Bayes-optimal partition as closely as possible.

A decision tree can thus be defined as a model $\phi : \mathcal{X} \rightarrow Y$, represented by a rooted, typically binary tree. Each internal node $t$ in the tree corresponds to a subspace $\mathcal{X}_t \subseteq \mathcal{X}$, and is associated with a split rule $s_t \in \mathcal{Q}$, where $\mathcal{Q}$ denotes the set of candidate split functions or questions. The split $s_t$ divides the node's subspace $\mathcal{X}_t$ into disjoint subsets corresponding to the node's children, thereby recursively partitioning the feature space [47].

### 3.3.2 Random Forest Classifier Construction

The Random Forest classifier extends the decision tree model by constructing an ensemble of $N$ decision trees, each trained independently on a bootstrapped sample drawn from the training dataset [10]. During training, at each node split, a random subset of features $m$ (where $m < p$, with $p$ denoting the total number of available features) is selected, and the optimal split is determined among these features. This randomization procedure enhances model robustness and reduces correlation between individual trees.

Each decision tree $T_i$ produces a class prediction for a given input $x$, and the final prediction of the Random Forest is obtained by majority voting across all trees:

$$\phi(x) = \text{Majority Vote}\{T_1(x), T_2(x), ..., T_N(x)\},$$

where $T_i(x) = \hat{y}_i \in \{c_1, c_2, ..., c_J\}$.

In regression tasks, the Random Forest outputs the average of the individual tree predictions:

$$\phi(x) = \frac{1}{N} \sum_{i=1}^N T_i(x).$$

This ensemble averaging mechanism contributes to a significant reduction in variance compared to single decision trees, yielding improved generalization performance.

### 3.3.3 Variance Reduction in Ensemble Learning

Let each base model (tree) have variance $\sigma^2$ and an average pairwise correlation $\rho$. The variance of the ensemble's aggregated prediction can be expressed as:

$$\text{Var}_{\text{ensemble}} = \rho\sigma^2 + \frac{1-\rho}{N}\sigma^2.$$

From this expression, it is evident that as the number of trees $N$ increases, the variance term associated with independent errors $\frac{1-\rho}{N}\sigma^2$ decreases. Furthermore, the introduction of random feature selection in Random Forests reduces the correlation $\rho$ between individual trees, further minimizing the overall ensemble variance [48].

## 3.4 Extreme Gradient Boosting (XGBoost)

### 3.4.1 Model Framework

In supervised learning, a model represents the mathematical relationship that maps input vectors $x_i$ to predictions $\hat{y}_i$. For instance, a linear model can be expressed as:

$$\hat{y}_i = \sum_j \theta_j x_{ij}$$

where $\theta_j$ denotes the model parameters (weights). The prediction $\hat{y}_i$ can take various forms depending on the problem—such as probabilities in logistic regression or continuous values in regression.

XGBoost generalizes this principle by building an additive model composed of multiple regression trees:

$$\hat{y}_i^{(t)} = \sum_{k=1}^t f_k(x_i), \quad f_k \in \mathcal{F}$$

where $\mathcal{F}$ is the space of regression trees, each $f_k$ representing a weak learner added sequentially to minimize the overall loss.

### 3.4.2 Ensemble Learning and Boosting

Ensemble learning combines multiple models to improve generalization and predictive performance [49]. Two major ensemble paradigms exist:

- **Bagging (Bootstrap Aggregating):** Constructs independent models on random subsamples of data to reduce variance, as seen in algorithms like Random Forests.
- **Boosting:** Builds models sequentially, where each new model corrects the errors of the previous one, thereby reducing bias.

AdaBoost [50] was one of the earliest practical boosting algorithms. It assigns higher weights to misclassified instances, forcing subsequent learners to focus on difficult cases. XGBoost extends this idea by performing gradient boosting, where weak learners are trained to fit the negative gradients (residual errors) of the loss function.

### 3.4.3 Mathematical Foundation

The optimization objective of XGBoost at iteration $t$ is defined as:

$$\mathcal{L}^{(t)}(\theta) = L(\theta) + \Omega(\theta), \quad (3.1)$$

where:

- $L(\theta)$ is the training loss measuring prediction error,
- $\Omega(\theta)$ is the regularization term penalizing model complexity.

**Training Loss**
For regression tasks:

$$L(\theta) = \sum_i (y_i - \hat{y}_i)^2. \quad (3.2)$$

For logistic regression (classification):

$$L(\theta) = \sum_i [y_i \ln(1+e^{-\hat{y}_i}) + (1-y_i) \ln(1+e^{\hat{y}_i})]. \quad (3.3)$$

In general:

$$L(\theta) = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) \quad (3.4)$$

where $l(\cdot)$ represents a differentiable loss function and $\hat{y}_i^{(t)}$ is the model prediction after $t$ boosting iterations.

**Additive Model Update**
XGBoost builds its model additively by adding one tree per iteration to minimize the loss:

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i). \quad (3.5)$$

Substituting this into the objective gives:

$$\mathcal{L}^{(t)} = \sum_i l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t). \quad (3.6)$$

Direct optimization is intractable, so XGBoost employs a second-order Taylor expansion around $\hat{y}_i^{(t-1)}$ [51]:

$$\mathcal{L}^{(t)} \approx \sum_i [l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t) \quad (3.7)$$

where:

$$g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2} \quad (3.8)$$

This enables XGBoost to exploit both first-order ($g_i$) and second-order ($h_i$) gradient information, resulting in faster convergence and improved accuracy.

**Regularization**
The regularization term controls model complexity:

$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2, \quad (3.9)$$

where:

- $T$ is the number of leaves,
- $w_j$ is the weight of leaf $j$,
- $\gamma$ and $\lambda$ are hyperparameters controlling tree complexity.

Explicit regularization distinguishes XGBoost from other gradient boosting methods, as it effectively mitigates overfitting while improving generalization [61].

**Optimal Leaf Weights and Objective Reduction**
For a given tree structure $q(x)$ assigning each sample to a leaf $j$, the optimal leaf weight is:

$$w_j^* = -\frac{G_j}{H_j + \lambda}, \quad (3.10)$$

where:

$$G_j = \sum_{i \in I_j} g_i, \quad H_j = \sum_{i \in I_j} h_i. \quad (3.11)$$

The corresponding optimal objective value is:

$$\mathcal{L}^* = -\frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j + \lambda} + \gamma T. \quad (3.12)$$

This efficient formulation enables the algorithm to evaluate potential tree splits effectively, guiding optimal tree construction.

## 3.5 K-Nearest Neighbor (KNN)

### 3.5.1 Voting Schemes

Given a labeled dataset:

$$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n, \quad x_i \in \mathbb{R}^d, \quad y_i \in \mathcal{Y} \quad (3.13)$$

and a query point $x \in \mathbb{R}^d$, the k-NN classifier estimates the label $\hat{y}$ based on the majority class among its nearest neighbors.

Classification:

$$\hat{y} = \arg\max_{c \in \mathcal{Y}} \sum_{i \in \mathcal{N}_k(x)} \mathbb{I}(y_i = c) \quad (3.14)$$

Regression:

$$\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i \quad (3.15)$$

where $\mathcal{N}_k(x)$ denotes the indices of the $k$ nearest samples to $x$, and

$$\mathbb{I}(\text{condition}) = \begin{cases} 1, & \text{if the condition holds.} \\ 0, & \text{otherwise.} \end{cases} \quad (3.16)$$

### 3.5.2 Distance Metrics

The effectiveness of k-NN is heavily influenced by the choice of the distance metric $d(x, x')$. Commonly used metrics include:

Euclidean Distance (L2 norm):

$$d(x, x') = \sqrt{\sum_{j=1}^d (x_j - x_j')^2} \quad (3.17)$$

Manhattan Distance (L1 norm):

$$d(x, x') = \sum_{j=1}^d |x_j - x_j'| \quad (3.18)$$

Minkowski Distance:

$$d(x, x') = \left(\sum_{j=1}^d |x_j - x_j'|^p\right)^{1/p} \quad (3.19)$$

Mahalanobis Distance:

$$d(x, x') = \sqrt{(x - x')^\top \Sigma^{-1} (x - x')} \quad (3.20)$$

which accounts for feature correlations through the covariance matrix $\Sigma$.

### 3.5.3 Weighting Schemes

To improve predictive performance—especially in non-uniform data distributions—weighted k-NN assigns higher influence to closer neighbors:

$$\hat{y} = \arg\max_{c \in \mathcal{Y}} \sum_{i \in \mathcal{N}_k(x)} w_i \cdot \mathbb{I}(y_i = c)$$$$w_i = \frac{1}{d(x, x_i) + \epsilon}, \quad (3.21)$$

where $\epsilon > 0$ prevents division by zero.

### 3.5.4 Theoretical Properties

The k-NN algorithm has several desirable theoretical guarantees:

**Consistency.** Under suitable conditions, k-NN is universally consistent [12, 52]:
If $k \rightarrow \infty$, $\frac{k}{n} \rightarrow 0$ then $\mathbb{P}(\hat{y} \neq y) \rightarrow \text{Bayes error}. \quad (3.22)$
This implies that its expected classification error approaches the minimum possible (Bayes optimal) error as $n$ increases.

**Curse of Dimensionality.** As the number of dimensions $d$ increases, the distinction between near and far points diminishes:
As $d \rightarrow \infty$, $\frac{|x - x_{\text{nearest}}|}{|x - x_{\text{farthest}}|} \rightarrow 1 \quad (3.23)$
leading to performance degradation due to sparse high-dimensional feature spaces.

### 3.5.5 Algorithmic Steps

1. **Compute Distances:** Evaluate the distances between the query point and all training points using a selected metric $d(x, x')$.
2. **Select Neighbors:** Identify the set $\mathcal{N}_k(x)$ containing the $k$ samples with the smallest distances.
3. **Aggregate Responses:** Perform majority voting (for classification) or averaging (for regression) to determine the final prediction $\hat{y}$.

# Chapter 4: Bayesian Optimization for Hyperparameter Tuning

## 4.1 Motivation and Positioning in the Thesis

Hyperparameter optimization is essential for ensuring that machine learning classifiers (M1-layer) generalize well and produce stable, economically useful signals for downstream reinforcement learning.

Conventional methods such as grid search or random search are inefficient for expensive, noisy, and conditional hyperparameter spaces.

Bayesian Optimization (BO) frames hyperparameter tuning as a black-box optimization problem and uses a probabilistic surrogate model to efficiently guide evaluations, balancing exploration and exploitation.

BO has become a standard approach for automating hyperparameter search and has strong theoretical and empirical support in machine learning literature [53, 62].

## 4.2 Mathematical Problem Statement

Let $\theta \in \Theta$ denote a hyperparameter vector for a given M1-layer classifier (e.g., $\theta = \{\text{n\_estimators}, \text{max\_depth}, ...\}$). Define the objective function $f : \Theta \rightarrow \mathbb{R}$ as the performance metric obtained via time-aware cross-validation (CV) on historical data:

$$f(\theta) = \text{Perf}_{CV}(\theta) \quad (\text{e.g., weighted combination of F1 and Sharpe ratios}). \quad (4.1)$$

The goal is to find:

$$\theta^* = \arg\max_{\theta \in \Theta} f(\theta). \quad (4.2)$$

Key challenges include:

- $f(\theta)$ is expensive to evaluate (each evaluation requires model training and validation),
- $f(\theta)$ is noisy due to CV variance and market non-stationarity,
- $\Theta$ contains continuous, integer, and categorical parameters with possible conditional dependencies.

BO addresses these challenges by building a probabilistic surrogate $\hat{f}$ and using an acquisition function $\alpha$ to propose promising hyperparameters sequentially [62].

## 4.3 Surrogate Models

### 4.3.1 Gaussian Process (GP) Surrogate

A Gaussian Process prior over functions is defined as:

$$f(\theta) \sim \mathcal{GP}(m(\theta), k(\theta, \theta')), \quad (4.3)$$

where $m(\theta)$ is the mean function (often zero) and $k$ is the covariance kernel.

Given observations $\mathcal{D}_t = \{(\theta_i, y_i)\}_{i=1}^t$, where $y_i = f(\theta_i) + \epsilon_i$ and $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$, the GP posterior mean and variance at a candidate $\theta$ are:

$$\mu_t(\theta) = k(\theta, \Theta_t)[K_t + \sigma_n^2I]^{-1}y_t \quad (4.4)$$$$\sigma_t^2(\theta) = k(\theta, \theta) - k(\theta, \Theta_t)[K_t + \sigma_n^2I]^{-1}k(\Theta_t, \theta), \quad (4.5)$$

where $K_t$ is the kernel matrix on $\Theta_t = \{\theta_1, ..., \theta_t\}$.

Typical kernel choices include the Matérn ($\nu = 5/2$) and RBF kernels with automatic relevance determination (ARD) [55].

### 4.3.2 Alternative Surrogates

- **Tree-structured Parzen Estimator (TPE)** models $p(\theta \vert{} y)$ non-parametrically, splitting observations into "good" and "bad" sets to improve performance with categorical and conditional variables [56].
- **Random forests and gradient-boosted trees** serve as nonparametric alternatives for large or mixed-type spaces.

## 4.4 Acquisition Functions

Given the surrogate posterior $(\mu_t(\theta), \sigma_t(\theta))$, the acquisition function $\alpha(\theta)$ balances exploration and exploitation.

(a) **Expected Improvement (EI):**

$$EI(\theta) = \mathbb{E}[\max(0, f(\theta) - y^+)] = (\mu_t(\theta) - y^+)\Phi(z) + \sigma_t(\theta)\phi(z), \quad (4.6)$$

where $y^+ = \max_{i \leq t} y_i$ and $z = \frac{\mu_t(\theta) - y^+}{\sigma_t(\theta)}$.

(b) **Upper Confidence Bound (UCB):**

$$UCB_\kappa(\theta) = \mu_t(\theta) + \kappa \cdot \sigma_t(\theta), \quad (4.7)$$

where $\kappa > 0$ controls the trade-off between exploration and exploitation.

(c) **Thompson Sampling:** Draw a function from the surrogate posterior and choose the maximizer, often effective for parallel BO setups [63, 57].

## 4.5 Practical Algorithm (BO Loop) for M1-layer

**Input:** Search space $\Theta$, budget $T$, initial design size $t_0$, CV routine for $f(\theta)$.
**Output:** Optimal parameters $\theta^*$ and evaluation history $\mathcal{D}_T$.

1. Initialize with $t_0$ points $\{\theta_i\}$ and evaluate $y_i = f(\theta_i)$ via time-series CV.
2. Fit the surrogate model to $\mathcal{D}_t$.
3. Optimize $\alpha(\theta | \mathcal{D}_t)$ to propose the next candidate $\theta_{t+1}$.
4. Evaluate $y_{t+1} = f(\theta_{t+1})$ and append $(\theta_{t+1}, y_{t+1})$ to $\mathcal{D}_{t+1}$.
5. Repeat until budget $T$ or convergence.
6. Return $\theta^* \approx \arg\max_{\theta \in \mathcal{D}_T} y_i$.

**Notes:**

- Use time-series aware CV (rolling or expanding windows) to avoid lookahead bias.
- Treat evaluations as noisy with variance $\sigma_n^2$ in the GP model.
- For parallel settings, use batch BO (q-EI) or asynchronous methods [63].

## 4.6 Objective Design for Finance

### 4.6.1 Composite Objective

A scalar objective combining classification and economic metrics:

$$f(\theta) = \lambda F1_{CV}(\theta) + (1-\lambda) \text{Sharpe}_{CV}(\theta), \quad (4.8)$$

where $\lambda \in [0, 1]$ balances predictive and financial goals.

### 4.6.2 Multi-objective BO

Alternatively, treat F1 and Sharpe as separate objectives and use multi-objective BO to obtain Pareto-efficient hyperparameter sets [62].

## 4.7 Handling Practical Complexities

(a) **Categorical and Conditional Parameters.** Use TPE or tree-based surrogates to handle conditional and categorical hyperparameters naturally [56].
(b) **High Dimensionality.** Use dimensionality reduction or hybrid search (random + BO). For large spaces, approximate or sparse GPs improve scalability.
(c) **Noisy Evaluations and Heteroscedasticity.** Model heteroscedastic noise by extending GP assumptions or using robust surrogates.
(d) **Multi-fidelity and Cost-aware BO.** Incorporate computational cost into acquisition optimization or use low-fidelity evaluations (e.g., fewer CV folds, reduced data) [63].

# Chapter 5: Extension Technique - Mixtures of Distributions Concept

In this work, we adopt an "extension technique" perspective based on mixtures of distributions. Mixture models represent the observed data density as a convex combination of component densities, allowing explicit modeling of latent sub-populations or regimes [24]. For time series, static mixtures (e.g., Gaussian Mixture Models) lack temporal persistence and can assign states independently for each observation; therefore, we consider dynamic extensions—Hidden Markov Models and Markov-switching formulations—that impose Markovian transitions on the latent state process and thus model regime persistence and transition dynamics more realistically [25, 9]. Estimation is performed using EM for finite mixtures and maximum likelihood / Bayesian methods for switching models; Bayesian nonparametric alternatives (Dirichlet Process mixtures) provide a principled route to let the data determine the number of mixture components [26]. These mixture and switching approaches are particularly relevant to our classification problem because they enable regime-conditional classifiers, observation-dependent gating (mixture-of-experts), and improved volatility/risk modeling, which together increase robustness and interpretability in non-stationary financial environments.

## 5.1 Introduction and Motivation

Mixtures of distributions provide a flexible way to model heterogeneity and multi-modality in data by representing an overall population density as a weighted combination of simpler component densities. In many real-world problems (finance, speech, biology), observations arise from several latent sub-populations (or regimes) whose characteristics differ; mixture models let the practitioner model this directly instead of forcing a single unimodal distribution [64].

## 5.2 Mathematical Formulation

A finite mixture model for a random vector $X$ assumes

$$p(x|\Theta) = \sum_{k=1}^K \pi_k f_k(x|\theta_k)$$

where $\pi_k \geq 0$, $\sum_{k=1}^K \pi_k = 1$ are mixture weights, each $f_k(\cdot|\theta_k)$ is a component density parameterized by $\theta_k$, and $\Theta = \{\pi_k, \theta_k\}_{k=1}^K$ denotes all parameters. Common choices are Gaussian components (Gaussian Mixture Models, GMMs), t-distributions (robust mixtures), or component-specific regression models [65, 79].

**Latent-variable view.** Introduce discrete latent variables $Z \in \{1, ..., K\}$ with $Pr(Z = k) = \pi_k$. Then the complete-data likelihood factorizes as

$$p(x, z|\Theta) = \prod_{i=1}^n \pi_{z_i} f_{z_i}(x_i|\theta_{z_i}), \quad Z \in \{1, ..., K\}$$

which is the basis for EM estimation [27, 82].

## 5.3 Parameter Estimation

### 5.3.1 Maximum Likelihood & EM Algorithm

Maximum likelihood estimation for mixtures is commonly performed with the Expectation-Maximization (EM) algorithm: the E-step computes posterior responsibilities $\gamma_{ik} = Pr(Z_i = k|x_i, \Theta^{(t)})$ and the M-step updates $\{\pi_k, \theta_k\}$ to maximize expected complete-data log-likelihood. EM is widely used because of its simplicity and general applicability, though it finds local maxima and needs careful initialization [27, 89].

### 5.3.2 Bayesian Approaches & MCMC / Variational Inference

Bayesian estimation places priors on weights and component parameters and uses MCMC or variational inference to obtain posterior distributions. This yields uncertainty quantification (posterior on $\pi_k, \theta_k$) and can mitigate overfitting through priors. For flexible, nonparametric mixtures, Dirichlet Process (DP) mixtures allow the number of components to be inferred from data [26].

## 5.4 Model Selection and Identifiability

Choosing K (number of components) is a standard problem: information criteria (AIC, BIC), integrated classification likelihood (ICL), cross-validation, and Bayesian model evidence are used in practice. Mixtures can suffer from identifiability issues (label switching) and multimodal likelihood surfaces; practical pipelines include multiple restarts, informative priors, or constrained parametrizations to address these issues [64].

## 5.5 Dynamic Extensions - Connecting Mixtures to Regimes

Mixture models as described above are cross-sectional (no temporal dependence). For time series and regime modelling, several extensions have been developed.

### 5.5.1 Hidden Markov Models (HMM) and Markov-Switching Models

Hidden Markov Models generalize mixtures by adding Markovian dynamics on the discrete latent state $Z_t$: the state sequence follows a Markov chain with transition matrix $P$, and observations are drawn from state-conditional densities $f_{Z_t}(x_t)$. In econometrics, Hamilton's Markov-switching model explicitly models time-varying autoregressive parameters via a discrete state process and has become a canonical framework for capturing persistent regime changes in macro and financial time series. HMM / Markov-switching models capture persistence and transition structure in regimes, which static GMMs cannot [25].

### 5.5.2 Regime-Switching Volatility Models (MS-GARCH)

For financial volatility, Markov-switching GARCH (and related MS variants) allow the conditional variance dynamics to shift by state; packages and recent literature implement ML and Bayesian inference for such models. These extensions combine mixture-like state-conditional densities with temporal dynamics for both mean and volatility [66].

### 5.5.3 Mixtures-of-Experts and Observation-Dependent Mixtures

The Mixture-of-Experts (MoE) family conditions component weights $\pi_k(x)$ on covariates (gating network), enabling observation-dependent allocation to components; hierarchical MoE and gating functions are useful when component assignment depends on observable features. MoE can be trained with EM-like procedures and connects mixture modelling to modular/ensemble learning [67].

## 5.6 Nonparametric & Flexible Mixtures

Dirichlet Process Mixtures (DPM) and other Bayesian nonparametric constructions allow an unbounded number of components with a prior encouraging parsimony; they are useful when the component count or shape is unknown. Practical inference uses truncation, stick-breaking, variational Bayes, or MCMC [68].

## 5.7 Applications in Classification (Especially Finance)

- **Pre-labelling / Regime Detection:** Use mixture clustering (GMM) or Markov-switching models to produce regime labels (e.g., high-volatility vs low-volatility), then train separate classifiers per regime or feed regime as a feature to a global classifier. Regime models that incorporate temporal dependence (HMM/ Markov switching) produce smoother, persistent regimes that are more realistic in financial markets [31].
- **Mixture-conditioned Decision Rules:** Mixture components may correspond to different conditional decision boundaries; classifier performance often improves when allowed to vary by component (mixture-of-experts style) [32].
- **Risk Modeling and Volatility Forecasting:** MS-GARCH and regime models improve volatility forecasts and risk-sensitive classification thresholds (e.g., when to enter/exit trades) by recognizing when conditional variance regimes change [66].

# Chapter 6: The Second Layer - RL model

## 6.1 Reinforcement Learning

### 6.1.1 Introduction

Reinforcement Learning (RL) is a class of machine learning techniques designed to solve sequential decision-making problems. In RL, an autonomous agent interacts with an environment by taking actions, receiving feedback in the form of rewards, and observing subsequent states. The overarching objective is to maximize the expected cumulative reward across time [6, 16].

Formally, RL can be expressed within the framework of a Markov Decision Process (MDP), defined by the tuple:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, r, \gamma)$$

where $\mathcal{S}$ denotes the state space, $\mathcal{A}$ the action space, $\mathcal{T}$ the transition probability kernel, $r$ the reward function, and $\gamma \in [0, 1)$ the discount factor.

### 6.1.2 The Agent-Environment Loop

At each discrete time step $t \in \mathbb{N}$, the following sequence unfolds [6]:

1. The agent observes the current state: $S_t \in \mathcal{S}$
2. Based on this state, the agent chooses an action: $A_t \in \mathcal{A}$ according to its policy $\pi$.
3. The environment transitions to a new state: $S_{t+1} \sim \mathcal{T}(\cdot | S_t, A_t)$ and produces an immediate reward, [91]: $R_{t+1} = r(S_t, A_t, S_{t+1})$

The tuple $(S_t, A_t, S_{t+1})$ defines one step of the RL process. Repetition of this loop yields a trajectory (episode):

$$\tau = (S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, ...)$$

### 6.1.3 Objective of Reinforcement Learning

The fundamental goal of RL stems from the reward hypothesis, which asserts that any sequential decision-making objective can be expressed as the maximization of the expected return [6].

For a trajectory $\tau$, the cumulative discounted reward (return) is defined as:

$$R(\tau) = \sum_{t=0}^\infty \gamma^t R_{t+1}$$

The task of the agent is to learn a policy that maximizes the expected return:

$$J(\pi) = \mathbb{E}_\pi[R(\tau)]$$

where the expectation is taken over trajectories induced by policy $\pi$ and environment dynamics $\mathcal{T}$. [73]

### 6.1.4 State and Action Spaces

**State Space**
The state space $\mathcal{S}$ encapsulates all information necessary to characterize the environment at a given time. In financial markets, a state $S_t$ may include:

$$S_t = (\text{Open}_t, \text{High}_t, \text{Low}_t, \text{Close}_t, \text{Volume}_t, \text{Fundamentals}_t, ...) \quad [78]$$

**Action Space**
The action space $\mathcal{A}$ defines the set of admissible decisions available to the agent. In trading applications, this often takes the form of a hybrid action space consisting of both discrete and continuous components [18]:

- Discrete: asset selection, e.g., $A_t^{(d)} \in \{\text{Stock A}, \text{Stock B}, ...\}$
- Continuous: trading quantity or allocation, e.g., $A_t^{(c)} \in [0, 1]$

Thus: $\mathcal{A} = \mathcal{A}^{(d)} \times \mathcal{A}^{(c)}$

### 6.1.5 Markov Decision Process

An environment satisfies the Markov property if the probability distribution of the next state depends only on the current state and action:

$$\mathbb{P}(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0, A_0) = \mathbb{P}(S_{t+1} | S_t, A_t)$$

Thus, the agent requires only the current state $S_t$ to determine the optimal action, rather than the entire history of states and actions [75, 96].

### 6.1.6 Rewards and Discounting

**Reward Function**
The reward function is defined as, [92]:

$$r: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$

or in transition-dependent form:

$$r: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$$

At time step $t$, the immediate reward is: $r_t = r(S_t, A_t, S_{t+1})$

**Discount Factor**
The discount factor $\gamma \in [0, 1)$ determines the present value of future rewards. Thus, the return becomes:

$$R(\tau) = r_0 + \gamma r_1 + \gamma^2 r_2 + ...$$

- For $\gamma = 0$: the agent is myopic, maximizing immediate rewards. [84]
- For $\gamma \rightarrow 1$: the agent is far-sighted, emphasizing long-term gains.

The discount factor ensures convergence in the infinite-horizon setting by geometrically bounding the cumulative reward [6].

### 6.1.7 Solution Approaches

Two principal families of algorithms exist for solving RL problems [17]:

**Policy-Based Methods**
A policy is a mapping $\pi : \mathcal{S} \rightarrow P(\mathcal{A})$, where $P(\mathcal{A})$ denotes the probability distribution over actions.

- Deterministic Policy: $a = \pi(s), \forall s \in \mathcal{S}$
- Stochastic Policy: $\pi(a\vert{}s) = P(A=a\vert{}S=s)$ [99]

Policy-based methods directly optimize $\pi$ to maximize $J(\pi)$.

**Value-Based Methods**
These methods rely on learning a value function that estimates the expected return from states (or state-action pairs):

$$v_\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} | S_t = s\right]$$$$q_\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} | S_t = s, A_t = a\right]$$

The optimal policy is derived by acting greedily with respect to the estimated value function. [74, 83]

## 6.2 Proximal Policy Optimization (PPO)

### 6.2.1 Introduction and Notation

Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, r, \gamma)$ be a Markov Decision Process (MDP) with state space $\mathcal{S}$, action space $\mathcal{A}$, transition kernel $\mathcal{T}(\cdot | s, a)$, reward function $r : \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$, and discount factor $\gamma \in [0, 1)$ [98].

A trajectory (episode) is given by: $\tau = (S_0, A_0, R_1, S_1, A_1, R_2, ...)$ and the (discounted) return from time $t$ is: $R_t(\tau) = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$.
A parametric policy $\pi_\theta(a|s)$ (with parameters $\theta \in \mathbb{R}^d$) induces a distribution over trajectories.
We denote the expectation under $\pi_\theta$ by $\mathbb{E}_{\tau \sim \pi_\theta}[\cdot]$.
The objective is the expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R_0(\tau)].$$

### 6.2.2 Policy Gradient - Principle and Estimator

**Policy Gradient Objective**
The optimization problem is: $\max_\theta J(\theta)$.
Under regularity conditions, the gradient of the objective can be expressed as [20]:

$$\nabla J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s, a)],$$

where $d^{\pi_\theta}$ is the discounted state-occupation distribution and $Q^{\pi_\theta}(s, a)$ the action-value function.

**REINFORCE Estimator**
A Monte Carlo estimator (Williams, REINFORCE) uses sampled returns [19]:

$$\hat{g}_t = \nabla_\theta \log \pi_\theta(A_t|S_t) G_t,$$

where $G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ...$

To reduce variance, a baseline $b(s)$ (often $V(s)$) is introduced, yielding the advantage estimator:

$$\hat{g}_t = \nabla_\theta \log \pi_\theta(A_t|S_t)(G_t - b(S_t))$$$$\approx \nabla_\theta \log \pi_\theta(A_t|S_t)(Q^\pi(S_t, A_t) - V^\pi(S_t))$$$$\approx \nabla_\theta \log \pi_\theta(A_t|S_t)\hat{A}_t,$$

where $\hat{A}_t$ estimates the advantage function $A^\pi(S_t, A_t)$. [87]

### 6.2.3 Variance Reduction: Baselines and GAE

Generalized Advantage Estimation (GAE) introduces a tunable bias-variance trade-off via $\lambda \in [0, 1]$ [115]:

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l},$$

where $\delta_t = R_t + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)$, where $V_\phi$ is a learned value function.
GAE reduces variance while maintaining sufficient learning signal for the policy gradient.

### 6.2.4 Multiple-Epoch Updates and Proximal Constraints

Naive policy-gradient methods perform one gradient step per batch. To improve sample efficiency, modern methods reuse data across multiple epochs. However, large updates can move the new policy far from the behavior policy, invalidating the gradient estimate. Trust-region and proximal methods address this by constraining policy updates per step [21]. PPO achieves this with a simple clipping mechanism [2].

### 6.2.5 PPO: Clipped Surrogate Objective

**Probability Ratio**
Define the probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

A naive update objective is:

$$\mathcal{L}(\theta) = \hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t],$$

which can explode if $r_t(\theta)$ deviates too far from 1.

**Clipped Objective**
PPO introduces the clipped objective:

$$\mathcal{L}^{CLIP}(\theta) = \hat{\mathbb{E}}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t)],$$

where $\epsilon > 0$ (commonly 0.1-0.3) defines the clipping range.
Clipping penalizes policy updates that change action probabilities by more than $\epsilon$ multiplicatively.

### 6.2.6 Additional PPO Components and Losses

In practice, PPO optimizes a combined loss [2]:

$$\mathcal{L}(\theta, \phi) = \underbrace{\hat{\mathbb{E}}_t[-\mathcal{L}^{CLIP}(\theta)]}_{\text{Policy Loss}} - c_1 \underbrace{\hat{\mathbb{E}}_t[(V_\phi(s_t) - \hat{R}_t)^2]}_{\text{Value Function Loss}} - c_2 \underbrace{\hat{\mathbb{E}}_t[H(\pi_\theta(\cdot|s_t))]}_{\text{Entropy Bonus}},$$

where $V_\phi$ is the critic network, $\hat{R}_t$ is an estimate of the return, $c_1$ and $c_2$ are loss weights, and $H(\pi)$ denotes entropy. Entropy regularization encourages exploration and prevents early convergence to deterministic policies [93].

### 6.2.7 Theoretical and Practical Remarks

1. **KL vs. Clipping:** TRPO uses an explicit KL constraint, while PPO approximates it via clipping for computational efficiency [21, 2].
2. **Bias-Variance Tradeoff:** GAE and baselines reduce variance; clipping introduces bias but improves stability [115].
3. **Hyperparameter Sensitivity:** PPO performance depends on $\epsilon$, learning rate, epoch count, minibatch size, $\lambda$, $\gamma$, and entropy/value weights.
4. **Extensions:** Further research refines PPO's theoretical guarantees, such as Truly Proximal PPO [23].

## 6.3 Long Short-Term Memory (LSTM) Networks as PPO Agent

### 6.3.1 Background and Motivation

Recurrent Neural Networks (RNNs) are a class of neural architectures designed to model sequential and temporal data by maintaining a hidden state that evolves over time. Despite their conceptual suitability for time-series and sequence modeling, standard RNNs suffer from the **vanishing and exploding gradient problem**, which severely limits their ability to learn long-term dependencies in practice [116].

To address this limitation, **Long Short-Term Memory (LSTM)** networks were introduced by Hochreiter and Schmidhuber [117]. The key innovation of LSTM lies in its explicit memory structure and gating mechanisms, which regulate information flow and enable stable gradient propagation across long time horizons.

### 6.3.2 Motivation for Integrating LSTM with Proximal Policy Optimization (PPO)

Financial markets and many real-world decision environments are inherently **sequential, stochastic, and partially observable**. At any given time step, the agent's observation typically represents only a noisy and incomplete snapshot of the true underlying state. Classical reinforcement learning algorithms that rely on fully observable Markov Decision Processes (MDPs) therefore face fundamental limitations in such settings.

Proximal Policy Optimization (PPO) is a policy-gradient method designed to provide stable and sample-efficient learning through clipped policy updates and trust-region-like constraints [118]. While PPO has demonstrated strong empirical performance across a wide range of continuous and discrete control tasks, its standard formulation assumes that the policy and value networks operate on instantaneous observations. This assumption may be suboptimal in environments where historical context is essential for inferring latent states.

To address partial observability, the decision process can be more accurately modeled as a **Partially Observable Markov Decision Process (POMDP)**, where the optimal action depends on the entire observation history rather than the current observation alone [119]. In this context, Long Short-Term Memory (LSTM) networks provide a principled mechanism for encoding temporal dependencies and maintaining a compact summary of past information through gated memory dynamics [117].

By integrating LSTM into the PPO architecture, the policy and value functions are endowed with implicit **belief-state estimation**, allowing the agent to infer latent market regimes, volatility conditions, or momentum persistence from observation sequences. Prior work has shown that recurrent policies significantly improve performance in partially observable environments by enabling history-dependent decision making [120].

Moreover, the combination of PPO and LSTM is particularly attractive from an optimization standpoint. PPO's clipped objective mitigates instability arising from high-variance policy gradients, while LSTM's gating mechanism stabilizes temporal credit assignment across long horizons. This complementary interaction yields a robust learning framework capable of handling noisy, non-stationary, and delayed-reward environments commonly encountered in financial trading and sequential decision problems.

### 6.3.3 LSTM Architecture

An LSTM network augments the traditional RNN structure by introducing a memory cell $C_t$ and a set of gates that control how information is stored, updated, and exposed. At each time step $t$, the LSTM processes an input vector $x_t$, the previous hidden state $h_{t-1}$, and the previous cell state $C_{t-1}$.

The gating mechanisms are defined as follows:

1. **Forget Gate**
Determines which information from the previous cell state should be discarded:
$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$$
2. **Input Gate**
Controls which new information should be written into the cell state:
$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$$
3. **Candidate Cell State**
Represents new candidate information to be added:
$$\tilde{C}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)$$
4. **Cell State Update**
Combines retained past information with new content:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
5. **Output Gate and Hidden State**
Determines the exposed hidden representation:
$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(C_t)$$

Here, $\sigma(\cdot)$ denotes the logistic sigmoid function and $\odot$ represents element-wise multiplication.

### 6.3.4 Key Properties and Theoretical Advantages

The central theoretical advantage of LSTM networks lies in their ability to maintain a **nearly constant error flow** through the cell state, thereby mitigating **vanishing gradients** during backpropagation through time (BPTT). This property allows LSTMs to learn dependencies over long temporal spans that are inaccessible to standard RNNs [121].

The gating structure also enables **selective memory**, allowing the model to dynamically adapt to non-stationary temporal patterns, a property particularly important in real-world sequential data such as speech, text, and financial time series.

# Chapter 7: Extra Alternative: Parallel Structure of Proximal Policy Optimization (PPO)

In the context of reinforcement learning, Proximal Policy Optimization (PPO) can be formulated as a parallelized actor-critic framework where multiple agents (or environments) collect experiences concurrently to stabilize and accelerate policy updates [2, 69]. Let the learning objective be defined over $N$ parallel environments, each producing a trajectory $\tau_i$.

## 7.1 Parallel Data Collection

At iteration $k$, the PPO framework samples trajectories in parallel:

$$\tau_i = \{(s_t^i, a_t^i, r_t^i, s_{t+1}^i)\}_{t=0}^{T-1}, \quad i = 1, 2, ..., N \quad (7.1)$$

where $N$ denotes the number of parallel environments, and each trajectory $\tau_i$ follows the current policy $\pi_{\theta_k}$.
The aggregated experience buffer is then constructed as:

$$\mathcal{D}_k = \bigcup_{i=1}^N \tau_i \quad (7.2)$$

This structure allows PPO to compute the policy gradient using a large and diverse sample set collected synchronously across environments.

## 7.2 Advantage Estimation (Parallelized GAE)

For each sampled transition in $\mathcal{D}_k$, the generalized advantage estimate (GAE) is computed in parallel [22]:

$$\hat{A}_t^i = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}^i \quad (7.3)$$

where

$$\delta_t^i = R_t^i + \gamma V_\phi(s_{t+1}^i) - V_\phi(s_t^i) \quad (7.4)$$

The advantage computation across $i=1, ..., N$ is inherently parallelizable, as each environment's trajectory is independent.

## 7.3 Surrogate Objective (Policy Update)

The clipped surrogate objective of PPO is defined as [2]:

$$\mathcal{L}^{CLIP}(\theta) = \hat{\mathbb{E}}_{(s_t, a_t) \sim \mathcal{D}_k}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)] \quad (7.5)$$

where the probability ratio $r_t(\theta)$ is computed as:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \quad (7.6)$$

The parallel structure allows this objective to be evaluated simultaneously across all $N$ environments, enabling efficient gradient estimation via minibatch stochastic gradient descent:

$$\nabla_\theta \mathcal{L}^{CLIP}(\theta) \approx \frac{1}{NT} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \min(r_t^i(\theta)\hat{A}_t^i, \text{clip}(r_t^i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t^i) \quad (7.7)$$

## 7.4 Value Function and Entropy Regularization

The total PPO loss incorporates both the value function and entropy terms:

$$\mathcal{L}_{\text{total}}(\theta, \phi) = \mathcal{L}^{CLIP}(\theta) - c_1 \hat{\mathbb{E}}[(V_\phi(s_t) - \hat{V}_t)^2] - c_2 \hat{\mathbb{E}}[H(\pi_\theta(\cdot|s_t))] \quad (7.8)$$

where $H(\pi_\theta)$ denotes the policy entropy and $\hat{V}_t$ the empirical return. Both $V_\phi$ and $H(\pi_\theta)$ are evaluated in parallel across the $N$ collected trajectories.

## 7.5 Parallel Update Scheme

The entire PPO training iteration can thus be represented as:

```
for k = 1, 2, ..., K:
    collect in parallel: \mathcal{D}_k = \bigcup_{i=1}^N \tau_i
    compute in parallel: \hat{A}_t^i, \hat{V}_t^i
    update: (\theta, \phi) \leftarrow (\theta, \phi) + \alpha \nabla_{(\theta, \phi)} \mathcal{L}_{\text{total}}(\theta, \phi) \quad (7.9)
```

This loop illustrates the parallel structure of PPO, where data collection, advantage computation, and gradient estimation are all distributed across $N$ concurrent environments, leading to both computational efficiency and improved sample diversity [44, 45].

## 7.6 Discussion: Motivation, Advantages, and Limitations of the Parallel PPO Structure

Mathematically, the parallel structure of PPO can be summarized as a multi-agent expectation over $N$ independent environment samplers [2, 44]:

$$
\mathbb{E}_{i=1}^N [\mathcal{L}^{CLIP, i}(\theta)] = \frac{1}{N} \sum_{i=1}^N \hat{\mathbb{E}}_{(s_t^i, a_t^i) \sim \tau_i}[\min(r_t^i(\theta)\hat{A}_t^i, \text{clip}(r_t^i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t^i)] \quad (7.10) 
$$

This formulation emphasizes that PPO operates as a synchronous parallel optimization process, combining multiple independent rollouts to approximate the true policy gradient more reliably.

### 7.6.1 Motivation and RationaleTraditional reinforcement learning algorithms, particularly those based on policy gradients, often suffer from high variance in gradient estimation and slow convergence due to limited and correlated samples from sequential data collection [69, 22].

To mitigate these challenges, the parallelized PPO framework introduces multiple concurrent environment instances $(E_1, E_2, ..., E_N)$ that operate independently under the same policy parameters $\pi_\theta$.

This parallel structure enhances both sample efficiency and gradient stability, as the aggregated experience buffer $\mathcal{D}_k = \bigcup_{i=1}^N \tau_i$ captures a more representative and less correlated distribution of state-action pairs.

Formally, this improves the approximation of the true policy gradient:

$$
\nabla_\theta J(\theta) \approx \frac{1}{NT} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^i\vert{}s_t^i)\hat{A}_t^i \quad (7.11) 
$$

The law of large numbers implies that as $N \rightarrow \infty$, the variance of the gradient estimate decreases:

$$
\text{Var}[\nabla_\theta J(\theta)] \propto \frac{1}{N} \quad (7.12) 
$$

Hence, the parallel design is theoretically justified for achieving a more stable policy improvement process.

### 7.6.2 Advantages of Parallel PPO

**(a) Improved Sample Efficiency:** 

Since data are collected concurrently from multiple trajectories, PPO can accumulate a substantially larger batch size per iteration without additional simulation time. This parallelism reduces the number of training iterations required for convergence, leading to faster policy learning:

$$
\vert{}\mathcal{D}_k\vert{} = N \times T \quad (\text{total samples per iteration}) \quad (7.13) 
$$

A higher $N$ yields a richer representation of the state space, enabling the policy to generalize more effectively across diverse market or environmental regimes [95].

**(b) Reduced Gradient Variance and Improved Stability:**

The PPO objective function already incorporates a clipping mechanism that constrains policy updates. When combined with parallel sampling, this further reduces oscillations during training by stabilizing the estimated advantage distribution across environments:

$$
\text{Var}[\hat{A}_t] = \frac{1}{N} \sum_{i=1}^N \text{Var}[\hat{A}_t^i] \quad (7.14) 
$$

This ensures smoother convergence and lowers the risk of catastrophic policy degradation.

**(c) Scalability and Computational Efficiency:** 

Parallel PPO maps naturally onto modern hardware, particularly multi-core CPUs and GPUs, where environment rollouts and gradient computations can be distributed efficiently. From a practical standpoint, the wall-clock time per iteration can be reduced approximately by a factor of $1/N$, assuming perfect parallel efficiency:

$$
T_{\text{parallel iteration}} \approx \frac{T_{\text{serial iteration}}}{N} \quad (7.15) 
$$

This enables PPO to scale effectively for large-scale problems such as financial trading, robotics, or autonomous decision-making. [76]

### 7.6.3 Limitations and Trade-offs

Despite its advantages, the parallel PPO structure introduces several computational and structural trade-offs that must be considered carefully.

**(a) Increased Memory Consumption:** 

Each environment maintains its own set of state, action, reward, and advantage buffers. The total memory requirement grows linearly with both $N$ (number of environments) and $T$ (trajectory length):

$$
\mathcal{O}_{\text{memory}} = \mathcal{O}(N \times T \times d) \quad (7.16) 
$$

where $d$ denotes the dimensionality of the observation space [94]. For high-dimensional observations (e.g., image-based or multi-feature financial states), this can quickly exhaust available GPU or system memory.

**(b) Communication Overhead and Synchronization Cost:**

Although the environments are executed in parallel, the model parameters must be synchronized across all environments after each update. This synchronization introduces communication overhead, particularly in distributed systems where inter-process communication (IPC) or network transfer is involved:

$$
T_{\text{update}} = T_{\text{compute}} + T_{\text{sync}} \quad (7.17) 
$$

where $T_{\text{sync}}$ increases with $N$, diminishing the theoretical linear speedup.

**(c) Diminishing Returns with Large N:**

While increasing $N$ improves experience diversity, the marginal benefit eventually declines because:

1. Policy updates rely on shared global parameters that may become overfitted to older trajectories when synchronization is delayed.
2. Gradient variance reduction exhibits diminishing improvement beyond a certain threshold. Empirically, optimal performance is often achieved for moderate values of $N \in [8, 32]$, depending on task complexity and hardware parallelism [45]. 

### 7.6.4 Computational Complexity Analysis

The time complexity per PPO iteration (including data collection and policy update) can be approximated as:

$$
\mathcal{O}_{\text{time}} = \mathcal{O}(N \times T \times f_{\text{env}}) + \mathcal{O}(B \times f_{\text{grad}}) \quad (7.18) 
$$

where:

$f_{\text{env}}$: cost of a single environment step,

$f_{\text{grad}}$: cost of a gradient update,

$B = N \times T$: total batch size.

Under ideal parallelization, the effective time complexity reduces to:

$$
\mathcal{O}_{\text{parallel time}} = \mathcal{O}(T \times f_{\text{env}}) + \mathcal{O}\left(\frac{B \times f_{\text{grad}}}{N_{GPU}}\right) \quad (7.19) 
$$

The space complexity, dominated by buffer storage and model parameters, is approximately:

$$
\mathcal{O}_{\text{space}} = \mathcal{O}(N \times T \times (\vert{}s\vert{} + \vert{}a\vert{} + 1)) + \mathcal{O}(\vert{}\theta\vert{} + \vert{}\phi\vert{}) \quad (7.20) 
$$

where $\vert{}\theta\vert{}$ and $\vert{}\phi\vert{}$ denote the parameter sizes of the policy and value networks, respectively. [90]

Overall, the parallel structure of PPO offers a balanced compromise between computational efficiency and learning stability. It enables large-scale data collection, smoother policy updates, and improved convergence properties—at the expense of increased memory demand and synchronization overhead. From a systems perspective, its time complexity benefits scale approximately linearly with hardware resources up to a moderate level of concurrency, making it one of the most practical algorithms for high-frequency or real-time decision-making environments.

# Chapter 8: Methodology

## 8.1 Research Design

This study adopts a two-layer modeling framework to address the challenges of short-term financial prediction and trading decision-making. The framework integrates supervised learning and reinforcement learning into a hierarchical structure.

In the first layer, classification models are developed to transform financial features into predictive trading signals. In the second layer, a reinforcement learning algorithm—specifically Proximal Policy Optimization (PPO)—is employed to map these signals, along with market feedback, into trading actions [2].

To enhance robustness under changing market conditions, an extension technique based on regime-switching is incorporated. By modeling market dynamics as mixtures of distributions, the framework adapts to non-stationarity such as volatility clustering and structural breaks [25, 9, 24].

## 8.2 Data Collection and Preprocessing

### 8.2.1 Data Source

The research relies on historical stock market data from the Vietnam Stock Exchange. The dataset includes the standard price and volume series (open, high, low, close, and volume).

To enrich predictive power, a set of technical indicators is derived using the Python ta library [4]. These indicators capture price momentum, volatility, and trend-following signals.

### 8.2.2 Preprocessing

The data undergo several preprocessing steps:

1. **Feature Construction:** Both continuous (value-based) and binary (signal-based) indicators are generated to represent trading rules and quantitative market states.
2. **Normalization:** Standardization (z-score) or min-max scaling is applied to ensure comparability and numerical stability across features.
3. **Time-Aware Splitting:** The dataset is divided into training, validation, and testing subsets while maintaining chronological order to avoid lookahead bias. [85]
4. **Regime Identification (optional extension):** Gaussian Mixture Models (GMM) or Markov-switching methods are employed to assign latent regime labels, distinguishing between high- and low-volatility environments [24, 25].

### 8.3 First Layer - Classification Models

The first layer focuses on predicting binary trading signals (buy vs. no-buy) from financial indicators. Four machine learning classifiers are implemented:

- Logistic Regression
- Random Forest
- XGBoost
- K-Nearest Neighbor (KNN)

### 8.3.1 Model-Oriented Feature Selection Framework

**Overview**

The Model-Oriented Feature Selection Framework is designed to identify the most effective subset of features from a high-dimensional dataset using a model-centric evaluation strategy. The framework focuses on optimizing the feature set for machine learning models by incorporating multiple feature selection methods, combining and intersecting their outputs, and systematically validating these subsets through performance metrics.

The framework is particularly built on this article, with the assumption that choosing feature selection methods without testing is ambiguous. In our opinion, testing for each feature set is needed to find out the best feature set that best fits the model.

**Initial Feature Set Generation**

The framework begins by constructing a comprehensive set of features. In the current implementation, a total of 91 features are initially generated. These features may originate from various domains such as financial technical indicators (in this article), text embeddings (in NLP), or statistical summaries (in time-series forecasting).

The goal of this step is to ensure that the dataset is rich and diverse enough to support various selection and filtering techniques in the subsequent steps.

**Application of Individual Feature Selection Methods**

Each feature selection method is applied independently to the full 91-feature dataset. These methods may include:

- **Filter-based methods:** (e.g., mutual information, variance threshold).
- **Wrapper-based methods:** (e.g., recursive feature elimination).
- **Embedded methods:** (e.g., feature importances from tree-based models).

Each method outputs a "sub-feature set", which contains a subset of features considered important or relevant based on the specific criteria of that method.

**Generation of Additional Subsets via Combinatorial Logic**

To enhance diversity and explore potential synergies between methods, the framework generates additional sub-feature sets through the following operations:

- **Union (Combination):** Combining the outputs of two or more selection methods to form a broader feature set.
- **Intersection:** Identifying the common features selected by multiple methods to create more conservative, consensus-based subsets.

This step significantly expands the pool of candidate feature sets, increasing the chances of discovering a highly effective configuration.

**Sub-Feature Set Packaging**

All generated sub-feature sets, including those directly selected by individual methods and those created via combination/intersection, are compiled into a structured package. This package serves as the input for the evaluation phase.

Each element in this package represents a distinct hypothesis about the optimal feature configuration.

**Model-Based Evaluation of Feature Sets**

Each candidate sub-feature set is evaluated using a selected machine learning model. The performance of the model with each feature set is estimated using K-Fold Cross-Validation (CV). This involves:

- Splitting the training data into $k$ folds.
- Training the model on $k-1$ folds and validating it on the remaining fold.
- Repeating this process $k$ times and averaging the results.

The K-Fold CV ensures a robust estimate of generalization performance and enables fair comparison across feature sets. The feature set yielding the best average performance metric (e.g., accuracy, F1-score, ROC AUC) is selected as the optimal subset for that model.

**Mathematical Setup**

Let the dataset be: $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$

Let:

- $\mathcal{D}_{train}^{(k)}$ be the training set of fold $k$.
- $\mathcal{D}_{test}^{(k)}$ be the validation set of fold $k$.
- $\hat{f}^{(k)}$ be model trained on fold $k$.

The CV risk estimator is:

$\hat{R}_{CV} = \frac{1}{K} \sum_{i=1}^K \text{Score}(y_{test}^{(k)}, \hat{f}^{(k)}(x_{test}^{(k)}))$

This approximates the true generalization error.

**Computational Cost Considerations**

During the model implementation, the manual F1-score is seen as a one validation method. This feature selection implementation is computationally intensive due to the manual validation cycles over multiple sub-feature sets. For this real-world trading application, this step can become a bottleneck and may necessitate parallelization or alternative evaluation strategies.

**Alternative Evaluation Method: Cross-Valid Package**

To mitigate the computational cost, an alternative efficiency estimation method is proposed using a simplified or adapted cross-validation package referred to as the Cross-Valid package. This package offers a lighter-weight evaluation method with reduced computational requirements compared to standard K-Fold Cross-Validation. It serves two purposes:

- As a fast pre-screening tool before committing to full CV.
- As a substitute evaluation method in constrained environments.This alternative may trade off some evaluation robustness for computational efficiency but is valuable for exploratory phases or repeated experiments.

**Final Output**

The framework concludes by recording the best-performing feature set based on the model evaluation. This feature set becomes the final selection for downstream model training and deployment. It represents the optimal balance between dimensionality reduction and predictive power for the selected algorithm.

**Conclusion**

This Model-Oriented Feature Selection Framework provides a structured, model-centric approach to selecting features. By integrating multiple selection strategies, combining results, and relying on model-based performance metrics, it aims to find the most effective feature subset under real-world constraints. Despite its computational cost, the inclusion of alternative evaluation methods allows it to remain practical and scalable. It is especially useful in complex domains where feature relevance is interdependent and best judged in the context of the model's learning dynamics.

### 8.3.2 Training Procedure:

Model hyperparameters are optimized using Bayesian Optimization [34], which efficiently explores high-dimensional search spaces. Model performance is assessed through both predictive metrics (accuracy, F1-score) and financial utility measures (Sharpe ratio), ensuring that statistical performance aligns with trading effectiveness. The output of this layer consists of predicted signals or probability scores, which serve as inputs to the reinforcement learning layer.

Mathematically, Let $\mathcal{M}(x; \theta, \lambda)$ denote a supervised learning model with parameters $\theta$ and hyperparameters $\lambda \in \Lambda \subset \mathbb{R}^d$. Given a training dataset $\mathcal{D} = \{(x_t, y_t)\}_{t=1}^T$, model training yields

$$
\theta^*(\lambda) = \arg\min_\theta \mathcal{L}(\theta | \lambda, \mathcal{D}), 
$$

where $\mathcal{L}(\cdot)$ denotes the empirical loss function.

Hyperparameter tuning is formulated as a black-box optimization problem:

$$
\lambda^* = \arg\max_{\lambda \in \Lambda} f(\lambda) 
$$

where

$$
f(\lambda) = \mathbb{E}[\mathcal{Q}(\mathcal{M}(x; \theta^*(\lambda), \lambda))] 
$$

represents model performance evaluated on a validation set.

Following [34], Bayesian Optimization models the unknown objective function $f(\lambda)$ using a probabilistic surrogate, typically a Gaussian Process (GP):

$$
f(\lambda) \sim \mathcal{GP}(m(\lambda), k(\lambda, \lambda')), 
$$

where $m(\cdot)$ and $k(\cdot, \cdot)$ denote the mean and covariance functions, respectively. 

At iteration $n$, the next hyperparameter configuration is selected by maximizing an acquisition function $\alpha(\cdot)$:

$$
\lambda_{n+1} = \arg\max_{\lambda \in \Lambda} \alpha(\lambda | \mathcal{D}_n), 
$$

which explicitly trades off exploration and exploitation [100].

Model performance is evaluated along both statistical and economic dimensions.

**Predictive Metrics.** Given predicted labels $\hat{y}_t$ and true labels $y_t$, classification performance is assessed via:

$$
\text{Accuracy} = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(\hat{y}_t = y_t), 
$$

and 

$$
\text{F1-score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}. 
$$

**Financial Utility Metric.** Let $r_t$ denote the realized return obtained by executing trades based on the model's predictions. Risk-adjusted performance is measured using the Sharpe ratio [101]:

$$
\text{Sharpe} = \frac{\mathbb{E}[r_t] - r_f}{\sqrt{\text{Var}(r_t)}} 
$$

where $r_f$ denotes the risk-free rate, which is set to zero for short-horizon trading.

The trained model produces either discrete trading signals or calibrated probability estimates:

$$
s_t = \mathcal{M}(x_t; \theta^*, \lambda^*) \in \{0, 1\} \quad \text{or} \quad p_t = \mathbb{P}(y_t = 1 | x_t). 
$$

These outputs define the input to the reinforcement learning layer:

$$
\mathcal{S}_t = \{x_t, s_t\} \quad \text{or} \quad \mathcal{S}_t = \{x_t, p_t\}, 
$$

thereby ensuring that statistical predictive performance is translated into sequential decision-making performance.

## 8.4 Extension Technique - Regime-Based Extension Framework

### 8.4.1 Overview

To address non-stationarity and regime heterogeneity in financial time series, this study adopts a regime-based extension framework built upon mixture and switching models. Rather than training a single global predictor, the proposed workflow decomposes the data into latent regimes, trains regime-specific predictive models, selects the optimal model within each regime, and finally aggregates regime-conditional predictions into a unified forecast sequence. This framework generalizes standard supervised learning by allowing both data distributions and model structures to vary across regimes.

### 8.4.2 Regime identification and data segmentation

Let $\{(x_t, y_t)\}_{t=1}^T$ denote the observed dataset, where $x_t \in \mathbb{R}^d$ represents the feature vector at time $t$ and $y_t$ denotes the corresponding target variable.

A regime identification method—such as a finite mixture model, Hidden Markov Model (HMM), or Markov-switching specification—is first applied to infer an unobserved regime variable:

$$
Z_t \in \{1, ..., K\}. 
$$

Inference yields either hard regime assignments,

$$
\hat{Z}_t = \arg\max_k Pr(Z_t = k | x_{1:t}) 
$$

or posterior regime probabilities,

$$
\gamma_{t,k} = Pr(Z_t = k | x_{1:t}). 
$$

Based on these inferred regimes, the dataset is partitioned into regime-specific subsets:

$$
\mathcal{D}_k = \{(x_t, y_t) : \hat{Z}_t = k\}, \quad k = 1, ..., K 
$$

This segmentation produces subsets that are approximately stationary, enabling more reliable model estimation within each regime.

### 8.4.3 Regime-specific model training

For each regime $k$, a collection of candidate predictive models $\mathcal{F} = \{\mathcal{F}_1, ..., \mathcal{F}_M\}$ is considered (e.g., Logistic Regression, Support Vector Machines, Random Forests, Gradient Boosting models).

Each candidate model $\hat{f}_{k,m}$ is trained exclusively on regime-specific data $\mathcal{D}_k$ by minimizing a regime-conditional loss function:

$$
\hat{f}_{k,m} = \arg\min_{f \in \mathcal{F}_m} \mathcal{L}_k(f), \quad m = 1, ..., M 
$$

This step allows model parameters to adapt to regime-dependent distributional characteristics.

### 8.4.4 Model selection within regimes

Model performance is evaluated using regime-consistent validation procedures. For each regime $k$, the optimal model is selected according to a predefined evaluation criterion (e.g., accuracy, F1-score, or a risk-adjusted financial metric):

$$
m_k^* = \arg\max_m \text{Score}(\hat{f}_{k,m} | \mathcal{D}_{val,k}). 
$$

The selected model for regime $k$ is denoted by:

$$
\hat{f}_k^* = \hat{f}_{k,m_k^*} 
$$

This step acknowledges that the model best suited for prediction may differ across regimes.

### 8.4.5 Regime-conditional prediction and aggregation

At each time point $t$, predictions are generated using the model corresponding to the inferred regime.

Under hard regime assignment, predictions are obtained as:

$$
\hat{y}_t = \hat{f}_{\hat{Z}_t}^*(x_t). 
$$

When posterior regime probabilities are available, a probabilistic aggregation is employed:

$$
\hat{y}_t = \sum_{k=1}^K \gamma_{t,k} \cdot \hat{f}_k^*(x_t). 
$$

This formulation accounts explicitly for regime uncertainty and yields smoother prediction dynamics.

### 8.4.6 Final prediction output

The final prediction sequence is given by:

$$
\hat{y} = (\hat{y}_1, \hat{y}_2, ..., \hat{y}_T) 
$$

which constitutes a piecewise-adaptive predictor whose structure and parameters vary across latent regimes.

Formally, the overall predictive function can be expressed as:

$$
\hat{f}_{\text{final}}(x_t) = \sum_{k=1}^K Pr(Z_t = k | x_{1:t}) \cdot \hat{f}_k^*(x_t), 
$$

demonstrating that the proposed method is a direct implementation of the mixture-of-experts principle with regime-dependent experts.

## 8.5 Second Layer - Reinforcement Learning (PPO)

### 8.5.1 Overview:

The second layer employs **Proximal Policy Optimization (PPO)** to transform classifier outputs, along with market states, into trading decisions. PPO is chosen for its balance between sample efficiency and training stability in sequential decision-making tasks [2, 33].

The trading environment is defined as follows:

- **State Space ($S_t$):** financial indicators, classifier outputs, and regime labels.
- **Action Space ($A_t$):** discrete actions (buy, hold/do nothing) or continuous portfolio allocations within [0, 1].
- **Reward Function ($R_t$):** realized returns, adjusted for risk exposure.

The PPO agent is trained using an LSTM-based actor-critic network, capable of capturing temporal dependencies in financial time series. Training incorporates Generalized Advantage Estimation (GAE) [70], entropy regularization, and the clipped surrogate loss to maintain stable policy updates.

### 8.5.2 Reward Function with Meta Label M1

**Concept**

Meta-labeling is employed as a second-stage decision mechanism whose objective is to improve the risk-adjusted performance of a primary trading model. Rather than forecasting market direction directly, the meta-model learns whether to accept or reject trade opportunities proposed by a base model. This framework follows the meta-labeling paradigm introduced by [102], which explicitly separates signal generation from execution control, allowing the meta-model to focus on trade quality, timing, and risk sensitivity.

In this study, the base model produces a binary trade proposal, indicating the presence of a potential trading opportunity. A reinforcement learning (RL) agent is subsequently trained as a meta-labeler to determine whether executing the proposed trade is optimal under prevailing market conditions, consistent with recent applications of RL for trade execution and filtering [104, 106].

**Base Model Signal Generation**

Let $s_t$ denote the base model signal at time $t$:

$$
s_t \in \{0, 1\}, 
$$

where:

- $s_t = 1$ indicates that the base model proposes a trade,
- $s_t = 0$ indicates no trade proposal.

The base model may consist of any supervised learning classifier or rule-based strategy (e.g., Random Forest, XGBoost, or technical indicator rules). Importantly, the base model is execution-agnostic, as it does not incorporate transaction-level risk considerations or position management [103].

**Meta-Labeling as a Reinforcement Learning Problem**

The meta-labeling task is formulated as a sequential binary decision problem and modeled as a Markov Decision Process (MDP) [58], which is solved using reinforcement learning.

**State Space**

At each timestep $t$, the agent observes a state vector

$$
X_t \in \mathbb{R}^d, 
$$

which may include:

- market features such as technical indicators and volatility measures,
- auxiliary outputs or confidence scores from the base model,
- recent price dynamics.

**Action Space**

The agent selects an action

$$
a_t \in \{0, 1\}, 
$$

where: 

- $a_t = 1$ corresponds to executing the trade,
- $a_t = 0$ corresponds to ignoring the trade.

**Reward Function Design**

The reward function is designed to align the agent's objective with realized trading performance, a standard principle in reinforcement learning-based trading systems [105]. Let:

- $r_t$ denote the realized return associated with executing a trade at time $t$,
- $s_t$ denote the base model signal,
- $a_t$ denote the agent's action.
- The reward at time $t$ is defined as:
    
    $$
    R_t = \begin{cases} r_t, & \text{if } s_t = 1 \text{ and } a_t = 1. \\ 0, & \text{if } s_t = 1 \text{ and } a_t = 0. \\ r_t, & \text{if } s_t = 0 \text{ and } a_t = 1. \\ 0, & \text{if } s_t = 0 \text{ and } a_t = 0. \end{cases}
    $$
    

This reward structure enforces profit-weighted execution and encourages selective trade filtering, consistent with utility-driven formulations of trading policies [107].

**Interpretation as Meta-Labels**

From a supervised learning perspective, the agent's decisions can be interpreted as latent meta-labels:

$$
y_t^{\text{meta}} = \begin{cases} 1, & \text{if executing the base signal is profitable.} \\ 0, & \text{otherwise.} \end{cases} 
$$

Unlike traditional meta-labeling approaches that rely on fixed return thresholds [102], the reinforcement learning agent implicitly infers these labels through interaction with the environment and reward feedback. This enables adaptation to non-stationary and regime-dependent market dynamics, a key advantage of reinforcement learning methods [58].

### 8.5.3 How LSTM Operates within Proximal Policy Optimization (PPO)

**Problem Setting: Partial Observability**

In many real-world environments, including financial markets, the agent does not observe the true underlying state $s_t$ directly. Instead, it receives a noisy observation $o_t$, making the problem more accurately described as a Partially Observable Markov Decision Process (POMDP). Under partial observability, optimal decisions depend on the history of observations, not solely on the current input. This motivates the use of belief-state approximations or recurrent architectures that summarize past observations into a compact internal state representation [108, 6].

Recurrent Policy RepresentationTo address this, PPO is augmented with an LSTM-based policy and value network. Rather than mapping a single observation to an action distribution, the policy is defined as:

$$
\pi_\theta(a_t|o_{1:t}) = \pi_\theta(a_t|h_t) \quad (8.1) 
$$

where $h_t$ is the hidden state produced by the LSTM after processing the observation sequence $o_{1:t}$.

At each time step:

$$
h_t, c_t = \text{LSTM}(o_t, h_{t-1}, c_{t-1}) \quad (8.2) 
$$

The hidden state $h_t$ acts as a compressed sufficient statistic of the observation history, enabling the agent to infer latent dynamics such as trends, volatility regimes, or structural changes in the environment. Recurrent policies of this form are a standard approach for handling POMDPs in deep reinforcement learning [109, 110].

**Actor-Critic with Shared or Separate LSTM**

In PPO, the LSTM can be incorporated in two common ways:

- **Shared LSTM Encoder:** A single LSTM processes observations and feeds its hidden state to both:
    - The actor head (policy logits or action parameters),
    - The critic head (state-value estimate). This design promotes representation sharing and reduces parameter count.
- **Separate LSTMs:** Independent LSTMs are used for the actor and critic to reduce gradient interference between policy optimization and value estimation. This separation can improve stability, particularly in environments with long temporal dependencies.

Both architectures are compatible with PPO's clipped objective and are widely used in recurrent actor-critic implementations [111, 112].

**Trajectory Collection with Recurrent States**

During environment interaction, the agent collects trajectories of the form:

$(o_t, a_t, r_t, h_t, c_t)$

Key points:

- LSTM hidden states are carried forward sequentially during rollout.
- At episode boundaries, hidden states are reset.
- For truncated rollouts, hidden states are stored and replayed during training. This procedure ensures that the recurrent computation graph used during optimization matches the one used during data collection, a requirement for unbiased gradient estimation in recurrent policies [113, 111].

**PPO Objective with Recurrent Policies**

The PPO clipped surrogate objective remains unchanged in form:

$$
\mathcal{L}^{CLIP}(\theta) = \hat{\mathbb{E}}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)] \quad (8.3) 
$$

where:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|h_t)}{\pi_{\theta_{\text{old}}}(a_t|h_t)} \quad (8.4) 
$$

The key distinction is that both the old and updated policies condition on identical LSTM hidden states, preserving PPO's trust-region-like constraint even in the presence of memory. This formulation is explicitly discussed in recurrent PPO implementations and empirical studies [111, 114].

**Advantage Estimation with Memory**

Advantages are computed using the recurrent value function:

$$
\hat{A}_t = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(h_{t+1}) - V(h_t) \quad (8.5) 
$$

Because the value function $V(\cdot)$ depends on LSTM hidden states, advantage estimation naturally incorporates historical context. This improves temporal credit assignment in partially observable environments and aligns with the theoretical motivation of Generalized Advantage Estimation (GAE) [115].

## 8.6 Evaluation Framework

The performance of the proposed framework is evaluated along three dimensions, [97]:

- Predictive Metrics: accuracy, F1-score, and precision-recall, assessing classification quality.
- Financial Metrics: cumulative return, Sharpe ratio, and maximum drawdown, measuring economic relevance.
- Robustness Metrics: performance stability across market regimes and under rolling-window evaluation.

Comparisons are made between baseline single-layer classifiers, the hybrid classifier-PPO framework, and the regime-extended models.

This multi-dimensional evaluation ensures the research contributes both statistically and practically to short-term trading strategies.

# Chapter 9: Implementation

## 9.1 Data Input

This study utilizes historical market data of Vietcombank Joint Stock Commercial Bank (VCB), one of the largest and most liquid equities listed on the Ho Chi Minh Stock Exchange (HOSE). The data are collected programmatically using the vnstock Python package, which provides structured access to Vietnamese equity market data through publicly available data providers.

The dataset is used as the primary input for all subsequent empirical analyses, including statistical predictability tests, feature construction, and reinforcement learning experiments. All data processing steps are conducted in a fully reproducible manner using Python-based data pipelines.

### 9.1.1 Data Sources

Historical price and trading information for VCB are retrieved via the vnstock package. This package aggregates data from official and widely used market information sources in the Vietnamese stock market ecosystem, such as exchange-reported price data and brokerage-provided market feeds.

The use of vnstock allows automated data acquisition with consistent formatting, enabling efficient updates and minimizing manual data handling errors. Only publicly available historical market data are used; no proprietary or confidential information is involved.

### 9.1.2 Data Description

The raw dataset consists of time-indexed observations sampled at a fixed market frequency. Each observation corresponds to one trading period and contains standard market variables, including open price, high price, low price, close price, and trading volume (OHLCV).

Let $P_t = \{O_t, H_t, L_t, C_t, V_t\}$ denote the market information observed at time $t$, where $O_t$, $H_t$, $L_t$, and $C_t$ represent the open, high, low, and close prices, respectively, and $V_t$ denotes trading volume.

To ensure statistical validity, raw price series are transformed into stationary representations when required, such as log-returns:

$$
r_t = \log(C_t) - \log(C_{t-1}). 
$$

The dataset is cleaned to remove non-trading days and missing values. Corporate actions such as stock splits or dividend adjustments are handled implicitly through the adjusted price series provided by the data source, when available.

This dataset serves as the foundational input for feature engineering, including the construction of technical indicators, volatility measures, and sequential state representations used by machine learning and reinforcement learning models. In particular, sequential windows of historical observations are constructed to form the state input for recurrent policies, such as LSTM-based PPO agents.

## 9.2 Data Preparation

### 9.2.1 Underlying Assumptions of Short-term Trading

Unlike fundamental analysis, which focuses on a company's intrinsic value based on financial statements and economic factors, technical analysis assumes that all relevant information is already reflected in market prices. Thus, it focuses solely on the evolution of price and volume data over time. [81]

**Historical Background**

The foundation of modern technical analysis is largely attributed to Charles H. Dow, co-founder of Dow Jones & Company and creator of the Dow Jones Industrial Average (DJIA). In a series of editorials published in The Wall Street Journal at the turn of the 20th century, Dow introduced a systematic approach to understanding market trends, which later became known as Dow Theory. [80]

Dow proposed two core principles that continue to underpin the philosophy of technical analysis today:

1. **Market efficiency in price representation:** Financial markets are believed to incorporate and reflect all factors—economic, political, or psychological—that can influence a security's price. Consequently, the current price is considered a fair representation of the collective market knowledge.
2. **Repetitive and pattern-based behavior:** Even seemingly random price movements often follow identifiable patterns or trends that recur over time, driven by underlying human psychology and behavioral tendencies.

These foundational ideas provided the conceptual framework for subsequent developments in market pattern recognition, trend-following strategies, and the statistical modeling of financial time series.

**Core Assumptions of Technical Analysis**

Building upon Dow's early insights, contemporary technical analysis operates under three fundamental assumptions that guide most empirical and algorithmic trading models [37, 38]:

1. **The Market Discounts Everything:** According to this assumption, all available information—including company fundamentals, macroeconomic variables, and investor psychology—is already embedded in the current market price. This view aligns closely with the Efficient Market Hypothesis (EMH) [39], suggesting that prices fully reflect all known information. Hence, the task of the technical analyst is not to interpret new information but to analyze price action as the resultant expression of market supply and demand dynamics.
2. **Prices Move in Trends:** Technical analysis posits that price movements are not purely random, but tend to exhibit persistent trends over time. These trends may manifest over various horizons (short-, medium-, or long-term), and most technical trading systems—such as moving average crossovers, momentum indicators, and trendline analyses—are based on this principle.
3. **History Tends to Repeat Itself:** Market behavior is strongly influenced by human emotions such as fear and greed, which lead to recurring psychological patterns. As a result, chart patterns—including formations such as head-and-shoulders, double tops, or triangles—are viewed as reflections of predictable behavioral cycles. These recurring formations allow analysts to anticipate probable future price movements based on historical analogies.

### 9.2.2 Technical Analysis

In order to capture market dynamics and generate predictive features for trading strategies, I employed the Python `ta` library, which provides a broad set of indicators widely used in financial analysis. Using the `add_all_ta_features` function, I constructed a comprehensive technical indicator feature set from the raw OHLCV data (Open, High, Low, Close, and Volume).

This ensures that the input to the models integrates diverse perspectives on market behavior, including trend, momentum, volatility, and volume-based signals. Missing values that typically occur at the beginning of rolling-window indicators were handled by filling them with valid defaults (`fillna=True`).

**Table 9.1: Sample of Historical Price and Momentum Data**

Datecloseopenvolumelog retvol adivol obvvol cmfdrcrtarget1981-02-270.09080.090814.76M0.0335-2.26x10^93.11x10^8-1.000-3.628-10.861.01981-03-020.09130.091311.76M0.0047-2.27x10^93.26x10^8-1.000-5.439-7.8250.01981-03-030.09000.090416.17M-0.0141-2.28x10^93.38x10^8-1.000-5.751-7.3900.02025-07-28214.05214.0237.85M0.0007-5.25x10^{10}2.25x10^{11}0.0040.0392168240.02025-07-29211.27214.1751.41M-0.0130-5.25x10^{10}2.25x10^{11}-0.0802.0342169960.02025-07-30209.05211.8945.51M-0.0105-5.25x10^{10}2.25x10^{11}-0.0901.2912141770.0

The generated feature set can be grouped into the following categories:

1. **Volatility Indicators:** Volatility indicators measure the degree of variation in prices, capturing market risk and uncertainty. These indicators have been widely applied in practice to detect periods of high and low uncertainty [36, 38].
    - **Bollinger Bands (BB, BB high, BB low, BB percentage, BB bandwidth):** measure price deviation relative to a moving average [40].
    - **Keltner Channel (KC, KC high, KC low, KC percentage, KC bandwidth):** volatility envelopes around an EMA-based average [36].
    - **Donchian Channel (DC, DC high, DC low, DC percentage, DC bandwidth):** captures the highest high and lowest low within a rolling window [38].
2. **Trend Indicators:** Trend indicators quantify price direction and help detect upward or downward market movements, a foundation of technical analysis [36].
    - **Average Directional Index (ADX, ADX positive, ADX negative):** measures trend strength [40].
    - **Moving Average Convergence Divergence (MACD, MACD diff, MACD signal):** relationship between short- and long-term moving averages, widely studied in academia [41].
    - **Vortex Indicator (Vortex positive, Vortex negative, Vortex diff):** captures positive and negative trend movements [38].
    - **TRIX:** triple smoothed EMA-based indicator of trend direction [40].
    - **Mass Index:** identifies trend reversals based on high-low range expansions [38].
    - **Commodity Channel Index (CCI):** detects overbought/oversold conditions relative to typical price [36].
    - **Detrended Price Oscillator (DPO):** removes long-term trend components to highlight short-term cycles [40].
3. **Momentum Indicators:** Momentum indicators evaluate the speed of price changes and are often used to detect overbought or oversold conditions [36, 42].
    - **Relative Strength Index (RSI):** measures price momentum on 0-100 scale [36].
    - **Stochastic Oscillator (STOCH, STOCH signal, STOCH RSI, STOCH RSI signal):** compares closing price relative to its recent high-low range [38].
    - **Williams %R:** momentum measure similar to RSI [40].
    - **Ultimate Oscillator (UO):** a weighted sum of three oscillators over different time periods [38].
    - **Rate of Change (ROC):** expresses price momentum as a percentage change [36].
4. **Volume Indicators:** Volume-based indicators measure buying/selling pressure using price-volume interactions, often applied in quantitative trading research [43].
    - **Accumulation/Distribution Index (ADI):** tracks cumulative flow of money into and out of an asset [36].
    - **On-Balance Volume (OBV):** cumulative measure of volume and price direction [38].
    - **Chaikin Money Flow (CMF):** captures buying and selling pressure over a specific period [40].
    - **Force Index:** combines price change and volume to assess the strength of a move [36].
    - **Ease of Movement (EOM, EMV):** measures how easily price moves relative to volume [38].
    - **Volume Price Trend (VPT):** cumulative measure combining price trend with volume [36].
    - **Money Flow Index (MFI):** RSI-like oscillator that incorporates volume [40].
5. **Other Indicators:** Additional derived features include simple or weighted combinations of price and volume measures that enhance predictive capacity.
    - **Daily Returns & Log Returns:** widely used in return-based forecasting models [43].
    - **Custom rolling averages:** generated from OHLCV data if needed.

## 9.3 First Layer Prediction

### 9.3.1 Input Data

The input dataset consists of OHLCV time-series data (Open, High, Low, Close, Volume) enriched with the full set of technical indicators derived using the Python `ta` library.

Before model ingestion, the following preprocessing steps were applied:

1. **Data Cleaning:**
    - Missing values (primarily from rolling-window indicators) were filled with valid defaults (`fillna=True`).
2. **Label Construction:**
    - The target variable was defined as a binary signal:
        
        $$
        y_t = \begin{cases} 1, & \text{if } P_{t+1} - P_t > 0 \\ 0, & \text{otherwise} \end{cases}
        $$
        
        representing whether the next period's price increases (UP) or decreases (DOWN).
        
    - This labeling approach aligns with short-term predictive trading horizons (e.g., 1-step or few-minute returns).
3. **Train-Validation-Test Split:**
    
    To ensure robust model evaluation and mitigate the effects of temporal leakage in financial time series, the dataset was divided into three main subsets: Training, Validation, and Testing. Each subset plays a specific role within the M1-Layers pipeline.
    
    **Training and Validation via Purged Cross-Validation:**
    
    The training-validation process was performed using a Purged K-Fold Cross-Validation (CV) procedure, specifically designed for time-series data. Unlike standard K-Fold CV, this approach ensures that:
    
    - No future information is leaked into the training process.
    - Temporal dependence between consecutive samples is mitigated by purging (i.e., removing) data near the boundary between training and validation folds.
    
    In each fold, the model was trained on a subset of data (`X_train`, `y_train`) and validated on a subsequent, non-overlapping subset (`X_test`, `y_test`). The cross-validation loop computed performance metrics (F1-score and Accuracy) across all folds to obtain an average estimate of generalization ability.
    
    Mathematically, if $R_t$ denotes the market regime at time $t$, the training data for each fold $k$ can be expressed as:
    
    $$
    \{(X_i, y_i) | R_i = r, i \in \text{Train}_k\} 
    $$
    
    and the corresponding validation set as:
    
    $$
    \{(X_j, y_j) | R_j = r, j \in \text{Valid}_k\} 
    $$
    
    where $r \in \{1, 2, ..., M\}$ represents the regime labels determined by the GMM.
    
    The average cross-validation F1-score across folds was used as the objective metric for hyperparameter tuning during the Bayesian Optimization process.
    
    **Testing and Out-of-Sample Evaluation:**
    
    After cross-validation and hyperparameter tuning, the model was re-trained on the entire training dataset and then evaluated on a held-out test set (`Valid_set`, `Valid_result`).
    
    The test phase followed the same regime-wise segmentation procedure to maintain consistency between training and evaluation.
    
    - For each regime, the model was trained on the corresponding subset of the training data and then used to predict the test data belonging to the same regime. [86]
    - Predictions were aggregated across regimes to form the final test predictions (`the_final_ypred`).
    - When a classification threshold was applied, predicted probabilities (`the_final_yproba`) were compared against the threshold to derive binary class outputs, allowing the evaluation of confidence-based decision rules.
    
    Performance on the test set was reported using:
    
    - Accuracy
    - F1-score
    - Classification report (precision, recall, F1 by class)
    
    Additionally, for threshold-based models, the probability distributions for both classes were visualized to assess the separability between positive and negative signals.
    

### 9.3.2 Feature Selection - Model-Oriented Feature Selection Framework

To prevent overfitting and enhance model interpretability, a model-specific feature selection strategy was applied for each classifier.

1. **Correlation Filtering:**
    - Features with pairwise correlation above a threshold ($\vert{}\rho\vert{} > 0.9$) were removed to reduce redundancy.
2. **Univariate Feature Importance:**
    - For tree-based models (e.g., Random Forest, XGBoost), the top-ranked features by Gini importance were retained.
    - For linear models (e.g., Logistic Regression, SVM), Recursive Feature Elimination (RFE) was used to select the most influential predictors. [77]
3. **Hybrid Selection (Statistical + Model-Based):**
    - The retained feature set combines statistically significant and model-relevant indicators, forming an optimized feature subset tailored for short-term prediction.

Let the dataset be: $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$

Let:

- $\mathcal{D}_{train}^{(k)}$ be the training set of fold $k$.
- $\mathcal{D}_{test}^{(k)}$ be the validation set of fold $k$.
- $\hat{f}^{(k)}$ be model trained on fold $k$.

The CV risk estimator is:

$$
\hat{R}_{CV} = \frac{1}{K} \sum_{i=1}^K \text{Score}(y_{test}^{(k)}, \hat{f}^{(k)}(x_{test}^{(k)})) 
$$

This approximates the true generalization error.

### 9.3.3 Regimes Process

Financial markets often exhibit different regimes—periods of distinct volatility or trend behaviors. 

To account for regime shifts, a time-series segmentation procedure was performed by **Trend Regime Detection:**

- A moving average slope or trend indicator (e.g., ADX) was used to identify uptrends versus downtrends.
- The model was trained and evaluated across these regimes to assess robustness and adaptability.

### 9.3.4 Output (Classification and Confidence / Probability)

Each M1 model outputs both:

- **Binary class label:** 0 (No Action) or 1 (Buy Signal), and
- **Predicted probability / confidence score:**
    
    $$
    \hat{y}_t = \mathbb{P}(y_t = 1 | X_t) \quad (9.1) 
    $$
    

These probabilities serve as meta-labeling inputs for the second layer (M2) or as confidence thresholds to convert predictions into actionable trading signals.

Threshold tuning was conducted using Bayesian Optimization to maximize the F1-score and Accuracy, balancing false positives and false negatives.

**Model Performance Summary (F1-scores)** 

- KNN: F1 = 0.3728
- LogR: F1 = 0.5759
- RF: F1 = 0.4576
- XGB: F1 = 0.5698
- RF: 0.4867
    - Test: (array ([0., 1.]), array([159, 101]))
    - Prediction: (array ([0., 1.]), array([61, 199]))
- XGB: 0.5761
    - Test: (array ([0., 1.]), array([206, 147]))
    - Prediction: (array ([0., 1.]), array([14, 339]))
- KNN: 0.3636
    - Test: (array ([0., 1.]), array([10, 7]))
    - Prediction: (array ([0., 1.]), array([13, 4]))
- RF: 0.5000
    - Test: (array([0., 1.]), array([5, 8]))
    - Prediction: (array([0., 1.]), array([5, 8]))

F1 = 0.5387

- Test: (array([0., 1.]), array([380, 263]))
- Prediction: (array([0., 1.]), array([93, 550]))

**Table 9.2: Classification performance of M1 models on the validation dataset.**

|  | **Precision** | **Recall** | **F1-Score** | **Support** |
| --- | --- | --- | --- | --- |
| **Class 0** | 0.53 | 0.13 | 0.21 | 380 |
| **Class 1** | 0.40 | 0.83 | 0.54 | 263 |
| **Accuracy** |  |  | **0.42** | 643 |
| **Macro Avg** | 0.46 | 0.48 | 0.37 | 643 |
| **Weighted Avg** | 0.47 | 0.42 | 0.34 | 643 |

## 9.4 Second Layer Prediction

### 9.4.1 Input Data

Inputs to the M2 (meta-labeler):

- **M1 model outputs:** predicted probabilities or discrete signals from the first-layer model, denoted $\hat{y}_t^{M1}$ (or simply "signal").
- **Market regime variables:** selected regime indicators (e.g. bull/bear flags, trend classification).
- **Volatility and momentum features:** continuous features such as realized volatility, ATR, or momentum scores.

The meta-labeler uses these inputs to filter false signals produced by M1 and output an actionable decision $d_t \in \{\text{no-action}, \text{execute}\}$ or a probability that a given M1 signal will be profitable. This reduces over-trading and improves the precision of executed trades.

### 9.4.2 Environment Setup

**Basic Environment**

The training is conducted within a custom Gym environment that simulates short-term trading dynamics.

- **Observation Space:** Normalized technical indicators corresponding to the feature set used by the LSTM layer. Each observation $s_t \in \mathbb{R}^n$ represents a vector of $n$ normalized features capturing market conditions at time $t$.
- **Action Space:** Binary decision space $\mathcal{A} = \{0, 1\}$, where $a_t = 1$ denotes a Buy action and $a_t = 0$ denotes Do Nothing.

**Reward Function**

The reward function is designed to align the agent's behavior with the profitability of trades while maintaining consistency with the base model's predictions.

Let:

- $b_t \in \{0, 1\}$ denote the base model's trading signal (1 = Buy, 0 = No trade),
- $a_t \in \{0, 1\}$ denote the agent's action (1 = Execute, 0 = Ignore),
- $\Delta p_t = p_{t+1} - p_t$ denote the price change or realized return.

Then, the reward function $r_t$ is defined as:

$$
r_t = \begin{cases} \Delta p_t, & \text{if } b_t = 1 \text{ and } a_t = 1 \text{ (execute profitable base trade)} \\ 0, & \text{if } b_t = 1 \text{ and } a_t = 0 \text{ (ignore base signal)} \\ \Delta p_t, & \text{if } b_t = 0 \text{ and } a_t = 1 \text{ (agent acts independently)} \\ 0, & \text{if } b_t = 0 \text{ and } a_t = 0 \text{ (no trade executed)} \end{cases} 
$$

**Interpretation**

- The agent receives a reward equal to the realized return only when it executes a trade ($a_t = 1$).
- When the base model suggests a trade ($b_t = 1$) and the agent acts ($a_t = 1$), the reward directly reflects the profit or loss from that decision.
- When the agent acts independently without base confirmation ($b_t = 0$, $a_t = 1$), the realized return determines the outcome—implicitly penalizing risky trades if they are unprofitable.
- All non-trading decisions ($a_t = 0$) yield zero reward, providing a neutral baseline.

This reward structure encourages the PPO agent to learn when to trust the base model's signal and when to abstain, effectively filtering low-confidence or unprofitable trades while maintaining flexibility to exploit profitable opportunities.

### 9.4.3 Agent Architecture

The agent layer employs the **Proximal Policy Optimization (PPO)** algorithm with an **LSTM-based Actor–Critic architecture** to learn dynamic and adaptive decision-making policies in the trading environment.

To effectively capture temporal dependencies and encourage robust exploration, the PPO agent integrates three architectural components:

**LSTM (Long Short-Term Memory)**

The LSTM module enables the agent to model sequential dependencies in financial time series, maintaining hidden states that capture evolving market dynamics and internal memory. This makes the actor–critic network recurrent, allowing policy updates conditioned on both current and historical observations.

The hidden state update is defined as:

$$
h_t, c_t = \text{LSTM}(x_t, (h_{t-1}, c_{t-1})) 
$$

where $h_t$ is the hidden state and $c_t$ is the cell state.

The policy and value function outputs are computed as:

$$
\pi_\theta(a_t|s_t, h_t) = \text{softmax}(f_{\text{actor}}(h_t)), \quad V_\phi(s_t, h_t) = f_{\text{critic}}(h_t) 
$$

This hybrid architecture—combining LSTM for temporal learning and NoisyNet for adaptive exploration—enables the PPO agent to capture temporal dependencies, maintain stability in policy optimization, and dynamically adjust exploration according to market uncertainty.

### 9.4.4 Output (Action)

The final output of the model is an action signal derived from both the meta-classifier and the PPO agent:

$$
a_t = \begin{cases} \text{Buy}, & \text{if confidence } > \tau \text{ and PPO policy suggests Buy,} \\ \text{Hold}, & \text{otherwise.} \end{cases} 
$$

These outputs are then translated into backtested trading positions to evaluate real-world performance.

### 9.4.5 Summary and Validation of the Proposed Framework

In this thesis, the primary objective is not to maximize absolute trading performance, but to construct a trading environment that is sufficiently well-specified for a reinforcement learning agent to learn meaningful decision-making behavior. In this sense, the study aims to build a "good enough" environment—one that preserves economically relevant structure, provides coherent reward signals, and supports consistent policy improvement. To validate the adequacy of the proposed environment, a Long Short-Term Memory (LSTM) network is adopted as the policy representation of the agent. LSTM networks possess an explicit memory mechanism and are theoretically capable of exploiting longer temporal dependencies as their hidden state dimensionality increases. Under a correctly specified environment and reward formulation, increasing the representational capacity of the agent—through larger LSTM hidden states—should lead to monotonic or at least non-degrading performance improvements.

Based on this principle, the environment is evaluated indirectly by examining whether agent performance improves as the LSTM hidden state size is increased during training. Observing consistent performance gains under this controlled increase in model capacity provides empirical evidence that the environment and reward structure supply informative and learnable signals, rather than noise-dominated or misleading feedback. In this context, successful scaling behavior of the agent serves as a validation criterion for the environment itself.

Furthermore, the framework integrates a meta-labeling mechanism into the Proximal Policy Optimization (PPO) reward function. Meta-labeling translates directional predictions—validated using classification metrics such as the F1-score or via rule-based trading strategies—into economically meaningful rewards based on realized investment outcomes. This design bridges the gap between statistical signal quality (upward or downward movement prediction) and financial performance (profit and loss).

The experimental results indicate that the meta-labeled reward formulation enables the PPO agent to effectively convert predictive signals into consistent trading decisions. As a result, the learning process reflects both predictive accuracy and economic relevance. The observed improvement in agent performance with increased LSTM capacity further supports the conclusion that the proposed framework provides a valid and functional decision-making environment. In summary, the thesis demonstrates that combining PPO with an LSTM-based policy and a meta-labeling reward mechanism yields a coherent reinforcement learning framework. The framework successfully aligns prediction-level validation with investment-level rewards, thereby validating the decision-making process and confirming the practical viability of the proposed environment.

# Chapter 10: Conclusion

## 10.1 Advantages

The proposed trading decision framework demonstrates an effective structural design for systematic decision-making in financial markets. The integration of a regime-switching model provides a dynamic and adaptive mechanism for capturing market conditions, representing a significant advancement over static approaches. Furthermore, the incorporation of meta-labeling substantially enhances the performance of the reinforcement learning (RL) model by improving signal reliability and reducing false-positive trades.

## 10.2 Limitations

Despite these promising results, several limitations remain. First, the hyperparameters tuning process for the Proximal Policy Optimization (PPO) algorithm requires further refinement to achieve optimal stability and convergence. Second, the performance of the M1-layer models, particularly when combined with various extension techniques, still presents room for improvement. Additionally, the framework exhibits relatively high computational complexity in terms of both time and memory usage, which may limit scalability for real-time applications. Finally, the current system primarily focuses on technical-based analysis; future extensions incorporating fundamental and alternative data sources could further enhance predictive robustness and generalization.