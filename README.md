# Hybrid Meta-Labeling & Policy Optimization with Regime-Switching for Adaptive Short-Term Trading

> **Graduation Thesis in Applied Mathematics (Financial Engineering & Risk Management)**
> 
> 
> **Author:** Tong Son Lam
> 
> 
> **Institution:** Department of Mathematics
> 

## 📌 Executive Summary

Traditional machine learning classifiers applied directly to financial asset price directional prediction often suffer from high false-positive rates, noise overfitting, and structural regime shifts. This repository provides an end-to-end quantitative framework that decouples **directional signal generation** from **signal execution and sizing** on the Vietnam Stock Exchange (HOSE).

The framework integrates three foundational pillars:

1. **Model-Oriented Feature Selection & Base Prediction (M1 Layer):** Exhaustive combination of filter, wrapper, and embedded selection across 91 technical indicators, optimized via **Bayesian Optimization** under **Purged K-Fold Cross-Validation**.
2. **Latent Regime-Switching Extension:** Unsupervised market regime clustering (Gaussian Mixture Models & Markov-Switching models) yielding regime-conditional predictive experts.
3. **Adaptive Meta-Labeling via Recurrent Reinforcement Learning (M2 Layer):** A **Proximal Policy Optimization (PPO)** agent utilizing an **LSTM Actor-Critic** architecture with parameter-space exploration (**NoisyNet**) to dynamically validate M1 signals and mitigate execution risk under a Partially Observable Markov Decision Process (POMDP).

## 🏗️ System Architecture

The overall pipeline executes a two-layer hierarchical decision process:

```
+---------------------------------------------------------------------------------------+
|                                1. DATA ACQUISITION & ETL                              |
|   Historical OHLCV (VCB / VN30) via `vnstock` + 91 Technical Indicators via `ta`     |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                 2. FIRST LAYER (M1): MODEL-ORIENTED CLASSIFICATION                    |
|  - Combinatorial Feature Selection (Union/Intersection of Filters, Wrappers & Trees)  |
|  - Bayesian Hyperparameter Optimization (Gaussian Process Surrogate)                  |
|  - Base Classifiers: Logistic Regression, Random Forest, XGBoost, KNN                |
|  - Purged K-Fold Time-Series Validation (Preventing Lookahead Leakage)                |
+-------------------------------------------+-------------------------------------------+
                                            |
                         +------------------+------------------+
                         |                                     |
                         v                                     v
+------------------------------------+  +-----------------------------------------------+
|     REGIME-SWITCHING EXTENSION     |  |         PRIMARY PREDICTIONS & CONFIDENCE      |
|  - Latent distribution segmentation|  |  - Directional Signals: s_t in {0, 1}         |
|    (GMM / HMM / MS-GARCH)          |  |  - Posterior Probabilities: p_t in [0, 1]     |
|  - Regime-conditional expert models|  +----------------------+------------------------+
+-----------------+------------------+                         |
                  |                                            |
                  +---------------------+----------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------------+
|                    3. SECOND LAYER (M2): RECURRENT META-LABELING                      |
|  - State Space: Scaled Technical Features + Regime ID + Base Signal + Probabilities   |
|  - Agent: Recurrent PPO (LSTM Shared Backbone + NoisyNet Actor/Critic Heads)          |
|  - Objective: Clipped Surrogate Loss with Generalized Advantage Estimation (GAE)      |
|  - Reward Engineering: Aligned with Realized Trade Return r_t * M1_valid             |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                                  4. EXECUTION ACTION                                  |
|          Filtered Output Action: a_t in {0 (Abstain / Hold), 1 (Execute Buy)}         |
+---------------------------------------------------------------------------------------+
```

## 🧮 Mathematical Formulations

### 1. Meta-Labeling Formulation

Given primary classifier prediction $f(x_t) \in \{0, 1\}$ and realized 1-step return $r_{t+1}$, the binary meta-label $y_{m,t}$ represents trade profitability:

$$y_{m,t} = \begin{cases} 1, & \text{if } f(x_t) = 1 \text{ and } r_{t+1} > 0 \\ 0, & \text{otherwise} \end{cases}$$

The secondary meta-policy $g(s_t, f(x_t))$ optimizes the trade filtering gate, converting high-recall/noisy signals into high-precision trading decisions.

### 2. Bayesian Hyperparameter Optimization

Hyperparameters $\theta^*$ for base classifiers are estimated over a composite objective function balancing classification accuracy and financial utility (Sharpe ratio):

$$\theta^* = \arg\max_{\theta \in \Theta} f(\theta), \quad \text{where } f(\theta) = \lambda \cdot \text{F1}_{\text{CV}}(\theta) + (1 - \lambda) \cdot \text{Sharpe}_{\text{CV}}(\theta)$$

The black-box objective is approximated sequentially using a Gaussian Process surrogate:

$$f(\theta) \sim \mathcal{GP}(m(\theta), k(\theta, \theta'))$$

evaluated with the Expected Improvement (EI) acquisition function.

### 3. Latent Regime Segmentation

Market returns are modeled as finite mixture distributions:

$$p(r_t \vert{} \Theta) = \sum_{k=1}^K \pi_k f_k(r_t \vert{} \mu_k, \sigma_k^2)$$

Parameters $\{\pi_k, \mu_k, \sigma_k\}_{k=1}^K$ are estimated via the Expectation-Maximization (EM) algorithm. Dynamic extensions incorporate Hidden Markov transitions to enforce temporal regime persistence.

### 4. Recurrent PPO Loss with GAE

To account for partial market observability (POMDP), the agent utilizes an LSTM hidden state $h_t = \text{LSTM}(o_t, h_{t-1})$. The clipped surrogate loss is formulated as:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta)\hat{A}_t, \, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

where $\rho_t(\theta) = \frac{\pi_\theta(a_t \vert{} h_t)}{\pi_{\theta_{\text{old}}}(a_t \vert{} h_t)}$, and $\hat{A}_t$ is computed using Generalized Advantage Estimation (GAE-$\lambda$):

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V, \quad \delta_t^V = R_t + \gamma V_\phi(h_{t+1}) - V_\phi(h_t)$$

## 📊 Experimental Results & Validation

The framework was evaluated on historical daily data of **Vietcombank (VCB)** listed on the Ho Chi Minh Stock Exchange (HOSE).

### M1 Classifier Performance (Out-of-Sample Validation)

| **Model** | **Feature Selection Strategy** | **Validation F1-Score** | **Status** |
| --- | --- | --- | --- |
| **K-Nearest Neighbors (KNN)** | Correlation + Variance Filter | 0.3728 | Baseline |
| **Random Forest (RF)** | Embedded Gini Importance | 0.4867 | Moderate |
| **Logistic Regression (LogR)** | Recursive Feature Elimination (RFE) | 0.5759 | High Recall |
| **Extreme Gradient Boosting (XGB)** | Model-Oriented Combinatorial Subset | **0.5761** | **Top Performer** |

### PPO Meta-Model Convergence across LSTM Sequence Lengths

To validate that the RL trading environment provides genuine learning signals rather than fitting noise, the policy's capacity was systematically scaled across sequence horizons $L \in \{5, 16, 20, 50\}$:

- **$L = 5$:** Fast initial learning, saturating around a cumulative reward of $\approx 35-40$.
- **$L = 16$:** Extended memory capacity pushes cumulative reward to $\approx 85$.
- **$L = 20$:** Enhanced stability in volatile periods; cumulative reward exceeds $105$.
- **$L = 50$:** Deep temporal persistence; the agent converges stably with average episode rewards reaching **$> 220$**.

This monotonic improvement under capacity scaling confirms the theoretical validity of embedding recurrent memory for financial POMDPs.

## 📁 Repository Structure

```
Hybrid-Meta-Labeling-on-the-Vietnam-Stock-Exchange/
│
├── .gitignore
├── requirements.txt                   # Dependency specifications
├── README.md                          # Repository documentation
│
├── data/
│   ├── raw/                           # Raw OHLCV price series (VCB, AAPL, MSFT)
│   └── processed/                     # Feature-engineered training/testing matrices
│
├── notebooks/
│   ├── 01_statistical_models/         # GMM, MSM & ARIMA-GARCH regime modeling
│   │   ├── ARIMA_GARCH_auto.ipynb
│   │   ├── GMM.ipynb
│   │   └── MSM.ipynb
│   ├── 02_machine_learning/           # Layer 1 Base Classifiers (KNN, LogR, RF, XGB)
│   │   ├── KNN(new_dataset)-classification.ipynb
│   │   ├── LogisticRegresion(new_dataset)-full_version.ipynb
│   │   ├── RandomForest(new_dataset)-full_version.ipynb
│   │   ├── XGB(new_dataset)-full_version.ipynb
│   │   └── multiple_modeling.ipynb
│   ├── 03_deep_learning/              # Sequence benchmarks (LSTM baselines)
│   │   ├── LTSM(new_dataset)-full_version.ipynb
│   │   └── Model_based_DL.ipynb
│   ├── 04_reinforcement_learning/     # Layer 2 Meta-Agents (Recurrent PPO)
│   │   ├── PPO.ipynb
│   │   ├── PPO_1.ipynb
│   │   ├── RL_main.ipynb
│   │   └── RLandDL_main.ipynb
│   └── 05_pipeline_orchestration/     # End-to-end pipeline & hyperparameter tuning
│       ├── main.ipynb
│       └── Historical_period_optimize.ipynb
│
├── saved_models/                      # Serialized weights and hyperparameter artifacts
│   ├── dl_models/                     # LSTM model checkpoints
│   ├── ml_models/                     # Fitted sklearn & xgboost classifiers per regime
│   └── rl_models/                     # PyTorch actor-critic checkpoints (.pth)
│
└── outputs/
    └── training_plots/                # Training curves, reward smoothing & diagnostics
```

## ⚡ Quick Start

### 1. Clone & Set Up Environment

```
git clone https://github.com/LamTong21/Hybrid-Meta-Labeling-on-the-Vietnam-Stock-Exchange.git
cd Hybrid-Meta-Labeling-on-the-Vietnam-Stock-Exchange

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Fetch Data & Build Features

Run the feature extraction pipeline to compute the 91 technical indicators:

```
from vnstock import stock_historical_data
import ta
import pandas as pd

# Fetch VCB daily data from HOSE
df = stock_historical_data("VCB", "2020-01-01", "2025-12-31", "1D", "stock")

# Clean column headers
df = df.rename(columns={"time": "Date", "open": "Open", "high": "High",
                        "low": "Low", "close": "Close", "volume": "Volume"})
df.set_index("Date", inplace=True)

# Generate comprehensive technical indicator matrix
df_features = ta.add_all_ta_features(
    df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
)
print(f"Constructed feature space dimension: {df_features.shape}")
```

### 3. Reproducing Experiments

1. **Regime Identification:** Open `notebooks/01_statistical_models/GMM.ipynb` to fit latent Gaussian mixtures and output market state partitions.
2. **Layer 1 Classifier Tuning:** Run `notebooks/02_machine_learning/multiple_modeling.ipynb` for Purged Cross-Validation and Bayesian optimization.
3. **Layer 2 Meta-PPO Training:** Execute `notebooks/04_reinforcement_learning/RL_main.ipynb` to train the recurrent Actor-Critic policy using the meta-label reward formulation.

## 📚 Citation

If you find this work or implementation useful in your research, please cite:

```
@bachelorthesis{lam2026hybridmetalabeling,
  author       = {Tong Son Lam},
  title        = {Hybrid Meta-Labeling and Policy Optimization with Regime-Switching for Adaptive Short-Term Trading in the Vietnam Stock Exchange Market},
  year         = {2026},
  month        = {January},
  type         = {Bachelor of Science Thesis},
  address      = {Ho Chi Minh City, Vietnam}
}
```
