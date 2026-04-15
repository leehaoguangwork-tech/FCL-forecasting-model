# FCL Forecasting: Architecture, Cap Optimisation, and LSTM Analysis

**Date:** April 15, 2026  
**Author:** Manus AI

This report documents the architectural design of the FCL forecasting model, explains the rationale behind the hybrid stacking approach, evaluates the optimal constraint limits (caps) for the Adjustment Engine, and analyses the failure of the experimental LSTM triple-stack.

---

## 1. Model Architecture and Rationale

The forecasting system employs a **Sequential Additive Stack** (hybrid model) comprising two distinct layers. Rather than training a single complex model to predict freight rates directly, the task is split into two specialised components.

### The Baseline Model
The first layer is a statistical time-series model (SARIMAX). Its purpose is to capture the linear, structural elements of the freight market:
- **Autoregression (AR):** The tendency of current rates to anchor to recent past rates.
- **Seasonality:** Recurring annual patterns (e.g., pre-Chinese New Year spikes, Golden Week lulls).
- **Macro Exogenous Factors:** Linear responses to external drivers such as Brent crude oil prices, USD/CNY exchange rates, and the BDRY dry bulk proxy.

The Baseline Model produces a stable, interpretable forecast. However, because it is strictly linear, it cannot capture threshold effects (e.g., Brent only impacting rates once it crosses $90/bbl) or complex interactions between variables.

### The Adjustment Engine
The second layer is a tree-based machine learning model (XGBoost). Its purpose is **residual correction**. It does not predict the freight rate directly; instead, it is trained to predict the *errors* (residuals) made by the Baseline Model. 

The Adjustment Engine looks at the same macro conditions, plus the Baseline Model's prediction itself, and learns patterns such as: *"When Brent is rising sharply and it is Q3, the Baseline Model typically under-predicts by $X."* It then applies this learned correction to the final forecast.

### Why This Stack Was Chosen
This sequential architecture was selected over meta-learning ensembles (which average the predictions of multiple standalone models) due to severe data constraints. With only ~54 monthly observations per lane, there are insufficient out-of-fold predictions to train a robust meta-learner. The sequential stack forces the machine learning component to focus purely on the residual error, minimising the risk of overfitting while still capturing non-linear market dynamics.

---

## 2. Adjustment Engine Constraints: 10% vs 30% Cap

Because the Adjustment Engine is a non-linear machine learning model trained on a small dataset, it is prone to over-extrapolating when faced with unprecedented macro conditions. To prevent it from dominating the stable Baseline Model, a hard constraint (cap) is applied to its correction output. 

Historically, this cap was set at **30%** of the Baseline Model's prediction. We evaluated tightening this cap to **10%** across the Jan 2025 – Mar 2026 validation period.

### Results: Model with COVID
For the model trained on the full dataset (including 2020–2021 volatility), tightening the cap from 30% to 10% generally **worsened** performance.

- **Average MAPE Change:** -0.71 percentage points (worse).
- **Winner:** The 30% cap won on 8 lanes, while the 10% cap won on only 3 lanes.

Because this model was exposed to extreme macro shocks during training, its Adjustment Engine learned robust, reliable corrections for volatile conditions. Restricting those corrections to 10% artificially truncates valid market signals.

### Results: Stable Model
For the model trained exclusively on stable, non-COVID data, tightening the cap to 10% significantly **improved** performance.

- **Average MAPE Change:** +2.83 percentage points (better).
- **Winner:** The 10% cap won on 7 lanes, while the 30% cap won on only 4 lanes.

Because the Stable Model never saw extreme volatility during training, its Adjustment Engine tends to panic and over-correct when faced with post-2024 shocks. The 30% cap allows too much of this unreliable correction through.

![MAPE Improvement Delta](cap_report_plots/cap_comparison_delta.png)

As shown above, tightening the cap to 10% rescued the Stable Model on highly volatile lanes, improving validation MAPE on AUSYD by **+18.17pp**, INNSA by **+15.27pp**, and GBFXT by **+14.06pp**.

**Conclusion:** The 30% cap is optimal for the Model with COVID, but the Stable Model requires a tighter 10% cap to prevent erratic over-corrections.

---

## 3. Analysis of the LSTM Triple-Stack Failure

As an experimental extension, a Long Short-Term Memory (LSTM) neural network was inserted between the Baseline Model and the Adjustment Engine. The hypothesis was that the LSTM could learn deep, non-linear temporal dependencies in the Baseline Model's residuals before passing them to the Adjustment Engine.

The architecture tested was:
> **Baseline Model → LSTM (6-month sequence lookback) → Adjustment Engine**

### Validation Results
The triple-stack was tested on the best-performing lane (THBKK) and the worst-performing lane (USLAX). In both cases, the LSTM severely degraded accuracy:

| Lane | Baseline + Adjustment Engine (Current) | Triple-Stack (with LSTM) | Degradation |
|---|---|---|---|
| **THBKK** | 4.6% MAPE | 14.4% MAPE | -9.8pp |
| **USLAX** | 31.0% MAPE | 76.3% MAPE | -45.3pp |

### Why It Failed
The failure of the LSTM is a classic example of **small-sample overfitting in deep learning**. 

1. **Insufficient Sequence Data:** LSTMs are designed for high-frequency data (daily or hourly) with thousands of observations. They require long histories to learn generalisable sequential patterns. With only ~54 monthly observations per lane, the LSTM simply memorised the specific sequence of the training data.
2. **Parameter Bloat:** An LSTM layer introduces hundreds of trainable weights. When applied to a dataset of 54 points, the model has more parameters than data points. It perfectly fits the training residuals but outputs pure noise on the unseen validation set.
3. **Redundancy:** The sequential patterns in the data (trend and seasonality) are already captured by the AR and MA terms of the Baseline Model. The remaining residuals are largely driven by concurrent macro shocks, not deep historical sequences, making the LSTM's memory mechanism redundant.

**Conclusion:** Deep learning sequence models like LSTM are structurally incompatible with low-frequency, small-N monthly macroeconomic datasets. The current two-layer statistical/tree-based stack represents the practical limit of model complexity for this data volume.
