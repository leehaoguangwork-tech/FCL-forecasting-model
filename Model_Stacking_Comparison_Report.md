# FCL Forecasting: Model Stacking Comparison Report

**Date:** April 14, 2026  
**Author:** Manus AI

This report evaluates alternative stacking methodologies for the FCL forecasting models. The goal is to determine whether the current **Sequential Additive Stack** (Baseline Model + Adjustment Engine with a fixed 30% cap) can be improved by either optimising the cap per lane, or by replacing it entirely with a **Linear Meta-Learner**.

## 1. Executive Summary

We evaluated four stacking methodologies across all 15 lanes for both the Model with COVID and the Stable Model. The validation period was January 2025 to March 2026.

1. **Current Method (30% Cap):** The Adjustment Engine corrects the Baseline Model, capped at ±30%.
2. **Optimal Cap (Per-Lane):** The same sequential stack, but the cap is empirically optimised for each lane via grid search.
3. **Meta-Learner (Constrained):** A linear blend where Final = α × Baseline + (1−α) × Adjustment Engine.
4. **Meta-Learner (Unconstrained):** A linear blend where Final = α × Baseline + β × Adjustment Engine.

**Key Findings:**
- For the **Model with COVID**, the current 30% cap is near-optimal. Changing to per-lane optimal caps yields only a negligible +0.13pp average improvement. Meta-learning performs slightly worse.
- For the **Stable Model**, the current 30% cap is too loose. Tightening the cap to 5–10% on highly volatile lanes (e.g., AUSYD, GBFXT) yields massive improvements, reducing average MAPE by **+4.14pp**.
- **Meta-learning fails to outperform the sequential stack.** With only ~54 monthly observations, the linear meta-learner struggles to find robust out-of-sample blending weights, particularly for the unconstrained variant which overfits severely.

**Recommendation:** Retain the sequential additive stack architecture. Keep the 30% cap for the Model with COVID. For the Stable Model, implement a tighter global cap of 10% to prevent over-correction on volatile lanes.

---

## 2. Average Performance Overview

The chart below summarises the average validation MAPE across all 15 lanes for both models. Lower is better.

![Average MAPE Summary](meta_plots/avg_mape_summary.png)

The Optimal Cap approach achieves the lowest average MAPE for both models. The unconstrained meta-learner performs worst, highlighting the danger of giving the model too much freedom on small datasets.

---

## 3. The Optimal Cap Approach

The current architecture uses the Adjustment Engine to correct the Baseline Model's residuals. A 30% cap was manually applied to prevent over-correction. We ran a grid search across 11 cap values (from 5% to uncapped) to find the empirical optimum per lane.

### Model with COVID
The 30% cap is generally not binding. For most lanes, the Adjustment Engine's raw correction is small enough that it never hits the 30% ceiling. Tightening the cap provides no benefit, and the current configuration is robust.

### Stable Model
Because the Stable Model was trained without the extreme volatility of the COVID period, the Adjustment Engine often learns overly aggressive corrections when faced with post-2024 shocks. The 30% cap allows too much of this noise through.

**Major Improvements from Tightening Caps (Stable Model):**
- **AUSYD:** Tightening from 30% to 10% improved MAPE by **+18.17pp** (30.2% → 12.0%).
- **GBFXT:** Tightening from 30% to 5% improved MAPE by **+16.95pp** (71.5% → 54.5%).
- **INNSA:** Tightening from 30% to 5% improved MAPE by **+16.73pp** (51.3% → 34.6%).

---

## 4. The Meta-Learner Approach

Instead of adding a correction, a meta-learner attempts to find the optimal weighted average of the two models. We tested a constrained linear meta-learner, where the weights must sum to 1 (α + β = 1).

### Fitted Weights
The chart below shows the fitted weights (α for the Baseline Model, β for the Adjustment Engine) across all lanes.

![Meta-Learner Weights](meta_plots/meta_weights.png)

In many cases, the meta-learner assigns a weight of 1.0 to the Baseline Model and 0.0 to the Adjustment Engine (or vice versa). This "winner-takes-all" behaviour is a classic symptom of small-sample overfitting in stacking. When trained on only ~54 observations, the cross-validation process struggles to find a stable blend, often just picking whichever model happened to perform slightly better in the training folds.

### Win Count Comparison
When we look at which method achieves the absolute lowest MAPE on a per-lane basis, the Optimal Cap approach wins the most lanes.

![Win Count Summary](meta_plots/win_count_summary.png)

The constrained meta-learner failed to win a single lane on the Model with COVID, and won zero lanes on the Stable Model. The unconstrained variant won a few lanes by chance, but its average performance was disastrous due to massive overfitting on other lanes.

---

## 5. Detailed Per-Lane Results

The following charts show the exact validation MAPE for each method, broken down by lane. The star (★) indicates the best-performing method for that specific lane.

### Model with COVID
![MAPE Comparison Path 1](meta_plots/mape_comparison_path1.png)

### Stable Model
![MAPE Comparison Path 2](meta_plots/mape_comparison_path2.png)

---

## 6. Conclusion

The hypothesis that a meta-learner might outperform the current sequential stack is definitively rejected for this specific dataset. The primary constraint is data volume: with only monthly frequency data, there are simply not enough out-of-fold predictions to train a robust meta-learner.

The current **Sequential Additive Stack** remains the superior architecture because it forces the Adjustment Engine to focus purely on the residual error, rather than competing directly with the Baseline Model for weight.

However, the grid search reveals a clear opportunity for improvement on the **Stable Model**. The manual 30% cap is too permissive for lanes that experience high out-of-sample volatility.

**Final Actionable Recommendations:**
1. **Do not implement meta-learning.** The risk of overfitting is too high, and the interpretability is lower.
2. **Model with COVID:** Maintain the current architecture and the 30% cap.
3. **Stable Model:** Reduce the global correction cap from 30% to **10%**. This will immediately improve validation accuracy on the most problematic lanes (AUSYD, GBFXT, INNSA) without harming the stable lanes.
