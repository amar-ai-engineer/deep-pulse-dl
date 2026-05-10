# DeepPulse — Teaching Notes

## Core concepts

### Remaining Useful Life (RUL) prediction

RUL is the number of operational cycles an asset can complete before it requires maintenance or fails. The goal is to predict this number from sensor readings, without actually running the engine to failure.

**Piecewise linear target (RUL cap):** Early in engine life, the RUL might be 300 cycles, but the sensors show no degradation — they look the same as at cycle 100. Capping RUL at 125 removes the "healthy flat zone" and focuses the model on the degradation signal it can actually learn from. This is standard practice in C-MAPSS benchmarks.

### NASA C-MAPSS dataset

C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) was released by NASA in 2008 as a public benchmark for predictive maintenance research. It simulates turbofan engines degrading under different operating conditions.

**Four sub-datasets:**
| Dataset | Operating conditions | Fault modes | Training engines | Test engines |
|---------|---------------------|-------------|-----------------|--------------|
| FD001   | 1                   | 1           | 100             | 100          |
| FD002   | 6                   | 1           | 260             | 259          |
| FD003   | 1                   | 2           | 100             | 100          |
| FD004   | 6                   | 2           | 249             | 248          |

FD001 is the simplest — single operating condition, one fault mode. It's the standard starting point.

**21 sensors:**
- s2, s3, s4: temperature measurements (LPC/HPC outlet)
- s7, s8, s11: pressure measurements
- s12, s13: fan and LPC speeds
- s14, s15, s17: fuel flow, efficiency indicators
- s1, s5, s6, s10, s16, s18, s19: constant throughout operation — no degradation signal, dropped

**The test set:** each test engine's sensor readings stop at some point before failure. The RUL files give the true remaining life at the last observed cycle. The model must predict this from the final window of readings.

### LSTM architecture

An LSTM (Long Short-Term Memory) network is a recurrent neural network designed to learn long-range dependencies in sequences.

**The problem with vanilla RNNs:** gradients vanish over long sequences. The error signal from cycle 100 barely reaches the model when backpropagating through cycle 70.

**LSTM solution:** a gating mechanism that controls what information is stored, forgotten, and output:
- **Forget gate:** decides what to throw away from the cell state
- **Input gate:** decides what new information to store
- **Output gate:** decides what part of the cell state to expose

For RUL prediction the input is a sliding window of the last 30 cycles. Each window becomes one training example. The LSTM reads the sequence of sensor readings and its final hidden state is passed to a dense output layer.

```
Input shape: (batch_size, 30, 14)
LSTM(128, return_sequences=True) → (batch_size, 30, 128)
Dropout(0.2)
LSTM(64, return_sequences=False) → (batch_size, 64)
Dropout(0.2)
Dense(1) → (batch_size, 1)  # predicted RUL
```

**LSTM vs GRU:** GRU (Gated Recurrent Unit) is a simpler variant with two gates instead of three. It's faster and often matches LSTM on shorter sequences. For C-MAPSS, both perform similarly. LSTM is the standard benchmark choice.

### Feature engineering

**Dropping flat sensors:** sensors s1, s5, s6, s10, s16, s18, s19 have zero or near-zero variance across the entire dataset. They carry no degradation signal and add noise to training. Check with `df[sensor].std()` — anything below 0.01 is a candidate to drop.

**Z-score normalization:** sensors have very different scales (temperature in hundreds of °R vs. pressure ratios near 1.0). Normalizing using training set statistics (μ, σ) ensures no single sensor dominates the loss function.

**Sliding window:** rather than one input per engine, generate overlapping windows:
- Engine 1, cycles 1-30 → target RUL at cycle 30
- Engine 1, cycles 2-31 → target RUL at cycle 31
- ...

This gives you far more training examples and forces the model to generalize across the full degradation trajectory.

### NASA scoring function

The official C-MAPSS scoring function is asymmetric:

```
s = sum(e^(-d/13) - 1 if d < 0 else e^(d/10) - 1)
where d = predicted_RUL - actual_RUL
```

Predicting too much life remaining (d > 0) is penalized more steeply than predicting too little. This reflects the real-world cost: predicting an engine is fine when it's about to fail is catastrophic; predicting it needs maintenance slightly early just wastes a service visit.

We report RMSE in this app because it's more interpretable ("off by 12.6 cycles on average"), but the NASA score is the academic benchmark.

### Comparison to other approaches

| Approach | RMSE FD001 | Notes |
|----------|-----------|-------|
| LSTM (this project) | ~13 | Standard benchmark result |
| Transformer | ~11 | Better but much larger model |
| CNN-LSTM | ~11 | Convolutional feature extraction + LSTM |
| Naive mean prediction | ~40 | Predict mean RUL for all engines |
| Support Vector Regression | ~20 | Classic ML baseline |

---

## Common questions

**Q: Why not just use a simple regression model?**
Linear regression on the last sensor reading ignores the *trajectory* — whether a sensor is getting worse faster. An LSTM can learn that a sensor trending steeply downward over the last 30 cycles means more risk than one that just happens to be low.

**Q: How do you handle multiple operating conditions (FD002, FD004)?**
Cluster the operating conditions first using k-means on (set1, set2, set3) — typically 6 clusters. Normalize each cluster separately. Then add the cluster label as an additional input feature.

**Q: What does 14 RMSE actually mean in practice?**
The model's prediction is off by ±14 flight cycles on average. If a flight cycle is ~4 hours, that's ±56 hours of prediction uncertainty. For a maintenance window planned 2 weeks out, that's accurate enough to be actionable.

**Q: How often do you retrain?**
In production, you'd monitor model performance over time. As the fleet's operating patterns change (new routes, climate, fuel quality), the model drifts. Monthly retraining on the latest sensor data is a common cadence.
