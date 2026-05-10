# DeepPulse AI — Presentation Script (7 slides)

---

## Slide 1: The Problem

"An unplanned engine failure on a commercial aircraft grounds the plane for 12-48 hours. At $150,000 per hour of downtime, a single unexpected maintenance event can cost $5M or more — not counting the reputational damage from stranded passengers.

Airlines currently handle this two ways: run-to-failure maintenance, which is obviously bad, and time-based maintenance, where you service every engine every X cycles regardless of condition. Time-based is safer but wasteful — you're often replacing parts that have half their life left."

---

## Slide 2: The Solution

"DeepPulse predicts Remaining Useful Life — how many more flight cycles an engine can run before it needs service.

Instead of a calendar-based schedule, you get a data-driven one. Engine 47 needs attention in 12 cycles. Engine 83 is fine for 90 more. You schedule maintenance for the right engine at the right time."

---

## Slide 3: The Data

"We trained this on NASA's C-MAPSS dataset — a widely used benchmark for predictive maintenance research that simulates turbofan engine degradation. 100 engines run to failure, with 21 sensor readings per flight cycle.

The interesting engineering challenge here: early in an engine's life, the sensors look perfectly healthy even though failure is eventually coming. We handle this by capping the target RUL at 125 cycles — that way the model focuses on the degradation window where sensor changes actually predict the remaining life."

---

## Slide 4: LSTM Architecture

"Standard machine learning models see each cycle in isolation. An LSTM sees the last 30 cycles as a sequence. It learns that a sensor trending steeply downward over 30 cycles means more risk than one that just happens to be low right now.

The architecture is two LSTM layers — 128 units, then 64 — feeding into a single dense output. Input is a sliding 30-cycle window of 14 sensors. The full training pipeline is in the Colab notebook."

---

## Slide 5: Live Demo

"Let me show you the Fleet Overview tab. All 100 test engines plotted by predicted RUL. The red ones — Critical, under 20 cycles — need attention this week. The yellow ones — Warning — need to be scheduled in the next two weeks.

[Select a critical engine in the Engine Deep Dive tab]

Here's Engine 67. The blue line is the actual RUL. The orange dashed line is what the LSTM predicted from sensor data alone. You can see it tracks the degradation accurately through the final 50 cycles — which is exactly the window where the prediction matters."

---

## Slide 6: Model Performance

"On the test set, the model achieves RMSE of 12.6 cycles — it's off by about 12 flight cycles on average. Compare that to the naive baseline, which just predicts the mean RUL for every engine and gets 40 cycles of error. We're 68% better than that.

R² of 0.90 means the model explains 90% of the variance in remaining useful life — not from engine specs or history, just from the last 30 sensor readings."

---

## Slide 7: Business Case

"For a mid-size airline managing 200 engines, a 1% reduction in unplanned downtime saves roughly $2M per year. That's the conservative estimate.

The model retrains on new data monthly. As you collect readings from your own fleet, it adapts to your specific operating conditions and degradation patterns — getting more accurate over time, not less.

The immediate deliverable: integrate with your existing sensor data pipeline, run a 3-month pilot, compare scheduled vs actual maintenance against the model's recommendations. The Colab notebook handles the training side; I handle the deployment."
