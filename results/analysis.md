
# Analysis and Discussion

## Task 5.1 — Largest Weight in Set A

The largest absolute feature weight in Set A is for **irradiation**.

The learned Normal Equation weight is:

**Irradiation = +8264.8794**

This positive weight is physically sensible because higher solar irradiation generally produces higher AC power from the solar plant.

The module temperature and ambient temperature weights are negative in the fitted model. These weights should not be interpreted independently because the features are related to each other. The sine and cosine terms together represent the daily time-of-day pattern.

---

## Task 5.2 — On-site Sensors vs Public Weather

The daytime test-set RMSE values using the Normal Equation were:

- **Set A (on-site sensors): 723.4147 kW**
- **Set B (Open-Meteo weather): 3417.6468 kW**

Therefore, Set B has a higher daytime RMSE by:

**3417.6468 - 723.4147 = 2694.2320 kW**

The training-set peak hourly AC power was approximately:

**27325.8987 kW**

The RMSE difference as a percentage of the plant peak is:

**9.86%**

This shows that the on-site sensor data provides a much more accurate prediction of plant output than the public weather data for this dataset. Public weather data can still be useful for preliminary estimates, but local sensors are preferable when accurate plant-level prediction is required.

---

## Task 5.3 — Normal Equation vs Batch GD vs SGD

For this dataset, the Normal Equation is the most convenient method because the training set contains only **628 rows** and six model parameters.

The Normal Equation directly calculates the optimal parameters without requiring a learning rate or multiple iterations.

Batch Gradient Descent also converges to essentially the same solution when enough iterations are used. After 5000 iterations with alpha = 0.001, the maximum difference between the Normal Equation and Batch GD parameters was approximately:

**0.000143**

The two methods therefore agree to at least two decimal places.

SGD updates the parameters one training example at a time. It can be useful for very large datasets where processing the complete dataset in every iteration is expensive.

---

## Task 5.4 — Batch GD vs SGD Learning Curves

The Batch Gradient Descent learning curve is smoother because the parameter update is calculated using the complete training dataset at each iteration.

The SGD learning curve is more irregular because the parameters are updated after individual training examples. This introduces more variation between updates.

Therefore, Batch GD generally gives a smoother cost reduction, while SGD can fluctuate more but can be useful for large datasets.

---

## Task 5.5 — Residuals vs Hour of Day

The residual plot shows that the largest prediction errors mainly occur during the daytime, particularly around approximately **10 AM to 3 PM**.

At night, AC power is close to zero, so the residuals are also generally close to zero.

The larger daytime residuals can be caused by rapid changes in solar irradiation, cloud conditions, temperature effects, and nonlinear behaviour of the photovoltaic system. The linear regression model cannot completely capture all of these effects.

Overall, the residual pattern indicates that the model performs better during low-power/night periods and has larger errors when solar generation is high and changing rapidly.
