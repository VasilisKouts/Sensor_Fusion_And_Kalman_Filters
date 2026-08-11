[README.md](https://github.com/user-attachments/files/30937055/README.md)
# Multiplicative EKF for IMU Attitude Estimation

A Multiplicative Extended Kalman Filter (MEKF) built from scratch, fusing gyroscope, accelerometer, and magnetometer measurements into a real time attitude estimate. Tested on synthetic ICM-20948/AK09916 calibrated sensor data, with validation that goes beyond accuracy plots into NEES consistency testing.

This started as a learning project, and the goal throughout was getting the underlying estimation theory right, not just working code. Every core equation below was checked independently against finite-difference Jacobians, closed form covariance identities, and Monte Carlo simulation before any of it was trusted. See [Validation Methodology](#validation-methodology) for details.

---

## Table of Contents

- [Headline Result](#headline-result)
- [Mathematical Formulation](#mathematical-formulation)
- [Project Structure](#project-structure)
- [Sensor Noise Model](#sensor-noise-model)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Validation Methodology](#validation-methodology)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## Headline Result

A correctly implemented filter can still be **statistically dishonest about its own accuracy** if its noise model is incomplete. Running the identical filter on identical data with two different accelerometer noise assumptions:

| Configuration | Roll RMSE | Pitch RMSE | Yaw RMSE | Mean NEES (dof=3, expected ≈3) |
|---|---|---|---|---|
| Naive (white-noise only) | 1.088° | 2.622° | 2.048° | **51,202** |
| Bias-aware (total error budget) | 0.572° | 0.892° | 1.773° | **663** |

The naive configuration builds its accelerometer measurement covariance $R$ purely from the ICM-20948's white noise datasheet figure, ignoring the sensor's much larger fixed turn on bias (roughly 9 times larger). The result is a filter that converges, looks accurate, and is *severely overconfident*: its reported uncertainty is orders of magnitude smaller than its actual error. Accounting for the full error budget improves both accuracy and consistency by roughly two orders of magnitude. It still doesn't fully fix the problem, and [Limitations](#limitations--future-work) explains why.

---

## Mathematical Formulation

### State representation

The filter separates a **nominal state**, which is exact and unconstrained, from a small **error state**, which is what the Kalman machinery actually operates on:

$$
\text{Nominal: } \quad \mathbf{x} = (\hat{\mathbf{q}},\ \hat{\mathbf{b}}_g) \in \mathbb{S}^3 \times \mathbb{R}^3
\qquad
\text{Error: } \quad \delta\mathbf{x} = (\delta\boldsymbol\theta,\ \delta\mathbf{b}_g) \in \mathbb{R}^6
$$

with the **local (right multiplicative) error convention**:

$$
\mathbf{q}_{true} = \hat{\mathbf{q}} \otimes \delta\mathbf{q}(\delta\boldsymbol\theta), \qquad \delta\mathbf{q}(\delta\boldsymbol\theta) \approx \begin{bmatrix}1 \\ \tfrac{1}{2}\delta\boldsymbol\theta\end{bmatrix}
$$

Quaternions use the Hamilton convention, scalar-first: $\mathbf{q} = [w,\,x,\,y,\,z]$, $\|\mathbf{q}\|=1$.

### Quaternion kinematics

The nominal quaternion is propagated via the *exact* nonlinear kinematic equation, not a linearization:

$$
\dot{\mathbf{q}} = \frac{1}{2}\,\mathbf{q} \otimes \begin{bmatrix}0 \\ \boldsymbol\omega\end{bmatrix}
\qquad\Longrightarrow\qquad
\mathbf{q}_{k+1} = \mathbf{q}_k \otimes \exp\!\left(\begin{bmatrix}0 \\ \tfrac{1}{2}\boldsymbol\omega\,\Delta t\end{bmatrix}\right)
$$

where the quaternion exponential of a pure rotation vector $\boldsymbol\phi$ has a closed form:

$$
\exp(\boldsymbol\phi) = \begin{bmatrix}\cos(\|\boldsymbol\phi\|/2) \\ \dfrac{\boldsymbol\phi}{\|\boldsymbol\phi\|}\sin(\|\boldsymbol\phi\|/2)\end{bmatrix}
$$

### Error-state dynamics and discretization

Linearizing the true vs. estimated kinematics under the local error convention gives the continuous time error dynamics:

$$
\dot{\delta\boldsymbol\theta} = -\hat{\boldsymbol\omega}\times\delta\boldsymbol\theta - \delta\mathbf{b}_g - \boldsymbol\eta_v,
\qquad
\dot{\delta\mathbf{b}}_g = \boldsymbol\eta_w
$$

Discretizing over one sample interval $\Delta t$, holding $\hat{\boldsymbol\omega}$ constant, gives a discrete transition matrix. Its attitude block is the *exact* matrix exponential, using the same rotation exponential defined above rather than a first order approximation:

$$
\Phi_k = \begin{bmatrix}\Phi_{11} & -\mathbf{I}_3\,\Delta t \\ \mathbf{0}_3 & \mathbf{I}_3\end{bmatrix},
\qquad
\Phi_{11} = \exp\!\big(-[\hat{\boldsymbol\omega}\times]\,\Delta t\big)
$$

and a discrete process noise covariance, derived by integrating $\Phi(\tau)\,G\,Q_c\,G^T\,\Phi(\tau)^T$ over the interval:

$$
Q_k = \begin{bmatrix}
\left(\sigma_v^2 \Delta t + \tfrac{1}{3}\sigma_w^2 \Delta t^3\right)\mathbf{I}_3 & -\tfrac{1}{2}\sigma_w^2 \Delta t^2\,\mathbf{I}_3 \\[4pt]
-\tfrac{1}{2}\sigma_w^2 \Delta t^2\,\mathbf{I}_3 & \sigma_w^2 \Delta t\,\mathbf{I}_3
\end{bmatrix}
$$

Here $\sigma_v$ is the gyroscope angle random walk density and $\sigma_w$ is the bias rate-random-walk density (continuous-time spectral densities, in rad/s/$\sqrt{\text{Hz}}$ and rad/s/$\sqrt{\text{s}}$ respectively). The covariance prediction step is then the standard

$$
P_k^- = \Phi_k\, P_{k-1}^+\, \Phi_k^T + Q_k
$$

### Measurement model

Both the accelerometer and magnetometer are modeled as noisy observations of a known nav frame reference vector $\mathbf{r}_{nav}$, rotated into the body frame:

$$
\hat{\mathbf{z}} = R(\hat{\mathbf{q}})^T\, \mathbf{r}_{nav}, \qquad \mathbf{y} = \mathbf{z}_{meas} - \hat{\mathbf{z}}
$$

For the accelerometer, $\mathbf{r}_{nav} = [0,0,g]^T$. Note that this reflects *specific force*, $\mathbf{f} = \mathbf{a}_{true} - \mathbf{g}$, not gravity directly. A stationary sensor reads $+g$ pointing up, which is the reaction force, not gravity itself. For the magnetometer, $\mathbf{r}_{nav}$ is Earth's local field vector, parameterized by magnitude and inclination.

The measurement Jacobian, re derived here from the local error convention rather than assumed from a reference, is:

$$
H = \begin{bmatrix}[\hat{\mathbf{z}}\times] & \mathbf{0}_3\end{bmatrix} \in \mathbb{R}^{3\times 6}
$$

Because a skew-symmetric matrix's null space is the vector it was built from, $H$ has a direct geometric reading: **rotation about the reference vector axis is unobservable from that sensor alone**. The accelerometer sees tilt but not heading, and the magnetometer sees heading but not tilt, which is exactly why both are needed.

### Correction step

$$
S = H P^- H^T + R, \qquad K = P^- H^T S^{-1}, \qquad \delta\mathbf{x} = K\,\mathbf{y}
$$

The correction is folded back into the nominal state **multiplicatively** (hence MEKF), and the covariance update uses Joseph form for numerical robustness against small gain computation errors:

$$
\hat{\mathbf{q}}^+ = \hat{\mathbf{q}}^- \otimes \delta\mathbf{q}(\delta\boldsymbol\theta), \qquad \hat{\mathbf{b}}_g^+ = \hat{\mathbf{b}}_g^- + \delta\mathbf{b}_g
$$

$$
P^+ = (I - KH)\,P^-\,(I-KH)^T + K R K^T
$$

### NEES as a validation metric

Filter consistency is assessed using the Normalized Estimation Error Squared, which compares actual error against the filter's *self reported* uncertainty rather than just its magnitude:

$$
\epsilon_k = \delta\mathbf{x}_k^T\, (P_k)^{-1}\, \delta\mathbf{x}_k \ \sim\ \chi^2_{n_x}
$$

A correctly calibrated filter's NEES should average to its error-state dimension $n_x$ (3 for attitude only, 6 for the full state). Large deviations reveal either overconfidence ($\epsilon \gg n_x$) or underconfidence ($\epsilon \ll n_x$). See [Headline Result](#headline-result) for what this looked like in practice.

---

## Project Structure

```
SensorFusion/
├── src/
│   ├── quaternion_utils.py   # Quaternion algebra: Hamilton convention, exp/log maps
│   ├── simulation.py         # Ground truth trajectory + ICM-20948/AK09916-calibrated synthetic sensors
│   ├── ekf.py                 # The MEKF: predict/correct, as derived above
│   └── metrics.py             # RMSE, NEES, and diagnostic plotting
├── main.py                    # Orchestration: simulate -> filter (naive & bias-aware) -> evaluate -> plot
└── requirements.txt
```

## Sensor Noise Model

Calibrated to the ICM-20948 (gyro/accel) and AK09916 (magnetometer) datasheets where those parameters are actually published, and flagged explicitly where they are not.

| Parameter | Value | Source |
|---|---|---|
| Gyroscope noise density | 0.015 °/s/√Hz | ICM-20948 Datasheet (TDK DS-000189 Rev 1.5), Table 1: **datasheet** |
| Accelerometer noise density | 230 µg/√Hz | Same datasheet, Table 2: **datasheet** |
| Accelerometer turn-on bias std | 15 mg (within the ±25 mg component-level tolerance) | Same datasheet: **datasheet-bounded** |
| Magnetometer noise (RMS) | ~0.5 µT | **Not published** by AKM, a representative assumption |
| Gyro bias instability / rate random walk | ~3.0 °/hr equivalent | **Not published**, a representative assumption for this MEMS class |

One subtlety is worth explaining here. Converting a continuous noise *density* to a discrete per-sample standard deviation uses different conventions depending on whether that noise is consumed as a direct measurement or *integrated* by the consumer. Point measurements, accelerometer and magnetometer, use the Nyquist convention: $\sigma = \text{density}\times\sqrt{f_s/2}$. Gyroscope noise is integrated by the attitude kinematics, so it needs the full-rate convention instead, $\sigma = \text{density}\times\sqrt{f_s}$, to stay consistent with the angle random walk property $\text{Var} = \text{density}^2 \times T$ that the filter's own $Q_k$ assumes. Using the wrong convention here silently desynchronizes the simulated noise from what the filter's math expects. This was a real bug, caught during Monte Carlo validation (see `_density_to_sigma` vs. `_random_walk_sigma` in `simulation.py`).

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.12. Three direct dependencies: `numpy`, `scipy`, `matplotlib`.

## Usage

```bash
python main.py
```

This runs a 60 s / 100 Hz synthetic trajectory, with a 10-second dynamic-acceleration disturbance injected at t=20-30s, fits the MEKF under both a naive and a bias aware accelerometer tuning, prints a comparison report, and writes plots plus `summary.txt` to `outputs/`.

## Validation Methodology

Every module was numerically verified before being trusted. Writing code that looks right isn't the same as confirming it is.

- **`quaternion_utils.py`**: round trip identities, orthogonality/determinant checks, and cross validation against `scipy.spatial.transform.Rotation`.
- **`simulation.py`**: deterministic (noise-free) physics checks, confirming specific force reproduces gravity exactly and magnetometer rotation exactly preserves field magnitude.
- **`ekf.py`**: the measurement Jacobian confirmed against finite differences. The attitude transition matrix confirmed exact, not merely first-order, under matched angular velocity via a group-conjugation identity. The bias-covariance growth confirmed against a closed form derived by hand. The process noise discretization confirmed via Monte Carlo simulation.
- **`metrics.py`**: NEES machinery statistically calibrated, so errors truly drawn from $N(0,P)$ recover a mean NEES within 5% of the theoretical value.

## Limitations & Future Work

- **No accelerometer-bias state.** The fixed turn on bias is only approximately absorbed via an inflated $R$. A constant body frame bias rotated by a changing attitude isn't statistically identical to isotropic white noise, so the bias-aware configuration's NEES (~663) doesn't fully return to the theoretical ~3. The complete fix is an explicit bias state, estimated the same way gyro bias already is.
- **No hard/soft-iron magnetometer distortion modeling.** A clean field was assumed for v1.
- **Single-rate fusion.** All three sensors run at a shared 100 Hz. Real hardware typically samples the magnetometer slower than gyro/accel, which would require asynchronous multi-rate correction.
- **EKF, not an exact nonlinear filter.** Linearization is well justified, since the per step rotation at 100 Hz is small, but an Unscented Kalman Filter would remove that approximation at a higher computational cost.
- **Synthetic data only.** There are no real ICM-20948/AK09916 hardware logs yet.
