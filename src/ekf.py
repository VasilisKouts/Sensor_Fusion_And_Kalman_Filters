"""Multiplicative Extended Kalman Filter (MEKF) for attitude estimation.
Nominal state: q (body -> nav quaternion), b_g (gyro bias)
Error state: delta_theta (local attitude error), delta_b_g
Error convention (local / right perturbation):
q_true = q_hat(x) delta_q(delta_theta), delta_q ~= [1, 0.5 * delta_theta]
This gives the measurement Jacobian its simplest form, H = [z_hat x, 0]"""

from dataclasses import dataclass, field
import numpy as np
from quaternion_utils import (
    quat_multiply, quat_normalize, quat_integrate, quat_to_rotmat,
    rotate_vector, rotvec_to_quat, skew_symmetric
)

@dataclass
class MEKFConfig:
    gravity: float = 9.80665
    mag_field_nav: np.ndarray = field(default_factory=lambda: np.array([0.0, 25.0, -43.3]))
    gyro_noise_density: float = np.deg2rad(0.015)
    gyro_bias_rrw: float = np.deg2rad(0.0035)
    accel_noise_std: float = 0.016
    mag_noise_std: float = 0.5
    initial_attitude_std_rad: float = np.deg2rad(10.0)
    initial_bias_std_rad: float = np.deg2rad(0.5)
    initial_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

class MEKF:
    """Multiplicative EKF fusing gyroscope (process model) with accelerometer
    and magnetometer reference-vector corrections."""
    def __init__(self, config: MEKFConfig):
        self.config = config
        self.q = quat_normalize(config.initial_quat.copy())
        self.bias = np.zeros(3)
        self.P = np.diag(np.concatenate([
            np.full(3, config.initial_attitude_std_rad ** 2),
            np.full(3, config.initial_bias_std_rad ** 2)
        ]))
    def predict(self, gyro_meas: np.ndarray, dt: float) -> None:
        omega_hat = gyro_meas - self.bias
        self.q = quat_integrate(self.q, omega_hat, dt)
        phi_11 = quat_to_rotmat(rotvec_to_quat(-omega_hat * dt))
        phi_12 = -np.eye(3) * dt
        Phi = np.block([
            [phi_11, phi_12],
            [np.zeros((3, 3)), np.eye(3)]
        ])
        sv2 = self.config.gyro_noise_density ** 2
        sw2 = self.config.gyro_bias_rrw ** 2
        Q = np.block([
            [(sv2 * dt + sw2 * dt ** 3 / 3.0) * np.eye(3), -0.5 * sw2 * dt ** 2 * np.eye(3)],
            [-0.5 * sw2 * dt ** 2 * np.eye(3), sw2 * dt * np.eye(3)]
        ])
        self.P = Phi @ self.P @ Phi.T + Q
    def _update(self, z_meas: np.ndarray, r_nav: np.ndarray, meas_std: float) -> None:
        z_hat = rotate_vector(self.q, r_nav)
        innovation = z_meas - z_hat
        H = np.zeros((3, 6))
        H[:, :3] = skew_symmetric(z_hat)
        R = np.eye(3) * meas_std ** 2
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        delta_x = K @ innovation
        delta_theta, delta_bias = delta_x[:3], delta_x[3:]
        self.q = quat_normalize(quat_multiply(self.q, rotvec_to_quat(delta_theta)))
        self.bias = self.bias + delta_bias
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
    def update_accel(self, accel_meas: np.ndarray) -> None:
        r_nav = np.array([0.0, 0.0, self.config.gravity])
        self._update(accel_meas, r_nav, self.config.accel_noise_std)
    def update_mag(self, mag_meas: np.ndarray) -> None:
        self._update(mag_meas, self.config.mag_field_nav, self.config.mag_noise_std)
    def run(self, gyro: np.ndarray, accel: np.ndarray, mag: np.ndarray, dt: float):
        n = len(gyro)
        quat_hist = np.zeros((n, 4))
        bias_hist = np.zeros((n, 3))
        P_hist = np.zeros((n, 6, 6))
        quat_hist[0] = self.q
        bias_hist[0] = self.bias
        P_hist[0] = self.P
        for k in range(1, n):
            self.predict(gyro[k - 1], dt)
            self.update_accel(accel[k])
            self.update_mag(mag[k])
            quat_hist[k] = self.q
            bias_hist[k] = self.bias
            P_hist[k] = self.P
        return quat_hist, bias_hist, P_hist