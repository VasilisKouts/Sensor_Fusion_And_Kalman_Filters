"""Sensor noise parameters calibrated to the ICM-20948 (gyro, accel) and
AK09916 (magnetometer) datasheets where published as representative assumptions for this MEMS grade sensor class.
Not published by either manufacturer representative assumptions:
gyro_bias_rrw_rad, mag_noise_std_uT"""

from dataclasses import dataclass, field
import numpy as np
from quaternion_utils import quat_integrate, quat_to_euler, rotate_vector
GRAVITY = 9.80665

@dataclass
class NoiseConfig:
    gyro_noise_density_rad: float = np.deg2rad(0.015)
    accel_noise_density_ms2: float = 230e-6 * GRAVITY
    mag_noise_std_uT: float = 0.5
    gyro_bias_rrw_rad: float = np.deg2rad(0.0035)
    accel_bias_std_ms2: float = 0.015 * GRAVITY

@dataclass
class SimulationConfig:
    duration_s: float = 60.0
    sample_rate_hz: float = 100.0
    seed: int = 42
    mag_field_magnitude_uT: float = 50.0
    mag_inclination_deg: float = 60.0
    omega_amplitudes_rad: tuple = (0.30, 0.25, 0.20)
    omega_frequencies_hz: tuple = (0.045, 0.07, 0.11)
    omega_phases_rad: tuple = (0.0, 1.0, 2.3)
    dynamic_phase_start_s: float = 20.0
    dynamic_phase_end_s: float = 30.0
    dynamic_phase_ramp_s: float = 1.0
    dynamic_accel_amplitude_ms2: float = 3.0
    dynamic_accel_frequency_hz: float = 0.5
    noise: NoiseConfig = field(default_factory=NoiseConfig)

@dataclass
class GroundTruth:
    time: np.ndarray
    quat: np.ndarray
    euler: np.ndarray
    omega: np.ndarray
    accel_nav: np.ndarray

@dataclass
class SensorData:
    gyro: np.ndarray
    accel: np.ndarray
    mag: np.ndarray
    true_gyro_bias: np.ndarray
    accel_bias: np.ndarray

@dataclass
class SimulationData:
    config: SimulationConfig
    ground_truth: GroundTruth
    sensors: SensorData

def _density_to_sigma(density: float, sample_rate_hz: float) -> float:

    return density * np.sqrt(sample_rate_hz / 2.0)


def _random_walk_sigma(density: float, sample_rate_hz: float) -> float:

    return density * np.sqrt(sample_rate_hz)

def _smooth_window(t: np.ndarray, t_start: float, t_end: float, ramp_s: float) -> np.ndarray:
    w = np.zeros_like(t)
    rise = (t >= t_start) & (t < t_start + ramp_s)
    plateau = (t >= t_start + ramp_s) & (t <= t_end - ramp_s)
    fall = (t > t_end - ramp_s) & (t <= t_end)
    w[rise] = 0.5 * (1 - np.cos(np.pi * (t[rise] - t_start) / ramp_s))
    w[plateau] = 1.0
    w[fall] = 0.5 * (1 + np.cos(np.pi * (t[fall] - (t_end - ramp_s)) / ramp_s))
    return w

def _true_angular_velocity(t: np.ndarray, config: SimulationConfig) -> np.ndarray:
    omega = np.zeros((len(t), 3))
    for axis in range(3):
        amp = config.omega_amplitudes_rad[axis]
        freq = config.omega_frequencies_hz[axis]
        phase = config.omega_phases_rad[axis]
        omega[:, axis] = amp * np.sin(2 * np.pi * freq * t + phase)
    return omega

def _true_translational_acceleration(t: np.ndarray, config: SimulationConfig) -> np.ndarray:
    window = _smooth_window(t, config.dynamic_phase_start_s, config.dynamic_phase_end_s,
                             config.dynamic_phase_ramp_s)
    signal = config.dynamic_accel_amplitude_ms2 * np.sin(
        2 * np.pi * config.dynamic_accel_frequency_hz * (t - config.dynamic_phase_start_s)
    )
    accel_nav = np.zeros((len(t), 3))
    accel_nav[:, 0] = signal * window  # East axis
    return accel_nav

def _magnetic_field_nav(config: SimulationConfig) -> np.ndarray:
    b0 = config.mag_field_magnitude_uT
    incl = np.deg2rad(config.mag_inclination_deg)
    return np.array([0.0, b0 * np.cos(incl), -b0 * np.sin(incl)])

def generate_ground_truth(config: SimulationConfig) -> GroundTruth:
    dt = 1.0 / config.sample_rate_hz
    n_samples = int(round(config.duration_s * config.sample_rate_hz)) + 1
    time = np.arange(n_samples) * dt
    omega = _true_angular_velocity(time, config)
    accel_nav = _true_translational_acceleration(time, config)
    quat = np.zeros((n_samples, 4))
    quat[0] = np.array([1.0, 0.0, 0.0, 0.0])
    for k in range(1, n_samples):
        quat[k] = quat_integrate(quat[k - 1], omega[k - 1], dt)

    euler = np.array([quat_to_euler(q) for q in quat])
    return GroundTruth(time=time, quat=quat, euler=euler, omega=omega, accel_nav=accel_nav)

def generate_sensor_data(ground_truth: GroundTruth, config: SimulationConfig,
    rng: np.random.Generator) -> SensorData:
    """Synthesize gyroscope, accelerometer, and magnetometer measurements
    from the ground-truth trajectory, using ICM-20948/AK09916-calibrated
    noise (see NoiseConfig for data provenance)."""
    n = len(ground_truth.time)
    dt = 1.0 / config.sample_rate_hz
    noise = config.noise
    bias_increments = rng.normal(0.0, noise.gyro_bias_rrw_rad * np.sqrt(dt), size=(n, 3))
    bias_increments[0] = 0.0
    true_gyro_bias = np.cumsum(bias_increments, axis=0)
    gyro_sigma = _random_walk_sigma(noise.gyro_noise_density_rad, config.sample_rate_hz)
    gyro_meas = ground_truth.omega + true_gyro_bias + rng.normal(0.0, gyro_sigma, size=(n, 3))
    accel_bias = rng.normal(0.0, noise.accel_bias_std_ms2, size=3)
    accel_sigma = _density_to_sigma(noise.accel_noise_density_ms2, config.sample_rate_hz)
    specific_force_nav = ground_truth.accel_nav + np.array([0.0, 0.0, GRAVITY])
    accel_meas = np.array([
        rotate_vector(ground_truth.quat[k], specific_force_nav[k]) for k in range(n)
    ]) + accel_bias + rng.normal(0.0, accel_sigma, size=(n, 3))
    mag_nav = _magnetic_field_nav(config)
    mag_meas = np.array([
        rotate_vector(ground_truth.quat[k], mag_nav) for k in range(n)
    ]) + rng.normal(0.0, noise.mag_noise_std_uT, size=(n, 3))
    return SensorData(gyro=gyro_meas, accel=accel_meas, mag=mag_meas,
                       true_gyro_bias=true_gyro_bias, accel_bias=accel_bias)

def simulate(config: SimulationConfig = None) -> SimulationData:
    config = config or SimulationConfig()
    rng = np.random.default_rng(config.seed)
    ground_truth = generate_ground_truth(config)
    sensors = generate_sensor_data(ground_truth, config, rng)
    return SimulationData(config=config, ground_truth=ground_truth, sensors=sensors)