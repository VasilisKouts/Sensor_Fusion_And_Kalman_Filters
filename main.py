"""End to end MEKF sensor fusion pipeline: simulate ICM-20948/AK09916-calibrated
gyro/accel/mag data, run the MEKF under two accelerometer noise tunings
(naive white noise only vs. bias-aware total budget), and report validation
metrics for both."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
from simulation import SimulationConfig, simulate
from ekf import MEKF, MEKFConfig
from metrics import evaluate, plot_euler_errors, plot_nees_comparison, plot_bias_estimate

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
def build_mekf_config(sim_config: SimulationConfig, accel_noise_std: float) -> MEKFConfig:
    noise = sim_config.noise
    incl = np.deg2rad(sim_config.mag_inclination_deg)
    mag_field_nav = np.array([
        0.0,
        sim_config.mag_field_magnitude_uT * np.cos(incl),
        -sim_config.mag_field_magnitude_uT * np.sin(incl),
    ])
    return MEKFConfig(
        mag_field_nav=mag_field_nav,
        gyro_noise_density=noise.gyro_noise_density_rad,
        gyro_bias_rrw=noise.gyro_bias_rrw_rad,
        accel_noise_std=accel_noise_std,
        mag_noise_std=noise.mag_noise_std_uT,
    )

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    report = []
    def log(line=""):
        print(line)
        report.append(line)
    sim_config = SimulationConfig()
    dt = 1.0 / sim_config.sample_rate_hz
    burn_in = int(5.0 * sim_config.sample_rate_hz)
    n_samples = int(round(sim_config.duration_s * sim_config.sample_rate_hz)) + 1
    log("=" * 72)
    log("Sensor Fusion MEKF -- Simulation and Validation Pipeline")
    log("=" * 72)
    log(f"Trajectory duration:      {sim_config.duration_s:.0f} s @ {sim_config.sample_rate_hz:.0f} Hz "
        f"({n_samples} samples)")
    log(f"Dynamic disturbance:      {sim_config.dynamic_phase_start_s:.0f}-"
        f"{sim_config.dynamic_phase_end_s:.0f} s (translational acceleration injected)")
    log("Sensor noise calibration: ICM-20948 (gyro/accel), AK09916 (magnetometer)")
    log("")
    data = simulate(sim_config)
    gt, sens = data.ground_truth, data.sensors
    naive_accel_std = sim_config.noise.accel_noise_density_ms2 * np.sqrt(sim_config.sample_rate_hz / 2)
    tuned_accel_std = np.sqrt(naive_accel_std ** 2 + sim_config.noise.accel_bias_std_ms2 ** 2)
    configs = {
        "naive (white-noise only)": build_mekf_config(sim_config, naive_accel_std),
        "bias-aware (total budget)": build_mekf_config(sim_config, tuned_accel_std),
    }
    results = {}
    for label, cfg in configs.items():
        filt = MEKF(cfg)
        quat_hist, bias_hist, P_hist = filt.run(sens.gyro, sens.accel, sens.mag, dt)
        result = evaluate(gt.quat, quat_hist, sens.true_gyro_bias, bias_hist, P_hist, burn_in=burn_in)
        results[label] = dict(result=result, quat=quat_hist, bias=bias_hist, P=P_hist)
    log(f"Accelerometer R: naive = {naive_accel_std:.4f} m/s^2 (white noise only), "
        f"bias-aware = {tuned_accel_std:.4f} m/s^2 (includes turn-on bias budget)")
    log("")
    log(f"{'Configuration':<28}{'Roll':>10}{'Pitch':>10}{'Yaw':>10}{'NEES':>10}{'%in-band':>11}")
    log(f"{'':<28}{'(deg)':>10}{'(deg)':>10}{'(deg)':>10}{'(3dof)':>10}{'(95%)':>11}")
    log("-" * 79)
    for label, r in results.items():
        rmse, nees = r["result"].rmse, r["result"].nees_attitude
        log(f"{label:<28}{rmse.roll_deg:>10.3f}{rmse.pitch_deg:>10.3f}{rmse.yaw_deg:>10.3f}"
            f"{nees.mean_nees:>10.1f}{nees.fraction_within_bounds * 100:>10.1f}%")
    log("")
    log("Interpretation: the naive filter trusts every accelerometer reading to within its stated")
    log("white-noise floor, ignoring the much larger fixed turn-on bias -- so its covariance shrinks")
    log("far below what its actual accuracy supports (severe overconfidence, NEES far above 3).")
    log("Accounting for the total accelerometer error budget substantially improves both accuracy")
    log("and covariance consistency, though not perfectly: a constant body-frame bias rotated by a")
    log("changing attitude is not statistically identical to isotropic white noise, so residual")
    log("overconfidence remains. A full fix would add an explicit accelerometer-bias state.")
    log("")
    best = results["bias-aware (total budget)"]
    result, bias_hist = best["result"], best["bias"]
    dynamic_window = (sim_config.dynamic_phase_start_s, sim_config.dynamic_phase_end_s)

    plot_euler_errors(gt.time, result.euler_err, highlight_window=dynamic_window,
                       save_path=str(OUTPUT_DIR / "euler_errors.png"))
    plot_bias_estimate(gt.time, sens.true_gyro_bias, bias_hist,
                        save_path=str(OUTPUT_DIR / "gyro_bias_estimate.png"))
    nees_comparison = {label: r["result"].nees_attitude for label, r in results.items()}
    plot_nees_comparison(gt.time, nees_comparison, save_path=str(OUTPUT_DIR / "nees_comparison.png"))

    log(f"Plots saved to: {OUTPUT_DIR}")
    log("  euler_errors.png       -- roll/pitch/yaw error (bias-aware config), dynamic phase shaded")
    log("  gyro_bias_estimate.png -- true vs. estimated gyro bias per axis")
    log("  nees_comparison.png    -- NEES consistency, naive vs. bias-aware tuning")
    log("=" * 72)
    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

if __name__ == "__main__":
    main()