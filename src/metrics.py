"""Validation metrics for MEKF attitude estimation: attitude/Euler error time
series, RMSE summaries, and NEES covariance consistency scoring."""

from dataclasses import dataclass
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from quaternion_utils import quat_multiply, quat_conjugate, quat_to_rotvec, quat_to_euler

def wrap_to_pi(angle: np.ndarray) -> np.ndarray:

    return (angle + np.pi) % (2 * np.pi) - np.pi

def attitude_error_rotvec(quat_true: np.ndarray, quat_est: np.ndarray) -> np.ndarray:
    n = len(quat_true)
    err = np.zeros((n, 3))
    for k in range(n):
        dq = quat_multiply(quat_conjugate(quat_est[k]), quat_true[k])
        err[k] = quat_to_rotvec(dq)
    return err

def euler_error(quat_true: np.ndarray, quat_est: np.ndarray) -> np.ndarray:
    euler_true = np.array([quat_to_euler(q) for q in quat_true])
    euler_est = np.array([quat_to_euler(q) for q in quat_est])
    return wrap_to_pi(euler_true - euler_est)

@dataclass
class RMSEResult:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    total_angle_deg: float

def compute_rmse(quat_true: np.ndarray, quat_est: np.ndarray, burn_in: int = 0) -> RMSEResult:
    eul_err = euler_error(quat_true, quat_est)[burn_in:]
    rotvec_err = attitude_error_rotvec(quat_true, quat_est)[burn_in:]
    total_angle = np.linalg.norm(rotvec_err, axis=1)
    def rmse(x):
        return np.sqrt(np.mean(x ** 2))
    return RMSEResult(
        roll_deg=np.rad2deg(rmse(eul_err[:, 0])),
        pitch_deg=np.rad2deg(rmse(eul_err[:, 1])),
        yaw_deg=np.rad2deg(rmse(eul_err[:, 2])),
        total_angle_deg=np.rad2deg(rmse(total_angle)),
    )

def compute_nees(error: np.ndarray, P: np.ndarray) -> np.ndarray:
    n = len(error)
    nees = np.zeros(n)
    for k in range(n):
        nees[k] = error[k] @ np.linalg.solve(P[k], error[k])
    return nees

@dataclass
class NEESResult:
    nees: np.ndarray
    dof: int
    mean_nees: float
    chi2_lower_95: float
    chi2_upper_95: float
    fraction_within_bounds: float
    fraction_above_upper: float
    fraction_below_lower: float

def evaluate_nees(error: np.ndarray, P: np.ndarray, dof: int, burn_in: int = 0) -> NEESResult:
    nees_full = compute_nees(error, P)
    nees = nees_full[burn_in:]
    lower = stats.chi2.ppf(0.025, dof)
    upper = stats.chi2.ppf(0.975, dof)
    return NEESResult(
        nees=nees_full,
        dof=dof,
        mean_nees=float(np.mean(nees)),
        chi2_lower_95=lower,
        chi2_upper_95=upper,
        fraction_within_bounds=float(np.mean((nees >= lower) & (nees <= upper))),
        fraction_above_upper=float(np.mean(nees > upper)),
        fraction_below_lower=float(np.mean(nees < lower)),
    )

@dataclass
class EvaluationResult:
    rmse: RMSEResult
    nees_attitude: NEESResult
    nees_full: NEESResult
    euler_err: np.ndarray
    rotvec_err: np.ndarray

def evaluate(quat_true: np.ndarray, quat_est: np.ndarray,
             bias_true: np.ndarray, bias_est: np.ndarray,
             P_hist: np.ndarray, burn_in: int = 0) -> EvaluationResult:
    eul_err = euler_error(quat_true, quat_est)
    rot_err = attitude_error_rotvec(quat_true, quat_est)
    bias_err = bias_true - bias_est
    full_err = np.hstack([rot_err, bias_err])
    rmse = compute_rmse(quat_true, quat_est, burn_in=burn_in)
    nees_att = evaluate_nees(rot_err, P_hist[:, :3, :3], dof=3, burn_in=burn_in)
    nees_full = evaluate_nees(full_err, P_hist, dof=6, burn_in=burn_in)
    return EvaluationResult(rmse=rmse, nees_attitude=nees_att, nees_full=nees_full, euler_err=eul_err, rotvec_err=rot_err)

def plot_euler_errors(time: np.ndarray, eul_err_rad: np.ndarray, highlight_window: tuple = None, save_path: str = None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["Roll", "Pitch", "Yaw"]
    for i, ax in enumerate(axes):
        ax.plot(time, np.rad2deg(eul_err_rad[:, i]), linewidth=0.8)
        if highlight_window is not None:
            ax.axvspan(*highlight_window, color="orange", alpha=0.15,
                       label="Dynamic disturbance" if i == 0 else None)
        ax.set_ylabel(f"{labels[i]} error (deg)")
        ax.grid(alpha=0.3)
    if highlight_window is not None:
        axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("MEKF Euler Angle Errors")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

def plot_nees(time: np.ndarray, nees_result: NEESResult, highlight_window: tuple = None, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, nees_result.nees, linewidth=0.6, label="NEES")
    if highlight_window is not None:
        ax.axvspan(*highlight_window, color="orange", alpha=0.15, label="Dynamic disturbance")
    ax.axhline(nees_result.dof, color="green", linestyle="--", label=f"Expected (dof={nees_result.dof})")
    ax.axhline(nees_result.chi2_lower_95, color="red", linestyle=":", label="95% band")
    ax.axhline(nees_result.chi2_upper_95, color="red", linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

def plot_nees_comparison(time: np.ndarray, nees_results: dict, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    dof = next(iter(nees_results.values())).dof
    for label, result in nees_results.items():
        ax.plot(time, result.nees, linewidth=0.6, label=label, alpha=0.8)
    ax.axhline(dof, color="green", linestyle="--", label=f"Expected (dof={dof})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("NEES Comparison Across Filter Tunings")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

def plot_bias_estimate(time: np.ndarray, bias_true: np.ndarray, bias_est: np.ndarray, save_path: str = None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["X", "Y", "Z"]
    for i, ax in enumerate(axes):
        ax.plot(time, np.rad2deg(bias_true[:, i]), linewidth=1.0, label="True bias", color="black")
        ax.plot(time, np.rad2deg(bias_est[:, i]), linewidth=0.8, label="Estimated bias", color="tab:blue")
        ax.set_ylabel(f"Bias {labels[i]} (deg/s)")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Gyroscope Bias: True vs. Estimated")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig