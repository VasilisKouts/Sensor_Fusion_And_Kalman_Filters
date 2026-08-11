"""
Quaternion algebra for attitude representation
Convention: Hamilton, scalar first, q = [w, x, y, z], ||q|| = 1
A quaternion q represents the rotation that takes a vector from the BODY
frame to the NAVIGATION frame
"""
import numpy as np

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near zero quaternion.")
    return q / norm

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
    ])

def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_rotmat(q).T @ v

def quat_to_euler(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x ** 2 + y ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y ** 2 + z ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])

def rotvec_to_quat(phi: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        return quat_normalize(np.array([1.0, *(0.5 * phi)]))
    axis = phi / angle
    return np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))])

def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:]
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-8:
        return 2.0 * v
    angle = 2.0 * np.arctan2(v_norm, w)
    return angle * (v / v_norm)

def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    delta_q = rotvec_to_quat(omega * dt)
    return quat_normalize(quat_multiply(q, delta_q))