import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import os

def get_local_basis(point: np.ndarray, target: np.ndarray) -> np.ndarray:
    direction = target - point
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    if abs(np.dot(direction, np.array([0, 0, 1]))) < 0.999:
        ref = np.array([0, 0, 1])
    else:
        ref = np.array([0, 1, 0])
    
    y_axis = np.cross(ref, direction)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    z_axis = np.cross(direction, y_axis)
    
    return np.stack([direction, y_axis, z_axis], axis=0)

def hermite_segment(p0: np.ndarray, p1: np.ndarray, v0: np.ndarray, v1: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[:, np.newaxis]
    t2 = t**2
    t3 = t**3
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1

def generate_trajectory(points: np.ndarray, n_interp: int = 12, curvature: float = 0.8, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    points = np.asarray(points)
    n_control = len(points)
    if n_control < 2:
        return points
    t_vals = np.linspace(0, 1, n_interp)
    trajectory = []
    
    for i in range(n_control - 1):
        p0, p1 = points[i], points[i+1]
        dist = np.linalg.norm(p1 - p0)
        basis0 = get_local_basis(p0, p1)
        basis1 = get_local_basis(p1, p0)
        lateral_strength = curvature * dist * 1.5
        weights0 = np.array([
            np.random.uniform(0.5, 1.0),
            np.random.uniform(-lateral_strength, lateral_strength),
            np.random.uniform(-lateral_strength, lateral_strength)
        ])
        weights1 = np.array([
            np.random.uniform(0.5, 1.0),
            np.random.uniform(-lateral_strength, lateral_strength),
            np.random.uniform(-lateral_strength, lateral_strength)
        ])
        
        v0 = np.sum(weights0[:, np.newaxis] * basis0, axis=0)
        v1 = -np.sum(weights1[:, np.newaxis] * basis1, axis=0)
        seg = hermite_segment(p0, p1, v0, v1, t_vals)
        if i < (n_control - 2):
            trajectory.append(seg[:-1])
        else:
            trajectory.append(seg)
    return np.vstack(trajectory)

