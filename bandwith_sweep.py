import numpy as np
from scipy.optimize import differential_evolution

k = np.linspace(-np.pi, np.pi, 60)
kx, ky = np.meshgrid(k, k)

def band_structure(params):
    lambda_c, M, t1, t2, t3 = params
    
    dx = np.sin(kx) + lambda_c * np.sin(2 * kx)
    dy = np.sin(ky) + lambda_c * np.sin(2 * ky)
    
    dz = (M 
          + t1 * (np.cos(kx) + np.cos(ky)) 
          + t2 * (np.cos(kx) * np.cos(ky)) 
          + t3 * (np.cos(2*kx) + np.cos(2*ky)))
    
    return -np.sqrt(dx**2 + dy**2 + dz**2)

def objective(params):
    lambda_c, M, t1, t2, t3 = params
    
    # Calculate dz at TRIM points to enforce topology
    dz_Gamma = M + 2*t1 + t2 + 2*t3
    dz_X     = M - t2 + 2*t3       # cos(pi)=-1, cos(0)=1
    dz_M     = M - 2*t1 + t2 + 2*t3
    
    # Strict C=-2 constraint: dz > 0 at Gamma/M, dz < 0 at X/Y
    if dz_Gamma <= 0 or dz_X >= 0 or dz_M <= 0:
        return 1e6  # Massive penalty for breaking topology
        
    E = band_structure(params)
    bandwidth = np.max(E) - np.min(E)
    gap = 2 * np.min(np.abs(E)) 
    
    if gap < 0.05:
        return 1e6 + (0.05 - gap) * 1e7
        
    return bandwidth / gap

bounds = [
    (-2.0, 2.0), # lambda_c
    (-3.0, 3.0), # M
    (-3.0, 3.0), # t1
    ( 0.0, 3.0), # t2 (Constrained positive to help X-point inversion)
    (-3.0, 3.0), # t3
]

print("Running topologically-constrained global parameter sweep...")
result = differential_evolution(
    objective, 
    bounds, 
    strategy='best1bin', 
    maxiter=1000, 
    popsize=20, 
    tol=1e-5,
    mutation=(0.5, 1.0),
    recombination=0.7,
    disp=True
)

opt_params = result.x
E_opt = band_structure(opt_params)
W_opt = np.max(E_opt) - np.min(E_opt)
Gap_opt = 2 * np.min(np.abs(E_opt))

print("\n--- Constrained Optimization Complete ---")
print(f"Optimal parameters [lambda_c, M, t1, t2, t3]:\n{np.round(opt_params, 4)}")
print(f"Bandwidth (W):     {W_opt:.5f}t")
print(f"Gap (Delta):       {Gap_opt:.5f}t")
print(f"Flatness Ratio:    {(Gap_opt / W_opt):.2f}")