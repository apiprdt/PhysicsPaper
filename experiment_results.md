# Experiment Report: Closed-Loop Physics Discovery (MOCK)

- **Proposer Backend**: MOCK
- **Success Rate**: 20/20 (100.0%)
- **Average Discovery Time**: 0.88 seconds per equation

## Performance Benchmark Table

| Problem Name | True Equation | Discovered Equation | Final NMSE | Status | Iterations | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Kinetic Energy | `0.5 * m * v**2` | `theta_0 * m * v**2` | 0.00e+00 | ✅ CONVERGED | 3 | 1.03 |
| Gravitational Force | `G * m * M / r**2` | `theta_0 * G * m * M / r**2` | 4.01e-37 | ✅ CONVERGED | 1 | 0.29 |
| Coulomb's Law | `theta_0 * q1 * q2 / r**2` | `theta_0 * q2 * q1 / r**2` | 7.24e-32 | ✅ CONVERGED | 1 | 1.21 |
| Spring Potential | `0.5 * k * r**2` | `theta_0 * k * r**2` | 0.00e+00 | ✅ CONVERGED | 2 | 1.47 |
| Simple Pendulum Period | `theta_0 * sqrt(r / 9.8)` | `theta_0 * sqrt(r)` | 1.17e-31 | ✅ CONVERGED | 1 | 0.61 |
| Linear Motion | `v * t` | `theta_0 * v * t` | 0.00e+00 | ✅ CONVERGED | 1 | 0.15 |
| Gravitational Potential Energy | `-G * m * M / r` | `theta_0 * G * m * M / r` | 2.28e-35 | ✅ CONVERGED | 1 | 0.19 |
| Relativistic Kinetic Energy | `0.5 * m * v**2 + theta_0 * m * v**4 / c**2` | `theta_0 * m * v**2` | 6.93e-16 | ✅ CONVERGED | 3 | 0.94 |
| Ideal Gas Law | `theta_0 * N * t` | `theta_0 * N * t` | 2.50e-32 | ✅ CONVERGED | 1 | 0.47 |
| Projectile Range | `v**2 * sin(theta_0 * t) / 9.8` | `theta_0 * v**2 * sin(theta_1 * t)` | 7.81e-32 | ✅ CONVERGED | 1 | 0.63 |
| Ohm's Law | `I * R` | `theta_0 * I * R` | 0.00e+00 | ✅ CONVERGED | 1 | 0.72 |
| Power Dissipation | `I**2 * R` | `theta_61 * R * I**2` | 0.00e+00 | ✅ CONVERGED | 1 | 0.57 |
| Capacitor Energy | `0.5 * C * V**2` | `theta_0 * V**2 * sin(theta_1 * C)` | 4.28e-15 | ✅ CONVERGED | 1 | 0.67 |
| Wave Speed | `f * lam` | `theta_0 * f * lam` | 0.00e+00 | ✅ CONVERGED | 1 | 0.69 |
| Doppler Effect | `f * (1 + v / c_s)` | `theta_0 * f * (1 + theta_1 * v / c_s)` | 1.17e-32 | ✅ CONVERGED | 1 | 1.05 |
| Stefan-Boltzmann | `sigma * A * T**4` | `theta_0 * sigma * A * T**4` | 9.01e-32 | ✅ CONVERGED | 1 | 0.86 |
| Wien's Law | `b / T` | `theta_0 * b / T` | 4.07e-35 | ✅ CONVERGED | 1 | 0.85 |
| Lorentz Force | `q * v * B` | `theta_0 * v * B * q**theta_1` | 1.49e-32 | ✅ CONVERGED | 1 | 1.13 |
| Buoyancy Force | `rho * g * V_f` | `theta_0 * g * rho * V_f` | 1.74e-32 | ✅ CONVERGED | 1 | 1.01 |
| Lens Equation | `d_o * d_i / (d_o + d_i)` | `theta_0 * d_o * d_i / (theta_1 * d_o + theta_2 * d_i)` | 0.00e+00 | ✅ CONVERGED | 2 | 3.02 |

## Detailed Discovered Parameters

### Kinetic Energy
- **Discovered Equation Structure**: `theta_0 * m * v**2`
- **Optimized Parameters (θ)**:
  - `theta_0`: 5.000000e-01
- **Final Normalized MSE**: 0.000000e+00
- **Time Taken**: 1.03s

### Gravitational Force
- **Discovered Equation Structure**: `theta_0 * G * m * M / r**2`
- **Optimized Parameters (θ)**:
  - `theta_0`: 1.000000e+00
- **Final Normalized MSE**: 4.005761e-37
- **Time Taken**: 0.29s

### Coulomb's Law
- **Discovered Equation Structure**: `theta_0 * q2 * q1 / r**2`
- **Optimized Parameters (θ)**:
  - `theta_0`: 8.990000e+09
- **Final Normalized MSE**: 7.238101e-32
- **Time Taken**: 1.21s

### Spring Potential
- **Discovered Equation Structure**: `theta_0 * k * r**2`
- **Optimized Parameters (θ)**:
  - `theta_0`: 5.000000e-01
- **Final Normalized MSE**: 0.000000e+00
- **Time Taken**: 1.47s

### Simple Pendulum Period
- **Discovered Equation Structure**: `theta_0 * sqrt(r)`
- **Optimized Parameters (θ)**:
  - `theta_0`: 2.007090e+00
- **Final Normalized MSE**: 1.165217e-31
- **Time Taken**: 0.61s

### Linear Motion
- **Discovered Equation Structure**: `theta_0 * v * t`
- **Optimized Parameters (θ)**:
  - `theta_0`: 1.000000e+00
- **Final Normalized MSE**: 0.000000e+00
- **Time Taken**: 0.15s

### Gravitational Potential Energy
- **Discovered Equation Structure**: `theta_0 * G * m * M / r`
- **Optimized Parameters (θ)**:
  - `theta_0`: -1.000000e+00
- **Final Normalized MSE**: 2.277263e-35
- **Time Taken**: 0.19s

### Relativistic Kinetic Energy
- **Discovered Equation Structure**: `theta_0 * m * v**2`
- **Optimized Parameters (θ)**:
  - `theta_0`: 5.000000e-01
- **Final Normalized MSE**: 6.929648e-16
- **Time Taken**: 0.94s

### Ideal Gas Law
- **Discovered Equation Structure**: `theta_0 * N * t`
- **Optimized Parameters (θ)**:
  - `theta_0`: 1.380000e-23
- **Final Normalized MSE**: 2.504562e-32
- **Time Taken**: 0.47s

### Projectile Range
- **Discovered Equation Structure**: `theta_0 * v**2 * sin(theta_1 * t)`
- **Optimized Parameters (θ)**:
  - `theta_0`: 1.020408e-01
  - `theta_1`: 1.000000e+00
- **Final Normalized MSE**: 7.806908e-32
- **Time Taken**: 0.63s

### Ohm's Law
- **Discovered Equation Structure**: `theta_0 * I * R`
- **Optimized Parameters (θ)**:
  - `theta_0`: 1.000000e+00
- **Final Normalized MSE**: 0.000000e+00
- **Time Taken**: 0.72s

### Power Dissipation
- **Discovered Equation Structure**: `theta_61 * R * I**2`
- **Optimized Parameters (θ)**:
  - `theta_61`: 1.000000e+00
- **Final Normalized MSE**: 0.000000e+00
- **Time Taken**: 0.57s

### Capacitor Energy
- **Discovered Equation Structure**: `theta_0 * V**2 * sin(theta_1 * C)`
- **Optimized Parameters (θ)**:
  - `theta_0`: 5.000000e-01
  - `theta_1`: 1.000000e+00
- **Final Normalized MSE**: 4.283621e-15
- **Time Taken**: 0.67s

### Wave Speed
- **Discovered Equation Structure**: `theta_0 * f * lam`
- **Optimized Parameters (θ)**:
  - `theta_0`: 1.000000e+00
- **Final Normalized MSE**: 0.000000e+00
- **Time Taken**: 0.69s

### Doppler Effect
- **Discovered Equation Structure**: `theta_0 * f * (1 + theta_1 * v / c_s)`
- **Optimized Parameters (θ)**:
  - `theta_0`: 1.000000e+00
  - `theta_1`: 1.000000e+00
- **Final Normalized MSE**: 1.168312e-32
- **Time Taken**: 1.05s

### Stefan-Boltzmann
- **Discovered Equation Structure**: `theta_0 * sigma * A * T**4`
- **Optimized Parameters (θ)**:
  - `theta_0`: 1.000000e+00
- **Final Normalized MSE**: 9.007306e-32
- **Time Taken**: 0.86s

### Wien's Law
- **Discovered Equation Structure**: `theta_0 * b / T`
- **Optimized Parameters (θ)**:
  - `theta_0`: 1.000000e+00
- **Final Normalized MSE**: 4.065721e-35
- **Time Taken**: 0.85s

### Lorentz Force
- **Discovered Equation Structure**: `theta_0 * v * B * q**theta_1`
- **Optimized Parameters (θ)**:
  - `theta_0`: 1.000000e+00
  - `theta_1`: 1.000000e+00
- **Final Normalized MSE**: 1.486525e-32
- **Time Taken**: 1.13s

### Buoyancy Force
- **Discovered Equation Structure**: `theta_0 * g * rho * V_f`
- **Optimized Parameters (θ)**:
  - `theta_0`: 1.000000e+00
- **Final Normalized MSE**: 1.738584e-32
- **Time Taken**: 1.01s

### Lens Equation
- **Discovered Equation Structure**: `theta_0 * d_o * d_i / (theta_1 * d_o + theta_2 * d_i)`
- **Optimized Parameters (θ)**:
  - `theta_0`: 1.000000e+00
  - `theta_1`: 1.000000e+00
  - `theta_2`: 1.000000e+00
- **Final Normalized MSE**: 0.000000e+00
- **Time Taken**: 3.02s

