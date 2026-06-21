"""
Compare published wide binary velocity boost measurements against
predictions of ADCD / Simple MOND / Standard MOND / RAR families.

Published measurements:
- Chae 2023 (ApJ 953, 153): velocity boost gamma_v ~ 1.20 +- 0.06
  (i.e. ~20% above Newtonian) at low acceleration (< a0)
- Chae 2024 reports acceleration boost gamma ~ 1.4-1.5
- Hernandez 2023: anomaly appears at separation > ~2000-7000 AU

For MOND: in the deep-MOND regime, the predicted velocity boost is
  v_MOND / v_Newton = sqrt(nu(x))
where nu(x) is the interpolating function and x = g_N / a0.
So velocity boost = sqrt(nu(x)).
"""
import numpy as np

# ADCD parameters (fitted on SPARC, no refit)
t0, t1 = 1.8281385981770948, 0.2615186295495788


def nu_adcd(x):
    return t0 * (np.sqrt(1 + t1 / x) - 1) + 1


def nu_simple(x):
    return (1 + np.sqrt(1 + 4 / x)) / 2


def nu_standard(x):
    return 1 / np.sqrt(1 - np.exp(-np.sqrt(x)))


def nu_rar(x):
    return 1 / (1 - np.exp(-np.sqrt(x)))


# Physical constants
G = 6.674e-11
Msun = 1.989e30
AU = 1.496e11
a0 = 1.2e-10

# Published measurements (Chae 2023, Chae 2024, Hernandez 2023)
# Velocity boost = v_obs / v_Newton
print("=" * 78)
print("WIDE BINARY TEST: ADCD vs MOND families vs PUBLISHED OBSERVATIONS")
print("=" * 78)
print()
print("Published measurements (Chae 2023/2024, Hernandez 2023):")
print("  Velocity boost gamma_v = v_obs/v_Newton")
print("  - Low-accel regime (s > 7 kAU): gamma_v ~ 1.20 +- 0.06  (Chae 2023)")
print("  - Acceleration boost gamma_a ~ 1.4-1.5                    (Chae 2024)")
print("  - Anomaly onset: separation > 2000 AU")
print()
print("For MOND: predicted velocity boost = sqrt(nu(x))")
print()

# Wide binary regime: separation -> x = g/a0
# Use M = 1.5 Msun typical
def sep_to_x(s_kAU, M_Msun=1.5):
    s_m = s_kAU * 1e3 * AU
    g_N = G * M_Msun * Msun / s_m ** 2
    return g_N / a0


print("=" * 78)
print("PREDICTED VELOCITY BOOST = sqrt(nu(x)) by separation")
print("=" * 78)
print()
hdr = f'{"s (kAU)":>9} {"x=g/a0":>8} {"ADCD":>7} {"Simple":>7} {"Stand":>7} {"RAR":>7} {"OBS(1.20)":>10}'
print(hdr)
print("-" * len(hdr))
for s in [1, 2, 5, 7, 10, 15, 20, 30, 50]:
    x = sep_to_x(s)
    a = np.sqrt(nu_adcd(x))
    sm = np.sqrt(nu_simple(x))
    st = np.sqrt(nu_standard(x))
    r = np.sqrt(nu_rar(x))
    print(f"{s:>9} {x:>8.3f} {a:>7.3f} {sm:>7.3f} {st:>7.3f} {r:>7.3f} {'~1.20':>10}")

print()
print("=" * 78)
print("VERDICT: where does OBSERVED 1.20 +/- 0.06 land?")
print("=" * 78)
print()
print("At s = 7-20 kAU (Hernandez anomaly zone, x ~ 0.05-0.2):")
for s in [7, 10, 15, 20]:
    x = sep_to_x(s)
    a = np.sqrt(nu_adcd(x))
    sm = np.sqrt(nu_simple(x))
    st = np.sqrt(nu_standard(x))
    r = np.sqrt(nu_rar(x))
    print(f"  s={s:>3} kAU (x={x:.3f}):  "
          f"ADCD={a:.3f}  Simple={sm:.3f}  Stand={st:.3f}  RAR={r:.3f}  |  Obs=1.20±0.06")

print()
print("=" * 78)
print("DISTANCE FROM OBSERVED 1.20 (in sigma, sigma=0.06)")
print("=" * 78)
print()
obs, sig = 1.20, 0.06
print(f'{"s (kAU)":>9} {"x":>8} {"ADCD":>10} {"Simple":>10} {"Stand":>10} {"RAR":>10}')
for s in [5, 7, 10, 15, 20, 30]:
    x = sep_to_x(s)
    preds = {
        "ADCD": np.sqrt(nu_adcd(x)),
        "Simple": np.sqrt(nu_simple(x)),
        "Stand": np.sqrt(nu_standard(x)),
        "RAR": np.sqrt(nu_rar(x)),
    }
    row = f"{s:>9} {x:>8.3f}"
    for name, p in preds.items():
        z = (p - obs) / sig
        row += f" {z:>+9.1f}σ"
    print(row)
