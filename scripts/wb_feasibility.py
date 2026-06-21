"""Decisive feasibility test: can wide binaries distinguish ADCD form from MOND?"""
import numpy as np

print("=" * 72)
print("DECISIVE TEST: Can wide binaries distinguish ADCD from MOND families?")
print("=" * 72)
print()
print("ADCD form = Simple MOND family (both are a + b*sqrt(1+c/x), a+b=1)")
print("So this test is REALLY: Simple-family vs Standard vs RAR vs Newtonian")
print()

G = 6.674e-11
Msun = 1.989e30
AU = 1.496e11
a0 = 1.2e-10

print("=== Wide binary acceleration regime ===")
print(f'{"sep (kAU)":>12} {"g_Newton":>16} {"x = g/a0":>12}')
for s_kAU in [0.5, 2, 7, 20]:
    s_m = s_kAU * 1e3 * AU
    g_N = G * 1.5 * Msun / s_m ** 2
    x = g_N / a0
    print(f"{s_kAU:>12.1f} {g_N:>16.3e} {x:>12.4f}")

print()
print("=== PREDICTED velocity boost nu(x) in wide binary regime ===")
print("This is THE discriminating observable.")
print()

t0, t1 = 1.8281385981770948, 0.2615186295495788


def nu_adcd(x):
    return t0 * (np.sqrt(1 + t1 / x) - 1) + 1


def nu_simple(x):
    return (1 + np.sqrt(1 + 4 / x)) / 2


def nu_standard(x):
    return 1 / np.sqrt(1 - np.exp(-np.sqrt(x)))


def nu_rar(x):
    return 1 / (1 - np.exp(-np.sqrt(x)))


header = f'{"x":>8} {"ADCD":>9} {"Simple":>9} {"Standard":>10} {"RAR":>8} {"ADCD-Simp":>11}'
print(header)
print("-" * len(header))
for x in [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
    a = nu_adcd(x)
    s = nu_simple(x)
    st = nu_standard(x)
    r = nu_rar(x)
    diff_pct = 100 * (a - s) / s
    print(f"{x:>8.3f} {a:>9.4f} {s:>9.4f} {st:>10.4f} {r:>8.4f} {diff_pct:>+10.1f}%")

print()
print("=== CRITICAL: distinguishability of ADCD vs Simple MOND ===")
print("ADCD and Simple MOND are the SAME family. Difference is only refit.")
print("At x~0.1 (s=7 kAU, Hernandez threshold):")
print()
x_test = np.array([0.05, 0.1, 0.2, 0.5, 1.0])
hdr = f'{"x":>8} {"ADCD":>8} {"Simple":>8} {"Stand":>8} {"RAR":>8} {"A-S":>8} {"S-St":>8}'
print(hdr)
print("-" * len(hdr))
for x in x_test:
    a = nu_adcd(x)
    s = nu_simple(x)
    st = nu_standard(x)
    r = nu_rar(x)
    print(
        f"{x:>8.3f} {a:>8.3f} {s:>8.3f} {st:>8.3f} {r:>8.3f} "
        f"{100*(a-s)/s:>+7.1f}% {100*(s-st)/st:>+7.1f}%"
    )

print()
print("=" * 72)
print("VERDICT ON FEASIBILITY")
print("=" * 72)
print()
print("Observational scatter in wide binaries (Hernandez 2023, Chae 2023):")
print("  - ~10-30% in relative velocity at s > 7 kAU")
print("  - Detection significance of MOND anomaly: ~2-4 sigma")
print()
print("Spread between model families at x=0.1:")
for x in [0.05, 0.1, 0.2]:
    a = nu_adcd(x); s = nu_simple(x); st = nu_standard(x); r = nu_rar(x)
    print(f"  x={x:.2f}: Simple vs Standard = {100*(s-st)/st:+.1f}%, Simple vs RAR = {100*(s-r)/r:+.1f}%")
