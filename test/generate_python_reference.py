"""
Generate reference data for Capse.jl lensing tests using the original Python ABCMB code.
- Wigner matrices: ells=2:3000, 200 μ values
- lensed_Cls reference: ellmax=600 AND ellmax=3000

Run from the Capse.jl root:
  python test/generate_python_reference.py
Output:
  test/python_lensing_reference.npz
"""
import sys, os
sys.path.insert(0, "/home/mbonici/Desktop/work/ABCMB")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from abcmb import ABCMBTools as tools

# ── Wigner matrix test grids ──────────────────────────────────────────────────
LMAX_WIGNER = 3000
N_MU        = 200

ells = jnp.arange(2, LMAX_WIGNER + 1)
mu   = jnp.linspace(-0.9999, 0.9999, N_MU)

print(f"Computing Wigner matrices: ells=2:{LMAX_WIGNER}, {N_MU} μ values ...")

n_gl_wigner = (LMAX_WIGNER + 500) + 70
print(f"Computing GL weights at n={n_gl_wigner} (for LensingConfig(2,{LMAX_WIGNER})) ...")
mu_gl, w_gl = tools.gauss_legendre_weights(n_gl_wigner)

data = {
    "gl_mu":  np.array(mu_gl),
    "gl_w":   np.array(w_gl),
    "gl_n":   np.array([n_gl_wigner]),
    "ells":   np.array(ells),
    "mu":     np.array(mu),
    "d00":    np.array(tools.d00(mu, ells)),
    "d11":    np.array(tools.d1n(mu, ells,  1)),
    "d1m1":   np.array(tools.d1n(mu, ells, -1)),
    "d22":    np.array(tools.d2n(mu, ells,  2)),
    "d2m2":   np.array(tools.d2n(mu, ells, -2)),
    "d20":    np.array(tools.d2n(mu, ells,  0)),
    "d31":    np.array(tools.d3n(mu, ells,  1)),
    "d3m1":   np.array(tools.d3n(mu, ells, -1)),
    "d3m3":   np.array(tools.d3n(mu, ells, -3)),
    "d40":    np.array(tools.d4n(mu, ells,  0)),
    "d4m2":   np.array(tools.d4n(mu, ells, -2)),
    "d4m4":   np.array(tools.d4n(mu, ells, -4)),
}

# ── Helper: compute lensed_Cls from Python for a given ellmax ─────────────────
def compute_lensed_cls_python(lmax, ClTT, ClTE, ClEE, Clpp):
    """Compute lensed CMB Cls using the Python ABCMB pipeline."""
    lensing_ells = jnp.arange(2, lmax + 500 + 1)
    n_gl = (lmax + 500) + 70
    mu_l, w_l = tools.gauss_legendre_weights(n_gl)
    mu_ext = jnp.concatenate([mu_l, jnp.array([1.])])
    w_ext  = jnp.concatenate([w_l,  jnp.array([0.])])

    print(f"  Computing 12 Wigner matrices at shape ({len(mu_ext)}, {len(lensing_ells)}) ...")
    d00_l  = tools.d00(mu_ext, lensing_ells)
    d11_l  = tools.d1n(mu_ext, lensing_ells,  1)
    d1m1_l = tools.d1n(mu_ext, lensing_ells, -1)
    d22_l  = tools.d2n(mu_ext, lensing_ells,  2)
    d2m2_l = tools.d2n(mu_ext, lensing_ells, -2)
    d20_l  = tools.d2n(mu_ext, lensing_ells,  0)
    d31_l  = tools.d3n(mu_ext, lensing_ells,  1)
    d3m1_l = tools.d3n(mu_ext, lensing_ells, -1)
    d3m3_l = tools.d3n(mu_ext, lensing_ells, -3)
    d40_l  = tools.d4n(mu_ext, lensing_ells,  0)
    d4m2_l = tools.d4n(mu_ext, lensing_ells, -2)
    d4m4_l = tools.d4n(mu_ext, lensing_ells, -4)

    print(f"  Running lensing pipeline ...")
    ells_j  = jnp.array(lensing_ells)
    ClTT_j  = jnp.array(ClTT);  ClTE_j = jnp.array(ClTE)
    ClEE_j  = jnp.array(ClEE);  Clpp_j = jnp.array(Clpp)

    llp1   = ells_j*(ells_j+1)
    two_l1 = 2.*ells_j+1
    coeff  = two_l1*llp1*Clpp_j

    Cgl    = 1./4./jnp.pi * jnp.sum(coeff*d11_l, axis=1)
    Cgl2   = 1./4./jnp.pi * jnp.sum(coeff*d1m1_l, axis=1)
    sigma2 = Cgl[-1] - Cgl
    Cgl2_c = Cgl2[:,None];  sigma2_c = sigma2[:,None]

    X000       = jnp.exp(-llp1*sigma2_c/4)
    X000_prime = -llp1/4.*X000
    X220       = 1./4.*jnp.sqrt((ells_j+2)*(ells_j-1)*ells_j*(ells_j+1))*jnp.exp(-(llp1-2)*sigma2_c/4.)
    X022       = jnp.exp(-(llp1-4)*sigma2_c/4)
    X022_prime = -(llp1-4)/4*X022
    X121       = -1./2.*jnp.sqrt((ells_j+2)*(ells_j-1))*jnp.exp(-(llp1-8./3.)*sigma2_c/4.)
    X132       = -1./2.*jnp.sqrt((ells_j+3)*(ells_j-2))*jnp.exp(-(llp1-20./3.)*sigma2_c/4.)
    X242       = 1./4.*jnp.sqrt((ells_j+4)*(ells_j+3)*(ells_j-2)*(ells_j-3))*jnp.exp(-(llp1-10.)*sigma2_c/4.)

    ksi  = 1./4./jnp.pi * jnp.sum((2.*ells_j+1)*ClTT_j*(X000**2*d00_l+8./llp1*Cgl2_c*X000_prime**2*d1m1_l+Cgl2_c**2*(X000_prime**2*d00_l+X220**2*d2m2_l)),axis=1)
    ksip = 1./4./jnp.pi * jnp.sum((2.*ells_j+1)*ClEE_j*(X022**2*d22_l+2*Cgl2_c*X132*X121*d31_l+Cgl2_c**2*(X022_prime**2*d22_l+X242*X220*d40_l)),axis=1)
    ksim = 1./4./jnp.pi * jnp.sum((2.*ells_j+1)*ClEE_j*(X022**2*d2m2_l+Cgl2_c*(X121**2*d1m1_l+X132**2*d3m3_l)+0.5*Cgl2_c**2*(2*X022_prime**2*d2m2_l+X220**2*d00_l+X242**2*d4m4_l)),axis=1)
    ksix = 1./4./jnp.pi * jnp.sum((2.*ells_j+1)*ClTE_j*(X022*X000*d20_l+Cgl2_c*2*X000_prime/jnp.sqrt(llp1)*(X121*d11_l+X132*d3m1_l)+0.5*Cgl2_c**2*((2*X022_prime*X000_prime+X220**2)*d20_l+X220*X242*d4m2_l)),axis=1)

    w_col   = w_ext[:,None]
    py_ClTT = 2*jnp.pi*jnp.sum(ksi[:,None]*d00_l*w_col, axis=0)
    py_ClTE = 2*jnp.pi*jnp.sum(ksix[:,None]*d20_l*w_col, axis=0)
    py_ClEE = jnp.pi*jnp.sum((ksip[:,None]*d22_l+ksim[:,None]*d2m2_l)*w_col, axis=0)

    return np.array(py_ClTT), np.array(py_ClTE), np.array(py_ClEE)


# ── lensed_Cls reference at ellmax=600 ────────────────────────────────────────
for lmax_ref in [600, 3000]:
    print(f"\nComputing lensed_Cls reference at ellmax={lmax_ref} ...")
    rng = np.random.default_rng(42)
    Nell = lmax_ref + 500 - 1  # lensing_ells = 2:(lmax+500), length = lmax+499
    ClTT = rng.random(Nell) * 1e-9
    ClTE = rng.random(Nell) * 1e-10
    ClEE = rng.random(Nell) * 1e-10
    Clpp = np.zeros(Nell)

    py_TT, py_TE, py_EE = compute_lensed_cls_python(lmax_ref, ClTT, ClTE, ClEE, Clpp)

    suffix = f"_{lmax_ref}"
    data.update({
        f"lensed_cls_lmax{suffix}":         np.array([lmax_ref]),
        f"lensed_cls_ClTT_unl{suffix}":     ClTT,
        f"lensed_cls_ClTE_unl{suffix}":     ClTE,
        f"lensed_cls_ClEE_unl{suffix}":     ClEE,
        f"lensed_cls_Clpp{suffix}":         Clpp,
        f"lensed_cls_ClTT_python{suffix}":  py_TT,
        f"lensed_cls_ClTE_python{suffix}":  py_TE,
        f"lensed_cls_ClEE_python{suffix}":  py_EE,
    })

# ── save ──────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_lensing_reference.npz")
np.savez(out, **data)
print(f"\nSaved to {out}")
print(f"Wigner shape (d00): {data['d00'].shape}")
print(f"lensed_Cls keys: {[k for k in data if k.startswith('lensed_cls_lmax')]}")
