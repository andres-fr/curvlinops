"""
Trace and diagonal estimation
=============================

In this example we will explore estimators for the trace and diagonal of a matrix.
:code:`curvlinops` implements different methods, and we will reproduce the results
from their original papers, using toy matrices with a power law spectrum.

Here are the imports:
"""

from itertools import count
from os import getenv
from typing import Tuple, Dict, Union

import matplotlib.pyplot as plt
import torch
from torch import (
    Tensor,
    arange,
    as_tensor,
    float64,
    int32,
    linspace,
    median,
    quantile,
    randn,
    stack,
)
from torch.linalg import qr
from tueplots import bundles

from curvlinops.examples import TensorLinearOperator
from skerch.algorithms import hutch as _hutch
from skerch.algorithms import xhutchpp as _xhutchpp


# LaTeX is not available on RTD and we also want to analyze smaller matrices
# to reduce build time
RTD = getenv("READTHEDOCS")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

PLOT_CONFIG = bundles.icml2024(
    column="full" if RTD else "half", usetex=not RTD, nrows=2
)

# Dimension of the matrices whose traces we will estimate
DIM = 200 if RTD else 1000
NUM_PLOT_POINTS = 30
# Number of repeats for the Hutchinson estimator to compute error bars
NUM_REPEATS = 50 if RTD else 200
SEEDS = [12345 + i for i in range(NUM_REPEATS)]

# %%
#
# Setup
# -----
#
# We will use power law matrices whose eigenvalues are given by
# :math:`\lambda_i = i^{-c}`, where :math:`i` is the index of the eigenvalue and
# :math:`c` is a constant that determines the rate of decay of the eigenvalues. A
# higher value of :math:`c` results in a faster decay of the eigenvalues.
#
# Here is a function that creates such a matrix:


def create_power_law_matrix(
    dim: int = DIM, c: float = 1.0, dtype: torch.dtype = float64
) -> Tensor:
    """Draw a matrix with a power law spectrum.

    Eigenvalues λ_i are given by λ_i = i^(-c), where i is the index of the eigenvalue
    and c is a constant that determines the rate of decay of the eigenvalues.
    A higher value of c results in a faster decay of the eigenvalues.

    Args:
        dim: Matrix dimension.
        c: Power law constant. Default is ``1.0``.

    Returns:
        A sample matrix with a power law spectrum.
    """
    # Create the diagonal matrix Λ with Λii = i^(-c)
    L = arange(1, dim + 1, dtype=dtype) ** (-c)

    # Generate a random Gaussian matrix and orthogonalize it to get Q
    Q, _ = qr(randn(dim, dim, dtype=dtype))

    # Construct the matrix A = Q^H Λ Q
    return (Q.H * L) @ Q


# %%
#
# The diagonal is a vector, which makes comparing the estimates by printing their
# entries tedious. Therefore, we will use the relative :math:`L_\infty` error, which is
# the maximum entry of the absolute difference between the estimated and exact diagonal
# entries, divided by the maximum absolute entry of the exact diagonal.


def relative_l_inf_error(est: Tensor, exact: Tensor) -> Tensor:
    """Compute the relative L-infinity error between two vectors.

    Args:
        est: Estimated vector.
        exact: Exact vector.

    Returns:
        Relative L-infinity error.
    """
    return (est - exact).abs().max() / exact.abs().max()


def hutch(lop, lop_device, lop_dtype, num_meas, seed):
    """Girard-Hutchinson estimator."""
    return _hutch(lop, lop_device, lop_dtype, num_meas, seed)


def hutchpp(lop, lop_device, lop_dtype, num_meas, seed):
    """Girard-Hutchinson with rank-deflation."""
    assert num_meas % 3 == 0, "num_meas must be a multiple of 3"
    # from xhutchpp we just use the deflation matrix Q here (num_meas // 3)
    Q = _xhutchpp(lop, lop_device, lop_dtype, num_meas // 3, 0, seed - num_meas)["Q"]
    result = _hutch(lop, lop_device, lop_dtype, num_meas // 3, seed, defl_Q=Q)
    result["diag"] += (Q.T * (Q.H @ lop)).sum(0)  # here another num_meas // 3
    result["tr"] = result["diag"].sum()
    return result


def xdiagtrace(lop, lop_device, lop_dtype, num_meas, seed):
    """XDiag/XTrace estimator."""
    assert num_meas % 2 == 0, "num_meas must be even"
    return _xhutchpp(lop, lop_device, lop_dtype, num_meas // 2, 0, seed)


# %%
#
# Trace estimation
# ----------------
#
# Basics
# ^^^^^^
#
# To get started, let's create a power law matrix and turn it into a linear operator:

Y_mat = create_power_law_matrix(dtype=DTYPE).to(DEVICE)
Y = TensorLinearOperator(Y_mat)

# %%
#
# For reference, let's compute the exact trace:

exact_trace = Y_mat.trace()
print(f"Exact trace: {exact_trace:.3f}")

# %%
#
# The simplest method for trace estimation is Hutchinson's method.
#
# The idea is to estimate the trace from matrix-vector products with random vectors.
# To obtain better estimates, we can use more queries.
# It is common to repeat this process multiple times to get error estimates.
#
# Let's estimate the trace and see if the estimate is decent:

# matrix-vector queries for one trace estimate
num_matvecs = 5

# Generate estimates, repeat process multiple times so we have error bars.
estimates = stack(
    [hutch(Y, Y.device, Y.dtype, num_matvecs, seed)["tr"] for seed in SEEDS]
)

# Calculate the median and quartiles (error bars) of the estimates
med = median(estimates)
quartile1 = quantile(estimates, 0.25)
quartile3 = quantile(estimates, 0.75)

# Print the exact trace and the statistical measures of the estimates
print(f"Exact trace: {exact_trace:.3f}")
print("Estimate:")
print(f"\t- Median: {med:.3f}")
print(f"\t- First quartile (25%): {quartile1:.3f}")
print(f"\t- Third quartile (75%): {quartile3:.3f}")

# Also print whether the true value lies between the quartiles
is_within_quartiles = quartile1 <= exact_trace <= quartile3
print(f"True value within interquartile range? {is_within_quartiles}")
assert is_within_quartiles


# %%
#
# Good! The estimate lies within the error bars.
#
# Comparison
# ^^^^^^^^^^
#
# In the following, we will look at Hutchinson's method and two other algorithms:
# Hutch++ and XTrace. Hutch++ combines vanilla Hutchinson with variance reduction,
# by deterministically computing the trace in a sub-space, and using Hutchinson's
# method in the remaining space. XTrace uses variance reduction from Hutch++,
# and the exchangeability principle (i.e. the estimate is identical when permuting
# the random test vectors). All methods are unbiased, but Hutch++ and XTrace require
# additional memory to store the basis in which the trace is computed exactly.
#
# For matrices whose trace is dominated by a few large eigenvalues, i.e. have fast
# spectral decay, Hutch++ and XTrace can converge faster than vanilla Hutchinson.
# For matrices with slow spectral decay, the benefits of Hutch++ and XTrace become
# less pronounced.
#
# Let's reproduce these results empirically.
#
# We will first consider a power law matrix with high decay rate :math:`c=2.0`:

Y_mat = create_power_law_matrix(c=2.0, dtype=DTYPE).to(DEVICE)


# %%
#
# As before, we will repeat the trace estimation to obtain error bars for each method,
# and investigate how their accuracy evolves as we increase the number of matrix-vector
# products. We use the relative error, which is the absolute value of the difference
# between the estimated and exact trace, divided by the exact trace's absolute value.
#
# Here is a function that computes these relative trace errors for a given matrix:

NUM_MATVECS_HUTCH = linspace(1, 100, NUM_PLOT_POINTS, dtype=int32).unique()
# Hutch++ requires matrix-vector products divisible by 3
NUM_MATVECS_HUTCHPP = (NUM_MATVECS_HUTCH + (3 - NUM_MATVECS_HUTCH % 3)).unique()
# XTrace/XDiag requires matrix-vector products divisible by 2
NUM_MATVECS_X = (NUM_MATVECS_HUTCH + (2 - NUM_MATVECS_HUTCH % 2)).unique()


def compute_relative_errors(
    Y_mat: Tensor,
) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Dict[str, Tensor]]]:
    """Compute the relative errors for Hutchinson, Hutch++, and XTrace.

    Args:
        Y_mat: Matrix to estimate the trace of.

    Returns:
        Dictionaries with the relative trace and diagonal errors.
    """
    Y = TensorLinearOperator(Y_mat)
    exact_diag = Y_mat.diag()
    exact_trace = exact_diag.sum()
    #
    tr_results, diag_results = {}, {}
    for name, method, num_matvecs_method in zip(
        ("Hutchinson", "Hutch++", "Exchanged"),
        (hutch, hutchpp, xdiagtrace),
        (NUM_MATVECS_HUTCH, NUM_MATVECS_HUTCHPP, NUM_MATVECS_X),
    ):
        tr_results[name] = {
            "med": [],
            "quartile1": [],
            "quartile3": [],
            "num_matvecs": [],
        }
        diag_results[name] = {
            "med": [],
            "quartile1": [],
            "quartile3": [],
            "num_matvecs": [],
        }
        for n in num_matvecs_method:
            tr_errors, diag_errors = [], []
            for seed in SEEDS:
                est = method(Y, Y.device, Y.dtype, int(n), seed)
                tr_errors.append((est["tr"] - exact_trace).abs() / abs(exact_trace))
                diag_errors.append(relative_l_inf_error(est["diag"], exact_diag))
            tr_errors, diag_errors = as_tensor(tr_errors), as_tensor(diag_errors)
            tr_results[name]["med"].append(tr_errors.median())
            tr_results[name]["quartile1"].append(tr_errors.quantile(0.25))
            tr_results[name]["quartile3"].append(tr_errors.quantile(0.75))
            tr_results[name]["num_matvecs"].append(n)
            diag_results[name]["med"].append(diag_errors.median())
            diag_results[name]["quartile1"].append(diag_errors.quantile(0.25))
            diag_results[name]["quartile3"].append(diag_errors.quantile(0.75))
            diag_results[name]["num_matvecs"].append(n)
    #
    return tr_results, diag_results


# %%
#
# Let's compute the relative trace errors and look at them:

tr_results, diag_results = compute_relative_errors(Y_mat)

print("Relative errors:")

for method, data in tr_results.items():
    print(f"-\t{method}:")

    num_matvecs = data["num_matvecs"]
    med = data["med"]
    quartile1 = data["quartile1"]
    quartile3 = data["quartile3"]

    # print the first 3 values
    for i in range(3):
        print(
            f"\t\t- {num_matvecs[i]} matvecs: median {med[i]:.5f}"
            + f" (quartiles {quartile1[i]:.3f} - {quartile3[i]:.3f})"
        )

# %%
#
# We should roughly see that the relative error decreases with more matrix-vector
# products.
#
# Let's visualize the convergence with the following function:


def plot_estimation_results(
    results: Dict[str, Dict[str, Tensor]], ax: plt.Axes, target: str = "trace"
) -> None:
    """Plot the trace estimation results on the given Axes.

    Args:
        results: Dictionary with the relative trace errors.
        ax: The matplotlib Axes to plot on.
        target: The property that is approximated (used in ylabel).
            Default is ``'trace'``.
    """
    ax.set_yscale("log")

    for name, data in results.items():
        print("\n\n\n plotting:", name, "from", results.keys())
        num_matvecs = data["num_matvecs"]
        med = data["med"]
        quartile1 = data["quartile1"]
        quartile3 = data["quartile3"]

        ax.plot(num_matvecs, med, label=name)
        ax.fill_between(num_matvecs, quartile1, quartile3, alpha=0.3)

    ax.set_xlabel("Matrix-vector products")
    ax.set_ylabel(f"Relative {target} error")
    ax.legend()


# %%
#
# We will analyze a matrix with fast spectral decay and a matrix with slow spectral
# decay.

# Compute results for matrices with different spectral decay rates
Y_mat_fast = create_power_law_matrix(c=2)  # Fast spectral decay
tr_results_fast, diag_results_fast = compute_relative_errors(Y_mat_fast)

Y_mat_slow = create_power_law_matrix(c=0.5)  # Slow spectral decay
tr_results_slow, diag_results_slow = compute_relative_errors(Y_mat_slow)


# Plot the results for both fast and slow spectral decay
with plt.rc_context(PLOT_CONFIG):
    fig, axes = plt.subplots(nrows=2, sharex=True)
    plot_estimation_results(tr_results_fast, axes[0])
    plot_estimation_results(tr_results_slow, axes[1])
    axes[0].set_title("Fast spectral decay ($c=2$)")
    axes[1].set_title("Slow spectral decay ($c=0.5$)")

    # Remove xlabel from the first, and legend from the second, plot
    axes[0].set_xlabel(None)
    axes[1].legend().remove()

    plt.savefig("./trace_estimation.pdf", bbox_inches="tight")


# %%
#
# For fast spectral decay, Hutch++ and XTrace yield more accurate trace estimates than
# vanilla Hutchinson. For slow spectral decay, the benefits of Hutch++ and XTrace
# disappear. Thankfully, many curvature matrices in deep learning exhibit a decaying
# spectrum, which may allow Hutch++ and XTrace to improve over Hutchinson.
#
# Diagonal estimation
# -------------------
#
# Basics
# ^^^^^^
#
# Diagonal estimation is similar to trace estimation.
#
# To give a concrete example, let's create a power law matrix, turn it into a linear
# operator, and compute its diagonal for reference:

Y_mat = create_power_law_matrix()
Y = TensorLinearOperator(Y_mat)
exact_diag = Y_mat.diag()

# %%
#
# The simplest method for diagonal estimation is Hutchinson's method.
#
# The idea is to estimate the diagonal from matrix-vector products with random vectors.
# To obtain better estimates, we can use more queries.
# It is common to repeat this process multiple times to get error estimates.
#
# Let's estimate the diagonal and see if the estimate is decent:

# matrix-vector queries for one diagonal estimate
num_matvecs = 5

# Generate estimates, repeat process multiple times so we have error bars.
estimates = [hutch(Y, Y.device, Y.dtype, num_matvecs, seed)["diag"] for seed in SEEDS]
errors = stack([relative_l_inf_error(e, exact_diag) for e in estimates])

# Calculate the median and quartiles (error bars) of the estimates
med = median(errors)
quartile1 = quantile(errors, 0.25)
quartile3 = quantile(errors, 0.75)

# Print error stats
print("Relative errors:")
print(f"\t- Median: {med:.3f}")
print(f"\t- First quartile (25%): {quartile1:.3f}")
print(f"\t- Third quartile (75%): {quartile3:.3f}")


# %%
#
# Comparison
# ^^^^^^^^^^
#
# We will compare Hutchinson's method for diagonal estimation with the XDiag method
# on matrices with fast and slow spectral decay.
#
# Here is a function that computes these relative diagonal errors for a given matrix:


# %%
#
# For plotting, we can re-purpose the function we used earlier to visualize the trace
# estimation results:


# Plot the results for both fast and slow spectral decay
with plt.rc_context(PLOT_CONFIG):
    fig, axes = plt.subplots(nrows=2, sharex=True)
    plot_estimation_results(diag_results_fast, axes[0], target="diagonal")
    plot_estimation_results(diag_results_slow, axes[1], target="diagonal")
    axes[0].set_title("Fast spectral decay ($c=2$)")
    axes[1].set_title("Slow spectral decay ($c=0.5$)")

    # Remove xlabel from the first, and legend from the second, plot
    axes[0].set_xlabel(None)
    axes[1].legend().remove()

    plt.savefig("./diagonal_estimation.pdf", bbox_inches="tight")


# %%
#
# For fast spectral decay, XDiag yields more accurate diagonal estimates than
# vanilla Hutchinson. For slow spectral decay, its benefits disappear.
#
# That's all for now.


"""
TODO: log all changes for PR explanation!

we need to explain that we compute trace and diagonal together: introduce the computation, metric and plotting there

also, compute the fast and slow altogether when we introduce the synthmat function


The rest is just discussion of results, first for trace and then for diag

also added diag++ to plot
"""
