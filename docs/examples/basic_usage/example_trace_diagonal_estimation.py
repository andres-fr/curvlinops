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
    [
        hutch(Y, Y.device, Y.dtype, num_matvecs, SEEDS[i])["tr"]
        for i in range(NUM_REPEATS)
    ]
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

NUM_MATVECS_HUTCH = linspace(1, 100, 50, dtype=int32).unique()
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
    results = {}
    for method, num_matvecs_method in zip(
        (hutch, hutchpp, xdiagtrace),
        (NUM_MATVECS_HUTCH, NUM_MATVECS_HUTCHPP, NUM_MATVECS_X),
    ):
        med = []
        quartile1 = []
        quartile3 = []

        for n in num_matvecs_method:
            """
            TODO: dispatch this cleanly
            """
            breakpoint()
            for seed in SEEDS:
                method(Y, Y.device, Y.dtype, int(n), seed)
            estimates = [method(Y, Y.device, Y.dtype, int(n), seed) for seed in SEEDS]

            print("====")
            breakpoint()
            errors = (estimates - exact_trace).abs() / abs(exact_trace)
            med.append(median(errors))
            quartile1.append(quantile(errors, 0.25))
            quartile3.append(quantile(errors, 0.75))

        results[name] = {
            "med": as_tensor(med),
            "quartile1": as_tensor(quartile1),
            "quartile3": as_tensor(quartile3),
            "num_matvecs": num_matvecs_method,
        }


aa = hutch(Y, Y.device, Y.dtype, 300, SEEDS[0])
bb = hutchpp(Y, Y.device, Y.dtype, 300, SEEDS[0])
cc = xdiagtrace(Y, Y.device, Y.dtype, 300, SEEDS[0])

compute_relative_errors(Y_mat)
breakpoint()


def compute_relative_diagonal_errors(Y_mat: Tensor) -> Dict[str, Dict[str, Tensor]]:
    """Compute the relative diagonal errors for Hutchinson's method and XDiag.

    Args:
        Y_mat: Matrix to estimate the diagonal of.

    Returns:
        Dictionary with the relative diagonal errors.
    """
    Y = TensorLinearOperator(Y_mat)
    exact_diag = Y_mat.diag()

    # compute median and quartiles for Hutchinson's method
    estimators = {
        "Hutchinson": hutchinson_diag,
        "XDiag": xdiag,
    }
    num_matvecs = [NUM_MATVECS_HUTCH, NUM_MATVECS_XDIAG]

    results = {}
    for (name, method), num_matvecs_method in zip(estimators.items(), num_matvecs):
        med = []
        quartile1 = []
        quartile3 = []

        for n in num_matvecs_method:
            estimates = [method(Y, n) for _ in range(NUM_REPEATS)]
            errors = stack([relative_l_inf_error(e, exact_diag) for e in estimates])
            med.append(median(errors))
            quartile1.append(quantile(errors, 0.25))
            quartile3.append(quantile(errors, 0.75))

        results[name] = {
            "med": as_tensor(med),
            "quartile1": as_tensor(quartile1),
            "quartile3": as_tensor(quartile3),
            "num_matvecs": num_matvecs_method,
        }

    return results


def compute_relative_trace_errors(Y_mat: Tensor) -> Dict[str, Dict[str, Tensor]]:
    """Compute the relative trace errors for Hutchinson's method, Hutch++, and XTrace.

    Args:
        Y_mat: Matrix to estimate the trace of.

    Returns:
        Dictionary with the relative trace errors.
    """
    Y = TensorLinearOperator(Y_mat)
    exact_trace = Y_mat.trace()

    # compute median and quartiles for Hutchinson's method
    estimators = {
        "Hutchinson": hutchinson_trace,
        "Hutch++": hutchpp_trace,
        "XTrace": xtrace,
    }
    num_matvecs = [NUM_MATVECS_HUTCH, NUM_MATVECS_HUTCHPP, NUM_MATVECS_XTRACE]

    results = {}
    for (name, method), num_matvecs_method in zip(estimators.items(), num_matvecs):
        med = []
        quartile1 = []
        quartile3 = []

        for n in num_matvecs_method:
            estimates = stack([method(Y, n) for _ in range(NUM_REPEATS)])
            errors = (estimates - exact_trace).abs() / abs(exact_trace)
            med.append(median(errors))
            quartile1.append(quantile(errors, 0.25))
            quartile3.append(quantile(errors, 0.75))

        results[name] = {
            "med": as_tensor(med),
            "quartile1": as_tensor(quartile1),
            "quartile3": as_tensor(quartile3),
            "num_matvecs": num_matvecs_method,
        }

    return results


# %%
#
# Let's compute the relative trace errors and look at them:

results = compute_relative_trace_errors(Y_mat)

print("Relative errors:")

for method, data in results.items():
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
Y_mat_fast = create_power_law_matrix()  # Fast spectral decay with c=2
results_fast = compute_relative_trace_errors(Y_mat_fast)

Y_mat_slow = create_power_law_matrix(c=0.5)  # Slow spectral decay with c=0.5
results_slow = compute_relative_trace_errors(Y_mat_slow)

# Plot the results for both fast and slow spectral decay
with plt.rc_context(PLOT_CONFIG):
    fig, axes = plt.subplots(nrows=2, sharex=True)
    plot_estimation_results(results_fast, axes[0])
    plot_estimation_results(results_slow, axes[1])
    axes[0].set_title("Fast spectral decay ($c=2$)")
    axes[1].set_title("Slow spectral decay ($c=0.5$)")

    # Remove xlabel from the first, and legend from the second, plot
    axes[0].set_xlabel(None)
    axes[1].legend().remove()

    plt.savefig("trace_estimation.pdf", bbox_inches="tight")

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
estimates = [hutchinson_diag(Y, num_matvecs) for _ in range(NUM_REPEATS)]
errors = stack([relative_l_inf_error(e, exact_diag) for e in estimates])

# Calculate the median and quartiles (error bars) of the estimates
med = median(errors)
quartile1 = quantile(errors, 0.25)
quartile3 = quantile(errors, 0.75)

# Print the exact trace and the statistical measures of the estimates
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
# on a matrices with fast and slow spectral decay.
#
# Here is a function that computes these relative diagonal errors for a given matrix:

NUM_MATVECS_HUTCH = linspace(1, 100, 50, dtype=int32).unique()
# XTrace requires matrix-vector products divisible by 2
NUM_MATVECS_XDIAG = (NUM_MATVECS_HUTCH + (2 - NUM_MATVECS_HUTCH % 2)).unique()


def compute_relative_diagonal_errors(Y_mat: Tensor) -> Dict[str, Dict[str, Tensor]]:
    """Compute the relative diagonal errors for Hutchinson's method and XDiag.

    Args:
        Y_mat: Matrix to estimate the diagonal of.

    Returns:
        Dictionary with the relative diagonal errors.
    """
    Y = TensorLinearOperator(Y_mat)
    exact_diag = Y_mat.diag()

    # compute median and quartiles for Hutchinson's method
    estimators = {
        "Hutchinson": hutchinson_diag,
        "XDiag": xdiag,
    }
    num_matvecs = [NUM_MATVECS_HUTCH, NUM_MATVECS_XDIAG]

    results = {}
    for (name, method), num_matvecs_method in zip(estimators.items(), num_matvecs):
        med = []
        quartile1 = []
        quartile3 = []

        for n in num_matvecs_method:
            estimates = [method(Y, n) for _ in range(NUM_REPEATS)]
            errors = stack([relative_l_inf_error(e, exact_diag) for e in estimates])
            med.append(median(errors))
            quartile1.append(quantile(errors, 0.25))
            quartile3.append(quantile(errors, 0.75))

        results[name] = {
            "med": as_tensor(med),
            "quartile1": as_tensor(quartile1),
            "quartile3": as_tensor(quartile3),
            "num_matvecs": num_matvecs_method,
        }

    return results


# %%
#
# For plotting, we can re-purpose the function we used earlier to visualize the trace
# estimation results:

# Compute results for matrices with different spectral decay rates
Y_mat_fast = create_power_law_matrix()  # Fast spectral decay with c=2
results_fast = compute_relative_diagonal_errors(Y_mat_fast)

Y_mat_slow = create_power_law_matrix(c=0.5)  # Slow spectral decay with c=0.5
results_slow = compute_relative_diagonal_errors(Y_mat_slow)

# Plot the results for both fast and slow spectral decay
with plt.rc_context(PLOT_CONFIG):
    fig, axes = plt.subplots(nrows=2, sharex=True)
    plot_estimation_results(results_fast, axes[0], target="diagonal")
    plot_estimation_results(results_slow, axes[1], target="diagonal")
    axes[0].set_title("Fast spectral decay ($c=2$)")
    axes[1].set_title("Slow spectral decay ($c=0.5$)")

    # Remove xlabel from the first, and legend from the second, plot
    axes[0].set_xlabel(None)
    axes[1].legend().remove()

    plt.savefig("diagonal_estimation.pdf", bbox_inches="tight")

# %%
#
# For fast spectral decay, XDiag yields more accurate diagonal estimates than
# vanilla Hutchinson. For slow spectral decay, its benefits disappear.
#
# That's all for now.
