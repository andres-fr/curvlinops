"""Autograd implementation of operations in ``lowrank.optim``."""

from test.implementation.autograd import AutogradExtensions

import torch
from backpack.utils.convert_parameters import vector_to_parameter_list


class AutogradOptimExtensions(AutogradExtensions):
    """Autograd implementation of optimizer functionality with similar API."""

    def gammas_ggn(self, top_k, subsampling_directions=None, subsampling_first=None):
        """First-order directional derivatives along the top-k GGN eigenvectors.

        Args:
            top_k (int): Number of leading eigenvectors used as directions. Will be
                clipped to ``[1, max]`` with ``max`` the maximum number of nontrivial
                eigenvalues.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_first ([int], optional): Sample indices used for individual
                gradients.
        """
        return super().gammas_ggn(
            top_k,
            ggn_subsampling=subsampling_directions,
            grad_subsampling=subsampling_first,
        )

    def lambdas_ggn(self, top_k, subsampling_directions=None, subsampling_second=None):
        """Second-order directional derivatives along the top-k GGN eigenvectors.

        Args:
            top_k (int): Number of leading eigenvectors used as directions. Will be
                clipped to ``[1, max]`` with ``max`` the maximum number of nontrivial
                eigenvalues.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_second ([int], optional): Sample indices used for individual
                curvature matrices.
        """
        return super().lambdas_ggn(
            top_k,
            ggn_subsampling=subsampling_directions,
            lambda_subsampling=subsampling_second,
        )

    def newton_step(
        self,
        top_k,
        damping,
        subsampling_directions=None,
        subsampling_first=None,
        subsampling_second=None,
    ):
        """Directionally-damped Newton step along the top-k GGN eigenvectors.

        Args:
            top_k (int): Number of leading eigenvectors used as directions. Will be
                clipped to ``[1, max]`` with ``max`` the maximum number of nontrivial
                eigenvalues.
            damping (lowrank.optim.damping.BaseDamping): Policy for selecting
                dampings along a direction from first- and second- order directional
                derivatives.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_first ([int], optional): Sample indices used for individual
                gradients.
            subsampling_second ([int], optional): Sample indices used for individual
                curvature matrices.
        """
        gammas, evecs = super().gammas_ggn(
            top_k,
            ggn_subsampling=subsampling_directions,
            grad_subsampling=subsampling_first,
            directions=True,
        )
        lambdas = super().lambdas_ggn(
            top_k,
            ggn_subsampling=subsampling_directions,
            lambda_subsampling=subsampling_second,
        )
        deltas = damping(gammas, lambdas)

        batch_axis = 0
        scale = -gammas.mean(batch_axis) / (lambdas.mean(batch_axis) + deltas)

        step = torch.einsum("d,id->i", scale, evecs)

        return vector_to_parameter_list(step, self.problem.model.parameters())
