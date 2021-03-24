"""

Note:
    This file is (almost) a copy of
    https://github.com/f-dangel/backpack/blob/development/test/extensions/implementation/autograd.py#L1-L119 # noqa: B950
"""
from test.implementation.base import ExtensionsImplementation

import torch
from backpack.hessianfree.ggnvp import ggn_vector_product, ggn_vector_product_from_plist
from backpack.hessianfree.rop import R_op
from backpack.utils.convert_parameters import vector_to_parameter_list
from torch.nn.utils.convert_parameters import parameters_to_vector


class AutogradExtensions(ExtensionsImplementation):
    """Extension implementations with autograd."""

    def centered_batch_grad(self):
        N_axis = 0
        centered_batch_grad = [bg - bg.mean(N_axis) for bg in self.batch_grad()]

        return centered_batch_grad

    def centered_gram_batch_grad(self):
        batch_grad_flat = self._batch_grad_flat()
        batch_grad_flat -= batch_grad_flat.mean(0)
        return torch.einsum("if,jf->ij", batch_grad_flat, batch_grad_flat)

    def gram_batch_grad(self):
        batch_grad_flat = self._batch_grad_flat()
        return torch.einsum("if,jf->ij", batch_grad_flat, batch_grad_flat)

    def batch_grad(self):
        N = self.problem.input.shape[0]
        batch_grads = [
            torch.zeros(N, *p.size()).to(self.problem.device)
            for p in self.problem.model.parameters()
        ]

        loss_list = torch.zeros((N))
        gradients_list = []
        for b in range(N):
            _, _, loss = self.problem.forward_pass(sample_idx=b)
            gradients = torch.autograd.grad(loss, self.problem.model.parameters())
            gradients_list.append(gradients)
            loss_list[b] = loss

        _, _, batch_loss = self.problem.forward_pass()
        factor = self.problem.get_reduction_factor(batch_loss, loss_list)

        for b, gradients in zip(range(N), gradients_list):
            for idx, g in enumerate(gradients):
                batch_grads[idx][b, :] = g.detach() * factor

        return batch_grads

    def batch_l2_grad(self):
        batch_grad = self.batch_grad()
        batch_l2_grads = [(g ** 2).flatten(start_dim=1).sum(1) for g in batch_grad]
        return batch_l2_grads

    def sgs(self):
        N = self.problem.input.shape[0]
        sgs = [
            torch.zeros(*p.size()).to(self.problem.device)
            for p in self.problem.model.parameters()
        ]

        loss_list = torch.zeros((N))
        gradients_list = []
        for b in range(N):
            _, _, loss = self.problem.forward_pass(sample_idx=b)
            gradients = torch.autograd.grad(loss, self.problem.model.parameters())
            loss_list[b] = loss
            gradients_list.append(gradients)

        _, _, batch_loss = self.problem.forward_pass()
        factor = self.problem.get_reduction_factor(batch_loss, loss_list)

        for _, gradients in zip(range(N), gradients_list):
            for idx, g in enumerate(gradients):
                sgs[idx] += (g.detach() * factor) ** 2
        return sgs

    def variance(self):
        batch_grad = self.batch_grad()
        variances = [torch.var(g, dim=0, unbiased=False) for g in batch_grad]
        return variances

    def diag_ggn(self):
        _, output, loss = self.problem.forward_pass()

        def extract_ith_element_of_diag_ggn(i, p):
            v = torch.zeros(p.numel()).to(self.problem.device)
            v[i] = 1.0
            vs = vector_to_parameter_list(v, [p])
            GGN_vs = ggn_vector_product_from_plist(loss, output, [p], vs)
            GGN_v = torch.cat([g.detach().view(-1) for g in GGN_vs])
            return GGN_v[i]

        diag_ggns = []
        for p in list(self.problem.model.parameters()):
            diag_ggn_p = torch.zeros_like(p).view(-1)

            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_ggn(parameter_index, p)
                diag_ggn_p[parameter_index] = diag_value

            diag_ggns.append(diag_ggn_p.view(p.size()))
        return diag_ggns

    def diag_h(self):
        _, _, loss = self.problem.forward_pass()

        def hvp(df_dx, x, v):
            Hv = R_op(df_dx, x, v)
            return [j.detach() for j in Hv]

        def extract_ith_element_of_diag_h(i, p, df_dx):
            v = torch.zeros(p.numel()).to(self.problem.device)
            v[i] = 1.0
            vs = vector_to_parameter_list(v, [p])

            Hvs = hvp(df_dx, [p], vs)
            Hv = torch.cat([g.detach().view(-1) for g in Hvs])

            return Hv[i]

        diag_hs = []
        for p in list(self.problem.model.parameters()):
            diag_h_p = torch.zeros_like(p).view(-1)

            df_dx = torch.autograd.grad(loss, [p], create_graph=True, retain_graph=True)
            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_h(parameter_index, p, df_dx)
                diag_h_p[parameter_index] = diag_value

            diag_hs.append(diag_h_p.view(p.size()))

        return diag_hs

    def sample_ggn(self, sample_idx=None):
        _, output, loss = self.problem.forward_pass(sample_idx=sample_idx)
        model = self.problem.model

        num_params = sum(p.numel() for p in model.parameters())
        ggn = torch.zeros(num_params, num_params).to(self.problem.device)

        for i in range(num_params):
            # GGN-vector product with i.th unit vector yields the i.th row
            e_i = torch.zeros(num_params).to(self.problem.device)
            e_i[i] = 1.0

            # convert to model parameter shapes
            e_i_list = vector_to_parameter_list(e_i, model.parameters())
            ggn_i_list = ggn_vector_product(loss, output, model, e_i_list)

            ggn_i = parameters_to_vector(ggn_i_list)
            ggn[i, :] = ggn_i

        return ggn

    def ggn(self, subsampling=None):
        if subsampling is None:
            return self.sample_ggn(sample_idx=subsampling)
        else:
            N_axis = 0
            return self.ggn_batch(subsampling=subsampling).sum(N_axis)

    def ggn_batch(self, subsampling=None):
        batch_size = self.problem.input.shape[0]

        # for determining the scaling factor stemming from reduction in the loss
        _, _, batch_loss = self.problem.forward_pass()
        loss_list = torch.zeros(batch_size, device=self.problem.device)

        if subsampling is None:
            subsampling = list(range(batch_size))

        batch_ggn = [None for _ in range(len(subsampling))]

        for b in range(batch_size):
            _, _, loss = self.problem.forward_pass(sample_idx=b)

            if b in subsampling:
                sample_idx = subsampling.index(b)
                ggn = self.sample_ggn(sample_idx=sample_idx)
                batch_ggn[sample_idx] = ggn

            loss_list[b] = loss

        factor = self.problem.get_reduction_factor(batch_loss, loss_list)

        return torch.stack(batch_ggn) * factor

    def diag_ggn_via_ggn(self):
        """Compute full GGN and extract diagonal. Reshape according to param shapes."""
        diag_ggn = self.ggn().diag()

        return vector_to_parameter_list(diag_ggn, self.problem.model.parameters())

    def ggn_mat_prod(self, mat_list):
        """Vectorized multiplication with the Generalized Gauss-Newton/Fisher.

        Uses multiplication with symmetric factors ``V``, ``Vᵀ``, and ``G = V @ Vᵀ``.

        Args:
            mat_list ([torch.Tensor]): Layer-wise split of matrices to be multiplied
                by the GGN. Each item has a free leading dimension, and shares the
                same trailing dimensions with the associated parameter.

        Returns:
            [torch.Tensor]: Result of multiplication with the GGN
        """
        _, output, loss = self.problem.forward_pass()

        parameters = [p for p in self.problem.model.parameters()]

        GGN_mat_list = [None for _ in mat_list]

        for V in range(mat_list[0].shape[0]):
            vec_list = [mat[V] for mat in mat_list]

            GGN_vec_list = ggn_vector_product_from_plist(
                loss, output, parameters, vec_list
            )
            GGN_vec_list = [ggn_v.unsqueeze(0) for ggn_v in GGN_vec_list]

            for idx, ggn_v in enumerate(GGN_vec_list):
                ggn_m = GGN_mat_list[idx]
                ggn_m = ggn_v if ggn_m is None else torch.cat([ggn_m, ggn_v])

                GGN_mat_list[idx] = ggn_m

        return GGN_mat_list
