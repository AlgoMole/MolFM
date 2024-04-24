import torch
import numpy as np
from torchdiffeq import odeint
from egnn import EGNN
from torch.distributions.categorical import Categorical
from torch import nn

def T(t):
    # 0   0, 1 beta_max
    beta_min = 0.1
    beta_max = 20
    return 0.5 * (beta_max - beta_min) * t**2 + beta_min * t


def T_hat(t):
    # 0 beta_min, 1 beta_max
    beta_min = 0.1
    beta_max = 20
    return (beta_max - beta_min) * t + beta_min


#Mask Functions
def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected

def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked

class Cnflows(torch.nn.Module):
    """
    The E(n) continous normalizing flows Module.
    """

    def __init__(
        self,
        dynamics,
        in_node_nf: int,
        n_dims: int,
        timesteps: int = 10000,
        parametrization="eps",
        time_embed=False,
        noise_schedule="learned",
        noise_precision=1e-4,
        loss_type="ot",
        norm_values=(1.0, 1.0, 1.0),
        norm_biases=(None, 0.0, 0.0),
        include_charges=True,
        discrete_path="OT_path",
        cat_loss="l2",
        cat_loss_step=-1,
        on_hold_batch=-1,
        sampling_method="vanilla",
        weighted_methods="jump",
        ode_method="dopri5",
        without_cat_loss=False,
        angle_penalty=False,
    ):
        super().__init__()

        # assert loss_type in {'ot'}
        self.set_odeint(method=ode_method)
        self.loss_type = loss_type
        self.include_charges = include_charges
        self._eps = 0.0  # TODO: fix the trace computation part
        self.discrete_path = discrete_path
        self.ode_method = ode_method

        self.cat_loss = cat_loss
        self.cat_loss_step = cat_loss_step
        self.on_hold_batch = on_hold_batch
        self.sampling_method = sampling_method
        self.weighted_methods = weighted_methods
        self.without_cat_loss = without_cat_loss
        self.angle_penalty = angle_penalty

        self.dynamics = dynamics
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges
        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.time_embed = time_embed
        self.register_buffer("buffer", torch.zeros(1))

        if time_embed:
            self.register_buffer(
                "frequencies", 2 ** torch.arange(self.frequencies) * torch.pi
            )

        # if noise_schedule != 'learned':
        #     self.check_issues_norm_values()

    def set_odeint(self, method="dopri5", rtol=1e-4, atol=1e-4):
        self.method = method
        self._atol = atol
        self._rtol = rtol
        self._atol_test = 1e-7
        self._rtol_test = 1e-7

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1.0 / max_norm_value:
            raise ValueError(
                f"Value for normalization value {max_norm_value} probably too "
                f"large with sigma_0 {sigma_0:.5f} and "
                f"1 / norm_value = {1. / max_norm_value}"
            )

    def phi(self, t, x, node_mask, edge_mask, context):
        # TODO: check the frequencies buffer. input is embedding to get better performance.
        if self.time_embed:
            t = self.frequencies * t[..., None]
            t = torch.cat((t.cos(), t.sin()), dim=-1)
            t = t.expand(*x.shape[:-1], -1)

        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)

        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(
            self.norm_values[0]
        )
        h_cat = (
            (h["categorical"].float() - self.norm_biases[1])
            / self.norm_values[1]
            * node_mask
        )
        h_int = (h["integer"].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {"categorical": h_cat, "integer": h_int}

        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):  # Check the unnormalize_z function
        # Parse from z
        x, h_cat = (
            z[:, :, 0 : self.n_dims],
            z[:, :, self.n_dims : self.n_dims + self.num_classes],
        )
        h_int = z[
            :, :, self.n_dims + self.num_classes : self.n_dims + self.num_classes + 1
        ]
        # print("unnormalize_", h_int.size(),x.size(), h_cat.size())
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    def sample_p_xh_given_z0(self, dequantizer, z0, node_mask):
        """Samples x ~ p(x|z0)."""

        x = z0[:, :, : self.n_dims]
        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        # if self.include_charges:
        x, h_cat, h_int = self.unnormalize(
            x, z0[:, :, self.n_dims : self.n_dims + self.num_classes], h_int, node_mask
        )
        tensor = dequantizer.reverse({"categorical": h_cat, "integer": h_int})
        one_hot, charges = tensor["categorical"], tensor["integer"]
        h = {"integer": charges, "categorical": one_hot}

        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps
            # get position, categorical, integer loss
            # Combining the terms
    
    def decode(self, z, node_mask, edge_mask, context):
        def wrapper(t, x):
            dx = self.phi(t, x, node_mask, edge_mask, context)
            if self.cat_loss_step > 0:
                if t > self.cat_loss_step:
                    dx[:, :, self.n_dims : -1] = 0
                else:
                    dx[:, :, self.n_dims : -1] = dx[:, :, self.n_dims : -1] / (
                        self.cat_loss_step
                    )
            if self.discrete_path == "VP_path":
                M_para = (
                    -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                )  # add epsilon to stable it
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path":
                M_para = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :] * M_para
            else:
                pass

            return dx

        
        t_list = [1.0, 0]
        t_list = torch.tensor(t_list, dtype=torch.float, device=z.device)

        return odeint(
            wrapper, z, t_list, method=self.method, rtol=self._rtol, atol=self._atol
        )

    def decode_chain(self, z, t, node_mask, edge_mask, context):
        # here t is all the model which we used to decode
        def wrapper(t, x):
            dx = self.phi(t, x, node_mask, edge_mask, context)
            if self.cat_loss_step > 0:
                if t > self.cat_loss_step:
                    dx[:, :, self.n_dims : -1] = 0
                else:
                    dx[:, :, self.n_dims : -1] = dx[:, :, self.n_dims : -1] / (
                        self.cat_loss_step
                    )
                # cat_mask = t.squeeze() < self.cat_loss_step
                # dx[~cat_mask][:,self.n_dims:-1] = 0
                # dx[cat_mask][:,self.n_dims:-1] = dx[cat_mask][:,self.n_dims:-1] / self.cat_loss_step # align the speed.
            if self.discrete_path == "VP_path":
                M_para = (
                    -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                )  # add epsilon to stable it
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path":
                M_para = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :] * M_para
            else:
                pass
            return dx

        t = torch.tensor(t, dtype=torch.float, device=z.device)

        return odeint(
            wrapper, z, t, method=self.method, rtol=self._rtol, atol=self._atol
        )

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def sample_cat_z0(self, xh, node_mask, edge_mask, context):
        """
        get the catgorical distribution according to coordinate and features.
        """
        # whether input use a xh or else.
        t = torch.zeros_like(xh[:, 0, 0]).view(-1, 1, 1)
        net_out = self.phi(0.0, xh, node_mask, edge_mask, context)
        z_h = net_out[
            :, :, self.n_dims : -1
        ]  # use the score function as the sampling direction. Instead of the ode results.
        xh[
            :, :, self.n_dims : -1
        ] = z_h  # replace the original xh with the sampled one.

        return xh



    @torch.no_grad()
    def sample(
        self,
        dequantizer,
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
        context,
        fix_noise=False,
    ):
        """
        Draw samples from the generative model.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask
            )

        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        # def decode(self, z,node_mask,edge_mask,context) -> Tensor:
        z_ = self.decode(z, node_mask, edge_mask, context)[-1]

        if self.sampling_method == "gradient":
            # time_step = [1e-2]
            # for i in range(time_step):
            init = z_[:, :, self.n_dims : -1]
            # print(init.norm(dim=2))
            categorical_steps = np.linspace(0.05, 0, 20)
            for i_ in categorical_steps:
                # slightly perturb
                gradient = self.phi(
                    torch.tensor([i_]), z_, node_mask, edge_mask, context
                )
                init = init + gradient[:, :, self.n_dims : -1] * (0.05 / 20)

            z_[:, :, self.n_dims : -1] = init
        elif self.sampling_method == "vanilla":
            pass
        else:
            raise NotImplementedError
        x, h = self.sample_p_xh_given_z0(dequantizer, z_, node_mask)

        assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(
                f"Warning cog drift with error {max_cog:.3f}. Projecting "
                f"the positions down."
            )
            x = remove_mean_with_mask(x, node_mask)

        return x, h

    @torch.no_grad()
    def sample_chain(
        self,
        dequantizer,
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
        context,
        keep_frames=None,
    ):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)
        if keep_frames is None:
            keep_frames = 100
        else:
            assert keep_frames <= 1000

        # chain = torch.zeros((keep_frames,) + z.size(), device=z.device)
        time_step = list(np.linspace(1, 0, keep_frames))

        chain_z = self.decode_chain(z, time_step, node_mask, edge_mask, context)

        for i in range(len(chain_z) - 1):
            ##fix chain sampling
            chain_z[i] = self.unnormalize_z(chain_z[i], node_mask)
            

        chain_z = reversed(chain_z)
        x, h = self.sample_p_xh_given_z0(
            dequantizer, chain_z[-1], node_mask
        )  # TODO this should be the reverse of our flow model
        assert_mean_zero_with_mask(x[:, :, : self.n_dims], node_mask)

        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
        # print(chain_z.size(),xh.size(),h['integer'], h['categorical'],chain_z[0])
        chain_z[0] = xh  # Overwrite last frame with the resulting x and h.
        chain_flat = chain_z.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

class EGNN_dynamics_QM9(nn.Module):
    def __init__(self,
                 in_node_nf,
                 context_node_nf,
                 n_dims,
                 hidden_nf=64,
                 device='cpu',
                 act_fn=torch.nn.SiLU(),
                 n_layers=4,
                 attention=False,
                 condition_time=True,
                 tanh=False,
                 mode='egnn_dynamics',
                 norm_constant=0,
                 inv_sublayers=2,
                 sin_embedding=False,
                 normalization_factor=100,
                 aggregation_method='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(in_node_nf=in_node_nf + context_node_nf,
                             in_edge_nf=1,
                             hidden_nf=hidden_nf,
                             device=device,
                             act_fn=act_fn,
                             n_layers=n_layers,
                             attention=attention,
                             tanh=tanh,
                             norm_constant=norm_constant,
                             inv_sublayers=inv_sublayers,
                             sin_embedding=sin_embedding,
                             normalization_factor=normalization_factor,
                             aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):

        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        # import pdb
        # pdb.set_trace()
        # print(node_mask)
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h,
                                         x,
                                         edges,
                                         node_mask=node_mask,
                                         edge_mask=edge_mask)
            vel = (
                x_final - x
            ) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [
                    torch.LongTensor(rows).to(device),
                    torch.LongTensor(cols).to(device)
                ]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)



class DistributionNodes:
    def __init__(self, histogram):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs



class UniformDequantizer(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(UniformDequantizer, self).__init__()

    def forward(self, tensor, node_mask, edge_mask, context):
        category, integer = tensor['categorical'], tensor['integer']
        zeros = torch.zeros(integer.size(0), device=integer.device)

        out_category = category + torch.rand_like(category) - 0.5
        out_integer = integer + torch.rand_like(integer) - 0.5

        if node_mask is not None:
            out_category = out_category * node_mask
            out_integer = out_integer * node_mask

        out = {'categorical': out_category, 'integer': out_integer}
        return out, zeros

    def reverse(self, tensor):
        categorical, integer = tensor['categorical'], tensor['integer']
        categorical, integer = torch.round(categorical), torch.round(integer)
        return {'categorical': categorical, 'integer': integer}

