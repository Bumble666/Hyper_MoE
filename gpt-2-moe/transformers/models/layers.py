import math
import torch
from torch import nn
from torch.distributions.normal import Normal
from .ad_layer import AdapterLayer

softplus = nn.Softplus()
softmax = nn.Softmax(1)

class SparseDispatcher(object):

    def __init__(self, n_experts, gates):

        self._gates = gates
        self._n_experts = n_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            _nonzero_gates = self._nonzero_gates.unsqueeze(-1) if stitched.dim() == 3 else self._nonzero_gates
            stitched = stitched.mul(_nonzero_gates)
        if stitched.dim() == 2:
            zeros = torch.zeros(self._gates.size()[0], expert_out[-1].size(1), requires_grad=True).to(stitched.device)
        else:
            zeros = torch.zeros(self._gates.size()[0], expert_out[-1].size(-2), expert_out[-1].size(-1),
                                requires_grad=True).to(stitched.device)

        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float()).to(stitched.device)
        # add eps to all zero values in order to avoid nans when going back to log space
        # combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined

    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)




class MoE(nn.Module):

    def __init__(self, config, layer_idx, expert_class, noisy_gating=True):
        super(MoE, self).__init__()
        self.layer_idx = layer_idx
        self.noisy_gating = noisy_gating
        self.n_experts, self.k = config.n_experts, config.k
        self.unselected_exp = self.n_experts - self.k
        self.input_size, self.output_size = config.hidden_size, config.hidden_size
        # instantiate experts
        self.w_gate = nn.Parameter(torch.zeros(config.hidden_size, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(config.hidden_size, config.n_experts), requires_grad=True)
        self.experts = nn.ModuleDict()
        for idx in range(config.n_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.expert_embedding_mode = config.experts_embedding
        if self.expert_embedding_mode:
            self.n_experts_embedding = nn.Embedding(config.n_experts, config.experts_embedding_dim)
            self.embedding_process = nn.Sequential(
                nn.Linear(config.experts_embedding_dim, config.process_dim),
                nn.ReLU(),
                nn.Linear(config.process_dim, config.hypernet_input),
            )
        if config.use_hypernet:
            self.adapter_layer = AdapterLayer(config, self.input_size, self.input_size)
        else:
            self.adapter_layer = None
        self.act = nn.ReLU()
        assert (self.k <= config.n_experts)

    def forward(self, x, param_gen=None, loss_coef=1e-2):
        if self.expert_embedding_mode:
            self.clear_adapters()
        res = x
        original_shape = list(x.shape[:-1])
        x = x.reshape(-1, self.input_size)
        gates, load, gates_out, expert_mask = noisy_top_k_gating_mixing(self, x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.n_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[f"expert_{i}"](expert_inputs[i]) for i in range(self.n_experts)]
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(original_shape + [self.output_size])

        index_out = torch.nonzero(gates_out)[:, -1:].flatten()  # batch*unselected_exp
        gates_out = gates_out[gates_out != 0]  # batch

        if self.expert_embedding_mode:
            embedding_input = torch.sum(
                self.n_experts_embedding(index_out).view(res.size(0), res.size(1),
                                                         self.unselected_exp, -1), dim=-2)
            self.apply_params_to_adapters(res.size(0), res.size(1),
                                          param_gen(self.embedding_process(embedding_input), self.layer_idx))
            out = self.adapter_layer(res) + y
            return out, loss


        return y, loss

    def apply_params_to_adapters(self, batch_size, length, generated_params):
        self.adapter_layer.apply_adapter_params(batch_size, length, *generated_params)  # param：batch * weight

    def clear_adapters(self):
        self.adapter_layer.clear_adapter()




def noisy_top_k_gating(layer, x, train, noise_epsilon=1e-2):
    clean_logits = x @ layer.w_gate
    if layer.noisy_gating and train:
        raw_noise_stddev = x @ layer.w_noise
        noise_stddev = ((softplus(raw_noise_stddev) + noise_epsilon))
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits
    else:
        logits = clean_logits
    # calculate topk + 1 that will be needed for the noisy gates
    top_logits, top_indices = logits.topk(min(layer.k + 1, layer.n_experts), dim=-1)
    top_k_logits = top_logits[:, : layer.k]
    top_k_indices = top_indices[:, : layer.k]
    top_k_gates = softmax(top_k_logits)
    zeros = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype).to(x.device)
    gates = zeros.scatter(1, top_k_indices, top_k_gates).to(x.device)
    if layer.noisy_gating and layer.k < layer.n_experts and train:
        load = (_prob_in_top_k(layer, clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
    else:
        load = _gates_to_load(gates)
    return gates, load

def noisy_top_k_gating_mixing(layer, x, train, noise_epsilon=1e-2):
    clean_logits = x @ layer.w_gate  # bs*length * expert_num
    if layer.noisy_gating and train:
        raw_noise_stddev = x @ layer.w_noise
        noise_stddev = ((softplus(raw_noise_stddev) + noise_epsilon))
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits
    else:
        logits = clean_logits
    # calculate topk + 1 that will be needed for the noisy gates
    # 1.softmax 2.topk
    logits = softmax(logits)
    top_logits, top_indices = logits.topk(min(layer.k + 1, layer.n_experts), dim=-1)  # bs * k+1
    top_k_logits = top_logits[:, : layer.k]  # bs*length * k
    top_k_indices = top_indices[:, : layer.k]
    # top_k_gates = softmax(top_k_logits)
    top_k_gates = top_k_logits/torch.sum(top_k_logits, dim=-1, keepdim=True)

    zeros = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype).to(x.device)  # bs * length * exp
    gates = zeros.scatter(1, top_k_indices, top_k_gates).to(x.device)

    # unselected_experts_mask:
    top_k_logits_ones = torch.ones_like(top_k_logits, requires_grad=True, dtype=top_k_logits.dtype).to(x.device)
    zeros_mask = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype).to(x.device)
    unselected_experts_mask = torch.ones_like(logits, requires_grad=True, dtype=top_k_gates.dtype).to(x.device) - zeros_mask.scatter(1, top_k_indices, top_k_logits_ones).to(x.device)
    gates_unselected = unselected_experts_mask/unselected_experts_mask.sum(dim=-1, keepdim=True)

    expert_mask = torch.nn.functional.one_hot(top_k_indices, num_classes=layer.n_experts).permute(2, 1, 0)

    # top_value_unselected = torch.ones_like(top_indices_unselected, requires_grad=True, dtype=top_k_gates.dtype)*(1/(exp_num-layer.k)).to(
    #     x.device)
    # zeros_unselected = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype).to(x.device)  # bs * exp
    # gates_unselected = zeros_unselected.scatter(1, top_indices_unselected, top_value_unselected).to(x.device)

    if layer.noisy_gating and layer.k < layer.n_experts and train:
        load = (_prob_in_top_k(layer, clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
    else:
        load = _gates_to_load(gates)
    return gates, load, gates_unselected, expert_mask


def _prob_in_top_k(layer, clean_values, noisy_values, noise_stddev, noisy_top_values):
    batch = clean_values.size(0)
    m = noisy_top_values.size(1)
    top_values_flat = noisy_top_values.flatten()
    normal = Normal(layer.mean, layer.std)
    threshold_positions_if_in = torch.arange(batch).to(clean_values.device) * m + layer.k
    threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
    is_in = torch.gt(noisy_values, threshold_if_in)
    threshold_positions_if_out = threshold_positions_if_in - 1
    threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
    # is each value currently in the top k.
    prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
    prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
    prob = torch.where(is_in, prob_if_in, prob_if_out)
    return prob

def cv_squared(x):
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.Tensor([0]).to(x.device)
    return x.float().var() / (x.float().mean() ** 2 + eps)

def _gates_to_load(gates):
    return (gates > 0).sum(0)


class TokenAtt(nn.Module):
    def __init__(self, dim, activation=nn.ReLU(), reduce_factor=64):
        super().__init__()
        self.down_proj = nn.Linear(dim, dim // reduce_factor)
        self.activation = activation
        self.up_proj = nn.Linear(dim // reduce_factor, dim)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        down = self.activation(self.down_proj(x))
        up = self.up_proj(down)
        return up
