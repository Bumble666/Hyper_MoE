import math

import torch
import torch.nn as nn


def hyperfanin_init_weight(linear_layer, hypernet_in, mainnet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in * mainnet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


def hyperfanin_init_bias(linear_layer, hypernet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


class SimpleGenerator(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        # output_dim: Main Network output_dim
        # input_dim: Main Network in_dim
        super().__init__()
        adapter_dim = config.adapter_dim
        self.input_dim = input_dim
        self.hidden_dim = config.hypernetwork_bottleneck
        self.linear1 = nn.Linear((config.hypernet_input + config.layer_emb_dim), self.hidden_dim)
        self.activation_fn = nn.ReLU()
        # output weights
        self.weight_up = nn.Linear(self.hidden_dim, output_dim * adapter_dim)
        self.weight_down = nn.Linear(self.hidden_dim, input_dim * adapter_dim)
        self.bias_up = nn.Linear(self.hidden_dim, output_dim)
        self.bias_down = nn.Linear(self.hidden_dim, adapter_dim)
        # init weights
        hyperfanin_init_weight(self.weight_up, self.hidden_dim, adapter_dim)
        hyperfanin_init_weight(self.weight_down, self.hidden_dim, input_dim)
        hyperfanin_init_bias(self.bias_up, self.hidden_dim)
        hyperfanin_init_bias(self.bias_down, self.hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        return (
            self.weight_up(x),
            self.weight_down(x),
            self.bias_up(x),
            self.bias_down(x),
        )
        # return (
        #     self.weight_up(x),
        #     self.weight_down(x),
        # )


class ParameterGenerator(nn.Module):
    def __init__(self, config, input_size, output_size):
        # output_dim: Main Network output_dim
        # input_dim: Main Network in_dim
        super().__init__()
        self.config = config
        self.layer_embed = nn.Embedding(config.num_hidden_layers, config.layer_emb_dim)
        self.decoder = SimpleGenerator(
            config, input_size, output_size
        )

    def forward(self, hidden_inputs, layer_idx):
        if self.config.use_fast_mode:
            layer_idx = torch.ones(self.config.num_experts, dtype=torch.long,
                                device=hidden_inputs.device) * layer_idx
        else:
            layer_idx = torch.ones(hidden_inputs.size(0), hidden_inputs.size(1), dtype=torch.long, device=hidden_inputs.device) * layer_idx
        layer_inputs = self.layer_embed(layer_idx)
        hidden_inputs = torch.cat([hidden_inputs, layer_inputs], dim=-1)
        out = self.decoder(hidden_inputs)
        return out