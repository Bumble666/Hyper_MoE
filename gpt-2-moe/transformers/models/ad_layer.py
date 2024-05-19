import torch
import torch.nn as nn


class AdapterLayer(nn.Module):
    def __init__(self, config, input_size, output_size):
        super().__init__()
        self.adapter_dim = config.adapter_dim
        self.input_dim = input_size
        self.output_dim = output_size
        # insertion weights
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None
        self.hidden_act = nn.ReLU()
        # learnt adapter + inits for it
        self.adapter_down_manual = nn.Linear(self.input_dim, self.adapter_dim)
        self.adapter_up_manual = nn.Linear(self.adapter_dim, self.output_dim)
        # self.adapter_down_manual = nn.Parameter(torch.zeros(self.input_dim, self.adapter_dim), requires_grad=True)
        # self.adapter_up_manual = nn.Parameter(torch.zeros(self.adapter_dim, self.output_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.adapter_up_manual.weight, gain=1e-4)
        nn.init.xavier_uniform_(self.adapter_down_manual.weight, gain=1e-4)
        nn.init.constant_(self.adapter_up_manual.bias, 0.0)
        nn.init.constant_(self.adapter_down_manual.bias, 0.0)
        # nn.init.kaiming_uniform_(self.adapter_down_manual, a=math.sqrt(5))
        # nn.init.zeros_(self.adapter_up_manual)
    def clear_adapter(self):
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None

    def apply_adapter_params(self, bsz, lg, uw, dw, ub, db):
        self.adapter_down_weight = dw.view(bsz, lg, self.input_dim, self.adapter_dim)
        self.adapter_down_bias = db.view(bsz, lg, self.adapter_dim)
        self.adapter_up_weight = uw.view(bsz, lg, self.adapter_dim, self.output_dim)
        self.adapter_up_bias = ub.view(bsz, lg, self.output_dim)


    def forward(self, x):
        if self.adapter_down_weight is not None:
            # x = (x @ self.adapter_down_weight)
            #x = (x @ self.adapter_down_weight) + self.adapter_down_bias.unsqueeze(1)  # x:batch * length * hid_dim  @  weight:batch*length * hid_dim * adapter_dim
            x = torch.einsum('bij,bijk->bik', x, self.adapter_down_weight) + self.adapter_down_bias
            # x = torch.einsum('bij,bijk->bijk', x1, x2)
            #  =  batch * length * adapter_dim
            x = self.hidden_act(x)
            # x = (x @ self.adapter_up_weight)
            #x = (x @ self.adapter_up_weight) + self.adapter_up_bias.unsqueeze(1)
            x = torch.einsum('bik,bikj->bij', x, self.adapter_up_weight) + self.adapter_up_bias
        else:
            # x = x @ self.adapter_down_manual
            # x = x @ self.adapter_up_manual
            x = self.adapter_down_manual(x)
            x = self.hidden_act(x)
            x = self.adapter_up_manual(x)
        return x  # no residual connection - we let the user of this layer decide that
