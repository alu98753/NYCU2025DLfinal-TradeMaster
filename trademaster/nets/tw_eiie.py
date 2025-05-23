import torch
import torch.nn as nn
from .builder import NETS
from .custom import Net
from trademaster.utils import build_conv2d

@NETS.register_module()
class EIIEConv(Net):
    def __init__(self,
                 input_dim=11,
                 company_count=42,
                 time_steps=50,
                 output_dim = 1,
                 kernel_size=[(1, 3),(1,48)],
                 dims=(32, 20)):
        super(EIIEConv, self).__init__()

        self.company_count = company_count
        self.time_steps = time_steps

        # Two convolutional layers: (3, 49, 50) -> (20, 49, 1)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dim, dims[0], kernel_size = kernel_size[0]),  # output: (32, 49, 48)
            nn.ReLU(),
            nn.Conv2d(dims[0], dims[1], kernel_size= kernel_size[1]),  # output: (20, 49, 1)
            nn.ReLU()
        )

        # Additional input: previous action (1), added as 1 channel
        # Output: (21, 49, 1)

        self.final_conv = nn.Conv2d(21, output_dim, kernel_size=(1, 1))  # output: (1, 49, 1)

        # Learnable bias for cash position
        self.cash_bias = nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x, prev_action):
        # x: (batch, 3, 49, 50), prev_action: (batch, 49, 1)
        x = self.conv_layers(x)  # (batch, 20, 49, 1)

        # Append previous action: reshape to match and concat along channel dim
        # prev_action = prev_action.unsqueeze(1)  # (batch, 1, 49, 1)
        # print(f"[EIIEConv] prev_action.shape before view: {prev_action.shape}")
        # print(f"[EIIEConv] prev_action.numel(): {prev_action.numel()}")
        prev_action = prev_action.squeeze(1)
        prev_action = prev_action[:, :-1]  
        prev_action = prev_action.view(prev_action.shape[0], 1, self.company_count, 1)
        # print(f"prev_action : {prev_action.shape}")
        # print(f"x : {x.shape}")
        x = torch.cat([x, prev_action], dim=1)  # (batch, 21, 49, 1)

        # Final 1x1 convolution to produce score per company
        x = self.final_conv(x)  # (batch, 1, 49, 1)
        x = x.squeeze(1)  # (batch, 49, 1)

        # Append cash bias
        cash = self.cash_bias.repeat(x.shape[0], 1, 1)  # (batch, 1, 1)
        x = torch.cat([x, cash], dim=1)  # (batch, 50, 1)

        # Softmax over company + cash
        x = torch.softmax(x.squeeze(-1), dim=1)  # (batch, 50)
        return x

@NETS.register_module()
class EIIECritic(Net):
    def __init__(self,
                 input_dim=3,
                 time_steps=50,
                 output_dim = 1,
                 company_count=42,
                 hidden_size=64):
        super(EIIECritic, self).__init__()

        self.flattened_input_dim = input_dim * company_count * time_steps

        self.mlp = nn.Sequential(
            nn.Linear(self.flattened_input_dim + company_count + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        # state: (batch, 3, 49, 50), action: (batch, 50)
        batch_size = state.shape[0]
        x = state.view(batch_size, -1)  # flatten all but batch
        x = torch.cat([x, action], dim=1)  # concat state and action
        x = self.mlp(x)  # output: (batch, 1)
        return x
