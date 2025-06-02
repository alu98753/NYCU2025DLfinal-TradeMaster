import torch
import torch.nn as nn
from .builder import NETS
from .custom import Net
from trademaster.utils import build_conv2d
from torch import Tensor
import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe is [max_len, 1, d_model], reshape for batch_first=True
        # We want [1, max_len, d_model] to add to [B, SeqLen, d_model]
        self.register_buffer('pe', pe.permute(1, 0, 2))

    def forward(self, x):
        # x shape: [Batch, SeqLen, Dim]
        # self.pe shape: [1, MaxLen, Dim]
        # Add positional encoding to x. self.pe is sliced to match x's SeqLen.
        return x + self.pe[:, :x.size(1), :]


@NETS.register_module()
class EIIEConv(Net): # This is your Transformer Actor
    def __init__(self, input_dim, time_steps, # F_in, T_max_len
                 d_model=128, n_heads=4, num_encoder_layers=2, dim_feedforward=256,
                 embed_dropout_p=0.1, transformer_dropout_p=0.1,
                 scoring_hidden_dim=64, scoring_dropout_p=0.1):
        super().__init__()
        self.d_model = d_model

        # 1. Embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        self.embed_layer_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(embed_dropout_p)

        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=time_steps)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout_p,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 4. Scoring Head (after pooling)
        self.scoring_head = nn.Sequential(
            nn.Linear(d_model, scoring_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(scoring_hidden_dim),
            nn.Dropout(scoring_dropout_p),
            nn.Linear(scoring_hidden_dim, 1)
        )
        # Cash bias parameter
        self.cash_bias_parameter = torch.nn.Parameter(torch.randn(1)) # Changed from ones to randn for better initial exploration

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name and p.dim() == 1: # Ensure bias is 1D
                nn.init.zeros_(p)
            elif "cash_bias_parameter" in name: # Specific init for cash bias if needed
                nn.init.normal_(p, mean=0.0, std=0.1) # Example: small random value

    def forward(self, x):
        # x: [B, N, T, input_dim] or [B, 1, N, T, input_dim]
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        B, N, T, _input_dim = x.shape # Use _input_dim to avoid conflict with class member

        x_reshaped = x.reshape(B * N, T, _input_dim)

        x_embedded = self.input_projection(x_reshaped)
        x_embedded = self.embed_layer_norm(x_embedded)
        x_embedded = self.embed_dropout(x_embedded)

        x_with_pos_enc = self.positional_encoding(x_embedded)
        
        encoded_sequence = self.transformer_encoder(x_with_pos_enc)
        # print("encoded_sequence:",encoded_sequence.std(), "max:",encoded_sequence.max(), "min:",encoded_sequence.min())


        # Using max pooling as in your provided Actor code
        # pooled_output shape: [B*N, d_model]
        pooled_output = encoded_sequence.max(dim=1).values 
        # print("pooled_output, std:",pooled_output.std(), "max:",pooled_output.max(), "min:",pooled_output.min())


        logits_flat = self.scoring_head(pooled_output) # Shape: [B*N, 1]
        
        stock_logits = logits_flat.view(B, N) # Shape: [B, N]
        # Expand cash_bias_parameter to match batch size
        cash_bias_expanded = self.cash_bias_parameter.expand(B, 1)

        # Concatenate stock logits with the cash bias (for print/debugging)
        combined_logits = torch.cat((stock_logits, cash_bias_expanded), dim=1)

        print("\n===== LOGIT INFO =====")
        print("Raw stock_logits:", stock_logits.detach().cpu().numpy().round(4))
        print("Cash bias:", self.cash_bias_parameter.item())

        # ---- Custom Top-5 Logic Start ----
        top_k = 10
        topk_vals, topk_indices = torch.topk(stock_logits, top_k, dim=1)  # Shape: [B, 5]

        # Apply softmax on top-k logits only
        topk_softmax = torch.softmax(topk_vals, dim=1)  # Shape: [B, 5]
        topk_scaled = topk_softmax * 0.9  # Rescale to sum to 0.9
        # Create zero tensor for all stocks
        stock_weights = torch.zeros_like(stock_logits)  # Shape: [B, N]

        # bocast weight
        for i in range(B):
            stock_weights[i].scatter_(0, topk_indices[i], topk_scaled[i])
            
        # Logging only batch 0
        order = torch.argsort(topk_indices[0], descending=True)
        print(f"\nBatch {0} Top-{top_k} Indices:", topk_indices[0][order].tolist())
        print(f"Batch {0} Top-{top_k} Weights:", [round(float(x), 4) for x in topk_scaled[0][order]])
        print(f"Batch {0} Sum of Top-{top_k} Weights: {topk_scaled[0].sum().item():.4f}")

        # Cash gets the remaining 0.1
        cash_weight = torch.full((B, 1), 0.1, device=stock_logits.device)

        # Combine stock weights and cash
        action_probs = torch.cat([stock_weights, cash_weight], dim=1)  # Shape: [B, N+1]

        print("\nFinal action_probs:", action_probs.detach().cpu().numpy().round(4))
        print("Sum of action_probs (should be 1.0):", action_probs.sum(dim=1).detach().cpu().numpy().round(4))
        # ---- Custom Top-5 Logic End ----

        return action_probs        
        # Expand cash_bias_parameter to match batch size
        # cash_bias_expanded = self.cash_bias_parameter.expand(B, 1)
        # # print("before combine:",stock_logits, "max:",stock_logits.max(), "min:",stock_logits.min())

        # # Concatenate stock logits with the cash bias
        # combined_logits = torch.cat((stock_logits, cash_bias_expanded), dim=1) # Shape: [B, N+1]
        # # print("before temperature: ",stock_logits.detach().cpu().numpy().round(4), "max:", stock_logits.max().item(), "min:", stock_logits.min().item())

        # # Optional: Temperature scaling (can be learned or fixed)
        # temperature =1 # Example fixed temperature
        # combined_logits = combined_logits / temperature
        # # print("after temperature: ",stock_logits.detach().cpu().numpy().round(4), "max:", stock_logits.max().item(), "min:", stock_logits.min().item())

        # action_probs = torch.softmax(combined_logits, dim=1) # Shape: [B, N+1]
        # action_probs_np = action_probs.detach().cpu().numpy()[0].flatten() 
        # # print("after softmax:",action_probs, "max:",action_probs.max(),"Second :",np.partition(action_probs_np, -2)[-2],"third :",np.partition(action_probs_np, -3)[-3], "min:",action_probs.min())
        
        # return action_probs


@NETS.register_module()
class EIIECritic(Net):
    def __init__(self,
                 input_dim,            # F_in (features per stock per time step)
                 action_dim,           # N (number of stocks, action will be N+1 for cash)
                 time_steps,           # T (sequence length for Transformer)
                 d_model=128,          # Transformer's embedding dimension
                 n_heads=4,            # Number of attention heads
                 num_encoder_layers=2, # Number of Transformer encoder layers for state
                 dim_feedforward=256,  # Dimension of FFN in Transformer
                 embed_dropout_p=0.1,
                 transformer_dropout_p=0.1,
                 q_head_hidden_dim=128, # Hidden dimension for the Q-value MLP
                 q_head_dropout_p=0.1
                 ):
        super(EIIECritic, self).__init__()
        self.d_model = d_model
        self.num_stocks = action_dim # N

        # --- State Processing Path (mirrors Actor's Transformer part) ---
        self.input_projection_state = nn.Linear(input_dim, d_model)
        self.embed_layer_norm_state = nn.LayerNorm(d_model)
        self.embed_dropout_state = nn.Dropout(embed_dropout_p)
        self.positional_encoding_state = PositionalEncoding(d_model, max_len=time_steps)

        encoder_layer_state = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout_p,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder_state = nn.TransformerEncoder(
            encoder_layer_state,
            num_layers=num_encoder_layers
        )
        # After per-stock processing, we need to aggregate them for a portfolio-level state.
        # The output of transformer_encoder_state will be [B*N, T, d_model].
        # After pooling (e.g. max or mean over T), it's [B*N, d_model].
        # We will reshape to [B, N, d_model] and then mean pool over N.

        # --- Action Processing Path (Optional: can directly use raw action) ---
        # Project action to a certain dimension if desired, or use it raw.
        # For simplicity, we'll concatenate raw action later.
        # Action 'a' has dimension N+1 (stocks + cash)

        # --- Q-Value Head ---
        # Input to Q-head: concatenated (pooled portfolio state representation, action)
        # Pooled portfolio state representation will be d_model.
        # Action dimension is N+1.
        q_head_input_dim = d_model + (self.num_stocks + 1)
        self.q_value_head = nn.Sequential(
            nn.Linear(q_head_input_dim, q_head_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(q_head_hidden_dim), # Normalization is good
            nn.Dropout(q_head_dropout_p),
            nn.Linear(q_head_hidden_dim, 1)  # Output single Q-value
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name and p.dim() == 1:
                nn.init.zeros_(p)

    def forward(self, state_x, action_a):
        # state_x: [B, N, T, input_dim] (or [B, 1, N, T, input_dim] if from certain buffers)
        # action_a: [B, N+1] (portfolio weights including cash)

        if state_x.dim() == 5 and state_x.size(1) == 1: # Handle potential extra dim
            state_x = state_x.squeeze(1)
        B, N, T, _input_dim = state_x.shape

        # --- Process State ---
        # Reshape for independent stock processing: [B*N, T, input_dim]
        x_reshaped = state_x.reshape(B * N, T, _input_dim)

        # Embedding: [B*N, T, d_model]
        x_embedded = self.input_projection_state(x_reshaped)
        x_embedded = self.embed_layer_norm_state(x_embedded)
        x_embedded = self.embed_dropout_state(x_embedded)

        # Positional Encoding: [B*N, T, d_model]
        x_with_pos_enc = self.positional_encoding_state(x_embedded)

        # Transformer Encoder for state: [B*N, T, d_model]
        encoded_state_sequence = self.transformer_encoder_state(x_with_pos_enc)

        # Temporal Pooling (e.g., max or mean over T) for each stock: [B*N, d_model]
        # Using max pooling to be consistent with your Actor's choice
        individual_stock_repr_flat = encoded_state_sequence.max(dim=1).values

        # Reshape to [B, N, d_model] to get per-stock representations for the batch
        individual_stock_repr_batch = individual_stock_repr_flat.view(B, N, self.d_model)

        # Aggregate per-stock representations into a single portfolio-level state representation
        # Using mean pooling over the N stocks.
        portfolio_state_representation = individual_stock_repr_batch.mean(dim=1) # Shape: [B, d_model]

        # --- Combine State Representation with Action ---
        # action_a is already [B, N+1]
        combined_features = torch.cat((portfolio_state_representation, action_a), dim=1)
        # Shape: [B, d_model + N + 1]

        # --- Get Q-Value ---
        q_value = self.q_value_head(combined_features) # Shape: [B, 1]

        return q_value
    
    
    
# @NETS.register_module()
# class EIIEConv(Net):
#     def __init__(self,
#                  input_dim,
#                  output_dim = 1,
#                  time_steps = 10,
#                  kernel_size = 3,
#                  dims = (32, )):
#         super(EIIEConv, self).__init__()

#         self.kernel_size = kernel_size
#         self.time_steps = time_steps

#         self.net = build_conv2d(
#             dims=[input_dim, *dims, output_dim],
#             kernel_size=[(1, self.kernel_size), (1, self.time_steps - self.kernel_size + 1)]
#         )
#         self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

#     def forward(self, x): # (batch_size, num_seqs, action_dim, time_steps, state_dim)
#         if len(x.shape) > 4:
#             x = x.squeeze(1)
#         x = x.permute(0, 3, 1, 2)
#         x = self.net(x)
#         x = x.view(x.shape[0], -1)

#         # print("combined_logits:",x)
#         para = self.para.repeat(x.shape[0], 1)
#         x = torch.cat((x, para), dim=1)
#         x = torch.softmax(x, dim=1)
#         # print("action_probs:",x)
#         return x

# @NETS.register_module()
# class EIIECritic(Net):
#     def __init__(self,
#                  input_dim,
#                  action_dim,
#                  output_dim=1,
#                  time_steps=10,
#                  num_layers= 1,
#                  hidden_size = 32,
#                  ):
#         super(EIIECritic, self).__init__()

#         self.time_steps = time_steps

#         self.lstm = nn.LSTM(input_size=input_dim * time_steps,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             batch_first=True)
#         self.linear1 = nn.Linear(hidden_size, output_dim)
#         self.act = nn.ReLU()
#         self.linear2 = nn.Linear(2 * (action_dim + 1), 1)
#         self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

#     def forward(self, x, a):
#         if len(x.shape) >= 4:
#             x = x.view(x.shape[0], x.shape[1], -1)
#         lstm_out, _ = self.lstm(x)
#         x = self.linear1(lstm_out)

#         x = self.act(x)

#         x = x.view(x.shape[0], -1)
#         para = self.para.repeat(x.shape[0], 1)

#         x = torch.cat((x, para, a), dim=1)
#         x = self.linear2(x)
#         # x = x.mean(dim = 1, keepdim=True)
#         return x