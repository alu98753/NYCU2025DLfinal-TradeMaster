class HCAR_Critic(Net):
    def __init__(self,
                 input_dim,
                 action_dim,
                 output_dim=1,
                 time_steps=10,
                 num_layers= 1,
                 hidden_size = 32,
                 ):
        super(HCAR_Critic, self).__init__()

        self.time_steps = time_steps

        self.lstm = nn.LSTM(input_size=input_dim * time_steps,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear1 = nn.Linear(hidden_size, output_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(2 * (action_dim + 1), 1)
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x, a):
        if len(x.shape) >= 4:
            x = x.view(x.shape[0], x.shape[1], -1)
        lstm_out, _ = self.lstm(x)
        x = self.linear1(lstm_out)

        x = self.act(x)

        x = x.view(x.shape[0], -1)
        para = self.para.repeat(x.shape[0], 1)

        x = torch.cat((x, para, a), dim=1)
        x = self.linear2(x)
        # x = x.mean(dim = 1, keepdim=True)
        return x