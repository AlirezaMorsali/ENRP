class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, new=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.new = new
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.a_1 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.a0 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w0 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.shift0 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.a1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.shift1 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        if self.new:
            # print('new >> ', self.new)
            before_activation = self.omega_0 * self.linear(input)
            after_activation = self.a_1 * before_activation + \
                                self.a0 * torch.sin(self.w0 * before_activation + self.shift0) + \
                                self.a1 * torch.cos(self.w1 * before_activation + self.shift1)
            return after_activation
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward2(self, input):
      before_activation = self.omega0 * self.linear(input)
      after_activation = self.a_1 * before_activation + \
                          self.a0 * torch.sin(self.w0 * before_activation + self.shift0) + \
                          self.a1 * torch.cos(self.w1 * before_activation + self.shift1)