class CSiren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.net = nn.ModuleList([])
        for i in range(out_features):
            self.net.append(Siren(in_features, hidden_features, hidden_layers, out_features=1, outermost_linear=outermost_linear, first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0))

    def forward(self, coords):
      x = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
      output = None
      for i in range(self.out_features):
        tmp, fff = self.net[i](x)
        if i == 0:
          output = tmp
        else:
          output = torch.cat((output, tmp), dim=2)
      return output, None