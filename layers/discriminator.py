import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h1, n_h2):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h1, n_h2, 1)
        self.act = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_c, h_pl, sample_list, s_bias1=None, s_bias2=None):
        sc_1 = torch.squeeze(self.f_k(h_pl, h_c), 2)
        sc_1 = self.act(sc_1)
        sc_2_list = []
        for i in range(len(sample_list)):
            h_mi = torch.unsqueeze(h_pl[0][sample_list[i]],0)
            sc_2_iter = torch.squeeze(self.f_k(h_mi, h_c), 2)
            sc_2_list.append(sc_2_iter)
        sc_2_stack = torch.squeeze(torch.stack(sc_2_list,1),0)
        sc_2 = self.act(sc_2_stack)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        return sc_1, sc_2