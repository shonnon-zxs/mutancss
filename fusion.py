import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AbstractFusion(nn.Module):

    def __init__(self,):
        super(AbstractFusion, self).__init__()

    def forward(self, input_v, input_q):
        raise NotImplementedError



class MutanFusion(AbstractFusion):

    def __init__(self, visual_embedding=True, question_embedding=True):
        super(MutanFusion, self).__init__()
        self.visual_embedding = visual_embedding
        self.question_embedding = question_embedding
        # Modules

        self.list_linear_hv = nn.ModuleList([
            nn.Linear(2048, 510)
            for i in range(5)])

        self.list_linear_hv2 = nn.ModuleList([
            nn.Linear(2048, 510)
            for i in range(5)])

        self.list_linear_hq = nn.ModuleList([
            nn.Linear(310, 510)
            for i in range(5)])

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 2:
            raise ValueError
        batch_size = input_v.size(0)

        x_v = input_v
        x_q = input_q

        x_mm = []
        for i in range(5):

            x_hv = F.dropout(x_v, p=0.5, training=self.training)
            x_hv = self.list_linear_hv2[i](x_hv)


            x_hq = F.dropout(x_q, p=0.5, training=self.training)
            x_hq = self.list_linear_hq[i](x_hq)

            x_mm.append(torch.mul(x_hq, x_hv))

        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, 510)

        return x_mm


class MutanFusion2d(MutanFusion):

    def __init__(self, visual_embedding=True, question_embedding=True):
        super(MutanFusion2d, self).__init__(visual_embedding,
                                            question_embedding)

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 3:
            raise ValueError
        batch_size = input_v.size(0)
        weight_height = input_v.size(1)

        if not input_v.is_contiguous():
            input_v = input_v.contiguous()
        if not input_q.is_contiguous():
            input_q = input_q.contiguous()

        x_v = input_v.view(batch_size * weight_height, 2048)
        x_q = input_q.view(batch_size * weight_height, 310)
        x_mm = super(MutanFusion2d, self).forward(x_v, x_q)
        x_mm = x_mm.view(batch_size, weight_height, 510)
        return x_mm

