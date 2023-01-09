import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w = nn.functional.softmax(logits, 1)
        # return w
        return logits

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

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
            nn.Linear(1024, 1024)
            for i in range(5)])

        self.list_linear_hv2 = nn.ModuleList([
            nn.Linear(1024, 1024)
            for i in range(5)])

        self.list_linear_hq = nn.ModuleList([
            nn.Linear(1024, 1024)
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
        x_mm = x_mm.sum(1).view(batch_size, 1024)

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

        x_v = input_v.view(batch_size * weight_height, 1024)
        x_q = input_q.view(batch_size * weight_height, 1024)
        x_mm = x_v * x_q
        x_mm = x_mm.view(batch_size, weight_height, 1024)
        return x_mm
