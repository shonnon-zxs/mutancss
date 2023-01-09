import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention, NewAttention, MutanFusion2d, MutanFusion
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from torch.autograd import Variable
import numpy as np
#np.set_printoptions(threshold=np.inf)

#torch.set_printoptions(threshold=np.inf)


def mask_softmax(x,mask):
    mask=mask.unsqueeze(2).float()
    x2=torch.exp(x-torch.max(x))
    x3=x2*mask
    epsilon=1e-5
    x3_sum=torch.sum(x3,dim=1,keepdim=True)+epsilon
    x4=x3/x3_sum.expand_as(x3)
    return x4


def att2d(input_v, input_q):
    if input_v.dim() != input_q.dim() and input_v.dim() != 2:
        raise ValueError

    list_linear_hv = nn.ModuleList([
        nn.Linear(2048, 1024)
        for i in range(5)])

    list_linear_hq = nn.ModuleList([
        nn.Linear(1024, 1024)
        for i in range(5)])

    batch_size = input_v.size(0)

    x_v = input_v
    x_q = input_q

    x_mm = []
    for i in range(5):
        x_hv = F.dropout(x_v, p=0.5, training=self.training)
        x_hv = list_linear_hv[i](x_hv)
        x_hq = F.dropout(x_q, p=0.5, training=self.training)
        x_hq = list_linear_hq[i](x_hq)
        x_mm.append(torch.mul(x_hq, x_hv))

    x_mm = torch.stack(x_mm, dim=1)
    x_mm = x_mm.sum(1).view(batch_size, 1024)

    return x_mm

def att3d(v, q):
    v_proj = FCNet([2048, 1024])
    q_proj = FCNet([1024, 1024])
    batch, k, _ = v.size()
    if not v.is_contiguous():
        v = v.contiguous()
    if not q.is_contiguous():
        q = q.contiguous()
    v = Variable(v).cuda().requires_grad_()
    q = Variable(q).cuda()
    v_proj = v_proj(v) # [batch, k, qdim]
    q_proj = q_proj(q).unsqueeze(1).repeat(1, k, 1)
    x_v = v_proj.view(batch * k, 2048)
    x_q = q_proj.view(batch * k, 1024)
    x_mm = att2d(x_v,x_q)
    logits = x_mm.view(batch_size, weight_height, 1024)

    return logits



class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, vatt, q_net, v_net, num_hid, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.vatt = vatt
        self.q_net = q_net
        self.v_net = v_net

        self.v_proj = FCNet([2048, 1024])
        self.q_proj = FCNet([1024, 1024])


        self.classifier = classifier
        self.debias_loss_fn = None
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)



    def forward(self, v, q, labels, bias,v_mask):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        batch, k, _ = v.size()
        v_proj = self.v_proj(v)  # [batch, k, qdim]
        q_proj = self.q_proj(q_emb).unsqueeze(1).repeat(1, k, 1)
        att = self.v_att(v_proj, q_proj)


        if v_mask is None:
            att = nn.functional.softmax(att, 1)
        else:
            att= mask_softmax(att,v_mask)

        v = self.v_net(v)
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = v_emb

        joint_repr = self.vatt(q_repr, v_repr)

        logits = self.classifier(joint_repr)

        if labels is not None:
            loss = self.debias_loss_fn(joint_repr, logits, bias, labels)

        else:
            loss = None
        return logits, loss,w_emb

def build_baseline0(dataset, num_hid):
    print(num_hid)
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, num_hid, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = MutanFusion2d(visual_embedding=False, question_embedding=False)
    vatt = MutanFusion(visual_embedding=False, question_embedding=False)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, vatt, q_net, v_net, num_hid, classifier)