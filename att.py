import torch
import torch.nn as nn
import torch.nn.functional as F
from language_model import WordEmbedding, QuestionEmbedding
import copy
import seq2vec
import fusion

class AbstractAtt(nn.Module):

    def __init__(self, dataset, num_hid):
        super(AbstractAtt, self).__init__()
        self.vocab_words = dataset.dictionary.ntoken
        self.num_classes = 1024
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
        self.q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        # Modules
        #self.seq2vec = seq2vec.factory(self.vocab_words)
        #self.seq2vec  = seq2vec.factory(self.vocab_words, self.opt['seq2vec'])
        # Modules for attention
        self.conv_v_att = nn.Conv2d(2048,
                                    2048, 1, 1)
        self.linear_q_att = nn.Linear(1024,
                                      310)
        self.conv_att = nn.Conv2d(510, 2, 1, 1)
        # Modules for classification
        self.list_linear_v_fusion = None
        self.linear_q_fusion = None
        self.linear_classif = None

    def _fusion_att(self, x_v, x_q):
        raise NotImplementedError

    def _fusion_classif(self, x_v, x_q):
        raise NotImplementedError

    def _attention(self, input_v, x_q_vec):
        # (16, 36, 2048)
        # (16, 1024)
        batch_size = input_v.size(0)
        width = 6
        height = 6

        # Process visual before fusion
        #x_v = input_v.view(batch_size*width*height, dim_features)

        x_v = input_v

        # Process question before fusion
        x_q = F.dropout(x_q_vec, p=0.5,training=self.training)
        x_q = self.linear_q_att(x_q)
        x_q = F.tanh(x_q)
        x_q = x_q.view(batch_size,
                       1,
                       310)
        x_q = x_q.expand(batch_size,
                         width * height,
                         310)

        # First multimodal fusion
        x_att = self._fusion_att(x_v, x_q)


        # Process attention vectors
        x_att = F.dropout(x_att, p=0.5, training=self.training)
        # can be optim to avoid two views and transposes
        x_att = x_att.view(batch_size,
                           width, height,
                           510)
        # x_att = x_att.transpose(1,2)
        x_att = x_att.transpose(2, 3).transpose(1, 2)
        x_att = self.conv_att(x_att) # 510,2,1,1
        x_att = x_att.view(batch_size,
                           2,
                           width * height)
        list_att_split = torch.split(x_att, 1, dim=1)
        list_att = []
        for x_att in list_att_split:
            x_att = x_att.contiguous()
            x_att = x_att.view(batch_size, width*height)
            x_att = F.softmax(x_att)
            list_att.append(x_att)

        self.list_att = [x_att.data for x_att in list_att]

        # Apply attention vectors to input_v
        x_v = input_v.view(batch_size, 2048, width * height)
        x_v = x_v.transpose(1,2)

        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(batch_size,
                               width * height,
                               1)
            x_att = x_att.expand(batch_size,
                                 width * height,
                                 2048)
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(batch_size, 2048)
            list_v_att.append(x_v_att)

        return list_v_att

    def _fusion_glimpses(self, list_v_att, x_q_vec):
        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att,
                            p=0.5,
                            training=self.training)
            
            x_v = self.list_linear_v_fusion[glimpse_id](x_v)
            #if 'activation_v' in self.opt['fusion']:
            x_v = F.tanh(x_v)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)

        # Process question
        x_q = F.dropout(x_q_vec,
                        p= 0.5,
                        training=self.training)
        x_q = self.linear_q_fusion(x_q)
        #if 'activation_q' in self.opt['fusion']:
        x_q = F.tanh(x_q)

        # Second multimodal fusion
        x = self._fusion_classif(x_v, x_q)
        return x

    def _classif(self, x):

        #if 'activation' in self.opt['classif']:
        #    x = getattr(F, self.opt['classif']['activation'])(x)
        x = F.dropout(x,
                      p=0.5,
                      training=self.training)
        x = self.linear_classif(x)
        return x

    def forward(self, input_v, input_q):
        #if input_v.dim() != 4 and input_q.dim() != 2:
        #    raise ValueError
        w_emb = self.w_emb(input_q)
        x_q = self.q_emb(w_emb)  # [batch, q_dim]
        x_q_vec = x_q
        list_v_att = self._attention(input_v, x_q_vec)
        x = self._fusion_glimpses(list_v_att, x_q_vec)
        x = self._classif(x)

        return x, w_emb




class MutanAtt(AbstractAtt):

    def __init__(self, dataset, num_hid):
        # TODO: deep copy ?
        super(MutanAtt, self).__init__(dataset, num_hid)
        # Modules for classification
        self.fusion_att = fusion.MutanFusion2d(
                                               visual_embedding=False,
                                               question_embedding=False)
        self.list_linear_v_fusion = nn.ModuleList([
            nn.Linear(2048,
                      int(1024))
            for i in range(2)])
        self.linear_q_fusion = nn.Linear(1024,
                                         310)
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
        self.q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        self.linear_classif = nn.Linear(510,
                                        self.num_classes)

        self.fusion_classif = fusion.MutanFusion(
                                                 visual_embedding=False,
                                                 question_embedding=False)

    def _fusion_att(self, x_v, x_q):
        return self.fusion_att(x_v, x_q)


    def _fusion_classif(self, x_v, x_q):
        return self.fusion_classif(x_v, x_q)
