import time
from transformers import AutoModel
import torch
from torch import nn
from torchcrf import CRF


class SqueezeEmbedding(nn.Module):
    """
    Squeeze sequence embedding length to the longest one in the batch
    当作组件来用就好
    """
    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack -> unpack ->unsort
        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        """unpack: out"""
        out = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)  # (sequence, lengths)
        out = out[0]
        """unsort"""
        out = out[x_unsort_idx]
        return out


class BertForAE(nn.Module):
    def __init__(self, opt, *args, **kwargs):
        super(BertForAE, self).__init__()

        self.model_name = 'BertForAE'
        self.opt = opt
        self.embedding_dim = opt.embedding_dim
        self.hidden_dim = opt.hidden_dim
        self.tagset_size = opt.tagset_size

        self.bert = AutoModel.from_pretrained(opt.model_path)

        self.dropout = torch.nn.Dropout(opt.dropout)

        self.squeeze_embedding = SqueezeEmbedding()

        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)

        self.hidden2tag = torch.nn.Linear(self.hidden_dim*2, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def calculate_cosin(self, context_output, att_hidden):
        '''
        context_output (batchsize, seqlen, hidden_dim)
        att_hidden (batchsize, hidden_dim)
        '''
        # [128, 64, 312]
        batchsize,seqlen,hidden_dim = context_output.size()
        # [128, 312] --> [128, 1, 312] --> [128, 64, 312]
        att_hidden = att_hidden.unsqueeze(1).repeat(1,seqlen,1)
        context_output = context_output.float()
        att_hidden = att_hidden.float()
        # [128, 64]
        # print(torch.sum(context_output*att_hidden, dim=-1).shape)
        cos = torch.sum(context_output*att_hidden, dim=-1)/(torch.norm(context_output, dim=-1)*torch.norm(att_hidden, dim=-1))
        # [128, 64, 1]
        cos = cos.unsqueeze(-1)
        cos_output = context_output*cos
        outputs = torch.cat([context_output, cos_output], dim=-1)

        return outputs

    def forward(self, inputs):
        context, att, target = inputs[0], inputs[1], inputs[2]
        context_len = torch.sum(context != 0, dim=-1).cpu()
        att_len = torch.sum(att != 0, dim=-1).cpu()

        context = self.squeeze_embedding(context, context_len)

        context = self.bert(context)
        context = context[0]

        # context = self.word_embeds(context)
        context_output, _ = self.lstm(context)

        att = self.squeeze_embedding(att, att_len)
        att = self.bert(att)
        att = att[0]
        # att = self.word_embeds(att)
        _, att_hidden = self.lstm(att)

        att_hidden = torch.cat([att_hidden[0][-2],att_hidden[0][-1]], dim=-1)
        outputs = self.calculate_cosin(context_output, att_hidden)
        outputs = self.dropout(outputs)

        outputs = self.hidden2tag(outputs)
        #CRF
        # outputs = outputs.transpose(0,1).contiguous()
        outputs = self.crf.decode(outputs)
        return outputs



    def log_likelihood(self, inputs):
        context, att, target = inputs[0], inputs[1], inputs[2]
        context_len = torch.sum(context != 0, dim=-1).cpu()
        att_len = torch.sum(att != 0, dim=-1).cpu()
        target_len = torch.sum(target != 0, dim=-1).cpu()
        target = self.squeeze_embedding(target, target_len)
        # target = target.transpose(0,1).contiguous()
        context = self.squeeze_embedding(context, context_len)
        context = self.bert(context)
        context = context[0]
        # context = self.word_embeds(context)
        context_output, _ = self.lstm(context)

        att = self.squeeze_embedding(att, att_len)
        att = self.bert(att)
        att = att[0]
        # att = self.word_embeds(att)
        _, att_hidden = self.lstm(att)

        att_hidden = torch.cat([att_hidden[0][-2],att_hidden[0][-1]], dim=-1)

        outputs = self.calculate_cosin(context_output, att_hidden)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        #CRF
        # outputs = outputs.transpose(0,1).contiguous()

        return - self.crf(outputs, target.long())

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = './checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name
