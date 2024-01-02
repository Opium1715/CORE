import torch
from torch import nn
import torch.nn.functional as F


class Encode_Ave(nn.Module):
    def __init__(self, emb_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size

    def forward(self, X, mask):
        seq_len = torch.sum(mask, dim=1, keepdim=True)
        X = torch.sum(X, dim=1) / seq_len
        return X


class Encode_Transformer(nn.Module):
    def __init__(self, emb_size, layer, dropout_tra, heads, layer_norm_eps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.layer = layer
        self.heads = heads
        self.inner_size = 256
        self.dropout_attn = dropout_tra
        self.layer_norm_eps = layer_norm_eps

        self.position_embedding = nn.Embedding(50, self.emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=self.emb_size,
                                                                                          nhead=self.heads,
                                                                                          dim_feedforward=self.inner_size,
                                                                                          dropout=self.dropout_attn,
                                                                                          activation='gelu',
                                                                                          layer_norm_eps=self.layer_norm_eps,
                                                                                          batch_first=True
                                                                                          ),
                                                 num_layers=self.layer
                                                 )
        self.layer_norm = nn.LayerNorm(self.emb_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(dropout_tra)
        self.fn = nn.Linear(self.emb_size, 1)

    def get_attention_mask(self, mask, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = mask.to(torch.bool)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, mask.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def forward(self, X, mask):
        pos_embedding = self.position_embedding(torch.arange(mask.shape[1], device='cuda', dtype=torch.int64))
        input_X = X + pos_embedding.unsqueeze(0).expand_as(X)
        input_X = self.layer_norm(input_X)
        input_X = self.dropout(input_X)

        # attention_mask = self.get_attention_mask(mask)
        attention_mask = mask
        # self-attn 正常遮罩
        attention_mask = attention_mask.unsqueeze(1).expand(-1, mask.size(-1), -1) * mask.unsqueeze(-1)
        attention_mask = torch.tril(attention_mask)
        attention_mask = torch.logical_not(attention_mask).repeat(2, 1, 1)

        attention_result = self.transformer(input_X, mask=attention_mask)

        alpha = self.fn(attention_result)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1)

        X = torch.sum(alpha * X, dim=-1)
        X = F.normalize(X, dim=-1)

        return X


class CORE(nn.Module):
    def __init__(self, emb_size, item_num, head_nums, dropout, transformer_layers, encoder_mode, tau, dropout_attn,
                 norm_eps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.item_num = item_num
        self.dropout = dropout
        self.encoder = None
        if encoder_mode == 'ave':
            self.encoder = Encode_Ave(self.emb_size)
        elif encoder_mode == 'tra':
            self.encoder = Encode_Transformer(self.emb_size, transformer_layers, dropout_attn, head_nums, norm_eps)
        self.item_embedding = nn.Embedding(item_num + 1, self.emb_size, padding_idx=0)
        self.item_dropout = nn.Dropout(dropout)
        self.sess_dropout = nn.Dropout(dropout)
        self.tau = tau

    def forward(self, session, mask):
        session_emb = self.item_embedding(session)
        session_emb = self.sess_dropout(session_emb)

        consistent_emb = self.encoder(session_emb, mask)
        consistent_emb = F.normalize(consistent_emb, dim=-1)

        all_item = F.normalize(self.item_dropout(
            self.item_embedding(torch.arange(1, self.item_num + 1, device='cuda', dtype=torch.int64))), dim=-1)
        score = torch.matmul(consistent_emb, all_item.transpose(1, 0)) / self.tau

        return score
