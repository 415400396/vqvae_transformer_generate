import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, seq_len = 28*28, n_embd = 128, n_head = 8, attn_pdrop = 0.1, resid_pdrop = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head


    def forward(self, x, layer_past=None):
        # pdb.set_trace()
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CausalAttention(nn.Module):
    def __init__(self, seq_len = 28*28, n_embd = 128, n_head = 8, attn_pdrop = 0.1, resid_pdrop = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len))
                                     .view(1, 1, seq_len, seq_len))
        self.n_head = n_head


    def forward(self, x, layer_past=None):
        # pdb.set_trace()
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Transformer_encoder(nn.Module):
    def __init__(self, seq_len = 28*28, num_layers = 6, n_embd = 128, n_head = 8, attn_pdrop = 0.1, resid_pdrop = 0.1, linear_drop = 0.1, ff_hid_dim = 4 * 64):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(n_embd, Attention(seq_len, n_embd, n_head, attn_pdrop, resid_pdrop)),
                PreNorm(n_embd, FeedForward(n_embd, ff_hid_dim, dropout = linear_drop))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer_decoder(nn.Module):
    def __init__(self, seq_len = 28*28, num_layers = 6, n_embd = 128, n_head = 8, attn_pdrop = 0.1, resid_pdrop = 0.1, linear_drop = 0.1, ff_hid_dim = 4 * 64):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(n_embd, CausalAttention(seq_len, n_embd, n_head, attn_pdrop, resid_pdrop)),
                PreNorm(n_embd, FeedForward(n_embd, ff_hid_dim, dropout = linear_drop))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Classification_GPT(nn.Module):
    def __init__(self, vocab_size = 256, num_classes = 10, seq_len = 28*28, n_embd = 256, n_head = 8, attn_pdrop = 0.1, resid_pdrop = 0.1, embd_pdrop = 0.1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # input embedding stem
        self.tok_emb1 = nn.Embedding(vocab_size, n_embd)
        self.tok_emb2 = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, n_embd))
        self.emdb_drop = nn.Dropout(embd_pdrop)
        # transformer
        self.transformer_encoder = Transformer_encoder(seq_len = seq_len, num_layers = 6, n_embd = n_embd, n_head = n_head, attn_pdrop = attn_pdrop, resid_pdrop = resid_pdrop, ff_hid_dim = 4 * 64)
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.last = nn.Linear(vocab_size, num_classes)

        self.seq_len = seq_len
        # self.apply(self._init_weights)

        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, bottom_feature, top_feature):
        top_feature = self.upsample(torch.unsqueeze(top_feature, 1))
        bottom_feature = torch.flatten(bottom_feature, start_dim=1).to(torch.int)
        top_feature = torch.flatten(top_feature, start_dim=1).to(torch.int)

        token_embeddings1 = self.tok_emb1(top_feature) # each index maps to a (learnable) vector
        token_embeddings2 = self.tok_emb2(bottom_feature)

        # token_embeddings = torch.cat((token_embeddings1, token_embeddings2), 2)
        token_embeddings = token_embeddings1 + token_embeddings2
        # forward the GPT model
        token_embeddings = self.emdb_drop(token_embeddings)
        position_embeddings = self.pos_emb[:, :self.seq_len, :] # each position maps to a (learnable) vector
        x = self.transformer_encoder(token_embeddings + position_embeddings)
        x = self.ln_f(x)
        logits = self.head(x)
        logits = self.last(logits.mean(dim = 1))

        return logits


class Autoregressive_GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, vocab_size = 512, seq_len = 8*8-1, n_embd = 128, n_head = 8, attn_pdrop = 0.1, resid_pdrop = 0.1, embd_pdrop = 0.1):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, n_embd))
        self.emdb_drop = nn.Dropout(embd_pdrop)
        # transformer
        self.transformer_decoder = Transformer_decoder(seq_len = seq_len, num_layers = 6, n_embd = n_embd, n_head = n_head, attn_pdrop = attn_pdrop, resid_pdrop = resid_pdrop, ff_hid_dim = 4 * 64)
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.seq_len = seq_len
        # self.apply(self._init_weights)

        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, seq):
        # forward the GPT model
        # pdb.set_trace()
        seq_len = seq.shape[1]
        token_embeddings = self.tok_emb(seq)
        token_embeddings = self.emdb_drop(token_embeddings)
        position_embeddings = self.pos_emb[:, :seq_len, :] # each position maps to a (learnable) vector
        x = self.transformer_decoder(token_embeddings + position_embeddings)
        x = self.ln_f(x)
        # pdb.set_trace()
        logits = self.head(x)
        return logits
