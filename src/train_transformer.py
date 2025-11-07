# train_transformer_fixed.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_from_disk

import math
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------
# 1. 模型组件
# -----------------------

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力模块（修正版）"""

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # 4 个线性变换： q, k, v, out
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        query/key/value: shape [seq_len, batch, d_model]
        attn_mask: can be
            - None
            - [tgt_len, tgt_len] (2D subsequent mask)
            - [batch, tgt_len, tgt_len] (3D)
        key_padding_mask: [batch, key_seq_len] bool mask where True indicates padding
        """
        seq_len_q, batch_size, _ = query.size()
        seq_len_k = key.size(0)

        # 1) 线性变换
        q = self.linears[0](query)  # [seq_q, batch, d_model]
        k = self.linears[1](key)    # [seq_k, batch, d_model]
        v = self.linears[2](value)  # [seq_k, batch, d_model]

        # 2) reshape -> split heads -> permute to [batch, num_heads, seq_len, d_k]
        def shape(x):
            # x: [seq, batch, d_model] -> [seq, batch, num_heads, d_k] -> [batch, num_heads, seq, d_k]
            seq, b, _ = x.size()
            x = x.view(seq, b, self.num_heads, self.d_k)
            x = x.permute(1, 2, 0, 3).contiguous()
            return x

        q = shape(q)
        k = shape(k)
        v = shape(v)
        # q,k,v: [batch, num_heads, seq, d_k]

        # 3) 计算 scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch, num_heads, seq_q, seq_k]

        # 4) key_padding_mask: [batch, seq_k] -> expand -> [batch, num_heads, seq_q, seq_k]
        if key_padding_mask is not None:
            # ensure bool tensor
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.bool()
            # mask: [batch, 1, 1, seq_k] -> expand
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch,1,1,seq_k]
            kpm = kpm.expand(-1, self.num_heads, scores.size(2), -1)  # [batch,num_heads,seq_q,seq_k]
            scores = scores.masked_fill(kpm, float('-inf'))

        # 5) attn_mask (subsequent mask for decoder self-attn)
        if attn_mask is not None:
            # attn_mask can be [tgt_len,tgt_len] or [batch, tgt_len, tgt_len]
            if attn_mask.dim() == 2:
                # [tgt_len, tgt_len] -> [1,1,tgt_len,tgt_len]
                am = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [batch, tgt_len, tgt_len] -> [batch,1,tgt_len,tgt_len]
                am = attn_mask.unsqueeze(1)
            else:
                am = attn_mask

            # move to same device
            am = am.to(scores.device)

            # If shapes differ, try expand broadcasting to scores shape
            if am.shape != scores.shape:
                try:
                    am = am.expand_as(scores)
                except Exception:
                    # as a safe fallback, raise informative error
                    raise RuntimeError(f"attn_mask shape {attn_mask.shape} cannot be broadcast to scores shape {scores.shape}")

            # ensure bool
            if am.dtype != torch.bool:
                am_bool = (am != 0)
            else:
                am_bool = am

            scores = scores.masked_fill(am_bool, float('-inf'))

        # 6) softmax
        p_attn = torch.softmax(scores, dim=-1)
        self.attn = p_attn  # for visualization if needed

        # 7) 加权求和
        x = torch.matmul(p_attn, v)  # [batch, num_heads, seq_q, d_k]
        # 合并 heads -> [batch, seq_q, d_model]
        x = x.permute(2, 0, 1, 3).contiguous().view(seq_len_q, batch_size, self.num_heads * self.d_k)
        # 最后线性映射
        x = self.linears[-1](x)  # [seq_q, batch, d_model]
        return x


class PositionwiseFeedForward(nn.Module):
    """逐点前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: [seq, batch, d_model]
        attn_output = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(attn_output))
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_output))
        return src


class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.src_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask=None, tgt_mask=None,
                memory_key_padding_mask=None, tgt_key_padding_mask=None):
        # tgt: [tgt_seq, batch, d_model], memory: [src_seq, batch, d_model]
        attn_output = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_output))

        # cross attention: query from tgt, key/value from memory
        src_attn_output = self.src_attn(tgt, memory, memory, attn_mask=src_mask,
                                        key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout2(src_attn_output))

        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_output))
        return tgt


# -----------------------
# 2. 手工搭建的完整 Transformer 模型
# -----------------------
class Encoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, memory, src_mask=None, tgt_mask=None,
                memory_key_padding_mask=None, tgt_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask)
        return self.norm(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads,
                 src_vocab_size, tgt_vocab_size, d_ff, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)
        self.decoder = Decoder(decoder_layer, num_decoder_layers)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        # src: [src_seq, batch], tgt: [tgt_seq, batch] (token ids)
        src_emb = self.pos_encoder(self.encoder_embedding(src))  # [src_seq, batch, d_model]
        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

        tgt_emb = self.pos_encoder(self.decoder_embedding(tgt))  # [tgt_seq, batch, d_model]
        output = self.decoder(tgt_emb, memory,
                              src_mask=None,
                              tgt_mask=tgt_mask,
                              memory_key_padding_mask=src_padding_mask,
                              tgt_key_padding_mask=tgt_padding_mask)

        return self.generator(output)  # [tgt_seq, batch, tgt_vocab_size]

    def encode(self, src, src_mask, src_padding_mask):
        return self.encoder(self.pos_encoder(self.encoder_embedding(src)), mask=src_mask,
                            src_key_padding_mask=src_padding_mask)

         # 消融实验一就移除self.pos_encoder 的应用

    def decode(self, tgt, memory, tgt_mask, tgt_padding_mask, memory_padding_mask):
        return self.decoder(self.pos_encoder(self.decoder_embedding(tgt)), memory,
                            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
                            memory_key_padding_mask=memory_padding_mask)
        # 消融实验一就移除self.pos_encoder 的应用

# -----------------------
# 3. 早停策略类
# -----------------------
class EarlyStopping:
    """早停法以防止过拟合"""
    def __init__(self, patience=5, verbose=False, delta=0, path='best_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# -----------------------
# 4. 数据处理与 mask 工具
# -----------------------
def generate_square_subsequent_mask(sz, device):
    """
    生成下三角 mask（bool），用于 decoder 自注意力，形状 [sz, sz]
    True 表示 mask（遮挡）
    """
    mask = torch.triu(torch.ones((sz, sz), device=device), diagonal=1).bool()
    return mask  # [sz, sz]


def create_padding_mask(seq, pad_idx):
    """
    seq: [seq_len, batch]
    返回： [batch, seq_len] 的 bool mask （True 表示 pad）
    """
    return (seq.transpose(0, 1) == pad_idx)  # [batch, seq_len]


def create_mask(src, tgt, pad_idx, device):
    """
    src: [src_seq, batch]
    tgt: [tgt_seq, batch]  (tgt is already shifted when passing)
    返回:
      src_mask (None),
      tgt_mask ([tgt_len, tgt_len]),
      src_padding_mask ([batch, src_len]),
      tgt_padding_mask ([batch, tgt_len])
    """
    src_mask = None
    tgt_len = tgt.size(0)
    tgt_mask = generate_square_subsequent_mask(tgt_len, device)
    src_padding_mask = create_padding_mask(src, pad_idx)
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# -----------------------
# 5. 训练 / 评估 函数
# -----------------------
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(torch.tensor(item['src_ids'], dtype=torch.long))
        tgt_batch.append(torch.tensor(item['tgt_ids'], dtype=torch.long))
    # pad_sequence 返回 [max_seq_len, batch]
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    losses = 0.0
    for src, tgt in tqdm(train_loader, desc="Training"):
        src = src.to(device)  # [src_seq, batch]
        tgt = tgt.to(device)  # [tgt_seq, batch]
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]

        # 正确生成 mask：src_mask=None，tgt_mask 为下三角，padding masks 为 [batch, seq_len]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX, device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
        optimizer.zero_grad()
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses += loss.item()
    return losses / len(train_loader)


def evaluate(model, criterion, val_loader, device):
    model.eval()
    losses = 0.0
    with torch.no_grad():
        for src, tgt in tqdm(val_loader, desc="Evaluating"):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:-1, :]
            tgt_out = tgt[1:, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input, PAD_IDX, device
            )
            logits = model(
                src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
            )
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                tgt_out.reshape(-1)
            )
            losses += loss.item()
    return losses / len(val_loader)


# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # 超参数
    D_MODEL = 256
    NUM_HEADS = 8   # 单头消融实验改为1
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    D_FF = 512
    DROPOUT = 0.1
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    NUM_EPOCHS = 40
    EARLY_STOPPING_PATIENCE = 5

    # 特殊 token idx
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    print("Loading and processing data...")
    dataset = load_from_disk("./data/iwslt2017_de_en")
    train_data, val_data = dataset['train'], dataset['validation']

    # tokenizer & vocab
    token_transform = {}
    token_transform['en'] = get_tokenizer('spacy', language='en_core_web_sm')
    token_transform['de'] = get_tokenizer('spacy', language='de_core_news_sm')

    def yield_tokens(data_iter, lang):
        for data in data_iter:
            yield token_transform[lang](data['translation'][lang])

    def build_vocabularies(train_iter):
        vocab_transform = {}
        for lang in ['en', 'de']:
            vocab_transform[lang] = build_vocab_from_iterator(
                yield_tokens(train_iter, lang),
                min_freq=2,
                specials=special_symbols,
                special_first=True
            )
            vocab_transform[lang].set_default_index(UNK_IDX)
        return vocab_transform

    vocab_transform = build_vocabularies(train_data)
    SRC_VOCAB_SIZE = len(vocab_transform['de'])
    TGT_VOCAB_SIZE = len(vocab_transform['en'])
    print(f"Source (de) vocab size: {SRC_VOCAB_SIZE}")
    print(f"Target (en) vocab size: {TGT_VOCAB_SIZE}")

    def process_data(data):
        processed = []
        for item in tqdm(data, desc="Processing data"):
            src_text = item['translation']['de']
            tgt_text = item['translation']['en']
            src_ids = [BOS_IDX] + [vocab_transform['de'][token] for token in token_transform['de'](src_text)] + [EOS_IDX]
            tgt_ids = [BOS_IDX] + [vocab_transform['en'][token] for token in token_transform['en'](tgt_text)] + [EOS_IDX]
            processed.append({'src_ids': src_ids, 'tgt_ids': tgt_ids})
        return processed

    train_set = process_data(train_data)
    val_set = process_data(val_data)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, D_MODEL, NUM_HEADS,
        SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_FF, DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True, path='best_model.pt')

    train_losses, val_losses, val_perplexities = [], [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(model, optimizer, criterion, train_loader, DEVICE)
        end_time = time.time()

        val_loss = evaluate(model, criterion, val_loader, DEVICE)
        val_perplexity = math.exp(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_perplexities.append(val_perplexity)

        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
              f"Val PPL: {val_perplexity:.3f}, Epoch time = {(end_time - start_time):.3f}s")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 绘图保存
    actual_epochs = len(train_losses)
    epochs_range = range(1, actual_epochs + 1)
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Transformer Training Metrics')

    axs[0].plot(epochs_range, train_losses, 'b-o', label='Train Loss')
    axs[0].plot(epochs_range, val_losses, 'r-o', label='Validation Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs_range, val_perplexities, 'g-o', label='Validation Perplexity')
    axs[1].set_title('Validation Perplexity')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Perplexity')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../results/baseline.png')
    print("Metrics curve saved to baseline.png")
