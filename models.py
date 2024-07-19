import torch.nn as nn
import torch
import math
import torch.nn.functional as F

########################
# Define the CNN model
class Conv1DModel(nn.Module):
    def __init__(self, input_channels, kernel_size=10):
        super(Conv1DModel, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, 64, kernel_size=10, padding='same')
        self.conv1d_2 = nn.Conv1d(64, 32, kernel_size=kernel_size // 2, padding='same' , dilation=2)
        self.conv1d_3 = nn.Conv1d(32, 8, kernel_size=2, padding='same')
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2000, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.conv1d_2(x)
        x = self.relu(x)
        x = self.conv1d_3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

################################
# Define the Transformer-Encoder

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # d_model % num_heads == 0 !!!!!!


        #500
        self.d_model = d_model
        #nr attn head
        self.num_heads = num_heads
        #dimension of  head key, query, and value
        self.d_k = d_model // num_heads


        self.W_q = nn.Linear(d_model, d_model) # query
        self.W_k = nn.Linear(d_model, d_model) # key
        self.W_v = nn.Linear(d_model, d_model) # value
        self.W_o = nn.Linear(d_model, d_model) # out

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        #calculating attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        #mask to ignore padded values in the data
        if mask is not None:
            #print('attention scores shape:',attn_scores.shape)

            #print('mask shape : ',mask.shape)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)


        # softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()


        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # linear transformations and split heads
        # print("MultiHeadAttention  shapes:")
        # print("Q shape:", Q.shape)
        # print("K shape:", K.shape)
        # print("V shape:", V.shape)

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        #print('PositionWiseFeedForward input shape:',x.shape)
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        device = x.device
        pe = self.pe.to(device)
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        #print("EncoderLayer input shape:", x.shape)
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x



class Transformer(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, src_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(d_model * max_seq_length, 2048)
        self.fc2 = nn.Linear(2048, 2)

    def generate_mask(self, src):


        #src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        seq_length = src.size(1)
        mask = torch.triu(torch.ones(seq_length, seq_length) == 1, diagonal=1).unsqueeze(0).unsqueeze(0).to(src.device)
        return mask.expand(src.size(0), 1, seq_length, seq_length)



        #return src_mask

    def forward(self, src):

        src_mask = self.generate_mask(src)
        src_embedded = self.dropout(self.positional_encoding(src))

        enc_output = src_embedded
        for idx, enc_layer in enumerate(self.encoder_layers, 1):
            enc_output = enc_layer(enc_output, src_mask)
            #print(f"encoder layer {idx} output shape:", enc_output.shape)



        flattened_output = self.flatten(enc_output)
        #print('flatten layer output shape: ', flattened_output.shape)
        fc1_output = self.fc1(flattened_output)
        #print('fc1 layer output shape: ', fc1_output.shape)
        output = self.fc2(fc1_output)
        #output = self.fc(enc_output)
        #print(" output shape:", output.shape)
        return output


####################
# Define BiGRU Model

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, layers, bidirectional_flag):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=layers, bidirectional=bidirectional_flag, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        if bidirectional_flag:
            self.fc1 = nn.Linear(2 * hidden_dim * 128, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.atten_weight_b = nn.Linear(hidden_dim, hidden_dim)
            self.atten_weight_f = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim * 128, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.bidirectional_used = bidirectional_flag

    def attention(self, output):
        if self.bidirectional_used:
            out_f, out_b = output[:, :, :self.hidden_dim], output[:, :, self.hidden_dim:]
            out_f, out_b = self.atten_weight_f(out_f), self.atten_weight_b(out_b)
            fwd_atten = torch.bmm(out_f, out_f.permute(0, 2, 1))
            bwd_atten = torch.bmm(out_b, out_b.permute(0, 2, 1))
            fwd_atten = F.softmax(fwd_atten, 1)
            bwd_atten = F.softmax(bwd_atten, 1)
            out_atten_f, out_atten_b = torch.bmm(fwd_atten, out_f), torch.bmm(bwd_atten, out_b)
            out_atten = torch.cat((out_atten_f, out_atten_b), dim=-1)
        else:
            out_atten = output

        return out_atten

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.attention(output)
        output = torch.flatten(output, start_dim=1)
        output = self.fc1(output)
        output = self.dropout(output)
        out = self.fc2(output)
        return out


##########################
##### Define CNN SELF ATTN

input_size = 768
class SelfAttentionPooling(nn.Module):
  def __init__(self, input_size):
    super(SelfAttentionPooling, self).__init__()
    self.W = nn.Linear(input_size, 1)
  def forward(self, batch):
    attention_weight = nn.functional.softmax(self.W(batch).squeeze(-1)).unsqueeze(-1)
    utter_rep = torch.sum(batch * attention_weight, dim=1)




    return utter_rep

class CNNSelfAttn(nn.Module):
  def __init__(self, embedding_dim, filter_sizes, output_dim):
    super(CNNSelfAttn, self).__init__()
    self.embedding_dim = embedding_dim
    self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim // 2, kernel_size=filter_sizes[0], padding='same')
    self.conv2 = nn.Conv1d(in_channels=embedding_dim // 2, out_channels=embedding_dim // 4, kernel_size=filter_sizes[1], padding='same')
    self.sa = SelfAttentionPooling(embedding_dim // 4)
    self.fc = nn.Linear(embedding_dim//4, output_dim)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()


  def forward(self, x):
    x = x.permute(0, 2, 1)
    x = self.relu(self.conv1(x))
  #  print('shape after first conv:', x.shape)
    x = self.relu(self.conv2(x))
 #   print('shape after sencond conv:', x.shape)
    x = x.permute(0, 2, 1)
    x = self.sa(x)
#    print('shape after self attn:', x.shape)
    x = self.fc(x)



    return x

