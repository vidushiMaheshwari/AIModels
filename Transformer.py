'''
@author: Vidushi Maheshwari
@version: 1.0

This is an implementation of 
'''

import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    '''
    Initialisation of Self Attention defines head_dim for the self attention block and creates various linear layers
    '''

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        # embed size is the dimension of the embedded space
        self.heads = heads # heads is number of heads
        self.head_dim = embed_size // heads # head_dim is the dimension of each heads

        assert(self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        #Defining the linear layers
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        #the above says that values is a linear layer which aaplies a linear transformation to an input size of self.head_dim and outputs a vector of head_dim size with no bias

        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size) # after concatanation..

    '''
    The name "forward" might be confusing but it is a part of multi-head attention block only.
    This block vector od dimension embed_size which gives a multiplication of value with attention (technically tells how important other pieces of sentence are with that particular word)
    '''

    def forward(self, values, keys, query, mask):
        N = query.shape[0] # Number of training examples (basically number of sentences we are putting in)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # The above len are the length of sentences (basically number of words)

        #split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)

        ## The above reshape can be thought of as I had N sentences, each sentences with value_len words and each word embed_size embeddings. Now the embed_size embedding have been further broken down to self.heads and self.head_dim matrix to effectively parallelize the task

        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd-->nhqk", [queries, keys])
        # energy is basically determining that for each word in the target (given by query), how much emphasis are we giving to the input using MatMul.
        # energy shape: (N, shape, query_len, key_len)
        # nqhd -> dimensions of query
        # nkhd -> dimension of keys

        # einsum performs matrix multiplicatioon by using the equation in quotation marks. It describes the input and output dimensions

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            # This says that if that particular area of the mask is zero (meaning we want to hide/ mask that particular area), we will fill the corresponding part of energy matrix with a very low float value.

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3) #direct formula from the paper

        out = torch.einsum("nhql,  nlhd -> nqhd", [attention, values]) #Taking the assumption that key_len == value_len (represented by l)

        # attention shape -> (N, heads, query_len, key_len)
        # values shape -> (N, value_len, heads, head_dim)

        # out -> (N, query_len, head, head_dim)
        # out is the multiplication of attention with values which is basically the entire point of attention

        out.reshape(N, query_len, self.heads*self.head_dim) # concat
        out = self.fc_out(out)
        return out

        # Take some time to notice that we are returning out which is the multiplication of attention and values. We were able to find attention using a softmax function and performing MatMul on key and query input. To make the entire thing usable and efficient we perfoorm parallization, which led us to use heads where heads = embed_size / head_dim. Finally we concatante the out from all the heads. The out technically shows how relevant the value is wrt other values. and has the dimension of embed_size. (not value_len or N)

'''
Transformer Block is a multi-head attention and feed forward latyer along with normalisation
'''

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size) # normalized shape: embed_size
        self.norm2 = nn.LayerNorm(embed_size)

        # These normalisation will be used after multi-head attention block and feed forward block

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        ) # Feed-forward block

        self.dropout = nn.Dropout(dropout) # useful for regularization

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query)) # we are doing attention + query because of the skip connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out

class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        # max_length is related to positional embedding.. Tells how long is the max sentence length
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size) ## nn.Embedding creates a table that stores embedding of a fixed dictionary and size
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # torch.arange returns a 1D vector of size seq_length and then exxpands it to a 2D vector of N spaces in seq_length vectors

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
            # As per the paper there can be more than just one encoder block

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.device = device
        self.attention = SelfAttention(embed_size, heads) ## This makes some linear layers
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        # The above was for the first masked multi-headed attention
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self,
        trg_vocab_size,
        max_length,
        dropout,
        trg_mask,
        forward_expansion,
        heads,
        embed_size,
        num_layers,
        device):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # x is the output embedding thing and enc_out is the output from the encoder
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc_out(x)

class Transformer(nn.Module):
    def __init__(self,
                src_vocab_size,
                trg_vocab_size,
                src_pad_idx, #padindex will be useful to calculate mask
                trg_pad_index,
                embed_size=256,
                num_layers=6,
                forward_expansion=4,
                heads=8,
                dropout=0,
                device="cuda",
                max_length=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)