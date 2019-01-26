import torch
import torch.nn as nn


class AttentionMechanism(nn.Module):
    """
    Attention Mechanism.
    """

    def __init__(self, encoder_size, decoder_size, attention_size):
        """
        Parameters
        ----------
        encoder_size: int, number of channels in encoder CNN output feature
            map (for MobileNetV2 it is 1280)
        decoder_size: int, number of features in the hidden state, i.e. LSTM 
            output size
        attention_size: int, size of MLP used to compute attention scores
        """
        super(AttentionMechanism, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.attention_size = attention_size

        # Linear layer to transform encoded features to attention size
        self.encoder_attn = nn.Linear(in_features=self.encoder_size,
                                      out_features=self.attention_size)

        # Linear layer to transform decoders (hidden state) output to attention size
        self.decoder_attn = nn.Linear(in_features=self.decoder_size,
                                      out_features=self.attention_size)

        # ReLU non-linearity
        self.relu = nn.ReLU()

        # Linear layer to compute attention scores at time t for L locations
        self.fc_attn = nn.Linear(in_features=self.attention_size, out_features=1)

        # Softmax layer to compute attention weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_out):
        """
        Parameters
        ----------
        encoder_out: PyTorch tensor, size: [M, L, D] where, L is feature
            map locations, and D is channels of encoded CNN feature map.
        decoder_out: PyTorch tensor, size: [M, h], where h is hidden
            dimension of the previous step output from decoder

        NOTE: M is batch size. k is attention size (see comments)

        Returns
        -------
        attn_weighted_encoding: PyTorch tensor, size: [M, D], attention weighted 
            annotation vector
        alpha: PyTorch tensor, size: [M, L], attention weights 
        """
        enc_attn = self.encoder_attn(encoder_out)  # size: [M, L, k]
        dec_attn = self.decoder_attn(decoder_out)  # size: [M, k]

        enc_dec_sum = enc_attn + dec_attn.unsqueeze(1)  # size: [M, L, k]

        # Compute attention scores for L locations at time t (Paper eq: 4)
        attn_scores = self.fc_attn(self.relu(enc_dec_sum))  # size: [M, L]

        # Compute for each location the probability that location i is the right
        # place to focus for generating next word (Paper eq: 5)
        alpha = self.softmax(attn_scores.squeeze(2))  # size: [M, L]

        # Compute attention weighted annotation vector (Paper eq: 6)
        attn_weighted_encoding = torch.sum(encoder_out * alpha.unsqueeze(2), dim=1)  # size: [M, D]

        return attn_weighted_encoding, alpha


class DecoderAttentionRNN(nn.Module):
    """
    RNN (LSTM) decoder to decode encoded images and generate sequences.
    """

    def __init__(self, encoder_size, decoder_size, attention_size, embedding_size, vocab_size, dropout_prob=0.5):
        """
        encoder_size: int, number of channels in encoder CNN output feature map (for MobileNetV2 it is 1280)
        decoder_size: int, number of features in the hidden state, i.e. LSTM  output size
        attention_size: int, size of MLP used to compute attention scores
        embedding_size: int, size of embedding
        vocab_size: int, vocabulary size
        dropout: float, dropout probability
        """
        super(DecoderAttentionRNN, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.attention_size = attention_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.dropout_prob = dropout_prob

        # Create attention mechanism
        self.attention = AttentionMechanism(self.encoder_size, self.decoder_size, self.attention_size)

        # Create embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)  # size: [V, E]

        # Create dropout module
        self.dropout = nn.Dropout(p=self.dropout_prob)

        # Create LSTM cell (uses for loop) for decoding
        self.rnn = nn.LSTMCell(input_size=self.embedding_size + self.encoder_size,
                               hidden_size=self.decoder_size, bias=True)

        # MLPs for LSTM cell's initial states
        self.init_h = nn.Linear(self.encoder_size, self.decoder_size)
        self.init_c = nn.Linear(self.encoder_size, self.decoder_size)

        # MLP to compute beta (gating scalar, paper section 4.2.1)
        self.f_beta = nn.Linear(self.decoder_size, 1)  # scalar

        # Sigmoid to compute beta
        self.sigmoid = nn.Sigmoid()

        # FC layer to compute scores over vocabulary
        self.fc = nn.Linear(self.decoder_size, self.vocab_size)

    def init_lstm_states(self, encoder_out):
        """
        Initialize LSTM's initial hidden and cell memory states based on encoded
        feature representation. NOTE: Encoded feature map locations mean is used.
        """
        # Compute mean of encoder output locations
        mean_encoder_out = torch.mean(encoder_out, dim=1)  # size: [M, L, D] -> [M, D]

        # Initialize LSTMs hidden state
        h0 = self.init_h(mean_encoder_out)  # size: [M, h]

        # Initialize LSTMs cell memory state
        c0 = self.init_c(mean_encoder_out)  # size: [M, h]

        return h0, c0

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Parameters
        ----------
        encoder_out: PyTorch tensor, size: [M, fs, fs, D] where, fs is feature
            map size, and D is channels of encoded CNN feature map.
        encoded_captions: PyTorch long tensor
        caption_lengths: PyTorch tensor
        """
        batch_size = encoder_out.size(0)

        # Flatten encoded feature maps from size [M, fs, fs, D] to [M, L, D]
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_size)
        num_locations = encoder_out.size(1)

        # Sort caption lengths in descending order
        caption_lengths, sorted_idx = torch.sort(caption_lengths.squeeze(1), dim=0,
                                                 descending=True)

        # Compute decode lengths to decode. Sequence generation ends when <END> token
        # is generated. A typical caption is [<START>, ..., <END>, <PAD>, <PAD>], caption
        # lengths only considers [<START>, ..., <END>], so when <END> is generated there
        # is no need to decode further. Decode lengths = caption lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Sort encoded feature maps and captions as per caption lengths. REASON: Since a
        # batch contains different caption lengths (and decode lengths). At each time step
        # up to max decode length T in a batch we need to apply attention mechanism to only
        # those images in batch whose decode length is greater than current time step
        encoder_out = encoder_out[sorted_idx]
        encoded_captions = encoded_captions[sorted_idx]

        # Get embeddings for encoded captions
        embeddings = self.embedding(encoded_captions)  # size: [M, T, E]

        # Initialize LSTM's states
        h, c = self.init_lstm_states(encoder_out)  # sizes: [M, h], [M, h]

        # Compute max decode length
        T = int(max(decode_lengths))

        # Create placeholders to store predicted scores and alphas (alphas for doubly stochastic attention)
        pred_scores = torch.zeros(batch_size, T, self.vocab_size)  # size: [M, T, V]
        alphas = torch.zeros(batch_size, T, num_locations)  # size: [M, T, L]

        # Decoding step: (1) Compute attention weighted encoding and attention weights
        # using encoder output, and initial hidden state; (2) Generate a new encoded word
        for t in range(T):
            # Compute batch size at step t (At step t how many decoding lengths are greater than t)
            batch_size_t = sum([dl > t for dl in decode_lengths])

            # Encoder output and encoded captions are already sorted by caption lengths
            # in descending order. So based on the number of decoding lengths that are
            # greater than current t, extract data from encoded output and initial hidden state
            # as input to attention mechanism.
            attn_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            # Compute gating scalar beta (paper section: 4.2.1)
            beta_t = self.sigmoid(self.f_beta(h[:batch_size_t]))  # size: [M, 1]

            # Multiply gating scalar beta to attention weighted encoding
            context_vector = beta_t * attn_weighted_encoding  # size: [M, D]

            # Concatenate embeddings and context vector, size: [M, E] and [M, D] -> [M, E+D]
            concat_input = torch.cat([embeddings[:batch_size_t, t, :], context_vector], dim=1)  # size: [M, E+D]

            # LSTM input states from time step t-1
            previous_states = (h[:batch_size_t], c[:batch_size_t])

            # Generate decoded word
            h, c = self.rnn(concat_input, previous_states)

            # Compute scores over vocabulary
            scores = self.fc(self.dropout(h))  # size: [M, V]

            # Populate placeholders for predicted scores and alphas
            pred_scores[:batch_size_t, t, :] = scores
            alphas[:batch_size_t, t, :] = alpha  # alpha size: [M, L]

        return pred_scores, encoded_captions, decode_lengths, alphas, sorted_idx

