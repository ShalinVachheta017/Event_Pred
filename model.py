# model.py

import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_size, num_layers=1, bidirectional=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(num_classes, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2  # Optional dropout
        )

    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        outputs, (hidden, cell) = self.lstm(embedded)
        if self.bidirectional:
            hidden = self._combine_directions(hidden)
            cell = self._combine_directions(cell)
        return hidden, cell

    def _combine_directions(self, h):
        num_directions = 2 if self.bidirectional else 1
        h = h.view(self.num_layers, num_directions, h.size(1), h.size(2))
        h = h.sum(dim=1)
        return h


class Decoder(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Embedding layer
        self.embedding = nn.Embedding(num_classes, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2  # Optional dropout
        )
        # Final output layer
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, input_tokens, hidden, cell):
        # input_tokens: (batch_size, 1)
        # (batch_size, 1, embedding_dim)
        embedded = self.embedding(input_tokens)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # Output logits
        logits = self.fc_out(outputs)  # (batch_size, 1, num_classes)
        return logits, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, sos_token):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_token = sos_token  # Start Of Sequence token index

    def forward(self, encoder_input, decoder_target_indices, teacher_forcing_ratio=0.5):
        batch_size = encoder_input.size(0)
        target_len = decoder_target_indices.size(1)
        num_classes = self.decoder.num_classes

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len,
                              num_classes).to(self.device)

        # Encode input sequence
        hidden, cell = self.encoder(encoder_input)

        # First input to the decoder is the <SOS> token
        input_token = torch.full(
            (batch_size, 1), self.sos_token, dtype=torch.long).to(self.device)

        for t in range(target_len):
            # Pass through decoder
            decoder_output, hidden, cell = self.decoder(
                input_token, hidden, cell)
            # Store output
            outputs[:, t, :] = decoder_output.squeeze(1)

            # Decide whether to use teacher forcing
            teacher_force = np.random.rand() < teacher_forcing_ratio

            # Get the highest predicted token
            top1 = decoder_output.argmax(2)  # Shape: (batch_size, 1)

            if teacher_force and t < target_len - 1:
                # Use actual next token from decoder_target_indices
                input_token = decoder_target_indices[:, t].unsqueeze(
                    1)  # Shape: (batch_size, 1)
            else:
                # Use predicted token as next input
                input_token = top1.detach()

        return outputs
