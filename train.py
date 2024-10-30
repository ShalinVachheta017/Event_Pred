# train.py

import torch
import torch.optim as optim
import torch.nn as nn

from model import Encoder, Decoder, Seq2Seq
from dataset import get_dataloaders
from utils import PAD_token, SOS_token


def train_model(train_loader, val_loader, num_classes, device):
    # Model parameters
    embedding_dim = 128
    hidden_size = 512
    num_layers = 2
    bidirectional = True

    # Initialize the models
    encoder = Encoder(num_classes, embedding_dim, hidden_size,
                      num_layers=num_layers, bidirectional=bidirectional).to(device)

    decoder_hidden_size = hidden_size * (1 if bidirectional else 1)
    decoder = Decoder(num_classes, embedding_dim,
                      decoder_hidden_size, num_layers=num_layers).to(device)

    model = Seq2Seq(encoder, decoder, device, sos_token=SOS_token).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    teacher_forcing_ratio = 0.5

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0

        for encoder_input, _, decoder_target in train_loader:
            encoder_input = encoder_input.to(device)
            decoder_target_indices = decoder_target.to(device)

            optimizer.zero_grad()
            outputs = model(encoder_input, decoder_target_indices,
                            teacher_forcing_ratio)

            # Flatten outputs and targets
            outputs_flat = outputs.view(-1, num_classes)
            decoder_target_flat = decoder_target_indices.view(-1)

            loss = criterion(outputs_flat, decoder_target_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_loss += loss.item()

            # Compute accuracy
            predictions = torch.argmax(outputs_flat, dim=1)
            mask = decoder_target_flat != PAD_token
            correct = (predictions == decoder_target_flat) & mask
            accuracy = correct.sum().item() / mask.sum().item()
            epoch_accuracy += accuracy

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_accuracy = epoch_accuracy / len(train_loader)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_accuracy:.4f}")

        # Optionally, evaluate on validation set each epoch
        # evaluate_model(model, val_loader, criterion, num_classes, device)

    return model
