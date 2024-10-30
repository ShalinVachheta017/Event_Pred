# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import PAD_token


class Seq2SeqDataset(Dataset):
    def __init__(self, encoder_inputs, decoder_inputs, decoder_targets):
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        return (self.encoder_inputs[idx], self.decoder_inputs[idx], self.decoder_targets[idx])


def collate_fn(batch):
    encoder_inputs, decoder_inputs, decoder_targets = zip(*batch)

    encoder_inputs_padded = pad_sequence(
        encoder_inputs, batch_first=True, padding_value=PAD_token)
    decoder_inputs_padded = pad_sequence(
        decoder_inputs, batch_first=True, padding_value=PAD_token)
    decoder_targets_padded = pad_sequence(
        decoder_targets, batch_first=True, padding_value=PAD_token)

    return encoder_inputs_padded, decoder_inputs_padded, decoder_targets_padded


def get_dataloaders(encoder_input_train, decoder_input_train, decoder_target_train,
                    encoder_input_val, decoder_input_val, decoder_target_val,
                    encoder_input_test, decoder_input_test, decoder_target_test,
                    batch_size):
    # Create datasets
    train_dataset = Seq2SeqDataset(
        encoder_input_train, decoder_input_train, decoder_target_train)
    val_dataset = Seq2SeqDataset(
        encoder_input_val, decoder_input_val, decoder_target_val)
    test_dataset = Seq2SeqDataset(
        encoder_input_test, decoder_input_test, decoder_target_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
