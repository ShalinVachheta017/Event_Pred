
import numpy as np
import torch

# Define special tokens
PAD_token = 0   # Padding token index
SOS_token = 1   # Start Of Sequence token index
EOS_token = 2   # End Of Sequence token index
UNK_token = 3   # Unknown token index (optional)


def build_vocab(train_sequences):
    # Flatten all sequences to get all unique feature vectors
    all_features = np.reshape(train_sequences, (-1, train_sequences.shape[2]))
    unique_features = np.unique(all_features, axis=0)

    # Map feature vectors to indices
    feature_to_index = {}
    index_to_feature = {}
    current_index = 4  # Starting index after special tokens

    for feature_vector in unique_features:
        feature_key = tuple(feature_vector)
        feature_to_index[feature_key] = current_index
        index_to_feature[current_index] = feature_vector
        current_index += 1

    num_classes = current_index  # Total number of classes including special tokens
    return feature_to_index, index_to_feature, num_classes


def convert_sequences_to_indices(sequences, feature_to_index):
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []

    for sequence in sequences:
        sequence_indices = []
        for feature_vector in sequence:
            feature_key = tuple(feature_vector)
            idx = feature_to_index.get(feature_key, UNK_token)
            sequence_indices.append(idx)

        # Prepare encoder input (without EOS)
        encoder_input = sequence_indices[:-1]  # All but the last time step

        # Prepare decoder input (with SOS at the beginning)
        decoder_input = [SOS_token] + sequence_indices[:-1]

        # Prepare decoder target (with EOS at the end)
        decoder_target = sequence_indices[1:] + [EOS_token]

        encoder_inputs.append(torch.tensor(encoder_input, dtype=torch.long))
        decoder_inputs.append(torch.tensor(decoder_input, dtype=torch.long))
        decoder_targets.append(torch.tensor(decoder_target, dtype=torch.long))

    return encoder_inputs, decoder_inputs, decoder_targets


def remove_features_from_sequences(sequences, features_to_delete_indices):
    sequences_adjusted = []
    for sequence in sequences:
        sequence_adjusted = sequence.copy()
        # Set specified features to zero
        sequence_adjusted[:, features_to_delete_indices] = 0
        sequences_adjusted.append(sequence_adjusted)
    return np.array(sequences_adjusted)
