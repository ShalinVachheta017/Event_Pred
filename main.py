# main.py

import torch
import numpy as np

from data_loader import load_data
from utils import build_vocab, convert_sequences_to_indices, remove_features_from_sequences, PAD_token
from dataset import get_dataloaders, Seq2SeqDataset, collate_fn  
from train import train_model
from evaluate import evaluate_model, compute_metrics

def main():
    # Load data
    sequences = load_data('event.csv')

    # Split data
    train_size = int(0.7 * len(sequences))
    val_size = int(0.15 * len(sequences))

    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:train_size + val_size]
    test_sequences = sequences[train_size + val_size:]

    # Build vocabulary
    feature_to_index, index_to_feature, num_classes = build_vocab(train_sequences)

    # Convert sequences to indices
    encoder_input_train, decoder_input_train, decoder_target_train = convert_sequences_to_indices(
        train_sequences, feature_to_index)
    encoder_input_val, decoder_input_val, decoder_target_val = convert_sequences_to_indices(
        val_sequences, feature_to_index)
    encoder_input_test, decoder_input_test, decoder_target_test = convert_sequences_to_indices(
        test_sequences, feature_to_index)

    # Get data loaders
    batch_size = 16
    train_loader, val_loader, test_loader = get_dataloaders(
        encoder_input_train, decoder_input_train, decoder_target_train,
        encoder_input_val, decoder_input_val, decoder_target_val,
        encoder_input_test, decoder_input_test, decoder_target_test,
        batch_size
    )

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    model = train_model(train_loader, val_loader, num_classes, device)

    # Evaluate on validation data
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_token)
    print("Validation Results:")
    evaluate_model(model, val_loader, criterion, num_classes, device)

    # Adjust test data by removing specific features
    features_to_delete = ['0', '1', '25', '26', '27', '28', '29', '30', '31']
    features_to_delete_indices = [int(col) for col in features_to_delete]
    test_sequences_adjusted = remove_features_from_sequences(test_sequences, features_to_delete_indices)

    # Prepare adjusted test sequences
    encoder_input_test_adj, decoder_input_test_adj, decoder_target_test_adj = convert_sequences_to_indices(
        test_sequences_adjusted, feature_to_index)

    # Create the adjusted test dataset and DataLoader directly
    test_dataset_adj = Seq2SeqDataset(
        encoder_input_test_adj, decoder_input_test_adj, decoder_target_test_adj)
    test_loader_adj = torch.utils.data.DataLoader(test_dataset_adj, batch_size=batch_size,
                                                  shuffle=False, collate_fn=collate_fn)

    # Evaluate on test data
    print("Test Results:")
    evaluate_model(model, test_loader_adj, criterion, num_classes, device)

    # Compute detailed metrics
    print("Detailed Metrics on Test Data:")
    compute_metrics(model, test_loader_adj, num_classes, device)

if __name__ == "__main__":
    main()
