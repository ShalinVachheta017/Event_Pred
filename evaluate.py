# evaluate.py

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import PAD_token


def evaluate_model(model, data_loader, criterion, num_classes, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for encoder_input, _, decoder_target in data_loader:
            encoder_input = encoder_input.to(device)
            decoder_target_indices = decoder_target.to(device)

            outputs = model(encoder_input, decoder_target_indices,
                            teacher_forcing_ratio=0.0)

            outputs_flat = outputs.view(-1, num_classes)
            decoder_target_flat = decoder_target_indices.view(-1)

            loss = criterion(outputs_flat, decoder_target_flat)
            total_loss += loss.item()

            predictions = torch.argmax(outputs_flat, dim=1)
            mask = decoder_target_flat != PAD_token
            correct = (predictions == decoder_target_flat) & mask
            accuracy = correct.sum().item() / mask.sum().item()
            total_accuracy += accuracy

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader)
    print(f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")


def compute_metrics(model, data_loader, num_classes, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for encoder_input, _, decoder_target in data_loader:
            encoder_input = encoder_input.to(device)
            decoder_target_indices = decoder_target.to(device)

            outputs = model(encoder_input, decoder_target_indices,
                            teacher_forcing_ratio=0.0)

            outputs_flat = outputs.view(-1, num_classes)
            decoder_target_flat = decoder_target_indices.view(-1)

            predictions = torch.argmax(outputs_flat, dim=1)

            mask = decoder_target_flat != PAD_token
            predictions = predictions[mask]
            targets = decoder_target_flat[mask]

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(
        all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions,
                          average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions,
                  average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
