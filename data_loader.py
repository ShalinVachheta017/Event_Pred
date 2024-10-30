

import pandas as pd
import numpy as np


def load_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Add a 'Cycle' column to identify each cycle
    data['Cycle'] = data.index // 81

    # Get event columns (all features)
    event_columns = [str(i) for i in range(34)]

    # Verify that each cycle has 81 events
    sequence_lengths = data.groupby('Cycle').size()
    assert sequence_lengths.nunique(
    ) == 1 and sequence_lengths.iloc[0] == 81, "Not all cycles have 81 events."

    # Collect sequences with all features
    sequences = []
    for _, group in data.groupby('Cycle'):
        sequence = group[event_columns].values  # Shape: (81, 34)
        sequences.append(sequence)

    sequences = np.array(sequences)  # Shape: (num_cycles, 81, 34)
    return sequences
