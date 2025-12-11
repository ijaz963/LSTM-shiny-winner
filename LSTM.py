#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

# --- Accessibility Setup: Ensure Colorblind-Safe Palette ---
try:
    # Use matplotlib's internal access for colormaps for maximum compatibility
    # Accessing the supported cblind map 'cb.solstice'
    cmap = mpl.colormaps.get_cmap("cb.solstice")

    # Extract the first two highly contrasting colors for the two plot lines
    COLOR_SCHEME = [
        cmap(0.0), # Color 1 (e.g., darkest tone)
        cmap(0.5)  # Color 2 (e.g., contrasting middle tone)
    ]
    print("Using colorblind-friendly 'cb.solstice' palette.")
except Exception:
    warnings.warn("Custom colormap failed. Using standard high-contrast fallback.", UserWarning)
    # Fallback: Blue and Orange (high contrast)
    COLOR_SCHEME = ['#0072B2', '#D55E00']

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Define data generation parameters
T_END = 100
N_POINTS = 1000
SEQ_LENGTH = 10 # How many past steps the LSTM looks at

# Generate sine wave data
t = np.linspace(0, T_END, N_POINTS)
data = np.sin(t)

# Prepare input-output sequences
def create_sequences(data, seq_length):
    """Converts a time series into (X, Y) pairs for sequence-to-value regression."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    # X shape: (N_samples, seq_length), Y shape: (N_samples,)
    return np.array(xs), np.array(ys)

X, y = create_sequences(data, SEQ_LENGTH)

# Convert to PyTorch tensors, adding feature dimension (None)
# trainX shape: (N_samples, seq_length, 1)
# trainY shape: (N_samples, 1)
trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)

print(f"Total sequences created: {len(X)}")
print(f"Input sequence length (L): {SEQ_LENGTH}")


# In[ ]:


# Define LSTM model (Sequence-to-Value Architecture)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # Core LSTM Layer: handles the sequence and long-term memory via the cell state.
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # FC Layer: maps the hidden state of the *last* sequence step to the output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        # Initialize hidden state (h0) and cell state (c0) if not provided
        if h0 is None or c0 is None:
            # States must be (layer_dim, batch_size, hidden_dim)
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        # LSTM Forward Pass: out shape is (batch, seq_len, hidden_dim)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # We only care about the last output step (out[:, -1, :]) for prediction
        out = self.fc(out[:, -1, :])

        # Return the prediction and the final states (hn, cn)
        return out, hn, cn

# Initialize model, loss, optimizer
model = LSTMModel(input_dim=1, hidden_dim=100, layer_dim=1, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"Model instantiated with {model.hidden_dim} hidden units.")


# In[ ]:


# Training loop
num_epochs = 100
h0, c0 = None, None # Start with None, states will be initialized inside forward pass

print("--- Starting Training ---")
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Pass states from previous step (or None if first epoch)
    outputs, h0, c0 = model(trainX, h0, c0)

    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()

    # DETACH states: Crucial for time-series continuity while preventing unnecessary BPTT across epochs
    h0, c0 = h0.detach(), c0.detach()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Evaluation and plotting (Figure 1)
model.eval()
# Perform final prediction using the learned final states
predicted, _, _ = model(trainX, h0, c0)

# Prepare data for plotting (remove the initial sequence steps not predicted)
original = data[SEQ_LENGTH:]
time_steps = np.arange(SEQ_LENGTH, len(data))

# Colorblind-friendly plotting using the scheme defined in Cell 1
plt.figure(figsize=(12, 6))
# Line 1: Original Data
plt.plot(time_steps,
         original,
         label='Original Data (Sine Wave)',
         color=COLOR_SCHEME[0],
         linewidth=2)
# Line 2: Predicted Data (using contrasting color and dashed line for visual distinction)
plt.plot(time_steps,
         predicted.detach().numpy().flatten(),
         label='Predicted Data (LSTM Forecast)',
         linestyle='--',
         color=COLOR_SCHEME[1],
         linewidth=2)

plt.title('Figure 1: LSTM Forecasting Performance on Sine Wave (Sequence-to-Value Regression)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('lstm_sine_prediction.pdf')
plt.show()

print("\nCode execution complete. Output PDF and Figure 1 generated.")


# In[ ]:




