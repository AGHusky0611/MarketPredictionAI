# train.py
import torch
import torch.nn as nn
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Fetch and Prepare Data
# In practice, you would load years of data from an API or database here
df = pd.read_csv("historical_btc_data.csv") 

# Engineer features using pandas_ta
df.ta.macd(append=True)
df.ta.rsi(length=14, append=True)
df.ta.bbands(append=True)
df.dropna(inplace=True)

# Select features and target (predicting the 'Close' price)
features = ['Close', 'MACD_12_26_9', 'RSI_14', 'BBL_5_2.0', 'BBU_5_2.0']
data = df[features].values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences (e.g., look back 60 days to predict the next day)
sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0]) # Predicting the 'Close' column

X, y = np.array(X), np.array(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 2. Define the Model Architecture
class MarketPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MarketPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = MarketPredictor(input_size=len(features), hidden_size=50, num_layers=2)

# 3. Train the Model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 4. Save the Model Weights
torch.save(model.state_dict(), 'vantage_market_model.pt')
print("Model saved successfully!")