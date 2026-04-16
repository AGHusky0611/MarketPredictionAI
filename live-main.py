# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np

app = FastAPI(title="Vantage Inference API")

# 1. Re-define the exact same architecture
class MarketPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MarketPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 2. Load the trained model into memory at startup
model = MarketPredictor(input_size=5, hidden_size=50, num_layers=2)
model.load_state_dict(torch.load('vantage_market_model.pt'))
model.eval() # Set to evaluation mode

class PredictionRequest(BaseModel):
    # Expecting a 2D array: 60 days of 5 features
    sequence_data: list[list[float]] 

@app.post("/predict")
async def predict_market(request: PredictionRequest):
    try:
        # Convert incoming JSON data to a PyTorch tensor
        # Shape must be (1, sequence_length, num_features)
        input_data = np.array(request.sequence_data)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        
        # 3. Run the prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            
        # Extract the value
        predicted_value = prediction.item()
        
        # Note: You would also need to apply the inverse_transform of your scaler here
        # to convert the predicted value back to a real dollar amount.
        
        return {
            "status": "success",
            "predicted_scaled_price": predicted_value,
            "signal": "Bullish" if predicted_value > input_data[-1] else "Bearish"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))