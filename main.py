from prediction import train_model, predict_winner

# Train model
model, qualifying_2025 = train_model()

# Predict winner
predicted_winner = predict_winner(model, qualifying_2025)

print(f"Predicted Winner: {predicted_winner}")