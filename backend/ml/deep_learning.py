#!/usr/bin/env python3
"""
🧠 Deep Learning Models for Advanced Fraud Detection
Gelişmiş fraud detection için derin öğrenme modelleri
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class AutoEncoder(nn.Module):
    """Anomaly detection için AutoEncoder"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16]):
        super(AutoEncoder, self).__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1][1:] + [input_dim]
        for hidden_dim in hidden_dims_reversed:
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU() if hidden_dim != input_dim else nn.Sigmoid(),
                ]
            )
            prev_dim = hidden_dim

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LSTMFraudDetector(nn.Module):
    """Sequence-based fraud detection için LSTM"""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super(LSTMFraudDetector, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.3
        )
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last output for classification
        output = self.classifier(attn_out[:, -1, :])
        return output


class TransformerFraudDetector(nn.Module):
    """Transformer-based fraud detection"""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 100,
    ):
        super(TransformerFraudDetector, self).__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = self._generate_positional_encoding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _generate_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        seq_len = x.size(1)
        x += self.pos_encoding[:, :seq_len, :].to(x.device)

        # Transformer forward pass
        transformer_out = self.transformer(x.transpose(0, 1))

        # Use mean pooling for classification
        pooled = transformer_out.mean(dim=0)
        output = self.classifier(pooled)

        return output


class GraphNeuralNetwork(nn.Module):
    """Graph-based fraud detection için GNN"""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(GraphNeuralNetwork, self).__init__()

        self.conv1 = self._graph_conv_layer(input_dim, hidden_dim)
        self.conv2 = self._graph_conv_layer(hidden_dim, hidden_dim)
        self.conv3 = self._graph_conv_layer(hidden_dim, 64)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _graph_conv_layer(self, in_dim: int, out_dim: int):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.2),
        )

    def forward(self, x, adjacency_matrix):
        # Graph convolution operations
        x = self.conv1(torch.matmul(adjacency_matrix, x))
        x = self.conv2(torch.matmul(adjacency_matrix, x))
        x = self.conv3(torch.matmul(adjacency_matrix, x))

        # Node-level classification
        output = self.classifier(x)
        return output


class EnsembleDeepLearning:
    """Multiple deep learning modellerinin ensemble'ı"""

    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()

    def add_model(self, name: str, model, weight: float = 1.0):
        """Ensemble'a model ekle"""
        self.models[name] = model
        self.weights[name] = weight

    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs: int = 100):
        """Ensemble modellerini eğit"""

        # Data preprocessing
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        results = {}

        for name, model in self.models.items():
            logger.info(f"Training {name} model...")

            if isinstance(
                model,
                (
                    AutoEncoder,
                    LSTMFraudDetector,
                    TransformerFraudDetector,
                    GraphNeuralNetwork,
                ),
            ):
                # PyTorch models
                results[name] = self._train_pytorch_model(
                    model, X_train_scaled, y_train, X_val_scaled, y_val, epochs
                )
            else:
                # TensorFlow/Keras models
                results[name] = self._train_tensorflow_model(
                    model, X_train_scaled, y_train, X_val_scaled, y_val, epochs
                )

        return results

    def _train_pytorch_model(self, model, X_train, y_train, X_val, y_val, epochs):
        """PyTorch model eğitimi"""

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.BCELoss()

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()

            if isinstance(model, GraphNeuralNetwork):
                # For GNN, we need adjacency matrix (simplified)
                adj_matrix = torch.eye(X_train_tensor.size(0))
                outputs = model(X_train_tensor, adj_matrix)
            else:
                outputs = model(X_train_tensor)

            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                if isinstance(model, GraphNeuralNetwork):
                    adj_matrix_val = torch.eye(X_val_tensor.size(0))
                    val_outputs = model(X_val_tensor, adj_matrix_val)
                else:
                    val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
                )

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
        }

    def _train_tensorflow_model(self, model, X_train, y_train, X_val, y_val, epochs):
        """TensorFlow model eğitimi"""

        # Compile model
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        ]

        # Train model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )

        return history.history

    def predict_ensemble(self, X):
        """Ensemble prediction"""
        X_scaled = self.scaler.transform(X)
        predictions = {}

        for name, model in self.models.items():
            if isinstance(
                model,
                (
                    AutoEncoder,
                    LSTMFraudDetector,
                    TransformerFraudDetector,
                    GraphNeuralNetwork,
                ),
            ):
                # PyTorch prediction
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled)
                    if isinstance(model, GraphNeuralNetwork):
                        adj_matrix = torch.eye(X_tensor.size(0))
                        pred = model(X_tensor, adj_matrix).numpy()
                    else:
                        pred = model(X_tensor).numpy()
                predictions[name] = pred.flatten()
            else:
                # TensorFlow prediction
                pred = model.predict(X_scaled, verbose=0)
                predictions[name] = pred.flatten()

        # Weighted ensemble
        ensemble_pred = np.zeros(len(X))
        total_weight = sum(self.weights.values())

        for name, pred in predictions.items():
            weight = self.weights[name] / total_weight
            ensemble_pred += weight * pred

        return ensemble_pred, predictions


def create_cnn_model(input_shape: Tuple[int, ...]) -> keras.Model:
    """CNN model for transaction sequence analysis"""

    model = keras.Sequential(
        [
            layers.Conv1D(64, 3, activation="relu", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            layers.Conv1D(128, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.Conv1D(128, 3, activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(0.4),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model


def create_variational_autoencoder(input_dim: int, latent_dim: int = 32) -> keras.Model:
    """Variational AutoEncoder for anomaly detection"""

    # Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # Decoder
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(decoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    decoder_output = layers.Dense(input_dim, activation="sigmoid")(x)

    # Models
    encoder = keras.Model(encoder_input, [z_mean, z_log_var, z])
    decoder = keras.Model(decoder_input, decoder_output)

    # VAE
    vae_output = decoder(encoder(encoder_input)[2])
    vae = keras.Model(encoder_input, vae_output)

    # Custom loss
    def vae_loss(x, x_decoded):
        reconstruction_loss = keras.losses.binary_crossentropy(x, x_decoded)
        reconstruction_loss *= input_dim
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        return tf.reduce_mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss(encoder_input, vae_output))

    return vae, encoder, decoder


# Usage example
if __name__ == "__main__":
    # Create ensemble
    ensemble = EnsembleDeepLearning()

    # Add models
    ensemble.add_model("autoencoder", AutoEncoder(input_dim=50), weight=1.0)
    ensemble.add_model("lstm", LSTMFraudDetector(input_size=50), weight=1.2)
    ensemble.add_model(
        "transformer", TransformerFraudDetector(input_dim=50), weight=1.1
    )
    ensemble.add_model("cnn", create_cnn_model((100, 50)), weight=0.9)

    print("🧠 Deep Learning Ensemble created successfully!")

    # Test with sample data
    try:
        import numpy as np

        sample_features = np.random.random((10, 50))
        sample_labels = np.random.choice([0, 1], 10)

        print("📊 Testing ensemble prediction...")
        predictions, individual_preds = ensemble.predict_ensemble(sample_features)
        print(f"✅ Ensemble predictions shape: {predictions.shape}")
        print(f"🎯 Individual predictions: {len(individual_preds)} models")

    except Exception as e:
        print(f"⚠️ Test failed: {e}")
