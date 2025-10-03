"""
Transformer-based Model for Carbon Market Forecasting
======================================================

This module implements a Temporal Fusion Transformer (TFT) architecture
for multi-step time series forecasting of carbon prices.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


class MultiHeadAttention(layers.Layer):
    """Multi-Head Attention layer for Transformer"""
    
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None, training=False):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        
        return output, attention_weights


class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.att = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False, mask=None):
        attn_output, _ = self.att(inputs, inputs, inputs, mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class PositionalEncoding(layers.Layer):
    """Positional encoding layer for Transformer"""
    
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


class TemporalFusionTransformer(keras.Model):
    """
    Temporal Fusion Transformer for time series forecasting
    
    This model combines:
    - Variable selection networks for feature importance
    - LSTM for local temporal processing
    - Multi-head attention for long-range dependencies
    - Quantile outputs for uncertainty estimation
    """
    
    def __init__(
        self,
        num_features: int,
        num_outputs: int,
        num_encoder_steps: int,
        num_decoder_steps: int,
        d_model: int = 128,
        num_heads: int = 4,
        dff: int = 512,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        num_quantiles: int = 3
    ):
        super(TemporalFusionTransformer, self).__init__()
        
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_encoder_steps = num_encoder_steps
        self.num_decoder_steps = num_decoder_steps
        self.d_model = d_model
        self.num_quantiles = num_quantiles
        
        # Input embedding layers
        self.input_embedding = layers.Dense(d_model, activation='relu')
        self.positional_encoding = PositionalEncoding(
            num_encoder_steps + num_decoder_steps, d_model
        )
        
        # Variable selection network
        self.variable_selection = keras.Sequential([
            layers.Dense(d_model, activation='relu'),
            layers.Dense(num_features, activation='softmax')
        ])
        
        # Encoder LSTM
        self.encoder_lstm = layers.LSTM(
            d_model, return_sequences=True, return_state=True
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Decoder LSTM
        self.decoder_lstm = layers.LSTM(
            d_model, return_sequences=True
        )
        
        # Output layers for quantile forecasts
        self.output_layers = []
        for _ in range(num_quantiles):
            self.output_layers.append(
                keras.Sequential([
                    layers.Dense(dff, activation='relu'),
                    layers.Dropout(dropout_rate),
                    layers.Dense(num_decoder_steps * num_outputs)
                ])
            )
        
        self.reshape_layer = layers.Reshape([num_decoder_steps, num_outputs])
    
    def call(self, inputs, training=False):
        # Input shape: (batch_size, time_steps, num_features)
        batch_size = tf.shape(inputs)[0]
        
        # Variable selection
        variable_weights = self.variable_selection(inputs)
        selected_inputs = inputs * variable_weights
        
        # Input embedding
        embedded = self.input_embedding(selected_inputs)
        embedded = self.positional_encoding(embedded)
        
        # Encoder
        encoder_outputs, state_h, state_c = self.encoder_lstm(embedded)
        
        # Apply transformer blocks
        transformer_output = encoder_outputs
        for transformer_block in self.transformer_blocks:
            transformer_output = transformer_block(
                transformer_output, training=training
            )
        
        # Decoder
        decoder_output = self.decoder_lstm(
            transformer_output, initial_state=[state_h, state_c]
        )
        
        # Generate quantile predictions
        quantile_outputs = []
        for output_layer in self.output_layers:
            output = output_layer(decoder_output[:, -1, :])
            output = self.reshape_layer(output)
            quantile_outputs.append(output)
        
        # Stack quantile outputs
        outputs = tf.stack(quantile_outputs, axis=-1)
        
        # Return mean prediction (middle quantile)
        return outputs[..., self.num_quantiles // 2]


class TransformerCarbonModel:
    """
    Wrapper class for Transformer-based carbon price prediction
    """
    
    def __init__(
        self,
        num_features: int,
        input_steps: int = 7,
        output_steps: int = 7,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.1
    ):
        self.num_features = num_features
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """Build the Transformer model"""
        
        # Simplified Transformer for easier training
        inputs = layers.Input(shape=(self.input_steps, self.num_features))
        
        # Embedding layer
        x = layers.Dense(self.d_model, activation='relu')(inputs)
        
        # Positional encoding
        pos_encoding = PositionalEncoding(self.input_steps, self.d_model)
        x = pos_encoding(x)
        
        # Transformer blocks
        for _ in range(self.num_layers):
            transformer_block = TransformerBlock(
                self.d_model, 
                self.num_heads, 
                self.d_model * 4, 
                self.dropout_rate
            )
            x = transformer_block(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Final output
        outputs = layers.Dense(self.output_steps * self.num_features)(x)
        outputs = layers.Reshape((self.output_steps, self.num_features))(outputs)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10
    ) -> keras.callbacks.History:
        """Train the Transformer model"""
        
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train model
        self.history = self.model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(data)
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model:
            self.model.save(f"{path}/transformer_model.keras")
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model = keras.models.load_model(
            f"{path}/transformer_model.keras",
            custom_objects={
                'MultiHeadAttention': MultiHeadAttention,
                'TransformerBlock': TransformerBlock,
                'PositionalEncoding': PositionalEncoding
            }
        )


def create_transformer_model(num_features: int, out_steps: int = 7) -> keras.Model:
    """
    Create a Transformer model compatible with existing pipeline
    
    Args:
        num_features: Number of input features
        out_steps: Number of prediction steps
    
    Returns:
        Compiled Keras model
    """
    transformer = TransformerCarbonModel(
        num_features=num_features,
        input_steps=7,
        output_steps=out_steps,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1
    )
    
    model = transformer.build_model()
    return model


def train_transformer_model(
    model: keras.Model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor,
    max_epochs: int = 50
):
    """
    Train Transformer model using existing data pipeline
    
    Compatible with existing train_model function signature
    """
    from utils.windowgenerator import WindowGenerator
    
    OUT_STEPS = 7
    INPUT_STEPS = 7
    
    # Create window generator
    multi_window = WindowGenerator(
        input_width=INPUT_STEPS,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_width=OUT_STEPS,
        shift=OUT_STEPS
    )
    
    # Train using preprocessor's compile_and_fit
    history = preprocessor.compile_and_fit(
        model, 
        multi_window, 
        use_early_stopping=True, 
        max_epochs=max_epochs
    )
    
    return history
