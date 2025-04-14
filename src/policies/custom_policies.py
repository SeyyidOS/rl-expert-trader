# src/policies/custom_policies.py
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Example of a custom feature extractor using LSTM.
class CustomLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_size=128):
        # For instance, assume observation_space.shape[0] is the feature dimension.
        super(CustomLSTMExtractor, self).__init__(observation_space, features_dim=hidden_size)
        self.lstm = nn.LSTM(input_size=observation_space.shape[0], hidden_size=hidden_size, batch_first=True)
        # Optionally add a fully connected layer or other layers.

    def forward(self, observations):
        # Assume observations shape is [batch, sequence_length, feature_dim].
        # If your observations are not sequences, you may need to reshape or adapt accordingly.
        lstm_out, _ = self.lstm(observations)
        # We take the output from the last time step
        return lstm_out[:, -1, :]


# Custom LSTM Policy
class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # Pass a custom feature extractor to the parent class.
        kwargs['features_extractor_class'] = CustomLSTMExtractor
        kwargs['features_extractor_kwargs'] = dict(hidden_size=128)
        super(CustomLSTMPolicy, self).__init__(*args, **kwargs)
        # Further customization can be added here if necessary.


# Example of a custom CNN extractor
class CustomCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNNExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        cnn_output = self.cnn(observations)
        return self.linear(cnn_output)


# Custom CNN Policy
class CustomCNNPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs['features_extractor_class'] = CustomCNNExtractor
        kwargs['features_extractor_kwargs'] = dict(features_dim=128)
        super(CustomCNNPolicy, self).__init__(*args, **kwargs)


# Example of a custom attention-based extractor (very simplified version)
class CustomAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomAttentionExtractor, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(input_dim, features_dim)

    def forward(self, observations):
        # Reshape observations into sequences if necessary.
        # For simplicity, treat the batch dimension as the sequence dimension.
        attn_output, _ = self.attention(observations, observations, observations)
        # Pooling (e.g., taking mean)
        pooled = th.mean(attn_output, dim=1)
        return self.fc(pooled)


# Custom Attention Policy
class CustomAttentionPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs['features_extractor_class'] = CustomAttentionExtractor
        kwargs['features_extractor_kwargs'] = dict(features_dim=128)
        super(CustomAttentionPolicy, self).__init__(*args, **kwargs)
