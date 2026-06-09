# custom_policy.py
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MapFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that processes LIDAR, position, map features, and local map
    """
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=256)
        
        # LIDAR feature extractor
        self.lidar_net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Position feature extractor
        self.position_net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # Map features extractor
        self.map_features_net = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Local map CNN
        self.local_map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Test local map CNN to get output size
        test_tensor = torch.zeros(1, 1, 20, 20)
        cnn_output_size = self.local_map_cnn(test_tensor).shape[1]
        
        # Combined features network
        combined_features_size = 64 + 32 + 64 + cnn_output_size
        self.combined_net = nn.Sequential(
            nn.Linear(combined_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Update features_dim
        self._features_dim = 128
    
    def forward(self, observations: dict) -> torch.Tensor:
        lidar_features = self.lidar_net(observations['lidar'])
        position_features = self.position_net(observations['position'])
        map_features = self.map_features_net(observations['map_features'])
        
        # Process local map
        local_map = observations['local_map'].unsqueeze(1)  # Add channel dimension
        local_map_features = self.local_map_cnn(local_map)
        
        # Combine all features
        combined = torch.cat([lidar_features, position_features, map_features, local_map_features], dim=1)
        
        return self.combined_net(combined)