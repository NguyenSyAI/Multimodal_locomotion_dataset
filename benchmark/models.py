from __future__ import annotations

import torch
import torch.nn as nn

from config import BaselineSpec


class VideoIMUModel(nn.Module):
    """1 IMU + video shared feature model with three outputs."""

    def __init__(self, spec: BaselineSpec, num_classes: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(len(spec.feature_cols))
        self.lstm = nn.LSTM(
            input_size=len(spec.feature_cols),
            hidden_size=spec.lstm_hidden_size,
            num_layers=spec.lstm_num_layers,
            batch_first=True,
            dropout=spec.lstm_dropout if spec.lstm_num_layers > 1 else 0.0,
            bidirectional=True,
        )
        lstm_dim = spec.lstm_hidden_size * 2
        self.imu_projection = nn.Sequential(
            nn.Linear(lstm_dim, spec.model_dim), nn.LayerNorm(spec.model_dim), nn.GELU()
        )

        self.video_net = nn.Sequential(
            nn.Conv3d(3, 16, (3, 5, 5), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True), nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            nn.Conv3d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.MaxPool3d(2, 2),
            nn.Conv3d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True), nn.MaxPool3d(2, 2),
        )
        self.video_projection = nn.Sequential(
            nn.Linear(64, spec.model_dim), nn.LayerNorm(spec.model_dim),
            nn.GELU(), nn.Dropout(spec.dropout),
        )
        self.shared = nn.Sequential(
            nn.Linear(spec.model_dim * 2, 128), nn.ReLU(inplace=True), nn.Dropout(spec.dropout),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(spec.dropout),
        )
        self.terrain_head = nn.Linear(64, num_classes)
        self.transition_head = nn.Linear(64, 1)
        self.offset_head = nn.Linear(64, 1)

    def forward(self, imu: torch.Tensor, frames: torch.Tensor | None = None):
        imu_tokens, _ = self.lstm(self.input_norm(imu))
        imu_feature = self.imu_projection(imu_tokens).mean(dim=1)

        x = frames.permute(0, 2, 1, 3, 4).contiguous()
        x = self.video_net(x)
        x = x.mean(dim=(-1, -2)).transpose(1, 2)
        video_feature = self.video_projection(x).mean(dim=1)

        shared = self.shared(torch.cat([imu_feature, video_feature], dim=-1))
        return (
            self.terrain_head(shared),
            self.transition_head(shared).squeeze(-1),
            torch.sigmoid(self.offset_head(shared).squeeze(-1)),
        )


class TwoIMUModel(nn.Module):
    """Shared BiLSTM + temporal attention with terrain/presence/time heads."""

    def __init__(self, spec: BaselineSpec, num_classes: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(len(spec.feature_cols))
        self.lstm = nn.LSTM(
            input_size=len(spec.feature_cols),
            hidden_size=spec.lstm_hidden_size,
            num_layers=spec.lstm_num_layers,
            batch_first=True,
            dropout=spec.lstm_dropout if spec.lstm_num_layers > 1 else 0.0,
            bidirectional=True,
        )
        output_dim = spec.lstm_hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(output_dim, spec.lstm_hidden_size), nn.Tanh(),
            nn.Dropout(spec.lstm_dropout), nn.Linear(spec.lstm_hidden_size, 1),
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(output_dim), nn.Dropout(spec.lstm_dropout),
            nn.Linear(output_dim, spec.lstm_hidden_size), nn.GELU(),
            nn.Dropout(spec.lstm_dropout),
        )
        self.terrain_head = nn.Linear(spec.lstm_hidden_size, num_classes)
        self.transition_head = nn.Linear(spec.lstm_hidden_size, 1)
        self.offset_head = nn.Linear(spec.lstm_hidden_size, 1)

    def forward(self, imu: torch.Tensor, frames: torch.Tensor | None = None):
        sequence, _ = self.lstm(self.input_norm(imu))
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        shared = self.shared(pooled)
        return (
            self.terrain_head(shared),
            self.transition_head(shared).squeeze(-1),
            torch.sigmoid(self.offset_head(shared).squeeze(-1)),
        )


def build_model(spec: BaselineSpec, num_classes: int) -> nn.Module:
    return VideoIMUModel(spec, num_classes) if spec.uses_video else TwoIMUModel(spec, num_classes)
