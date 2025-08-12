import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicAMTCNN(nn.Module):
    def __init__(self, input_length, n_frames, n_notes=128):
        super(BasicAMTCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, stride=3, padding=7)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=11, stride=3, padding=5)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=3, padding=4)
        self.bn3 = nn.BatchNorm1d(64)

        # Compute the resulting time dimension after convolutions
        dummy_input = torch.zeros(1, 1, input_length)
        with torch.no_grad():
            dummy_output = self.forward_features(dummy_input)
        self.output_len = dummy_output.shape[2]

        self.pool = nn.AdaptiveAvgPool1d(output_size=n_frames)

        # Fully connected layer maps to piano roll frame-wise predictions
        self.fc = nn.Linear(64, n_notes)

    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self.forward_features(x)  # [B, C, T']
        x = self.pool(x)              # [B, C, FRAMES_PER_CLIP]
        x = x.permute(0, 2, 1)        # [B, T, C]
        x = self.fc(x)                # [B, T, 128]
        return x
