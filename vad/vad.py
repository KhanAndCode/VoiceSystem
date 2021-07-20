import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa


class VAD(nn.Module):
    #  [128, 1, 80], expected input[256, 128, 150]
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(5, 64, 16)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(p=0.1)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        logits = self.out(x)
        return logits


class KHANVAD():
    def __init__(self):
        self.model = self.__eval__()

    def __eval__(self):
        model = VAD()
        model.load_state_dict(torch.load("vad\\vad.pt"))
        model.eval()
        return model

    def is_voice(self, audio):
        mfcc = librosa.feature.mfcc(audio, 8000, n_mfcc=5)
        inputs = torch.from_numpy(mfcc)
        output = self.model(inputs.unsqueeze(0))
        output = torch.exp(output)
        prediction = int(torch.max(output.data, 1)[1].numpy())
        if prediction == 1:
            return True
        return False
