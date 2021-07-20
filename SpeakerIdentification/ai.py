import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from SpeakerIdentification import data_loader as dl
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset, val_dataset, n_class = dl.get_dataset()


class VoiceCnn(nn.Module):
    #  [128, 1, 80], expected input[256, 128, 150]
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(120, 64, 4, 2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(4, 2)

        self.conv3 = nn.Conv1d(256, 512, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(512, 768, 1)
        self.bn4 = nn.BatchNorm1d(768)
        self.pool4 = nn.MaxPool1d(4, 2)

        self.conv5 = nn.Conv1d(768, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)
        self.conv6 = nn.Conv1d(1024, 1536, 1)
        self.bn6 = nn.BatchNorm1d(1536)
        self.pool6 = nn.MaxPool1d(4, 2)

        self.conv7 = nn.Conv1d(1536, 2048, 1)
        self.bn7 = nn.BatchNorm1d(2048)
        self.pool7 = nn.AvgPool1d(4, 2)

        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 8192)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(8192, 12288)
        self.out = nn.Linear(12288, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        x = self.pool6(x)
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.pool7(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        logits = self.out(x)
        return logits


def fit_epoch(model, train_loader, criterion, optimizer):
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in train_loader:
        inputs = inputs.float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)

    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    return train_loss, train_acc


def eval_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in val_loader:
        inputs = inputs.float()
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size
    return val_loss, val_acc


def train(model, epochs, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        opt = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, nesterov=True)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            train_loss, train_acc = fit_epoch(model, train_loader, criterion, opt)
            print("loss", train_loss)

            val_loss, val_acc = eval_epoch(model, val_loader, criterion)
            history.append((train_loss, train_acc, val_loss, val_acc))

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss, \
                                           v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))

    return history


def evaluate(n_classes=n_class, filepath='SpeakerIdentification\\models\\voice_92.pt'):
    model = VoiceCnn(n_classes)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model


def identify(model, mfcc=None):
    output = model(mfcc)
    output = torch.nn.functional.softmax(output, dim=1)
    prob = torch.max(output.data, 1)[0].numpy()[0]
    prediction = torch.max(output.data, 1)[1].numpy()
    return prediction, prob
