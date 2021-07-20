from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from SpeakerIdentification import data_loader as dl
from torch.utils.data import DataLoader
from joblib import dump, load
import numpy as np


def fit(model, batch_size=128, epochs=100):
    train_dataset, val_dataset, classes = dl.get_dataset(dl=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(epochs):
        print(epoch)
        print("Train...")
        for batch in train_loader:
            X, y = batch
            X, y = X.numpy(), y.numpy()
            model.partial_fit(X, y, classes=classes)
        for batch in val_loader:
            X, y = batch
            X, y = X.numpy(), y.numpy()
            model.partial_fit(X, y, classes=classes)
    return model


def eval_svm():
    clf = load('SpeakerIdentification\\models\\svm_sgd.sav')
    return clf

def ml_classify(model, mfcc):
    mfcc = mfcc.numpy()
    a = mfcc.shape[1]
    b = mfcc.shape[2]
    mfcc = mfcc.reshape((1, a*b))
    pred = model.predict(mfcc)
    return pred

def eval_pac():
    clf = load('SpeakerIdentification\\models\\pac_70_98.sav')
    return clf

# model = fit(PassiveAggressiveClassifier(C=0.5), epochs=50)


# model = fit(PassiveAggressiveClassifier(C=0.5))
