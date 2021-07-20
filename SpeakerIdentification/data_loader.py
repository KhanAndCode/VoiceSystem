from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pickle
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random
from mongoengine import connect, disconnect
from pathlib import Path
from mongoengine import Document, ListField, StringField, FloatField, IntField, ObjectIdField

DATA_MODES = ['ml', 'dl']


class VoicePrint(Document):
    _id = ObjectIdField()
    name = StringField(required=True, max_length=50)
    voiceprint = ListField(FloatField(), required=True)
    label = IntField(required=False)
    mode = StringField(required=True, max_length=10)
    sample_rate = IntField(required=True)

class User(Document):
    _id = ObjectIdField()
    name = StringField(required=True, max_length=50)
    label = IntField(required=True, unique=True)
    phone = IntField(required=True)


class VoiceDataset(Dataset):
    def __init__(self, files, mode):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files, key=lambda voiceprint: voiceprint.label)
        # режим работы
        self.mode = mode

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        # self.label_encoder = LabelEncoder()

        # if self.mode != 'test':
        self.labels = [voiceprint.label for voiceprint in self.files]
        # self.label_encoder.fit(self.labels)
        with open('labels.pkl', 'wb') as le_dump_file:
            pickle.dump(self.labels, le_dump_file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        y = self.labels[index]
        x = self.get_sample(self.files[index])
        if self.mode == 'ml':
            row, col = x.shape
            x = x.reshape(row * col)
        return x, y

    def get_sample(self, voiceprint):
        file = VoicePrint.objects(_id=voiceprint._id).only("voiceprint").only("sample_rate").first()
        amp, sr = np.asarray(file.voiceprint, dtype=float), file.sample_rate
        mfcc = librosa.feature.mfcc(y=amp, sr=sr, n_mfcc=20)
        return mfcc


def get_dataset():
    # host = from file...
    disconnect()
    connect(host="mongodb://127.0.0.1:27017/call_center")
    train_files = VoicePrint.objects(mode='train').only("_id").only("label")
    train_labels = [voiceprint.name for voiceprint in train_files]
    valid_files = VoicePrint.objects(mode='valid').only("_id").only("label")
    test_files = VoicePrint.objects(mode='test').only("_id").only("label")
    val_dataset = VoiceDataset(valid_files, mode='dl')
    train_dataset = VoiceDataset(train_files, mode='dl')
    n_classes = User.objects.count()
    return train_dataset, val_dataset, n_classes


def upload_new_user(TRAIN_DIR=Path('SpeakerIdentification\\dataset_1\\train'),
                    VAL_DIR=Path('SpeakerIdentification\\dataset_1\\val'),
                    TEST_DIR=Path('SpeakerIdentification\\dataset_1\\test')):
    train_files = sorted(list(TRAIN_DIR.rglob('*.wav')))
    val_files = sorted(list(VAL_DIR.rglob('*.wav')))
    test_files = sorted(list(TEST_DIR.rglob('*.wav')))
    names = np.unique([path.parent.name for path in train_files])
    counter = 0
    for name in names:
        User(name=name, label=counter, phone=random.randint(89000000000,89999999999)).save()
        for file in train_files:
            if name == file.parent.name:
                amplitudes, sr = librosa.load(file, 8000)
                VoicePrint(name=name, voiceprint=amplitudes, label=counter, mode="train", sample_rate=sr).save()
        for file in val_files:
            if name == file.parent.name:
                amplitudes, sr = librosa.load(file, 8000)
                VoicePrint(name=name, voiceprint=amplitudes, label=counter, mode="valid", sample_rate=sr).save()
        for file in test_files:
            if name == file.parent.name:
                amplitudes, sr = librosa.load(file, 8000)
                VoicePrint(name=name, voiceprint=amplitudes, label=counter, mode="test", sample_rate=sr).save()
        counter+=1
        print(name + " added")



def save_recorded():
    recorded_dir=Path("SpeakerIdentification/recorded/")
    subdirs = [x for x in recorded_dir.iterdir() if recorded_dir.is_dir()]
    number_of_ids  = User.objects.count()
    for subdir in subdirs:
        info_path = list(subdir.glob("info.txt"))
        voiceprints_path = list(subdir.glob("*.wav"))
        train_files, val_test_files = train_test_split(voiceprints_path, test_size=0.2)
        test_files, val_files = train_test_split(val_test_files, test_size=0.5)
        info = open(info_path[0], "r")
        id = int(info.readline())
        name = "undefined"
        if id == -1:
            name = info.readline().strip()
            phone = int(info.readline())
            User(name=name, label=number_of_ids, phone=phone).save()
            id = number_of_ids
        for file in train_files:
            amplitudes, sr = librosa.load(file, 8000)
            VoicePrint(name=name, voiceprint=amplitudes, label=id, mode="train", sample_rate=sr).save()
        for file in val_files:
            amplitudes, sr = librosa.load(file, 8000)
            VoicePrint(name=name, voiceprint=amplitudes, label=id, mode="valid", sample_rate=sr).save()
        for file in test_files:
            amplitudes, sr = librosa.load(file, 8000)
            VoicePrint(name=name, voiceprint=amplitudes, label=id, mode="test", sample_rate=sr).save()
    print("Done")




