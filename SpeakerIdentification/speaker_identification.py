from SpeakerIdentification import ai
from SpeakerIdentification import data_loader
import torch
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
from mongoengine import connect
from mongoengine import disconnect
import random
from pathlib import Path
import torchaudio
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from SpeakerIdentification import ml


class Identifier():
    def __init__(self):
        self.database_connected, self.n_speakers = self.server_connecter()
        self.model = ai.evaluate(n_classes=self.n_speakers)
        self.phone = -1
        self.svm = ml.eval_svm()
        self.pac = ml.eval_pac()
        self.preds = []
        self.probs = []
        self.id = -1
        self.identified = False
        self.saving_voiceprints = True
        self.directory = self.__make_dir__()

    def server_connecter(self):
        connect(host="mongodb://127.0.0.1:27017/call_center")
        users = data_loader.User.objects.count()
        return True, users

    def identify(self, sr, audio):
        audio = torch.from_numpy(audio).view(1, 8000)
        mfcc = torchaudio.transforms.MFCC(sr, 120, log_mels=False,
                                          melkwargs={'n_fft': 512, 'n_mels': 120, 'win_length': 200, 'hop_length': 80,
                                                     'f_min': 70.0, 'f_max': 4000.0})(audio)
        pred, prob = ai.identify(self.model, mfcc)
        self.preds.append(pred[0])
        # print("CNN думает, что говорит клиент:", pred)
        pred = ml.ml_classify(self.svm, mfcc)
        self.preds.append(pred[0])
        mfcc = torchaudio.transforms.MFCC(sr, 70, log_mels=False,
                                          melkwargs={'n_fft': 512, 'n_mels': 70, 'win_length': 200, 'hop_length': 80,
                                                     'f_min': 70.0, 'f_max': 4000.0})(audio)
        pred = ml.ml_classify(self.pac, mfcc)
        self.preds.append(pred[0])
        print(self.preds)
        # print("SVM думает, что говорит клиент:", pred)
        # if len(self.preds) >= 6:
        #     self.__conclude__()
        # else:
        #     print("Для распознавания нужно еще " + str((6 - len(self.preds))/2) + " сек. голосовой информации." )

    def __make_dir__(self):
        path = "SpeakerIdentification/recorded/" + str(datetime.now().timestamp()) + "/"
        Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def __conclude__(self):
        self.identified = True

        if len(np.unique(self.preds)) - len(self.preds) > -2:
            print(np.unique(self.preds))
            print("Похоже это новый клиент, хотите его добавить? (y/n)")
            answer = input()
            if answer == "y":
                self.add_speaker()
            else:
                self.saving_voiceprints = False
            return
        else:
            self.identified = True
            result = max(set(self.preds), key = self.preds.count)
            user = data_loader.User.objects(label = result).first()
            print(" Я думаю это: ", user.name)
        print(self.preds)
        return

    def voiceprint_save(self, audio):
        sf.write(self.directory + str(datetime.now().timestamp()) + ".wav", audio, len(audio))

    def add_speaker(self):
        print("Введите имя")
        name = input()
        phone = random.randint(89000000000, 89999999999)
        f = open(self.directory + "/info.txt", "w")
        f.write(str(self.id) + "\n")
        f.write(name + "\n")
        f.write(str(phone))
        f.close()


def upload_users():
    data_loader.upload_new_user()


# Думаю создать отдельный server который бы обучал, добавлял пользователей в бд и т.д., чтоб не
# мешать обучение и работу

def fit():
    model = ai.VoiceCnn()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    history = ai.train(model=model, epochs=6, batch_size=64)
    torch.save(model.state_dict(), 'SpeakerIdentification\\models\\voice.pt')
    with open('history.pkl', 'wb') as le_dump_file:
        pickle.dump(history, le_dump_file)
    print("done")
