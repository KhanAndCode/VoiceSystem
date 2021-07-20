# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from SpeakerIdentification import speaker_identification
from mongoengine import connect
from mongoengine import disconnect
from pathlib import Path
import librosa
from mongoengine import Document, ListField, StringField, FloatField, IntField, ObjectIdField
import numpy as np
from datetime import datetime
import time
import recorder
from SpeakerIdentification import data_loader
import warnings

if __name__ == '__main__':
    start_time = datetime.now()
    disconnect()
    connect(host="mongodb://127.0.0.1:27017/call_center")
    # data_loader.save_recorded()
    # recorder.lets_go()
    # speaker_identification.fit()
    # speaker_identification.upload_users()
    warnings.filterwarnings("ignore")
    rec = recorder.Recorder()
    rec.start_recording()
    print(datetime.now() - start_time)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
