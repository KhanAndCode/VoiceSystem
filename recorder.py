from datetime import datetime
import sounddevice as sd

import argparse
import queue
import sys
import matplotlib;

matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from SpeakerIdentification import speaker_identification as si
from vad import vad


class Recorder():
    def __init__(self):
        self.parser = argparse.ArgumentParser(add_help=False)
        self.args = self.__init_parser__()
        self.identifier = si.Identifier()
        self.vad = vad.KHANVAD()

    def __audio_callback__(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        audio = np.reshape(indata, [8000])

        if not self.identifier.identified:
            if self.vad.is_voice(audio):
                self.identifier.voiceprint_save(audio)
                self.identifier.identify(8000, audio)
            else:
                self.noise = audio
                print("Сейчас тишина")

        #

    def __int_or_str__(self, text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    def __init_parser__(self):
        self.parser.add_argument(
            '-l', '--list-devices', action='store_true',
            help='show list of audio devices and exit')
        args, remaining = self.parser.parse_known_args()
        if args.list_devices:
            print(sd.query_devices())
            self.parser.exit(0)
        self.parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[self.parser])
        self.parser.add_argument(
            'channels', type=int, default=1, nargs='*', metavar='CHANNEL',
            help='input channels to plot (default: the first)')
        self.parser.add_argument(
            '-d', '--device', type=self.__int_or_str__,
            help='input device (numeric ID or substring)')
        self.parser.add_argument(
            '-w', '--window', type=float, default=200, metavar='DURATION',
            help='visible time slot (default: %(default)s ms)')
        self.parser.add_argument(
            '-i', '--interval', type=float, default=30,
            help='minimum time between plot updates (default: %(default)s ms)')
        self.parser.add_argument(
            '-b', '--blocksize', type=int, help='block size (in samples)')
        self.parser.add_argument(
            '-r', '--samplerate', type=float, help='sampling rate of audio device')
        self.parser.add_argument(
            '-n', '--downsample', type=int, default=10, metavar='N',
            help='display every Nth sample (default: %(default)s)')
        args = self.parser.parse_args(remaining)
        return args

    def start_recording(self):
        try:
            if self.args.samplerate is None:
                device_info = sd.query_devices(self.args.device, 'input')
                self.args.samplerate = device_info['default_samplerate']

            stream = sd.InputStream(
                device=self.args.device, channels=1,
                samplerate=8000, callback=self.__audio_callback__, blocksize=8000)

            with stream:
                print('Идет запись...')
                while True:
                    if not self.identifier.saving_voiceprints:
                        return
        except Exception as e:
            self.parser.exit(type(e).__name__ + ': ' + str(e))
