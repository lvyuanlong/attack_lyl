import pyaudio
import wave


CHUNK = 1024
FILENAME = '/data/lvyuanlong/datasets/librispeech_out/test-clean/wav/1089-134686-0000.wav'

def play(filename = FILENAME):
    wf = wave.open(filename, 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while data != b'':
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()

play(FILENAME)