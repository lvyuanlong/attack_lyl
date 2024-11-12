# 导入所需的库
import pyaudio
import wave
import threading

# 定义一些参数
CHUNK = 1024 # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16 # 采样位数
CHANNELS = 1 # 单声道
RATE = 44100 # 采样频率
RECORD_SECONDS = 10 # 需要录制的时间
WAVE_INPUT_FILENAME = "input.wav" # 输入的文件名
WAVE_OUTPUT_FILENAME = "output.wav" # 输出的文件名

# 初始化 PyAudio 对象
p = pyaudio.PyAudio()

# 定义一个播放音频的函数
def play_audio():
    # 打开输入的 WAV 文件
    wf = wave.open(WAVE_INPUT_FILENAME, 'rb')
    # 打开音频流，传入相应的参数
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # 读取数据
    data = wf.readframes(CHUNK)
    # 播放音频
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)
    # 关闭流和文件
    stream.stop_stream()
    stream.close()
    wf.close()

# 定义一个录制音频的函数
def record_audio():
    # 打开音频流，传入相应的参数
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    # 打开输出的 WAV 文件，设置相应的参数
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    # 开始录音
    print("开始录音...")
    frames = [] # 存储音频数据的列表
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK) # 读取数据
        frames.append(data) # 将数据添加到列表中
    print("结束录音...")
    # 关闭流和文件
    stream.stop_stream()
    stream.close()
    wf.writeframes(b''.join(frames))
    wf.close()

# 创建两个线程，分别执行播放和录制函数
t1 = threading.Thread(target=play_audio)
t2 = threading.Thread(target=record_audio)
# 启动线程
t1.start()
t2.start()
# 等待线程结束
t1.join()
t2.join()
# 关闭 PyAudio 对象
p.terminate()
