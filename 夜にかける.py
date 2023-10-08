import pyaudio
import numpy as np
import streamlit as st 
#サンプリングレートを定義
RATE=44100

#BPMや音長を定義
BPM=120
L1=(60/BPM*4)
L2,L4,L8=(L1/2,L1/4,L1/8)

#ドレミ...の周波数を定義
C,C_s,D,D_s,E,F,F_s,G,G_s,A,A_s,B,C2,D2,D2_s=(
    261.626,277.183,293.665,311.127,329.628,
    349.228,369.994,391.995,415.305,440.000,466.164,
    493.883,523.251,587.330,622.254
)

#サイン波を生成
def tone(freq,length,gain):
    slen=int(length*RATE)
    t=float(freq)*np.pi*2/RATE
    return np.sin(np.arange(slen)*t)*gain

#再生
def play_wave(stream,samples):
    stream.write(samples.astype(np.float32).tostring())
    
#出力用のストリームを開く
p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paFloat32,
             channels=1,
             rate=RATE,
              frames_per_buffer=1024,
              output=True
             )
#音を再生
print("play")
play_wave(stream,tone(G,L8,1.0))
play_wave(stream,tone(A_s,L8,1.0))
play_wave(stream,tone(C2,L4,1.0))
play_wave(stream,tone(G_s,L8,1.0))
play_wave(stream,tone(G,L8,1.0))
play_wave(stream,tone(F,L8,1.0))
play_wave(stream,tone(D_s,L8,1.0))
play_wave(stream,tone(F,L8,1.0))
play_wave(stream,tone(C2,L8,1.0))
play_wave(stream,tone(A_s,L8,1.0))
play_wave(stream,tone(C2,L8,1.0))
play_wave(stream,tone(G,L8,1.0))
play_wave(stream,tone(F,L8,1.0))
play_wave(stream,tone(D_s,L4,1.0))
play_wave(stream,tone(C,L8,1.0))
play_wave(stream,tone(A_s,L8,1.0))
play_wave(stream,tone(G_s,L8,1.0))
play_wave(stream,tone(G,L8,1.0))
play_wave(stream,tone(F,L8,1.0))
play_wave(stream,tone(D_s,L8,1.0))
play_wave(stream,tone(D,L8,1.0))
play_wave(stream,tone(D_s,L8,1.0))
play_wave(stream,tone(F,L8,1.0))
play_wave(stream,tone(G_s,L8,1.0))
play_wave(stream,tone(G,L8,1.0))
play_wave(stream,tone(F,L8,1.0))
play_wave(stream,tone(G,L8,1.0))
play_wave(stream,tone(C2,L8,1.0))
play_wave(stream,tone(A_s,L4,1.0))
play_wave(stream,tone(G,L8,1.0))
play_wave(stream,tone(A_s,L8,1.0))
play_wave(stream,tone(C2,L4,1.0))
play_wave(stream,tone(G_s,L8,1.0))
play_wave(stream,tone(G,L8,1.0))
play_wave(stream,tone(F,L8,1.0))
play_wave(stream,tone(D2,L8,1.0))
play_wave(stream,tone(C2,L8,1.0))
play_wave(stream,tone(A_s,L8,1.0))
play_wave(stream,tone(A_s,L8,1.0))
play_wave(stream,tone(C2,L8,1.0))
play_wave(stream,tone(D2,L8,1.0))
play_wave(stream,tone(D2_s,L4,1.0))
play_wave(stream,tone(G,L8,1.0))
play_wave(stream,tone(F,L8,1.0))
play_wave(stream,tone(D_s,L4,1.0))
play_wave(stream,tone(C,L8,1.0))
play_wave(stream,tone(D_s,L8,1.0))
play_wave(stream,tone(G_s,L8,1.0))
play_wave(stream,tone(G,L8,1.0))
play_wave(stream,tone(D_s,L8,1.0))
play_wave(stream,tone(F,L8,1.0))
play_wave(stream,tone(D_s,L4,1.0))


stream.close()
