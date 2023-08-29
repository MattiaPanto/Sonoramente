import subprocess
import time
import librosa
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import datetime

def __convert_time(seconds):
    return str(datetime.timedelta(seconds = seconds))

def __extract_audio(input_file, output_file, sr = 22050):
    st = time.time()
    ffmpeg_cmd = f"ffmpeg -hide_banner -loglevel info -y -i {input_file} -vn -threads 4 -c:a aac -b:a 96k -ar {sr} -ac 1 {output_file}"
    extract = subprocess.Popen(ffmpeg_cmd.split())
    extract.communicate()
    elapsed_time = time.time() - st
    return elapsed_time

def cut_video(input_file, output_dir, clip, yaw = 0):
    st = time.time()
    foldername = os.path.basename(input_file).replace('.mp4','_clips')
    clipFolder = os.path.join(output_dir, foldername)
    if not os.path.exists(clipFolder):
        os.system('mkdir ' + str(clipFolder))

    ffmpeg_cmd = f'ffmpeg -hide_banner -loglevel info -y -i {input_file} -threads 4'
    for idx, (s,d) in enumerate(clip):
        ffmpeg_cmd = ffmpeg_cmd + f' -ss {s} -t {d} -c copy {clipFolder}/clip{idx}.mp4'

    cut = subprocess.Popen(ffmpeg_cmd.split())
    cut.communicate()

    elapsed_time = time.time() - st
    return elapsed_time

        
def __segment(input_file, sr = 1000, max_thresh = 1, min_thresh = 0.4,  min_time=10, min_distance = 10, overlap = 10, plot=False):
    """
    Trova i punti di inizio e fine delle canzoni presenti in un video.

    :param input_file:        percorso del file audio da analizzare
    :param sr:                sampling-rate
    :param thresh:            soglia della rms sotto la quale i frame vengono scartati
    :param min_time:          il tempo minimo di durata per essere considerata una clip valida
    :param min_distance:      il tempo sotto il quale due clip vicine vengono unite
    :param overlap:           secondi di aumento della clip
    """
    def hysteresis(vect):
        count=0
        for idx, x in enumerate(vect):
            if(x == 1):
                count = count+1
            if(x==0):
                count=0
            if(x == 2):
                #converto in strong i weak precendenti
                for h in range(idx-count,idx):
                    if(vect[h] == 1):
                        vect[h] = 2

                #converto in strong i weak successivi
                for h in range(idx+1,len(vect)-1):
                    if(vect[h] == 1):
                        vect[h] = 2
                    else:
                        break  
                count=0
                    
        vect = np.where(vect == 1, 0, vect)    
        vect = np.where(vect == 2, 1, vect)          
        return vect
    
    segments = []

    y, _ = librosa.load(input_file, dtype="int16", sr=sr)
    min_time = librosa.time_to_frames(min_time, sr = sr)
    min_distance = librosa.time_to_frames(min_distance, sr = sr)
    overlap = librosa.time_to_frames(overlap, sr = sr)

    # Calcolo spettrogramma Mel
    n_mels = 26
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S[int(n_mels/2):] = 0
    # Calcolo RMS dello spettrogramma Mel
    rms = np.reshape(librosa.feature.rms(S=S, frame_length = 50),-1)
    # Applica un filtro mediano
    dim = librosa.time_to_frames(30, sr = sr)
    if (dim%2 == 0):
        dim = dim + 1

    rms = medfilt(rms, kernel_size=dim)



    #assegna 1 ai valori deboli e 2 ai valori forti
    btm = np.where((rms > min_thresh), 1, 0)
    btm = btm + np.where(rms > max_thresh, 1, 0)
    

    btm = hysteresis(btm)

    #elimino le serie di valori true più corte di un tempo "min_time"
    count = 0   
    for idx, x in enumerate(btm):
        if(x == 1):
            count = count+1
        if(x == 0 or idx == len(btm)-1):
            if(count < min_time and count != 0):
                for h in range(idx-count,idx):
                    btm[h] = 0
            count=0

    if(plot):
        fig, ax = plt.subplots(nrows=2, sharex=True)

        times = librosa.times_like(rms, sr = sr)       
        ax[0].semilogy(times, rms, label='RMS')
        ax[0].semilogy(times, btm, label='clips')
        ax[0].set(xticks=[])
        ax[0].legend()
        ax[0].label_outer()

        img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr = sr,
                                y_axis='log', x_axis='time', ax=ax[1])
        ax[1].set(title='log Power spectrogram')
        ax[0].axhline(y=max_thresh, color='red', linestyle=':')
        ax[0].axhline(y=min_thresh, color='blue', linestyle=':')
        plt.pause(-1)

    #trovo inizio e fine delle clip e stampo risultati
    segments = []
    start = None
    for i in range(len(btm)):
        if btm[i] == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                s = int(librosa.frames_to_time(start - overlap, sr = sr))
                e = int(librosa.frames_to_time(i-1 + overlap, sr = sr))
                if s < 0:
                    s=0
                if e > len(btm):
                    e = len(btm)
                segments.append((s, e-s))
                start = None
    if start is not None:
        s = int(librosa.frames_to_time(start, sr = sr))
        e = int(librosa.frames_to_time(len(btm)-1, sr = sr))
        segments.append((s, e-s))

    return segments, sr

def __writeToCsv(segmentation,output_dir):
    headers = ['Nome','Inizio','Fine']
    csvPath =os.path.join(output_dir, 'songCut.csv')
    if os.path.exists(csvPath):
        os.remove(csvPath)
    with open(csvPath, 'w') as f:
      writer = csv.writer(f)

      writer.writerow(headers)
      i = 0
      for (s,e) in segmentation:
        writer.writerow([f'clip{i}.mp4',__convert_time(s),__convert_time(s+e)])
        i+=1
        
def extract_songs(input_video, output_dir, overlap = 10, plot = False):
    aac_file = os.path.join(output_dir, 'audio.aac')
    __extract_audio(input_video, aac_file)
    seg, _ = __segment(aac_file, overlap = overlap, plot=plot, max_thresh = 0.7, min_thresh = 0.4)
    os.remove(aac_file)
    __writeToCsv(seg,output_dir)
    return seg