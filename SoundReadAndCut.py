import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import csv 
from matplotlib.animation import FuncAnimation

# constants
SampleTime = 0.1  # seconds
link="sounds/star.wav"

#Data storage
#Time domain
Amplitude = []
Time = []
#Frequency domain
Frequency = []
Magnitude = []

#Functions

def readAudio(link):
    sampleRate, dataInitial = wavfile.read(link)

    # Convert stereo to mono if needed
    if len(dataInitial.shape) > 1:
        dataMono = dataInitial.mean(axis=1)
    else:
        dataMono = dataInitial

    # Number of samples per chunk
    N = int(SampleTime * sampleRate)

    return dataMono, N, sampleRate

def PerformDFT(link):
    sound, N, sampleRate = readAudio(link)

    # Precompute DFT matrix
    n = np.arange(N)
    k = n.reshape((N, 1))

    DFT_matrix = np.exp(-2j * np.pi * k * n / N)
    freqs = np.arange(N) * sampleRate / N

    # Process chunks
    for i in range(0, len(sound), N):

        sample = sound[i:i + N]

        if len(sample) != N:
            continue

        # Time axis
        time_axis = np.arange(N) / sampleRate

        # Manual DFT using matrix multiplication
        X = DFT_matrix @ sample

        #magnitude = np.abs(X)
        magnitude = np.abs(X[:N//2]) * 2 / N
        freqs = freqs[:N//2]

        # Store results
        Amplitude.append(sample)
        Time.append(time_axis)
        Frequency.append(freqs)
        Magnitude.append(magnitude)
def sortFrequencyMagnitude(Frequency, Magnitude):
    sorted_Frequency = []
    sorted_Magnitude = []

    for freqs_t, mags_t in zip(Frequency, Magnitude):
        # Sort indices by magnitude (descending)
        idx = np.argsort(mags_t)[::-1]

        # Apply sorting
        sorted_freqs_t = freqs_t[idx]
        sorted_mags_t = mags_t[idx]

        sorted_Frequency.append(sorted_freqs_t)
        sorted_Magnitude.append(sorted_mags_t)
        
        
    #print(sorted_Magnitude[0][0], sorted_Magnitude[0][1])
    #print(sorted_Frequency[0][0], sorted_Frequency[0][1])
    
    Required_frequencies = np.array([freqs_t[:2] for freqs_t in sorted_Frequency])
    Required_magnitudes = np.array([mags_t[:2] for mags_t in sorted_Magnitude])
    for x in range(len(Required_frequencies)):
        
        line = ""
        if Required_magnitudes[x][0] > 1500:
            line = line + str(Required_frequencies[x][0]) + " "
        else: 
            line = line + "pause1 "
            Required_frequencies[x][0] = 0
        if Required_magnitudes[x][1] > 1500:
            
            line = line + str(Required_frequencies[x][1]) + " "
        else: 
            line = line + "pause 2 "
            Required_frequencies[x][1] = 0
        #print(line)
        #print(Required_frequencies[x][0], Required_frequencies[x][1], Required_magnitudes[x][0], Required_magnitudes[x][1])
    #print(len(Required_frequencies))
    
    

    data = pd.read_csv("note_freq.csv")
    note_freq = data["Frequency"].values
    note  = data["Note"].values
    octave = data['Octave'].values
    
    song_note1 = []
    song_octave1 = []
    song_time1 = []
    
    song_note2 = []
    song_octave2 = []
    song_time2 = []
    
    for freq in Required_frequencies:
        for x in range(len(note_freq) -1):
            if freq[0] == 0:
                song_note1.append('P')
                song_octave1.append(0)
                song_time1.append(SampleTime)
                break
            if freq[0] < note_freq[x]:
                if abs(note_freq[x] - freq[0]) < abs(note_freq[x+1]-freq[0]):
                    song_note1.append(note[x])
                    song_octave1.append(int(octave[x]))
                    song_time1.append(SampleTime)
                else:
                    song_note1.append(note[x])
                    song_octave1.append(int(octave[x]))
                    song_time1.append(SampleTime)
                break
        for x in range(len(note_freq) -1):
            if freq[1] == 0:
                song_note2.append('P')
                song_octave2.append(0)
                song_time2.append(SampleTime)
                break
            if freq[1] < note_freq[x]:
                if abs(note_freq[x] - freq[1]) < abs(note_freq[x+1]-freq[1]):
                    song_note2.append(note[x])
                    song_octave2.append(int(octave[x]))
                    song_time2.append(SampleTime)
                else:
                    song_note2.append(note[x])
                    song_octave2.append(int(octave[x]))
                    song_time2.append(SampleTime)
                break
                    
    #print(song_note1)
    
    #print(song_note2)
    
    while x in range(len(song_note1)-1):
        if song_note1[x+1] == song_note1[x] and song_octave1[x+1] == song_octave1[x]:
            song_time1[x] += song_time1[x+1]
            song_time1[x] = round(song_time1[x], 2)
            del song_note1[x+1]
            del song_octave1[x+1]
            del song_time1[x+1]
        else:
            x+=1
    x = 0
    while x in range(len(song_note2)-1):
        if song_note2[x+1] == song_note2[x] and song_octave2[x+1] == song_octave2[x]:
            song_time2[x] += song_time2[x+1]
            song_time2[x] = round(song_time2[x], 2)
            del song_note2[x+1]
            del song_octave2[x+1]
            del song_time2[x+1]
        else:
            x+=1
        #print(x)
            
    
    
    outputData = pd.DataFrame({
       "Note": song_note2,
       "Octave": song_octave2,
       "Time": song_time2
    })
    
    outputData.to_csv("SongChannel2.csv", index=False)
            
    outputData = pd.DataFrame({
        "Note": song_note1,
        "Octave": song_octave1,
        "Time": song_time1
    })
    
    outputData.to_csv("SongChannel1.csv", index=False)
    

    return Required_frequencies

def animate(FrequencyLimit):
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    ax.set_xlim(0, FrequencyLimit)
    ax.set_ylim(0, np.max(Magnitude))

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Dynamic Frequency Spectrum")

    def init():
        line.set_data([], [])
        return line,

    def update(frame):

        line.set_data(Frequency[frame], Magnitude[frame])

        ax.set_title(f"Spectrum at t = {frame * SampleTime:.2f}s")

        return line,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(Frequency),
        init_func=init,
        interval=SampleTime*1000,   # milliseconds
        blit=True
    )

    plt.show()

def showFirst(n):
    # Plot spectrum
    plt.plot(Frequency[0], Magnitude[n])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Spectrum")


    plt.plot(Frequency[0], Magnitude[n+1])
    plt.show()

PerformDFT(link)
Required_frequencies = sortFrequencyMagnitude(Frequency, Magnitude)
#showFirst(100)
animate(1000)