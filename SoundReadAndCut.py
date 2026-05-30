import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd
import csv 
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

# constants
SampleTime = 0.2    # seconds
HopTime = SampleTime      # seconds
link="sounds/bitute.wav"
    
cutOff = 0.02
octaveOffset = 0

#experimental filtering
HanningOn = True
#good

FrequencyFilter = True
MinFreq = 300
MaxFreq = 15000

LogFrequencyWeight = False
StrongCurve = False

HarmonicScore = True #Highly doubt its effectivenss - it reduces sound to 1 or 2 frequencies
max_harmonics=4

#Data storage
#Time domain
Amplitude = []
Time = []
#Frequency domain
Frequency = []
Magnitude = []

#Animation storage
AnimationFrequency = []
AnimationMagnitude = []

#Functions

def readAudio(link):
    sampleRate, dataInitial = wavfile.read(link)

    # Convert stereo to mono if needed
    if len(dataInitial.shape) > 1:
        dataMono = dataInitial.mean(axis=1)
    else:
        dataMono = dataInitial

    # Normalize to [-1, 1]
    dataMono = dataMono.astype(np.float32)
    dataMono /= np.max(np.abs(dataMono))

    # Number of samples per chunk
    N = int(SampleTime * sampleRate)

    return dataMono, N, sampleRate

def bandpass_filter(signal, sampleRate, lowcut, highcut, order=4):
    nyq = 0.5 * sampleRate
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# def harmonic_product_spectrum(magnitude):

#     hps = magnitude.copy()

#     for h in range(2, max_harmonics + 1):
#         # downsample spectrum
#         downsampled = magnitude[::h]

#         # trim to match size
#         hps[:len(downsampled)] *= downsampled

#     return hps

def harmonic_product_spectrum(magnitude):
    hps = magnitude.copy()
    x = np.arange(len(magnitude))

    for h in range(2, max_harmonics + 1):
        f = interp1d(x, magnitude, kind='linear',
                     bounds_error=False, fill_value=0)
        hps *= f(x * h)

    return hps

def PerformDFT(link):
    sound, N, sampleRate = readAudio(link)

    if (FrequencyFilter):
        sound = bandpass_filter(sound, sampleRate, MinFreq, MaxFreq)

    hopN = int(HopTime * sampleRate)

    n = np.arange(N)
    k = n.reshape((N, 1))

    DFT_matrix = np.exp(-2j * np.pi * k * n / N)
    #freqs = np.arange(N) * sampleRate / N
    full_freqs = np.arange(N) * sampleRate / N

    for i in range(0, len(sound) - N, hopN):

        sample = sound[i:i + N]

        if len(sample) != N:
            continue

        time_axis = np.arange(N) / sampleRate

        # Apply Hann window
        if(HanningOn):
            window = np.hanning(N)
            windowed_sample = sample * window
            X = DFT_matrix @windowed_sample

        else:
            X = DFT_matrix @ sample

        magnitude = np.abs(X[:N//2]) * 2 / N
        freqs = full_freqs[:N//2]

        if(HarmonicScore):
            magnitude = harmonic_product_spectrum(magnitude)
            magnitude = magnitude / (np.max(magnitude) + 1e-9)

        if (LogFrequencyWeight):
            freqs_safe = np.maximum(freqs, 1e-6)
            weight = np.log1p(freqs_safe)
            if(StrongCurve):
                weight = np.log1p(freqs_safe) ** 2
            magnitude = magnitude * weight

        #Save data for animation
        AnimationFrequency.append(freqs)
        AnimationMagnitude.append(magnitude)

        #Save data for analysis
        if i % N == 0:
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
        if Required_magnitudes[x][0] > cutOff:
            line = line + str(Required_frequencies[x][0]) + " "
        else: 
            line = line + "pause1 "
            Required_frequencies[x][0] = 0
        if Required_magnitudes[x][1] > cutOff:
            
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
                song_time1.append(SampleTime*1000)
                break
            if freq[0] < note_freq[x]:
                if abs(note_freq[x] - freq[0]) < abs(note_freq[x+1]-freq[0]):
                    song_note1.append(note[x] + str(int(octave[x])+octaveOffset))
                    song_octave1.append(int(octave[x]))
                    song_time1.append(SampleTime*1000)
                else:
                    song_note1.append(note[x] + str(int(octave[x])+octaveOffset))
                    song_octave1.append(int(octave[x])+2)
                    song_time1.append(SampleTime*1000)
                break
        for x in range(len(note_freq) -1):
            if freq[1] == 0:
                song_note2.append('P')
                song_octave2.append(0)
                song_time2.append(SampleTime*1000)
                break
            if freq[1] < note_freq[x]:
                if abs(note_freq[x] - freq[1]) < abs(note_freq[x+1]-freq[1]):
                    song_note2.append(note[x] + str(int(octave[x])+octaveOffset))
                    song_octave2.append(int(octave[x]))
                    song_time2.append(SampleTime*1000)
                else:
                    song_note2.append(note[x] + str(int(octave[x])+octaveOffset))
                    song_octave2.append(int(octave[x]))
                    song_time2.append(SampleTime*1000)
                break
                    
    #print(song_note1)
    
    #print(song_note2)
    
    while x in range(len(song_note1)-1):
        if song_note1[x+1] == song_note1[x] and song_octave1[x+1] == song_octave1[x]:
            song_time1[x] += song_time1[x+1]
            song_time1[x] = int(song_time1[x])
            del song_note1[x+1]
            del song_octave1[x+1]
            del song_time1[x+1]
        else:
            x+=1
    x = 0
    while x in range(len(song_note2)-1):
        if song_note2[x+1] == song_note2[x] and song_octave2[x+1] == song_octave2[x]:
            song_time2[x] += song_time2[x+1]
            song_time2[x] = int(song_time2[x])
            del song_note2[x+1]
            del song_octave2[x+1]
            del song_time2[x+1]
        else:
            x+=1
        #print(x)
            
    ints = [int(v) for v in song_time1]
    song_time1 = ints
    
    ints2 = [int(v) for v in song_time2]
    song_time2 = ints2
    
    outputData = pd.DataFrame({
       "Note": song_note2,
       "Time": song_time2
    })
    
    outputData.to_csv("SongChannel2.csv", index=False)
            
    outputData = pd.DataFrame({
        "Note": song_note1,
        "Time": song_time1
    })
    
    outputData.to_csv("SongChannel1.csv", index=False)
    

    return Required_frequencies

def animate(FrequencyLimit):
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    ax.set_xlim(1, FrequencyLimit)
    #ax.set_xscale('log')
    ax.set_ylim(0, np.max(AnimationMagnitude))

    ax.axhline(y = cutOff, color = 'red', linestyle='--')
    ax.axvline(x=440, linestyle='--', color='red')
    ax.axvline(x=220, linestyle='--', color='red')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Dynamic Frequency Spectrum")

    def init():
        line.set_data([], [])
        return line,

    def update(frame):

        line.set_data(AnimationFrequency[frame], AnimationMagnitude[frame])

        ax.set_title(f"Spectrum at t = {frame * SampleTime:.2f}s")

        return line,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(AnimationFrequency),
        init_func=init,
        interval=HopTime*1000,   # milliseconds
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
animate(25000)