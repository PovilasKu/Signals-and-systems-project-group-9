import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
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
showFirst(2)
animate(3000)