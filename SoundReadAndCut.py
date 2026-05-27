import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# constants
SampleTime = 0.1  # seconds
link="sounds/song1.wav"

#Data storage
Amplitude = []
Time = []
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

def animate():
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    ax.set_xlim(0, 2000)
    ax.set_ylim(0, np.max(Magnitude))

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Dynamic Frequency Spectrum")

    def init():
        line.set_data([], [])
        return line,

    def update(frame):

        x = Frequency[frame]
        y = Magnitude[frame]

        # Optional: show only positive frequencies
        half = len(x) // 2

        line.set_data(x[:half], y[:half])

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
    # Plot waveform
    plt.plot(Time[0], Amplitude[n])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    plt.show()

    # Plot spectrum
    plt.plot(Frequency[0], Magnitude[n])
    plt.xlim(0, 10000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Spectrum")
    plt.show()

    plt.plot(Frequency[0], Magnitude[n])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Spectrum")
    plt.show()

PerformDFT(link)
#showFirst(100)
animate()