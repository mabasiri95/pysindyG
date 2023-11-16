import numpy as np
import matplotlib.pyplot as plt

# Generate a sample signal
fs = 256  # Sampling frequency
t = np.arange(0, 5, 1/fs)  # Time vector
f = 3  # Frequency of the signal
signal = np.sin(2 * np.pi * f * t) + 0.5 * np.random.randn(len(t))  # Sinusoidal signal with noise
plt.plot(t, signal)
# Calculate PSD using matplotlib.mlab.psd
frequencies, psd = plt.psd(signal, NFFT=1024, Fs=fs)

# Plot the PSD
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.show()




N = 1280
x = signal
X = np.fft.fft(x)

PSD = (1.0 / N) * np.abs(X[:N//2])**2

sample_rate = 256  # Replace with your actual sample rate
freq_axis = np.fft.fftfreq(N, 1/sample_rate)[:N//2]


import matplotlib.pyplot as plt

plt.plot(freq_axis, PSD)  # Convert to dB for better visualization
plt.plot(freq_axis, 10 * np.log10(PSD))  # Convert to dB for better visualization
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.title('Power Spectral Density')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate a sample signal
fs = 256  # Sampling frequency
t = np.arange(0, 5, 1/fs)  # Time vector

# Example signal
signal = np.sin(2 * np.pi * 4 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
plt.plot(t, signal)

scales = np.arange(1, 128)  # Adjust the range of scales as needed
coefficients, frequencies = pywt.cwt(signal, scales, 'cmor')

# Plot the wavelet transform at scale 90
scale_to_plot = 90
scale_index = np.argmin(np.abs(scales - scale_to_plot))
wavelet_at_scale_90 = coefficients[scale_index, :]

plt.plot(t, wavelet_at_scale_90, label=f'Scale {scale_to_plot}')



plt.imshow(np.abs(coefficients), aspect='auto', extent=[0, 1, 1, 128], cmap='jet')
plt.colorbar(label='Magnitude')
plt.ylabel('Scale')
plt.xlabel('Time')
plt.title('Continuous Wavelet Transform')
plt.show()




import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate a sample signal (brain signal)
fs = 256  # Sampling frequency
t = np.arange(0, 5, 1/fs)  # Time vector
signal = np.random.randn(len(t))  # Replace with your actual brain signal

# Choose wavelet and scales
wavelet = 'cmor'
nyquist = 0.5 * fs  # Nyquist frequency
max_frequency = 49  # Maximum frequency of interest

# Calculate the maximum meaningful scale
max_scale = int(round(nyquist / max_frequency))
scales = np.arange(2, max_scale + 1)

# Perform Continuous Wavelet Transform
coefficients, frequencies = pywt.cwt(signal, scales, wavelet)

# Print the dimensions of the coefficients array
print("Dimensions of coefficients array:", coefficients.shape)

# Plot the wavelet transform
plt.imshow(np.abs(coefficients), aspect='auto', extent=[0, 5, 2, max_scale], cmap='jet')
plt.colorbar(label='Magnitude')
plt.ylabel('Scale')
plt.xlabel('Time (seconds)')
plt.title('Continuous Wavelet Transform')
plt.show()

