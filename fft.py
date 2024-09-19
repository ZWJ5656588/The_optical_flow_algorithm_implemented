import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data from the Excel file (first column)
data = pd.read_excel(r'D:\Project\振动实验数据\降采样数据\1706hz_1.xlsx', header=None)

# Extract the time series data (first column)
timeseries = data.iloc[:, 0].values

# Sampling frequency
sampling_frequency = 1706  # Hz
N = len(timeseries)  # Number of samples

# Calculate FFT
fft_result = np.fft.fft(timeseries)

# Generate frequency axis
freqs = np.fft.fftfreq(N, 1 / sampling_frequency)

# Compute amplitude spectrum
amplitude_spectrum = np.abs(fft_result)

# Find the peak in the amplitude spectrum
peak_amplitude_idx = np.argmax(amplitude_spectrum[:N // 2])
peak_amplitude_freq = freqs[peak_amplitude_idx]
peak_amplitude_value = amplitude_spectrum[peak_amplitude_idx]

# Plot the amplitude spectrum
plt.figure(1)
plt.plot(freqs[:N // 2], amplitude_spectrum[:N // 2], label='Amplitude Spectrum')
plt.scatter(peak_amplitude_freq, peak_amplitude_value, color='red', label=f'Peak: {peak_amplitude_freq:.2f} Hz')
plt.title('Amplitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

# Compute power spectral density (PSD)
psd = (np.abs(fft_result) ** 2) / (sampling_frequency * N)
psd[1:N // 2] = 2 * psd[1:N // 2]

# Find the peak in the PSD
peak_psd_idx = np.argmax(psd[:N // 2])
peak_psd_freq = freqs[peak_psd_idx]
peak_psd_value = psd[peak_psd_idx]

# Plot the PSD
plt.figure(2)
plt.plot(freqs[:N // 2], psd[:N // 2], label='PSD')
plt.scatter(peak_psd_freq, peak_psd_value, color='red', label=f'Peak: {peak_psd_freq:.2f} Hz')
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (PSD)')
plt.legend()

# Show the plots
plt.show()
