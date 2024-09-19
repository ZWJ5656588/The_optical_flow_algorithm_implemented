import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import resample, detrend

# Load acceleration data
file_path = r'D:\Project\振动实验数据\降采样数据\1706hz_1.xlsx'  # Example file path
sheet_name = 'Sheet1'  # Adjust if the data is in another worksheet

# Load data
data = pd.read_excel(file_path, sheet_name=sheet_name)
acceleration_data = data.iloc[:, 0].values  # Assuming acceleration data is in the first column

# Original sampling frequency and target sampling frequency
original_sampling_frequency = 1706  # Hz
target_sampling_frequency = 30  # Hz
time_step_target = 1 / target_sampling_frequency

# Resample to the target frequency
n_target = int(len(acceleration_data) * target_sampling_frequency / original_sampling_frequency)
acceleration_data_resampled = resample(acceleration_data, n_target)

# Generate time axis for resampled data
time_resampled = np.arange(n_target) * time_step_target

# Remove trend from acceleration data
acceleration_data_resampled = detrend(acceleration_data_resampled, type='constant')

# Time-domain integration (velocity and displacement)
velocity_time_resampled = np.cumsum(acceleration_data_resampled) * time_step_target

# Detrend velocity to remove any drift before calculating displacement
velocity_time_resampled = detrend(velocity_time_resampled, type='constant')

# Integrate velocity to get displacement
displacement_time_resampled = np.cumsum(velocity_time_resampled) * time_step_target

# Correct the initial displacement to zero
displacement_time_resampled -= displacement_time_resampled[0]

# Perform Fourier transform for frequency domain integration
acceleration_fft_resampled = fft(acceleration_data_resampled)
frequencies_resampled = fftfreq(n_target, time_step_target)

# Integrate in the frequency domain by dividing by (2 * pi * f)^2
epsilon = 1e-4  # Small value to prevent division by zero
displacement_fft_resampled = acceleration_fft_resampled / (-(2 * np.pi * frequencies_resampled)**2 + epsilon)

# Perform inverse Fourier transform to return to the time domain
displacement_freq_resampled = ifft(displacement_fft_resampled).real

# Use high-pass filter to remove low-frequency drift (adjusted cutoff frequency)
cutoff_frequency = .0  # Adjust this value to control drift removal
high_pass_filter_resampled = np.abs(frequencies_resampled) > cutoff_frequency
displacement_freq_filtered_resampled = ifft(displacement_fft_resampled * high_pass_filter_resampled).real

# Plotting
plt.figure(figsize=(12, 8))

# Plot resampled acceleration time history
plt.subplot(3, 1, 1)
plt.plot(time_resampled, acceleration_data_resampled, label="Resampled Acceleration (30Hz)", color='blue')
plt.title('Resampled Acceleration Time History (30Hz)')
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s²)')
plt.grid(True)

# Plot time-domain integrated displacement time history
plt.subplot(3, 1, 2)
plt.plot(time_resampled, displacement_time_resampled, label="Time Domain Displacement (Corrected)", color='green')
plt.title('Displacement Time History (Time Domain Integration, Corrected, 30Hz)')
plt.xlabel('Time (seconds)')
plt.ylabel('Displacement (m)')
plt.grid(True)

# Plot frequency-domain integrated displacement with drift removal
plt.subplot(3, 1, 3)
plt.plot(time_resampled, displacement_freq_filtered_resampled, label="Frequency Domain Displacement (Drift Removed, 30Hz)", color='red')
plt.title('Displacement Time History (Frequency Domain Integration and Drift Removal, 30Hz)')
plt.xlabel('Time (seconds)')
plt.ylabel('Displacement (m)')
plt.grid(True)

plt.tight_layout()
plt.show()
