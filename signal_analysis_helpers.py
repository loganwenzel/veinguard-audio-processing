import numpy as np
from scipy.signal import butter, lfilter, find_peaks


# Butterworth Low Pass Filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Normalize and amplify signal
def normalize(data, max_amp):
    normalized_data = data / max_amp
    return normalized_data


def apply_selective_gain(signal, threshold, gain):
    """
    Applies gain to parts of the signal that are within the specified threshold.

    Parameters:
    - signal: An array containing the audio signal data.
    - threshold: The threshold for applying the gain.
    - gain: The gain multiplier applied to the signal within the threshold.

    Returns:
    - A new array with the selectively adjusted audio signal.
    """
    # Initialize the output signal with the original signal
    adjusted_signal = np.copy(signal)

    # Create a mask for signal components that are within -threshold and +threshold
    within_threshold_mask = (adjusted_signal > -threshold) & (
        adjusted_signal < threshold
    )

    # Apply gain to these components
    adjusted_signal[within_threshold_mask] *= gain

    return adjusted_signal


def max_amp_over_period(calibration_data):
    print("Starting Amplitude Calibration...")
    max_amplitude_channel1 = 0
    max_amplitude_channel2 = 0

    # Split the stereo audio data into two channels
    channel1 = calibration_data[::2]
    channel2 = calibration_data[1::2]

    # Update the maximum amplitude for each channel
    max_amplitude_channel1 = np.max(np.abs(channel1))
    max_amplitude_channel2 = np.max(np.abs(channel2))

    print(
        f"Calibration complete. Max amplitude channel 1: {max_amplitude_channel1}, channel 2: {max_amplitude_channel2}"
    )
    return max_amplitude_channel1, max_amplitude_channel2


def average_delay_over_period(calibration_data, rate):
    print("Starting Standard Peak Delay Calibration...")

    if len(calibration_data) < 2:
        print("Error: Data is too short.")
        return 0, 0

    # Split the stereo data
    data_ch1 = calibration_data[0::2]
    data_ch2 = calibration_data[1::2]

    # Find peaks for both channels
    peaks_ch1, _ = find_peaks(data_ch1, height=np.max(data_ch1) / 4, distance=rate / 2)
    peaks_ch2, _ = find_peaks(data_ch2, height=np.max(data_ch2) / 4, distance=rate / 2)

    print(len(peaks_ch1))
    print(len(peaks_ch2))

    # Find troughs for both channels (by finding peaks in the inverted signals)
    troughs_ch1, _ = find_peaks(
        -data_ch1, height=-np.min(data_ch1) / 4, distance=rate / 2
    )
    troughs_ch2, _ = find_peaks(
        -data_ch2, height=-np.min(data_ch2) / 4, distance=rate / 2
    )

    # Calculate peak delays
    peak_delays = []
    for peak1 in peaks_ch1:
        closest_peak2 = min(peaks_ch2, key=lambda peak2: abs(peak2 - peak1))
        peak_delays.append((closest_peak2 - peak1) / rate)

    # Calculate trough delays
    trough_delays = []
    for trough1 in troughs_ch1:
        closest_trough2 = min(troughs_ch2, key=lambda trough2: abs(trough2 - trough1))
        trough_delays.append((closest_trough2 - trough1) / rate)

    # Calculate average delays in milliseconds
    avg_peak_delay_ms = round(np.mean(peak_delays) * 1000, 2) if peak_delays else 0
    avg_trough_delay_ms = (
        round(np.mean(trough_delays) * 1000, 2) if trough_delays else 0
    )

    print(
        f"Calibration complete. Average peak delay: {avg_peak_delay_ms} ms, Average trough delay: {avg_trough_delay_ms} ms"
    )

    return avg_peak_delay_ms, avg_trough_delay_ms


def read_calibration_sample(source, rate, duration, is_live):
    total_samples = int(duration * rate) * 2  # Multiply by 2 for stereo data
    samples_read = 0
    data_buffer = []

    while samples_read < total_samples:
        if is_live:
            data = source.read(
                total_samples - samples_read, exception_on_overflow=False
            )
        else:
            raw_data = source.readframes(total_samples - samples_read)
            if len(raw_data) == 0:  # End of file
                break
            data = np.frombuffer(raw_data, dtype=np.int16)

        data_buffer.append(data)
        samples_read += len(data)

    return np.concatenate(data_buffer)


# Function to read chunks from WAV file
def read_wav_chunk(wav_file, chunk_size):
    raw_data = wav_file.readframes(chunk_size)
    if len(raw_data) == 0:  # End of file
        return None
    return np.frombuffer(raw_data, dtype=np.int16)
