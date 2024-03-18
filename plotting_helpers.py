from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np
from scipy.signal import butter, lfilter

from signal_analysis_helpers import (
    butter_lowpass,
    butter_lowpass_filter,
    normalize,
    max_amp_over_period,
    apply_selective_gain,
)

# from stream_recorder import perform_signal_analysis_from_wav


def find_audio_device(p, desired_device_name):

    # Print available audio devices
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(
            f"{i}: {device_info['name']} (Input channels: {device_info['maxInputChannels']})\n"
        )

    # Find the index of the desired device
    input_device_index = None
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info["name"] == desired_device_name:
            input_device_index = i
            break

    if input_device_index is None:
        raise Exception(f"Device '{desired_device_name}' not found")

    return input_device_index


def create_plots(window_seconds, rate):
    x_ticks = [[(rate * i, str(i)) for i in range(window_seconds + 1)]]

    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Veinguard")
    win.setBackground("w")
    win.resize(1000, 600)

    # Define plot pen styles
    plot_pen = pg.mkPen(color=(0, 0, 0), width=2)  # Black color, thicker line
    peak_pen = pg.mkPen(color=(0, 255, 0), width=2)  # Green color, thicker line
    trough_pen = pg.mkPen(color=(255, 0, 0), width=2)  # Red color, thicker line

    # First plot
    plot1 = win.addPlot(title="Channel 1 - Upper Arm")
    curve1 = plot1.plot(pen=plot_pen)
    plot1.setYRange(-1, 1, padding=0.1)
    plot1.setLabel("left", "Amplitude")
    plot1.setLabel("bottom", "Time (seconds)")
    plot1.getAxis("bottom").setTickSpacing(
        rate, rate
    )  # Dynamic tick spacing based on rate

    # Scatter plots for peaks and troughs in the first plot
    peaks_scatter1 = pg.ScatterPlotItem(
        pen=peak_pen, symbol="o", size=5, brush=pg.mkBrush(0, 255, 0)
    )
    troughs_scatter1 = pg.ScatterPlotItem(
        pen=trough_pen, symbol="o", size=5, brush=pg.mkBrush(255, 0, 0)
    )
    plot1.addItem(peaks_scatter1)
    plot1.addItem(troughs_scatter1)

    win.nextRow()

    # Second plot
    plot2 = win.addPlot(title="Channel 2 - Wrist")
    curve2 = plot2.plot(pen=plot_pen)
    plot2.setYRange(-1, 1, padding=0.1)
    plot2.setLabel("left", "Amplitude")
    plot2.setLabel("bottom", "Time (seconds)")
    plot2.getAxis("bottom").setTickSpacing(rate, rate)

    # Scatter plots for peaks and troughs in the second plot
    peaks_scatter2 = pg.ScatterPlotItem(
        pen=peak_pen, symbol="o", size=5, brush=pg.mkBrush(0, 255, 0)
    )
    troughs_scatter2 = pg.ScatterPlotItem(
        pen=trough_pen, symbol="o", size=5, brush=pg.mkBrush(255, 0, 0)
    )
    plot2.addItem(peaks_scatter2)
    plot2.addItem(troughs_scatter2)

    win.nextRow()

    # Third row layout
    layout = win.addLayout()

    # Define the font properties
    font = QtGui.QFont()
    font.setPixelSize(30)
    font.setBold(True)
    label_color = "black"

    ### SECTION 1
    blood_velocity_label = pg.LabelItem(
        text="Blood Velocity: ", color=label_color, font=font
    )
    blood_velocity_label.setFont(font)
    heart_rate_label = pg.LabelItem(text="Heart Rate: ", color=label_color)
    heart_rate_label.setFont(font)
    layout.addItem(blood_velocity_label, row=0, col=0)
    layout.addItem(heart_rate_label, row=1, col=0)

    ### SECTION 2
    avg_peak_delay_label = pg.LabelItem(text="Average Peak Delay: ", color=label_color)
    avg_peak_delay_label.setFont(font)
    avg_trough_delay_label = pg.LabelItem(
        text="Average Trough Delay: ", color=label_color
    )
    avg_trough_delay_label.setFont(font)
    layout.addItem(avg_peak_delay_label, row=0, col=1)
    layout.addItem(avg_trough_delay_label, row=1, col=1)  # Placed in a different row

    ### SECTION 3
    percent_difference_from_calibration_label = pg.LabelItem(
        text="Percent difference from calibration: ", color=label_color
    )
    percent_difference_from_calibration_label.setFont(font)
    layout.addItem(percent_difference_from_calibration_label, row=0, col=2)

    stenosis_risk_label = pg.LabelItem(text="Stenosis Risk", color=label_color)

    layout.addItem(stenosis_risk_label, row=1, col=2)

    return (
        app,
        win,
        plot1,
        curve1,
        peaks_scatter1,
        troughs_scatter1,
        plot2,
        curve2,
        peaks_scatter2,
        troughs_scatter2,
        blood_velocity_label,
        heart_rate_label,
        avg_peak_delay_label,
        avg_trough_delay_label,
        percent_difference_from_calibration_label,
        stenosis_risk_label,
    )


def perform_signal_analysis(
    WINDOW_SECONDS,
    RATE,
    CHUNK,
    data_buffer_1,
    data_buffer_2,
    source,
    max_amp_channel_1,
    max_amp_channel_2,
    lpf_cut_off,
    is_live,
):
    if is_live:
        # Read a chunk of data from the live stream
        new_data = np.frombuffer(
            source.read(CHUNK, exception_on_overflow=False), dtype=np.int16
        )
    else:
        # Use the provided NumPy array (source is already a NumPy array)
        new_data = source

    channel1 = new_data[::2]
    channel2 = new_data[1::2]
    # print(f"C1: {(channel1)}, C2: {(channel2)}\n")

    ###### Operations #####
    ### Normalize
    channel1 = normalize(channel1, max_amp_channel_1)
    channel2 = normalize(channel2, max_amp_channel_2)

    # channel1 = apply_selective_gain(channel1, max_amp_channel_1*0.6, 0.3)
    # channel2 = apply_selective_gain(channel2, max_amp_channel_2*0.6, 0.3)

    # Apply low-pass filter
    # channel1 = butter_lowpass_filter(channel1, cutoff=lpf_cut_off, fs=RATE)
    # channel2 = butter_lowpass_filter(channel2, cutoff=lpf_cut_off, fs=RATE)
    #######################

    # Remove first elements
    data_buffer_1 = data_buffer_1[CHUNK:]
    data_buffer_2 = data_buffer_2[CHUNK:]

    # append the new ones
    data_buffer_1 = np.append(data_buffer_1, channel1)
    data_buffer_2 = np.append(data_buffer_2, channel2)

    return data_buffer_1, data_buffer_2


def calculate_delays(
    peaks_channel1, troughs_channel1, peaks_channel2, troughs_channel2, sample_rate
):
    """
    Calculates the delays between corresponding peaks and troughs in two channels.

    Parameters:
    - peaks_channel1: Indices of peaks in the first channel.
    - troughs_channel1: Indices of troughs in the first channel.
    - peaks_channel2: Indices of peaks in the second channel.
    - troughs_channel2: Indices of troughs in the second channel.
    - sample_rate: The sample rate of the signal.

    Returns:
    - A dictionary with lists of peak and trough delays in seconds.
    """
    peak_delays = []
    trough_delays = []

    # Calculate peak delays
    for i in range(min(len(peaks_channel1), len(peaks_channel2))):
        delay = (peaks_channel2[i] - peaks_channel1[i]) / sample_rate
        peak_delays.append(delay)

    # Calculate trough delays
    for i in range(min(len(troughs_channel1), len(troughs_channel2))):
        delay = (troughs_channel2[i] - troughs_channel1[i]) / sample_rate
        trough_delays.append(delay)

    return {"peak_delays": peak_delays, "trough_delays": trough_delays}


def calculate_heart_rate(global_peaks, rate):
    if len(global_peaks) > 1:
        time_intervals = np.diff(global_peaks[:, 0]) / rate
        average_time_interval = np.mean(time_intervals)
        heart_rate = round(60 / average_time_interval)
    else:
        heart_rate = 0
    return heart_rate
