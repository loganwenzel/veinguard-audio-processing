from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton

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
    # Generate x_ticks with even spacing
    tick_interval = max(
        1, window_seconds // 10
    )  # Ensure at least 1 second between ticks
    x_ticks = [
        [(i * rate, str(i)) for i in range(0, window_seconds + 1, tick_interval)]
    ]

    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Veinguard")
    win.setBackground("w")
    win.resize(1000, 600)

    # Define plot pen styles
    plot_pen = pg.mkPen(color=(0, 0, 0), width=2)
    peak_pen = pg.mkPen(color=(0, 255, 0), width=2)
    trough_pen = pg.mkPen(color=(255, 0, 0), width=2)

    # First plot
    plot1 = win.addPlot(title="Channel 1 - Upper Arm")
    curve1 = plot1.plot(pen=plot_pen)
    plot1.setYRange(-1, 1, padding=0.1)
    plot1.setLabel("left", "Amplitude")
    plot1.setLabel("bottom", "Time (seconds)")
    plot1.getAxis("bottom").setTicks(x_ticks)

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
    plot2.getAxis("bottom").setTicks(x_ticks)

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

    # Define the CSS styling
    label_style = """
        QLabel {
            color: #000;
            border-radius: 10px;
            padding: 5px;
            text-align: center;
            height: 50px;
            background-color: #eeeeee;
        }
    """

    # Create and style QLabel widgets for each label
    def create_styled_label(text):
        label_widget = QtWidgets.QLabel(text)
        label_widget.setStyleSheet(label_style)
        label_widget.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Bold))
        label_widget.setAlignment(QtCore.Qt.AlignCenter)
        label_widget.setMinimumHeight(150)
        label_widget.setMaximumHeight(150)
        label_widget.setMinimumWidth(400)
        proxy_widget = pg.QtWidgets.QGraphicsProxyWidget()
        proxy_widget.setWidget(label_widget)
        return label_widget, proxy_widget

    # Column 1
    blood_velocity_label, blood_velocity_proxy = create_styled_label("Calibrating")
    blood_velocity_label.setAlignment(QtCore.Qt.AlignCenter)
    heart_rate_label, heart_rate_proxy = create_styled_label("Calibrating")
    heart_rate_label.setAlignment(QtCore.Qt.AlignCenter)
    layout.addItem(blood_velocity_proxy, row=0, col=0)
    layout.addItem(heart_rate_proxy, row=1, col=0)

    # Column 2
    avg_peak_delay_label, avg_peak_delay_proxy = create_styled_label("Calibrating")
    avg_peak_delay_label.setAlignment(QtCore.Qt.AlignCenter)
    avg_trough_delay_label, avg_trough_delay_proxy = create_styled_label("Calibrating")
    avg_trough_delay_label.setAlignment(QtCore.Qt.AlignCenter)
    layout.addItem(avg_peak_delay_proxy, row=0, col=1)
    layout.addItem(avg_trough_delay_proxy, row=1, col=1)

    # Column 3
    percent_difference_from_calibration_label, percent_difference_proxy = (
        create_styled_label("Calibrating")
    )
    percent_difference_from_calibration_label.setAlignment(QtCore.Qt.AlignCenter)
    layout.addItem(percent_difference_proxy, row=0, col=2)

    stenosis_risk_label, stenosis_risk_proxy = create_styled_label("Calibrating")
    stenosis_risk_label.setAlignment(QtCore.Qt.AlignCenter)
    layout.addItem(stenosis_risk_proxy, row=1, col=2)

    # positions = [0, 44100, 88200, 132300, 176400, 220500, 264600, 308700, 352800, 396900, 441000]

    def update_x_axis(current_time):
        tick_start = max(0, current_time - window_seconds + 1)
        tick_labels = [str(i) for i in range(tick_start, current_time + 1)]

        # Generate positions for the ticks based on the window size
        positions = [i * rate for i in range(window_seconds + 1)]

        # Associate each position with the corresponding label, ensuring labels match their positions
        x_ticks = [
            (positions[i - tick_start], label)
            for i, label in enumerate(tick_labels, start=tick_start)
        ]

        plot1.getAxis("bottom").setTicks([x_ticks])
        plot2.getAxis("bottom").setTicks([x_ticks])

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
        update_x_axis,
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
    new_data = source

    channel1 = new_data[::2]
    channel2 = new_data[1::2]
    
    # Calculate alpha for the low-pass filter
    tau = 1.0 / (2 * np.pi * lpf_cut_off)
    alpha = 1.0 / RATE / tau

    ###### Operations #####
    # # Apply low-pass filter
    # channel1 = butter_lowpass_filter(channel1, cutoff=lpf_cut_off, fs=RATE)
    # channel2 = butter_lowpass_filter(channel2, cutoff=lpf_cut_off, fs=RATE)

    # Apply low-pass filter
    channel1 = simple_lowpass_filter(channel1, alpha)
    channel2 = simple_lowpass_filter(channel2, alpha)

    # Amplify
    filtered_channel1 = simple_lowpass_filter(channel1, alpha, data_buffer_1[-1])
    filtered_channel2 = simple_lowpass_filter(channel2, alpha, data_buffer_2[-1])

    ### Normalize
    channel1 = normalize(channel1, max_amp_channel_1)
    channel2 = normalize(channel2, max_amp_channel_2)

    # channel1 = apply_selective_gain(channel1, max_amp_channel_1*0.6, 0.3)
    # channel2 = apply_selective_gain(channel2, max_amp_channel_2*0.6, 0.3)
    #######################

    # Remove first elements
    data_buffer_1 = data_buffer_1[CHUNK:]
    data_buffer_2 = data_buffer_2[CHUNK:]

    # append the new ones
    data_buffer_1 = np.append(data_buffer_1, channel1)
    data_buffer_2 = np.append(data_buffer_2, channel2)

    return data_buffer_1, data_buffer_2

def simple_lowpass_filter(data, alpha, last_value=0):
    # data is your input signal array.
    # alpha is the smoothing factor, which you can calculate using alpha = dt / tau. dt is the inverse of the sampling rate (1 / RATE), and tau is related to the cutoff frequency of the filter (tau = 1 / (2 * np.pi * cutoff_frequency)).
    # filtered_data is the output of the filtered signal.

    # Initialize the filtered data array
    filtered_data = np.zeros_like(data)
    filtered_data[0] = last_value

    # Apply the filter to the data
    for i in range(1, len(data)):
        filtered_data[i] = filtered_data[i-1] + alpha * (data[i] - filtered_data[i-1])
    
    return filtered_data

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


def select_distance():
    app = QApplication([])

    # Create a window
    window = QWidget()
    window.setWindowTitle("Select Distance")
    layout = QVBoxLayout()

    # Create a dropdown for distance selection
    distance_dropdown = QComboBox()
    distance_dropdown.addItems(["3.0", "3.5", "4.0", "4.5", "5.0"])
    layout.addWidget(distance_dropdown)

    # Create a button to start recording
    start_button = QPushButton("Begin Recording")
    layout.addWidget(start_button)

    # Function to handle button click
    def on_start_button_clicked():
        distance = float(distance_dropdown.currentText())
        window.close()
        app.exit()  # Close the application
        return distance

    # Connect the button click event to the function
    start_button.clicked.connect(lambda: on_start_button_clicked())

    # Set the layout and show the window
    window.setLayout(layout)
    window.show()

    # Start the application event loop
    app.exec_()

    # Return the selected distance after the window is closed
    return float(distance_dropdown.currentText())
