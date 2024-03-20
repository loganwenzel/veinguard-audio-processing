import pyaudio
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
import wave
from scipy.signal import butter, lfilter, find_peaks

from handel_initial_inputs import (
    get_init_values,
)

from signal_analysis_helpers import (
    max_amp_over_period,
    average_delay_over_period,
    read_wav_chunk,
    read_calibration_sample,
    butter_lowpass_filter,
)

from plotting_helpers import (
    create_plots,
    perform_signal_analysis,
    find_audio_device,
    apply_selective_gain,
    calculate_delays,
    calculate_heart_rate,
)
from stream_recorder import record_stream

######## Initial Parameters ########
FORMAT = pyaudio.paInt16
DISTANCE = 3.5
CHANNELS = 2  # both the left and right channel
RATE = 44100  # sampling rate
REFRESH_PERIOD = 500  # number of milliseconds between plot updates
CHUNK = int(RATE * (REFRESH_PERIOD / 1000))  # chunk size for processing
WINDOW_SECONDS = 10  # Window length in seconds
CALIBRATION_DURATION = 10  # Calibration duration in seconds
live = False  # Set to True for live data, False for saved data
# desired_device_name = "Scarlett 2i2 USB"
desired_device_name = "Microphone (Scarlett 2i2 USB)"
low_pass_filter_cut_off = 10
stenosis_risk_levels = {"low": 25, "medium": 50, "high": 75}
saved_file = "/Users/ayden/Desktop/unfiltered_signal_2_from_cad.wav"
# saved_file = "C:/Users/wenze/source/repos/veinguard/veinguard-audio-processing/recordings/ayden/A1_NOCOMP_35_WITH_CALIBRATION.wav"
saved_file = "C:/Users/wenze/source/repos/veinguard/veinguard-audio-processing/recordings/ayden/unfiltered_march_20.wav"
# saved_file = "/Users/ayden/Desktop/unfiltered_signal_2_from_cad.wav"
# saved_file = "/Users/ayden/Desktop/rec/wav/ayden/A2_2.5COMP_3.5.wav"
#############################

# Call the function and store the returned distance
params = get_init_values(
    default_distance=DISTANCE,
    default_refresh_period=REFRESH_PERIOD,
    default_window_seconds=WINDOW_SECONDS,
    default_calibration_duration=CALIBRATION_DURATION,
    default_live=live,
    default_saved_file=saved_file,
)

DISTANCE = params["distance"] / 4
REFRESH_PERIOD = params["refresh_period"]
WINDOW_SECONDS = params["window_seconds"]
CALIBRATION_DURATION = params["calibration_duration"]
live = params["live"]

######## Global Values ########
global_data_buffer_1 = np.zeros(int(WINDOW_SECONDS * RATE), dtype=np.int16)
global_data_buffer_2 = np.zeros(int(WINDOW_SECONDS * RATE), dtype=np.int16)

global_peaks_c1 = np.array([])
global_peaks_c2 = np.array([])

global_troughs_c1 = np.array([])
global_troughs_c2 = np.array([])

max_amp_channel_1 = 0
max_amp_channel_2 = 0

calibration_peak_delay = 0
calibration_trough_delay = 0
#############################

if DISTANCE is not None:
    print(f"Selected distance: {DISTANCE}")

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    if live:
        # Desired device name
        input_device_index = find_audio_device(p, desired_device_name)
        # Open audio stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=CHUNK,
        )
    else:
        # Open the saved file
        wav_file = wave.open(saved_file, "rb")

    source = stream if live else wav_file

    def calibrate():
        global max_amp_channel_1, max_amp_channel_2, calibration_peak_delay, calibration_trough_delay

        ######## Calibration ########
        # Get chunk of calibration samples (not the issue)
        calibration_data_channel_1, calibration_data_channel_2 = (
            read_calibration_sample(source, RATE, CALIBRATION_DURATION, live)
        )

        # Apply low pass filter to calibration data (not the issue)
        calibration_data_channel_1 = butter_lowpass_filter(
            calibration_data_channel_1, low_pass_filter_cut_off, RATE
        )
        calibration_data_channel_2 = butter_lowpass_filter(
            calibration_data_channel_2, low_pass_filter_cut_off, RATE
        )

        # Get max amp for each channel from calibration data
        max_amp_channel_1 = max_amp_over_period(calibration_data_channel_1)
        max_amp_channel_2 = max_amp_over_period(calibration_data_channel_2)

        # Get average peak and trough delays during calibration for percent difference
        calibration_peak_delay, calibration_trough_delay = average_delay_over_period(
            calibration_data_channel_1, calibration_data_channel_2, RATE
        )
        #############################

    calibrate()

    ######## Plotting ########
    total_frames_processed = 0
    # Create a PyQtGraph window
    (
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
        stenosis_risk_widget,
        update_x_axis,
        recalibrate_button,
    ) = create_plots(WINDOW_SECONDS, RATE)

    recalibrate_button.clicked.connect(calibrate)

    def update_data():
        global total_frames_processed
        global global_data_buffer_1, global_data_buffer_2
        global global_peaks_c1, global_peaks_c2
        global global_troughs_c1, global_troughs_c2
        try:
            # Perform signal analysis for each new chunk
            if live:
                new_data = stream.read(CHUNK, exception_on_overflow=False)
                new_data = np.frombuffer(new_data, dtype=np.int16)
            else:
                new_data = read_wav_chunk(wav_file, CHUNK)
                if new_data is None:  # End of WAV file
                    print("End of WAV file")
                    return

            data_buffer_1, data_buffer_2 = perform_signal_analysis(
                WINDOW_SECONDS,
                RATE,
                CHUNK,
                global_data_buffer_1,
                global_data_buffer_2,
                new_data,
                max_amp_channel_1,
                max_amp_channel_2,
                low_pass_filter_cut_off,
                live,
            )

            prev_time = total_frames_processed // RATE
            total_frames_processed += CHUNK
            cur_time = total_frames_processed // RATE

            if cur_time > prev_time and cur_time > WINDOW_SECONDS:
                update_x_axis(cur_time)  # Update the x-axis ticks

            global_data_buffer_1 = data_buffer_1
            global_data_buffer_2 = data_buffer_2

            # Update the plot curves
            curve1.setData(global_data_buffer_1)
            curve2.setData(global_data_buffer_2)

            # Find peaks
            peaks_c1, _ = find_peaks(
                global_data_buffer_1,
                height=0.5,
                distance=RATE / 2,
            )
            peaks_c2, _ = find_peaks(
                global_data_buffer_2,
                height=0.5,
                distance=RATE / 2,
            )

            # Don't use peaks with amplitude above max amplitude. Uncomment once ready to test on Jes
            # peaks_c1 = [
            #     peak
            #     for peak in peaks_c1
            #     if global_data_buffer_1[peak] < max_amp_channel_1
            # ]
            # peaks_c2 = [
            #     peak
            #     for peak in peaks_c2
            #     if global_data_buffer_2[peak] < max_amp_channel_2
            # ]

            # Find troughs (negative peaks of the inverted signal)
            troughs_c1, _ = find_peaks(
                -global_data_buffer_1,
                height=-np.min(global_data_buffer_1) / 4,
                distance=RATE / 2,
            )
            troughs_c2, _ = find_peaks(
                -global_data_buffer_2,
                height=-np.min(global_data_buffer_2) / 4,
                distance=RATE / 2,
            )

            global_peaks_c1 = np.array([[i, global_data_buffer_1[i]] for i in peaks_c1])
            global_peaks_c2 = np.array([[i, global_data_buffer_2[i]] for i in peaks_c2])
            global_troughs_c1 = np.array(
                [[i, global_data_buffer_1[i]] for i in troughs_c1]
            )
            global_troughs_c2 = np.array(
                [[i, global_data_buffer_2[i]] for i in troughs_c2]
            )

            # Update the scatter plots with peak and trough coordinates
            peaks_scatter1.setData(pos=global_peaks_c1)
            troughs_scatter1.setData(pos=global_troughs_c1)
            peaks_scatter2.setData(pos=global_peaks_c2)
            troughs_scatter2.setData(pos=global_troughs_c2)

        except Exception as e:
            print(f"Error: {e}")
            return

    def update_calculations():
        try:
            if (
                len(global_peaks_c1) < 3
                or len(global_troughs_c1) < 3
                or len(global_peaks_c2) < 3
                or len(global_troughs_c2) < 3
            ):
                # print("One or more arrays do not have enough peaks/troughs. Skipping delay calculation.")
                return

            # Calculate delays
            delays = calculate_delays(
                global_peaks_c1[1:-1, 0],
                global_troughs_c1[1:-1, 0],
                global_peaks_c2[1:-1, 0],
                global_troughs_c2[1:-1, 0],
                RATE,
            )

            def filter_delays_std(delays, m=1):
                """Filter delays using mean and standard deviation to exclude outliers."""
                mean_delay = np.mean(delays)
                std_delay = np.std(delays)
                return [
                    delay
                    for delay in delays
                    if mean_delay - m * std_delay <= delay <= mean_delay + m * std_delay
                ]

            # Filter peak and trough delays using standard deviation
            filtered_peak_delays = filter_delays_std(delays["peak_delays"], m=2)
            filtered_trough_delays = filter_delays_std(delays["trough_delays"], m=2)

            # Calculate averages with filtered delays
            average_peak_delay_ms = (
                round(np.mean(filtered_peak_delays) * 1000, 2)
                if filtered_peak_delays
                else 0
            )
            average_trough_delay_ms = (
                round(np.mean(filtered_trough_delays) * 1000, 2)
                if filtered_trough_delays
                else 0
            )

            # calibration_average_time_delay_ms = (
            #     calibration_peak_delay + calibration_trough_delay / 2
            # )
            # current_average_time_delay_ms = (
            #     average_peak_delay_ms + average_trough_delay_ms / 2
            # )

            # Trough delay can be more unpredictable. Just use peak for blood velocity calculation for now
            calibration_average_time_delay_ms = calibration_peak_delay
            current_average_time_delay_ms = average_peak_delay_ms

            calibration_blood_velocity = round(
                (DISTANCE * 1000) / (calibration_average_time_delay_ms)
            )
            current_blood_velocity = round(
                (DISTANCE * 1000) / (current_average_time_delay_ms)
            )

            percent_difference_from_calibration = round(
                abs((1 - (calibration_blood_velocity / current_blood_velocity)) * 100)
            )
            if len(filtered_peak_delays) > 3:
                percent_difference_from_calibration_label.setText(
                    f'<p style="font-size: 16px;"> Difference from No Stenosis blood velocity ({calibration_blood_velocity} cm/s) </p> <h2> {percent_difference_from_calibration}% </h2>'
                )

            # Set stenosis risk label based on stenosis_risk_levels dictionary
            if len(filtered_peak_delays) > 3:
                if percent_difference_from_calibration < stenosis_risk_levels["low"]:
                    # Green
                    stenosis_risk_widget.setText(
                        '<span> Stenosis Risk: None </span> <div> <img src="assets/NoRisk.svg" width="80" height="80"> </div>'
                    )
                    stenosis_risk_widget.setStyleSheet(
                        """
                    QLabel {
                        color: #000;
                        border: 2px solid green;
                        background-color: #CCFFCC;
                    }
                    """
                    )
                elif (
                    percent_difference_from_calibration < stenosis_risk_levels["medium"]
                ):
                    # Yellow
                    stenosis_risk_widget.setText(
                        '<span> Stenosis Risk: Low </span> <div> <img src="assets/LowRisk.svg" width="80" height="80"> </div>'
                    )
                    stenosis_risk_widget.setStyleSheet(
                        """
                    QLabel {
                        color: #000;
                        border: 2px solid yellow;
                        background-color: #FFFFCC;
                    }
                    """
                    )
                elif percent_difference_from_calibration < stenosis_risk_levels["high"]:
                    # Orange
                    stenosis_risk_widget.setText(
                        '<span> Stenosis Risk: Medium </span> <div> <img src="assets/MediumRisk.svg" width="80" height="80"> </div>'
                    )
                    stenosis_risk_widget.setStyleSheet(
                        """
                    QLabel {
                        color: #000;
                        border: 2px solid orange;
                        background-color: #FFD699;
                    }
                    """
                    )
                else:
                    # Red
                    stenosis_risk_widget.setText(
                        '<span> Stenosis Risk: High </span> <div> <img src="assets/HighRisk.svg" width="80" height="80"> </div>'
                    )
                    stenosis_risk_widget.setStyleSheet(
                        """
                    QLabel {
                        color: #000;
                        border: 2px solid red;
                        background-color: #FFCCCC;
                        }
                    """
                    )

            # Calculate heart rate
            heart_rate_c1 = calculate_heart_rate(global_peaks_c1[1:-1], RATE)
            heart_rate_c2 = calculate_heart_rate(global_peaks_c2[1:-1], RATE)
            heart_rate = (heart_rate_c1 + heart_rate_c2) / 2

            # Update text items with delay values and heart rate
            avg_peak_delay_label.setText(
                f'<p style="font-size: 16px;"> Average Peak Delay </p> <h2> {average_peak_delay_ms} ms </h2>'
            )
            avg_trough_delay_label.setText(
                f'<p style="font-size: 16px;"> Average Trough Delay </p> <h2> {average_trough_delay_ms} ms </h2>'
            )
            if len(filtered_peak_delays) > 3:
                blood_velocity_label.setText(
                    f'<p style="font-size: 16px;"> Blood Velocity </p> <h2> {current_blood_velocity} cm/s </h2>'
                )
            heart_rate_label.setText(
                f'<p style="font-size: 16px;"> Heart Rate </p> <h2> {heart_rate} BPM </h2>'
            )

        except Exception as e:
            print(f"Error: {e}")
            return

    # Set up a timer for updating the plot
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update_data)
    timer.start(REFRESH_PERIOD)  # Update the plot

    # Set up a second timer for performing all calculations
    calculations_timer = pg.QtCore.QTimer()
    calculations_timer.timeout.connect(update_calculations)
    calculations_timer.start(REFRESH_PERIOD * 5)  # Update peaks

    # Start the PyQtGraph application
    win.showMaximized()  # Use showFullScreen() for symposium
    QtWidgets.QApplication.instance().exec_()

    # Close the stream and PyAudio
    if live:
        stream.stop_stream()
        stream.close()
    p.terminate()
