from PyQt5 import QtWidgets, QtCore
import sys

def get_init_values(
    default_distance,
    default_refresh_period,
    default_window_seconds,
    default_calibration_duration,
    default_live,
    default_saved_file
):
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout()
    distance_label = QtWidgets.QLabel("Enter distance between microphones (cm):")
    distance_input = QtWidgets.QLineEdit(str(default_distance))
    refresh_period_label = QtWidgets.QLabel("Enter refresh period (ms):")
    refresh_period_input = QtWidgets.QLineEdit(str(default_refresh_period))
    window_seconds_label = QtWidgets.QLabel("Enter window length (seconds):")
    window_seconds_input = QtWidgets.QLineEdit(str(default_window_seconds))
    calibration_duration_label = QtWidgets.QLabel("Enter calibration duration (seconds):")
    calibration_duration_input = QtWidgets.QLineEdit(str(default_calibration_duration))
    live_label = QtWidgets.QLabel("Live mode (True/False):")
    live_input = QtWidgets.QLineEdit(str(default_live))
    saved_file_label = QtWidgets.QLabel("Enter path to saved file:")
    saved_file_input = QtWidgets.QLineEdit(str(default_saved_file))

    button = QtWidgets.QPushButton("Start Recording")
    button.clicked.connect(lambda: window.close())

    layout.addWidget(distance_label)
    layout.addWidget(distance_input)
    layout.addWidget(refresh_period_label)
    layout.addWidget(refresh_period_input)
    layout.addWidget(window_seconds_label)
    layout.addWidget(window_seconds_input)
    layout.addWidget(calibration_duration_label)
    layout.addWidget(calibration_duration_input)
    layout.addWidget(live_label)
    layout.addWidget(live_input)
    layout.addWidget(saved_file_label)
    layout.addWidget(saved_file_input)
    layout.addWidget(button)

    window.setLayout(layout)
    window.show()
    app.exec_()

    distance = float(distance_input.text())
    refresh_period = int(refresh_period_input.text())
    window_seconds = int(window_seconds_input.text())
    calibration_duration = int(calibration_duration_input.text())
    live = True if live_input.text().lower() == "true" else False
    saved_file = saved_file_input.text()

    return {
        "distance": distance,
        "refresh_period": refresh_period,
        "window_seconds": window_seconds,
        "calibration_duration": calibration_duration,
        "live": live,
        "saved_file": saved_file
    }
