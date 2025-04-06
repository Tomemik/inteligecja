import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QSpinBox,
                             QFormLayout, QHBoxLayout, QGroupBox, QComboBox, QCheckBox, QDoubleSpinBox)
from PyQt6.QtCore import QProcess
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas)
import re
import sys

sys.stdout.reconfigure(line_buffering=True)


class GAApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep learning and Computational intelligence - EA Project")
        self.setGeometry(100, 100, 1200, 600)

        self.history = []
        self.best_chromosome = None
        self.maximum_fitness = None
        self.process = None

        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.plot_line, = self.ax.plot([], [], 'b-')

        self.initUI()

    def start_algorythm(self):
        if self.process is None:
            self.process = QProcess()
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stderr)
            self.process.stateChanged.connect(self.handle_state)
            self.process.finished.connect(self.process_finished)

            params = [
                "algo.py",
                "--iterations", str(self.gen_spin.value()),
                "--population_size", str(self.pop_size_spin.value()),
                "--mutation_rate", str(self.mutation_rate_spin.value() / 100),
                "--label", self.attack_info_combobox.currentText(),
                "--stagnation", str(self.stagnation_spin.value()),
                "--elitism", "1" if self.elitism_check.isChecked() else "0",
                # "--full_random", "1" if self.full_random_check.isChecked() else "0"
            ]
            self.process.start(sys.executable, params)

    def message(self, s):
        print("[INFO] " + s)

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.message(stderr)

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")

        new_list = parse_fitness_list(stdout)
        if len(new_list) != 0:
            self.history = new_list
            self.update_chart()

        new_max = parse_maximum_fitness_on_test(stdout)
        if len(new_max) != 0:
            self.maximum_fitness = new_max[0]
            self.update_maximum_fitness_on_test(self.maximum_fitness)

        new_best = parse_best_chromosome_fitness(stdout)
        if len(new_best) != 0:
            self.best_chromosome = new_best[0]
            self.update_best_chromosome_label(self.best_chromosome)

        self.message(stdout)

    def handle_state(self, state):
        states = {
            QProcess.ProcessState.NotRunning: 'Not running',
            QProcess.ProcessState.Starting: 'Starting',
            QProcess.ProcessState.Running: 'Running',
        }
        state_name = states[state]
        if state_name == 'Starting':
            self.do_cleanup()
        self.update_algorythm_state_label(state_name)
        self.message(f"State changed: {state_name}")

    def process_finished(self):
        self.message("Process finished.")
        self.process = None

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()

        config_panel = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.attack_info_combobox = QComboBox()
        self.attack_info_combobox.addItems(get_attack_labels())
        form_layout.addRow("Attack type:", self.attack_info_combobox)

        self.pop_size_spin = QSpinBox()
        self.pop_size_spin.setRange(5, 5000)
        self.pop_size_spin.setValue(100)
        form_layout.addRow("Population Size:", self.pop_size_spin)

        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(1, 10000)
        self.gen_spin.setValue(100)
        form_layout.addRow("Generations:", self.gen_spin)

        self.stagnation_spin = QSpinBox()
        self.stagnation_spin.setRange(0, 50)
        self.stagnation_spin.setValue(5)
        form_layout.addRow("Stagnation limit:", self.stagnation_spin)

        self.mutation_rate_spin = QDoubleSpinBox()
        self.mutation_rate_spin.setRange(0.0, 100.0)
        self.mutation_rate_spin.setSingleStep(0.1)
        self.mutation_rate_spin.setValue(5.0)
        form_layout.addRow("Mutation Rate (%):", self.mutation_rate_spin)

        # self.full_random_check = QCheckBox("Yes")
        # form_layout.addRow("Full random:", self.full_random_check)

        self.elitism_check = QCheckBox("Yes")
        form_layout.addRow("Elitism:", self.elitism_check)

        self.used_selection_label = QLabel("Tournament")
        form_layout.addRow("Selection method:", self.used_selection_label)

        self.used_crossover_label = QLabel("One-point crossover")
        form_layout.addRow("Crossover method:", self.used_crossover_label)

        self.used_mutation_label = QLabel("Mutation of a random 1 feature")
        form_layout.addRow("Mutation method:", self.used_mutation_label)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.start_algorythm)

        config_layout.addLayout(form_layout)
        config_layout.addWidget(self.run_button)
        config_panel.setLayout(config_layout)

        viz_panel = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()

        info_panel = QWidget()
        info_layout = QVBoxLayout()

        self.algorythm_state = QLabel("Algorythm state: Not running")
        info_layout.addWidget(self.algorythm_state)

        self.best_chromosome_label = QLabel("Best chromosome fitness: N/A")
        info_layout.addWidget(self.best_chromosome_label)

        self.max_fitness_label = QLabel("Maximum fitness on test set: N/A")
        info_layout.addWidget(self.max_fitness_label)
        info_panel.setLayout(info_layout)
        viz_layout.addWidget(info_panel, 1)

        chart_panel = QWidget()
        chart_layout = QVBoxLayout()
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)
        chart_panel.setLayout(chart_layout)
        viz_layout.addWidget(chart_panel, 20)

        viz_panel.setLayout(viz_layout)

        main_layout.addWidget(config_panel, 3)
        main_layout.addWidget(viz_panel, 5)

        central_widget.setLayout(main_layout)


    def do_cleanup(self):
        self.update_best_chromosome_label("N/A")
        self.update_maximum_fitness_on_test("N/A")
        self.ax.clear()
        self.canvas.draw()

    def update_algorythm_state_label(self, state: str):
        self.algorythm_state.setText(f"Algorythm state: {state}")

    def update_best_chromosome_label(self, value):
        if isinstance(value, str):
            self.best_chromosome_label.setText(f"Best chromosome fitness: {value}")
        else:
            self.best_chromosome_label.setText(f"Best chromosome fitness: {value:.4}")

    def update_maximum_fitness_on_test(self, value):
        if isinstance(value, str):
            self.max_fitness_label.setText(f"Maximum fitness on test set: {value}")
        else:
            self.max_fitness_label.setText(f"Maximum fitness on test set: {value:.4}")

    def update_chart(self):
        x = list(range(1, len(self.history) + 1))
        y = self.history

        self.ax.clear()
        self.ax.plot(x, y, 'b-', label="Fitness")

        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Fitness")
        self.ax.set_title("Fitness history")
        self.ax.legend(loc='lower right')
        self.ax.relim()
        self.ax.autoscale_view()

        self.canvas.draw()
        QApplication.processEvents()




def parse_fitness_list(line: str):
    matches = re.findall(r'np\.float64\(([\d\.eE+-]+)\)', line)
    if matches is None:
        return []
    return [float(val) for val in matches]

def parse_best_chromosome_fitness(line: str):
    matches = re.findall(r'Best chromosome fitness:\s*([\d.]+)', line)
    if matches is None:
        return []
    return [float(val) for val in matches]

def parse_maximum_fitness_on_test(line: str):
    matches = re.findall(r'Maximum fitness on test set:\s*([\d.]+)', line)
    if matches is None:
        return []
    return [float(val) for val in matches]


def get_attack_labels():
    return ["smurf",
            "neptune",
            "normal",
            "satan",
            "ipsweep",
            "portsweep",
            "nmap",
            "back",
            "warezclient",
            "teardrop",
            "pod",
            "guess_passwd",
            "buffer_overflow",
            "land",
            "warezmaster",
            "imap",
            "rootkit",
            "loadmodule",
            "ftp_write",
            "multihop",
            "phf",
            "perl",
            "spy"]


def main():
    app = QApplication([])
    main_window = GAApp()
    main_window.show()
    app.exec()


if __name__ == "__main__":
    main()
