import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QSpinBox,
                             QFormLayout, QHBoxLayout, QGroupBox, QComboBox, QCheckBox, QStackedWidget, QDoubleSpinBox)
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas)


class GAApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep learning and Computational intelligence - EA Project")
        self.setGeometry(100, 100, 1200, 600)

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()


        config_panel = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.pop_size_spin = QSpinBox()
        self.pop_size_spin.setRange(10, 5000)
        self.pop_size_spin.setValue(100)
        self.pop_size_spin.valueChanged.connect(self.update_tournament_size)
        form_layout.addRow("Population Size:", self.pop_size_spin)

        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(10, 10000)
        self.gen_spin.setValue(100)
        form_layout.addRow("Generations:", self.gen_spin)


        self.selection_combo = QComboBox()
        self.selection_combo.addItems(["Tournament", "Roulette Wheel", "Rank Selection", "Boltzmann Selection"])
        self.selection_combo.currentIndexChanged.connect(self.update_selection_config)
        form_layout.addRow("Selection Method:", self.selection_combo)


        self.selection_config_stack = QStackedWidget()



        tournament_panel = QWidget()
        tournament_layout = QFormLayout()
        self.tournament_size_spin = QSpinBox()
        self.tournament_size_spin.setRange(2, self.pop_size_spin.value())
        self.tournament_size_spin.setValue(5)
        self.tournament_best_check = QCheckBox("Return Best Individual")
        tournament_layout.addRow("Tournament Size:", self.tournament_size_spin)
        tournament_layout.addRow("Tournament Best:", self.tournament_best_check)
        tournament_panel.setLayout(tournament_layout)
        self.selection_config_stack.addWidget(tournament_panel)


        roulette_panel = QWidget()
        roulette_layout = QVBoxLayout()
        roulette_panel.setLayout(roulette_layout)
        self.selection_config_stack.addWidget(roulette_panel)


        rank_panel = QWidget()
        rank_layout = QVBoxLayout()
        rank_panel.setLayout(rank_layout)
        self.selection_config_stack.addWidget(rank_panel)


        boltzmann_panel = QWidget()
        boltzmann_layout = QFormLayout()
        self.boltzmann_temperature_spin = QDoubleSpinBox()
        self.boltzmann_temperature_spin.setRange(0.1, 10.0)
        self.boltzmann_temperature_spin.setSingleStep(0.1)
        self.boltzmann_temperature_spin.setValue(1.0)
        boltzmann_layout.addRow("Temperature:", self.boltzmann_temperature_spin)
        boltzmann_panel.setLayout(boltzmann_layout)
        self.selection_config_stack.addWidget(boltzmann_panel)

        form_layout.addRow("Selection Parameters:", self.selection_config_stack)


        self.crossover_combo = QComboBox()
        self.crossover_combo.addItems(["Single Point", "Two Point", "Uniform"])
        form_layout.addRow("Crossover Method:", self.crossover_combo)


        self.mutation_combo = QComboBox()
        self.mutation_combo.addItems(["Bit Flip", "Swap", "Scramble"])
        form_layout.addRow("Mutation Method:", self.mutation_combo)

        self.mutation_rate_spin = QDoubleSpinBox()
        self.mutation_rate_spin.setRange(0.0, 100.0)
        self.mutation_rate_spin.setSingleStep(1.0)
        self.mutation_rate_spin.setValue(5.0)
        form_layout.addRow("Mutation Rate (%):", self.mutation_rate_spin)

        self.mutation_always_check = QCheckBox("Always Mutate")
        form_layout.addRow("Force Mutation:", self.mutation_always_check)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_ga)

        config_layout.addLayout(form_layout)
        config_layout.addWidget(self.run_button)
        config_panel.setLayout(config_layout)


        viz_panel = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()

        info_panel = QWidget()
        info_layout = QVBoxLayout()
        self.info_label = QLabel("Best Solution: None")
        info_layout.addWidget(self.info_label)
        info_panel.setLayout(info_layout)
        viz_layout.addWidget(info_panel, 1)

        chart_panel = QWidget()
        chart_layout = QVBoxLayout()
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)
        chart_panel.setLayout(chart_layout)
        viz_layout.addWidget(chart_panel, 4)

        viz_panel.setLayout(viz_layout)

        main_layout.addWidget(config_panel, 3)
        main_layout.addWidget(viz_panel, 5)

        central_widget.setLayout(main_layout)

    def update_selection_config(self, index):
        self.selection_config_stack.setCurrentIndex(index)

    def update_tournament_size(self):
        self.tournament_size_spin.setMaximum(self.pop_size_spin.value())

    def run_ga(self):
        pop_size = self.pop_size_spin.value()
        generations = self.gen_spin.value()

        self.ax.clear()
        self.ax.set_title("Genetic Algorithm")
        self.ax.set_xlabel("Population")
        self.ax.set_ylabel("Fitness")
        best_fitness = []

        for gen in range(generations):
            fitness = np.random.rand(pop_size)
            best_fitness.append(np.max(fitness))

            self.info_label.setText(f"Best Solution: {max(best_fitness):.4f}")
            self.ax.clear()
            self.ax.set_title("Genetic Algorithm")
            self.ax.set_xlabel("Population")
            self.ax.set_ylabel("Fitness")
            self.ax.plot(range(len(best_fitness)), best_fitness, label="Best Fitness", color='blue')
            self.ax.legend(loc='lower right')
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
            QApplication.processEvents()


def main():
    app = QApplication([])
    main_window = GAApp()
    main_window.show()
    app.exec()


if __name__ == "__main__":
    main()
