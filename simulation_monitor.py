# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pyAerial library - Simulation monitor for notebook examples."""
from collections import defaultdict
import datetime

import matplotlib.pyplot as plt
import numpy as np


class SimulationMonitor:
    """Helper class to show the progress and results of the simulation."""

    markers = ["d", "o", "s"]
    linestyles = ["-", "--", ":"]
    colors = ["blue", "black", "red"]

    def __init__(self, cases, esno_db_range):
        """Initialize the SimulationMonitor.

        Initialize the figure and the results table.
        """
        self.cases = cases
        self.esno_db_range = esno_db_range
        self.current_esno_db_range = []

        self.start_time = None
        self.esno_db = None
        self.bler = defaultdict(list)

        self._print_headers()

    def step(self, esno_db):
        """Start next Es/No value."""
        self.start_time = datetime.datetime.now()
        self.esno_db = esno_db
        self.current_esno_db_range.append(esno_db)

    def update(self, num_tbs, num_tb_errors):
        """Update current state for the current Es/No value."""
        self._print_status(num_tbs, num_tb_errors, False)

    def _print_headers(self):
        """Print result table headers."""
        cases_str = " " * 21
        separator = " " * 21
        for case in self.cases:
            cases_str += case.center(20) + " "
            separator += "-" * 20 + " "
        print(cases_str)
        print(separator)
        title_str = "Es/No (dB)".rjust(12) + "TBs".rjust(8) + " "
        for case in self.cases:
            title_str += "TB Errors".rjust(12) + "BLER".rjust(8) + " "
        title_str += "ms/TB".rjust(8)
        print(title_str)
        print(("=" * 20) + " " + ("=" * 20 + " ") * len(self.cases) + "=" * 8)

    def _print_status(self, num_tbs, num_tb_errors, finish):
        """Print simulation status in a table."""
        end_time = datetime.datetime.now()
        t_delta = end_time - self.start_time

        if finish:
            newline_char = '\n'
        else:
            newline_char = '\r'
        result_str = f"{self.esno_db:9.2f}".rjust(12) + f"{num_tbs:8d}".rjust(8) + " "
        for case in self.cases:
            result_str += f"{num_tb_errors[case]:8d}".rjust(12)
            result_str += f"{(num_tb_errors[case] / num_tbs):.4f}".rjust(8) + " "
        result_str += f"{(t_delta.total_seconds() * 1000 / num_tbs):6.1f}".rjust(8)
        print(result_str, end=newline_char)

    def finish_step(self, num_tbs, num_tb_errors):
        """Finish simulating the current Es/No value and add the result in the plot."""
        self._print_status(num_tbs, num_tb_errors, True)
        for case_idx, case in enumerate(self.cases):
            self.bler[case].append(num_tb_errors[case] / num_tbs)

    def finish(self):
        """Finish simulation and plot the results."""
        self.fig = plt.figure()
        for case_idx, case in enumerate(self.cases):
            plt.plot(
                self.current_esno_db_range,
                self.bler[case],
                marker=SimulationMonitor.markers[case_idx],
                linestyle=SimulationMonitor.linestyles[case_idx],
                color=SimulationMonitor.colors[case_idx],
                markersize=8,
                label=case
            )
        plt.yscale('log')
        plt.ylim(0.001, 1)
        plt.xlim(np.min(self.esno_db_range), np.max(self.esno_db_range))
        plt.title("Receiver BLER Performance vs. Es/No")
        plt.ylabel("BLER")
        plt.xlabel("Es/No [dB]")
        plt.grid()
        plt.legend()
        plt.show()
