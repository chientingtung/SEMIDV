# MIT License
# 
# Copyright (c) 2025 Chien-Ting Tung
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
from copy import deepcopy

class solve:
    def __init__(self, device, boundary, damping=1, recombination=False, quantum=False, Kq=0):
        self.device = device
        self.boundary = boundary
        self.damping = damping
        self.recombination = recombination
        self.quantum = quantum
        self.Kq = Kq

    def update_boundary(
        self, V1=None, V2=None, V1_contact_name=None, V2_contact_name=None
    ):
        """
        Dynamically update the boundary conditions with the new V1 and V2 values,
        allowing for different boundary types (Dirichlet, Neumann, etc.).
        """
        for condition in self.boundary:
            contact_name = condition.get("name")

            if contact_name == V1_contact_name:
                condition["value"] = V1
            elif contact_name == V2_contact_name:
                condition["value"] = V2

    def IVsweep(self, V1_contact_name, V1_sweep, V2_contact_name, V2_sweep, results):
        """
        Perform an IV sweep of the device, where V1 and V2 are identified by their contact names.
        """
        if results == {}:
            empty = True
        else:
            empty = False

        for i, V2 in enumerate(V2_sweep):
            for j, V1 in enumerate(V1_sweep):
                V2 = np.round(V2, decimals=3)
                V1 = np.round(V1, decimals=3)

                # Update initial guesses based on previous results
                if empty == True:
                    if j == 1:
                        Ec_guess = results[
                            (
                                np.round(V1_sweep[j - 1], decimals=3),
                                np.round(V2_sweep[i], decimals=3),
                            )
                        ]["model"].Ec
                        Efn_guess = results[
                            (
                                np.round(V1_sweep[j - 1], decimals=3),
                                np.round(V2_sweep[i], decimals=3),
                            )
                        ]["model"].Efn
                        Efp_guess = results[
                            (
                                np.round(V1_sweep[j - 1], decimals=3),
                                np.round(V2_sweep[i], decimals=3),
                            )
                        ]["model"].Efp
                        self.device.update_initial_guess(Ec_guess, Efn_guess, Efp_guess)
                    if j > 1:
                        dV = (V1_sweep[j] - V1_sweep[j - 1]) / (
                            V1_sweep[j - 1] - V1_sweep[j - 2]
                        )
                        Ec_guess = results[
                            (
                                np.round(V1_sweep[j - 1], decimals=3),
                                np.round(V2_sweep[i], decimals=3),
                            )
                        ]["model"].Ec + dV * (
                            results[
                                (
                                    np.round(V1_sweep[j - 1], decimals=3),
                                    np.round(V2_sweep[i], decimals=3),
                                )
                            ]["model"].Ec
                            - results[
                                (
                                    np.round(V1_sweep[j - 2], decimals=3),
                                    np.round(V2_sweep[i], decimals=3),
                                )
                            ]["model"].Ec
                        )
                        Efn_guess = results[
                            (
                                np.round(V1_sweep[j - 1], decimals=3),
                                np.round(V2_sweep[i], decimals=3),
                            )
                        ]["model"].Efn + dV * (
                            results[
                                (
                                    np.round(V1_sweep[j - 1], decimals=3),
                                    np.round(V2_sweep[i], decimals=3),
                                )
                            ]["model"].Efn
                            - results[
                                (
                                    np.round(V1_sweep[j - 2], decimals=3),
                                    np.round(V2_sweep[i], decimals=3),
                                )
                            ]["model"].Efn
                        )
                        Efp_guess = results[
                            (
                                np.round(V1_sweep[j - 1], decimals=3),
                                np.round(V2_sweep[i], decimals=3),
                            )
                        ]["model"].Efp + dV * (
                            results[
                                (
                                    np.round(V1_sweep[j - 1], decimals=3),
                                    np.round(V2_sweep[i], decimals=3),
                                )
                            ]["model"].Efp
                            - results[
                                (
                                    np.round(V1_sweep[j - 2], decimals=3),
                                    np.round(V2_sweep[i], decimals=3),
                                )
                            ]["model"].Efp
                        )
                        self.device.update_initial_guess(Ec_guess, Efn_guess, Efp_guess)
                    if j == 0:
                        if i == 1:
                            Ec_guess = results[
                                (
                                    np.round(V1_sweep[0], decimals=3),
                                    np.round(V2_sweep[i - 1], decimals=3),
                                )
                            ]["model"].Ec
                            Efn_guess = results[
                                (
                                    np.round(V1_sweep[0], decimals=3),
                                    np.round(V2_sweep[i - 1], decimals=3),
                                )
                            ]["model"].Efn
                            Efp_guess = results[
                                (
                                    np.round(V1_sweep[0], decimals=3),
                                    np.round(V2_sweep[i - 1], decimals=3),
                                )
                            ]["model"].Efp
                            self.device.update_initial_guess(
                                Ec_guess, Efn_guess, Efp_guess
                            )
                        elif i > 1:
                            dV = (V2_sweep[i] - V2_sweep[i - 1]) / (
                                V2_sweep[i - 1] - V2_sweep[i - 2]
                            )
                            Ec_guess = results[
                                (
                                    np.round(V1_sweep[0], decimals=3),
                                    np.round(V2_sweep[i - 1], decimals=3),
                                )
                            ]["model"].Ec + dV * (
                                results[
                                    (
                                        np.round(V1_sweep[0], decimals=3),
                                        np.round(V2_sweep[i - 1], decimals=3),
                                    )
                                ]["model"].Ec
                                - results[
                                    (
                                        np.round(V1_sweep[0], decimals=3),
                                        np.round(V2_sweep[i - 2], decimals=3),
                                    )
                                ]["model"].Ec
                            )
                            Efn_guess = results[
                                (
                                    np.round(V1_sweep[0], decimals=3),
                                    np.round(V2_sweep[i - 1], decimals=3),
                                )
                            ]["model"].Efn + dV * (
                                results[
                                    (
                                        np.round(V1_sweep[0], decimals=3),
                                        np.round(V2_sweep[i - 1], decimals=3),
                                    )
                                ]["model"].Efn
                                - results[
                                    (
                                        np.round(V1_sweep[0], decimals=3),
                                        np.round(V2_sweep[i - 2], decimals=3),
                                    )
                                ]["model"].Efn
                            )
                            Efp_guess = results[
                                (
                                    np.round(V1_sweep[0], decimals=3),
                                    np.round(V2_sweep[i - 1], decimals=3),
                                )
                            ]["model"].Efp + dV * (
                                results[
                                    (
                                        np.round(V1_sweep[0], decimals=3),
                                        np.round(V2_sweep[i - 1], decimals=3),
                                    )
                                ]["model"].Efp
                                - results[
                                    (
                                        np.round(V1_sweep[0], decimals=3),
                                        np.round(V2_sweep[i - 2], decimals=3),
                                    )
                                ]["model"].Efp
                            )
                            self.device.update_initial_guess(
                                Ec_guess, Efn_guess, Efp_guess
                            )
                else:
                    Ec_guess = results[
                        (
                            np.round(V1_sweep[j], decimals=3),
                            np.round(V2_sweep[i], decimals=3),
                        )
                    ]["model"].Ec
                    Efn_guess = results[
                        (
                            np.round(V1_sweep[j], decimals=3),
                            np.round(V2_sweep[i], decimals=3),
                        )
                    ]["model"].Efn
                    Efp_guess = results[
                        (
                            np.round(V1_sweep[j], decimals=3),
                            np.round(V2_sweep[i], decimals=3),
                        )
                    ]["model"].Efp
                    self.device.update_initial_guess(Ec_guess, Efn_guess, Efp_guess)

                # Update boundary conditions dynamically for V1 and V2 based on contact names
                self.update_boundary(
                    V1=V1,
                    V2=V2,
                    V1_contact_name=V1_contact_name,
                    V2_contact_name=V2_contact_name,
                )
                self.device.assign_boundary_conditions(self.boundary)
                self.device.QCn = np.zeros([self.device.nx, self.device.ny])
                self.device.QCp = np.zeros([self.device.nx, self.device.ny])
                self.device.DriftDiffusion(recombination=self.recombination)
                self.device.getmatrix()
                if self.quantum == 'e':
                    self.device.quantum_getmatrix()
                if self.quantum == 'h':
                    self.device.quantum_getmatrix()

                # Run the simulation loop until convergence
                if self.quantum == False:
                    print("Solve Poisson-Drift-Diffusion")
                else:
                    print("Solve Quantum-Poisson-Drift-Diffusion")
                error = 1
                while error > self.device.tolerance:
                    self.device.Poisson(damping=self.damping)
                    if self.quantum == 'e':
                        self.device.equantum(Kq=self.Kq)
                    if self.quantum == 'h':
                        self.device.hquantum(Kq=self.Kq)
                    self.device.DriftDiffusion(recombination=self.recombination)
                    error = self.device.error
                    print(
                        f"{V1_contact_name}: {V1}, {V2_contact_name}: {V2}, Error: {error}"
                    )
                print("Converge!")



                # Store the results
                self.device.getcurrent()
                self.device.getcharge()

                results[(V1, V2)] = {
                    "model": deepcopy(self.device),
                    "current": self.device.current.copy(),
                    "charge": self.device.charge.copy(),
                }
        return results
