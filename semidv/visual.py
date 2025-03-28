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

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import random


class visual:
    def __init__(self, device):
        """
        Initialize the MaterialGridPlotter with the device object containing material structure and grid information.

        Parameters:
        - device: A device object that contains the material grid (structure) and grid spacing (dx, dy, nx, ny).
        """
        # Extract necessary properties from the device object
        self.device = device
        self.materials = self.device.structure  # Material grid (2D array)
        self.dx = self.device.dx  # x-axis spacing
        self.dy = self.device.dy  # y-axis spacing
        self.nx = self.device.nx  # Number of grid points in the x direction
        self.ny = self.device.ny  # Number of grid points in the y direction

        # Generate the unique material types in the structure
        self.unique_materials = np.unique(self.materials)

        # Mapping materials to random colors
        self.material_to_color = self._generate_random_colors()

        # Mapping materials to indices
        self.material_to_index = self._create_material_index_mapping()

        # Convert the material structure to a matrix of indices
        self.index_matrix = self._convert_materials_to_indices()

    def _generate_random_colors(self):
        """
        Generate random colors for each unique material.

        Returns:
        - Dictionary mapping each material to a random color.
        """
        random.seed(1023)  # Set seed for reproducibility (optional)
        return {
            material: random.choice(list(mcolors.CSS4_COLORS.values()))
            for material in self.unique_materials
        }

    def _create_material_index_mapping(self):
        """
        Create a mapping from material names to unique indices.

        Returns:
        - Dictionary mapping each material to an index.
        """
        return {material: idx for idx, material in enumerate(self.unique_materials)}

    def _convert_materials_to_indices(self):
        """
        Convert the materials array to a matrix of indices based on the material-to-index mapping.

        Returns:
        - 2D numpy array with indices representing materials.
        """
        return np.vectorize(self.material_to_index.get)(self.materials)

    def plot_structure(self, save=False):
        """
        Plot the material grid with randomly assigned colors for each material.
        """
        # Create a figure and axes
        fig, ax = plt.subplots()

        # Create a colormap from the randomly generated colors
        cmap = mcolors.ListedColormap(
            [self.material_to_color[material] for material in self.unique_materials]
        )

        # Create a normalization for the colormap (for discrete colors)
        norm = mcolors.BoundaryNorm(
            boundaries=np.arange(len(self.unique_materials) + 1),
            ncolors=len(self.unique_materials),
        )

        # Display the material grid with colors
        cax = ax.imshow(
            self.index_matrix,
            cmap=cmap,
            norm=norm,
            extent=[0, (self.ny - 1) * self.dy, 0, (self.nx - 1) * self.dx],
        )

        # Add a color bar with material names
        cbar = plt.colorbar(cax, ticks=np.arange(len(self.unique_materials)))
        cbar.set_ticklabels(self.unique_materials)

        # Title and labels
        ax.set_title("Material")
        ax.set_xlabel(f"Y")
        ax.set_ylabel(f"X")
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        # Show the plot
        plt.show()


    def _plot_boundaries(self, ax):
        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                if self.materials[i, j] != self.materials[i + 1, j]:
                    ax.plot(
                        [j * self.dy, (j + 1) * self.dy],
                        [(i + 1) * self.dx, (i + 1) * self.dx],
                        "k-",
                        linewidth=1.5,
                    )
                if self.materials[i, j] != self.materials[i, j + 1]:
                    ax.plot(
                        [(j + 1) * self.dy, (j + 1) * self.dy],
                        [i * self.dx, (i + 1) * self.dx],
                        "k-",
                        linewidth=1.5,
                    )

    def plot_potential(self, save=False):
        """
        Plot the contour of the potential Ec / q.
        """
        Ec = self.device.Ec  # Electric potential (2D array)
        xi = self.device.xi
        q = self.device.q  # Charge (scalar)

        # Calculate the potential -Ec / q
        potential = -(Ec + xi) / q
        potential = potential - np.min(potential)

        # Create the contour plot
        fig, ax = plt.subplots()

        # Generate contours
        contour = ax.contourf(
            np.arange(self.ny) * self.dy,
            np.arange(self.nx) * self.dx,
            potential,
            levels=50,
            cmap="viridis",
        )

        # Add color bar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Electrostatic Potential (V)")

        # Set labels and title
        ax.set_title("Electrostatic Potential (V)")
        ax.set_xlabel(f"Y")
        ax.set_ylabel(f"X")
        ax.set_aspect(self.nx / self.ny)
        # Show the plot
        self._plot_boundaries(ax)
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()


    def plot_efermi(self, save=False):
        """
        Plot the contour of the e fermi energy Efn.
        """
        Efn = self.device.Efn  # Fermi level (2D array)
        q = self.device.q  # Charge (scalar)

        # Calculate the e fermi -Ec / q
        efermi = -Efn / q

        # Create the contour plot
        fig, ax = plt.subplots()

        # Generate contours
        contour = ax.contourf(
            np.arange(self.ny) * self.dy,
            np.arange(self.nx) * self.dx,
            efermi,
            levels=50,
            cmap="viridis",
        )

        # Add color bar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("eFermi Potential (eV)")

        # Set labels and title
        ax.set_title("eFermi Potential (eV)")
        ax.set_xlabel(f"Y")
        ax.set_ylabel(f"X")
        ax.set_aspect(self.nx / self.ny)
        # Show the plot
        self._plot_boundaries(ax)
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_hfermi(self, save=False):
        """
        Plot the contour of the h fermi energy Efp.
        """
        Efp = self.device.Efp  # Fermi level (2D array)
        q = self.device.q  # Charge (scalar)

        # Calculate the e fermi -Ec / q
        efermi = -Efp / q

        # Create the contour plot
        fig, ax = plt.subplots()

        # Generate contours
        contour = ax.contourf(
            np.arange(self.ny) * self.dy,
            np.arange(self.nx) * self.dx,
            efermi,
            levels=50,
            cmap="viridis",
        )

        # Add color bar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("hFermi Potential (eV)")

        # Set labels and title
        ax.set_title("hFermi Potential (eV)")
        ax.set_xlabel(f"Y")
        ax.set_ylabel(f"X")
        ax.set_aspect(self.nx / self.ny)
        # Show the plot
        self._plot_boundaries(ax)
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_doping(self, save=False):
        """
        Plot the contour of the doping.
        """
        doping = np.abs(self.device.NB * 1e-6)
        # Create the contour plot
        fig, ax = plt.subplots()

        # Generate contours
        contour = ax.contourf(
            np.arange(self.ny) * self.dy,
            np.arange(self.nx) * self.dx,
            doping,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )

        # Add color bar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Doping (cm^-3)")

        # Set labels and title
        ax.set_title("Doping (cm^-3)")
        ax.set_xlabel(f"Y")
        ax.set_ylabel(f"X")
        ax.set_aspect(self.nx / self.ny)
        # Show the plot
        self._plot_boundaries(ax)
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_n(self, save=False):
        """
        Plot the contour of the n.
        """
        n = self.device.n * 1e-6
        n = np.ma.masked_outside(n, 1e10, 1e22)

        # Create the contour plot
        fig, ax = plt.subplots()

        # Generate contours
        contour = ax.contourf(
            np.arange(self.ny) * self.dy,
            np.arange(self.nx) * self.dx,
            n,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )

        # Add color bar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Electron (cm^-3)")

        # Set labels and title
        ax.set_title("Electron (cm^-3)")
        ax.set_xlabel(f"Y")
        ax.set_ylabel(f"X")
        ax.set_aspect(self.nx / self.ny)
        # Show the plot
        self._plot_boundaries(ax)
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_p(self, save=False):
        """
        Plot the contour of the p.
        """
        p = self.device.p * 1e-6
        p = np.ma.masked_outside(p, 1e10, 1e22)

        # Create the contour plot
        fig, ax = plt.subplots()

        # Generate contours
        contour = ax.contourf(
            np.arange(self.ny) * self.dy,
            np.arange(self.nx) * self.dx,
            p,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )

        # Add color bar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Hole (cm^-3)")

        # Set labels and title
        ax.set_title("Hole (cm^-3)")
        ax.set_xlabel(f"Y")
        ax.set_ylabel(f"X")
        ax.set_aspect(self.nx / self.ny)
        # Show the plot
        self._plot_boundaries(ax)
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_Efield(self, save=False):
        Efield = np.gradient(
            (self.device.Ec + self.device.xi) / self.device.q, self.dx, self.dy
        )
        E_mag = np.sqrt(Efield[0] ** 2 + Efield[1] ** 2) + 0.01
        E_x = Efield[0] / E_mag
        E_y = Efield[1] / E_mag
        fig, ax = plt.subplots()
        plt.quiver(
            np.arange(self.ny) * self.dy,
            np.arange(self.nx) * self.dx,
            E_y,
            E_x,
            E_mag,
            cmap="inferno",
            scale=50,
        )
        plt.xlabel("Y")
        plt.ylabel("X")
        ax.set_title("Electric field")
        self._plot_boundaries(ax)
        # Draw the outer boundary
        xmax = self.ny * self.dy
        ymax = self.nx * self.dx

        # Left and Right boundaries
        ax.plot([0, 0], [0, ymax], "k-", linewidth=1.5)  # Left
        ax.plot([xmax, xmax], [0, ymax], "k-", linewidth=1.5)  # Right

        # Bottom and Top boundaries
        ax.plot([0, xmax], [0, 0], "k-", linewidth=1.5)  # Bottom
        ax.plot([0, xmax], [ymax, ymax], "k-", linewidth=1.5)  # Top
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_Jn(self, save=False):
        J_mag = np.sqrt(self.device.Jnx**2 + self.device.Jny**2) + 1e-15
        J_x = -self.device.Jnx / J_mag
        J_y = -self.device.Jny / J_mag
        fig, ax = plt.subplots()
        plt.quiver(
            np.arange(self.ny) * self.dy,
            np.arange(self.nx) * self.dx,
            J_y,
            J_x,
            J_mag,
            cmap="inferno",
            scale=50,
        )
        plt.xlabel("Y")
        plt.ylabel("X")
        ax.set_title("Electron Current")
        self._plot_boundaries(ax)
        # Draw the outer boundary
        xmax = self.ny * self.dy
        ymax = self.nx * self.dx

        # Left and Right boundaries
        ax.plot([0, 0], [0, ymax], "k-", linewidth=1.5)  # Left
        ax.plot([xmax, xmax], [0, ymax], "k-", linewidth=1.5)  # Right

        # Bottom and Top boundaries
        ax.plot([0, xmax], [0, 0], "k-", linewidth=1.5)  # Bottom
        ax.plot([0, xmax], [ymax, ymax], "k-", linewidth=1.5)  # Top
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_Jp(self, save=False):
        J_mag = np.sqrt(self.device.Jpx**2 + self.device.Jpy**2) + 1e-15
        J_x = self.device.Jpx / J_mag
        J_y = self.device.Jpy / J_mag
        fig, ax = plt.subplots()
        plt.quiver(
            np.arange(self.ny) * self.dy,
            np.arange(self.nx) * self.dx,
            J_y,
            J_x,
            J_mag,
            cmap="inferno",
            scale=50,
        )
        plt.xlabel("Y")
        plt.ylabel("X")
        ax.set_title("Hole Current")
        self._plot_boundaries(ax)
        # Draw the outer boundary
        xmax = self.ny * self.dy
        ymax = self.nx * self.dx

        # Left and Right boundaries
        ax.plot([0, 0], [0, ymax], "k-", linewidth=1.5)  # Left
        ax.plot([xmax, xmax], [0, ymax], "k-", linewidth=1.5)  # Right

        # Bottom and Top boundaries
        ax.plot([0, xmax], [0, 0], "k-", linewidth=1.5)  # Bottom
        ax.plot([0, xmax], [ymax, ymax], "k-", linewidth=1.5)  # Top
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_band_xcut(self, x_coord=0, save=False):
        q = self.device.q
        x_index = int(x_coord / self.dx)
        if x_index < 0 or x_index >= self.nx:
            raise ValueError("x_coord is out of bounds")

        y_values = np.arange(self.ny) * self.dy
        Ec = self.device.Ec[x_index, :]
        Ev = self.device.Ev[x_index, :]
        Efn = self.device.Efn[x_index, :]
        Efp = self.device.Efp[x_index, :]

        plt.figure()
        plt.plot(y_values, Ec / q, label="Ec", linestyle="-", color="b")
        plt.plot(y_values, Ev / q, label="Ev", linestyle="-", color="r")
        plt.plot(y_values, Efn / q, label="Efn", linestyle="--", color="g")
        plt.plot(y_values, Efp / q, label="Efp", linestyle="--", color="m")
        plt.xlabel("y Coordinate")
        plt.ylabel("Energy (eV)")
        plt.title(f"Energy Levels at x = {x_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_band_ycut(self, y_coord=0, save=False):
        q = self.device.q
        y_index = int(y_coord / self.dy)
        if y_index < 0 or y_index >= self.ny:
            raise ValueError("x_coord is out of bounds")

        x_values = np.arange(self.nx) * self.dx
        Ec = self.device.Ec[:, y_index]
        Ev = self.device.Ev[:, y_index]
        Efn = self.device.Efn[:, y_index]
        Efp = self.device.Efp[:, y_index]

        plt.figure()
        plt.plot(x_values, Ec / q, label="Ec", linestyle="-", color="b")
        plt.plot(x_values, Ev / q, label="Ev", linestyle="-", color="r")
        plt.plot(x_values, Efn / q, label="Efn", linestyle="--", color="g")
        plt.plot(x_values, Efp / q, label="Efp", linestyle="--", color="m")
        plt.xlabel("x Coordinate")
        plt.ylabel("Energy (eV)")
        plt.title(f"Energy Levels at y = {y_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_efield_xcut(self, x_coord=0, save=False):
        Efield = np.gradient(
            (self.device.Ec + self.device.xi) / self.device.q, self.dx, self.dy
        )
        x_index = int(x_coord / self.dx)
        if x_index < 0 or x_index >= self.nx:
            raise ValueError("x_coord is out of bounds")

        y_values = np.arange(self.ny) * self.dy
        Ey = self.device.Ec[1][x_index, :]

        plt.figure()
        plt.plot(y_values, Ey, label="Ey", linestyle="-", color="b")
        plt.xlabel("y Coordinate")
        plt.ylabel("Electric Field (V/m)")
        plt.title(f"Electric Field at x = {x_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_efield_ycut(self, y_coord=0, save=False):
        Efield = np.gradient(
            (self.device.Ec + self.device.xi) / self.device.q, self.dx, self.dy
        )
        y_index = int(y_coord / self.dy)
        if y_index < 0 or y_index >= self.ny:
            raise ValueError("y_coord is out of bounds")

        x_values = np.arange(self.nx) * self.dx
        Ex = self.device.Ec[0][:, y_index]

        plt.figure()
        plt.plot(x_values, Ex, label="Ex", linestyle="-", color="b")
        plt.xlabel("y Coordinate")
        plt.ylabel("Electric Field (V/m)")
        plt.title(f"Electric Field at y = {y_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_doping_xcut(self, x_coord=0, save=False):
        x_index = int(x_coord / self.dx)
        if x_index < 0 or x_index >= self.nx:
            raise ValueError("x_coord is out of bounds")

        y_values = np.arange(self.ny) * self.dy
        doping = self.device.NB[x_index, :] * 1e-6

        plt.figure()
        plt.semilogy(y_values, np.abs(doping), label="Nd-Na", linestyle="-", color="b")
        plt.xlabel("y Coordinate")
        plt.ylabel("Doping Concentration (cm^-3)")
        plt.title(f"Doping Concentration at x = {x_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_doping_ycut(self, y_coord=0, save=False):
        y_index = int(y_coord / self.dy)
        if y_index < 0 or y_index >= self.ny:
            raise ValueError("x_coord is out of bounds")

        x_values = np.arange(self.nx) * self.dx
        doping = self.device.NB[:, y_index] * 1e-6

        plt.figure()
        plt.semilogy(x_values, np.abs(doping), label="Nd-Na", linestyle="-", color="b")
        plt.xlabel("y Coordinate")
        plt.ylabel("Doping Concentration (cm^-3)")
        plt.title(f"Doping Concentration at y = {x_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_q_xcut(self, x_coord=0, save=False):
        x_index = int(x_coord / self.dx)
        if x_index < 0 or x_index >= self.nx:
            raise ValueError("x_coord is out of bounds")

        y_values = np.arange(self.ny) * self.dy
        n = self.device.n[x_index, :] * 1e-6
        p = self.device.p[x_index, :] * 1e-6
        n = np.ma.masked_outside(n, 0e10, 1e22)
        p = np.ma.masked_outside(p, 0e10, 1e22)

        plt.figure()
        plt.plot(y_values, n, label="n", linestyle="-", color="b")
        plt.plot(y_values, p, label="p", linestyle="-", color="r")
        plt.xlabel("y Coordinate")
        plt.ylabel("Carrier Density (cm^-3)")
        plt.title(f"Carrier Density at x = {x_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_q_ycut(self, y_coord=0, save=False):
        y_index = int(y_coord / self.dy)
        if y_index < 0 or y_index >= self.ny:
            raise ValueError("x_coord is out of bounds")

        x_values = np.arange(self.nx) * self.dx
        n = self.device.n[:, y_index] * 1e-6
        p = self.device.p[:, y_index] * 1e-6
        n = np.ma.masked_outside(n, 0e10, 1e22)
        p = np.ma.masked_outside(p, 0e10, 1e22)

        plt.figure()
        plt.plot(x_values, n, label="n", linestyle="-", color="b")
        plt.plot(x_values, p, label="p", linestyle="-", color="r")
        plt.xlabel("x Coordinate")
        plt.ylabel("Carrier Density (cm^-3)")
        plt.title(f"Carrier Density at y = {y_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_J_xcut(self, x_coord=0, save=False):
        x_index = int(x_coord / self.dx)
        if x_index < 0 or x_index >= self.nx:
            raise ValueError("x_coord is out of bounds")

        y_values = np.arange(self.ny) * self.dy
        Jny = self.device.Jnx[x_index, :]
        Jpy = self.device.Jpx[x_index, :]

        plt.figure()
        plt.plot(y_values, Jny * 1e-4, label="Jny", linestyle="-", color="b")
        plt.plot(y_values, Jpy * 1e-4, label="Jpy", linestyle="-", color="r")
        plt.xlabel("y Coordinate")
        plt.ylabel("Current Density (A/cm^2)")
        plt.title(f"Current Density at x = {x_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_J_ycut(self, y_coord=0, save=False):
        y_index = int(y_coord / self.dy)
        if y_index < 0 or y_index >= self.ny:
            raise ValueError("x_coord is out of bounds")

        x_values = np.arange(self.nx) * self.dx
        Jnx = self.device.Jnx[:, y_index]
        Jpx = self.device.Jpx[:, y_index]

        plt.figure()
        plt.plot(x_values, Jnx * 1e-4, label="Jnx", linestyle="-", color="b")
        plt.plot(x_values, Jpx * 1e-4, label="Jpx", linestyle="-", color="r")
        plt.xlabel("x Coordinate")
        plt.ylabel("Current Density (A/cm^2)")
        plt.title(f"Current Density at y = {y_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_vn_xcut(self, x_coord=0, save=False):
        x_index = int(x_coord / self.dx)
        if x_index < 0 or x_index >= self.nx:
            raise ValueError("x_coord is out of bounds")

        y_values = np.arange(self.ny) * self.dy
        vny = self.device.vny[x_index, :]

        plt.figure()
        plt.plot(y_values, vny * 1e2, label="vny", linestyle="-", color="b")
        plt.xlabel("y Coordinate")
        plt.ylabel("Electron Velocity (cm/s)")
        plt.title(f"Electron Velocity at x = {x_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_vn_ycut(self, y_coord=0, save=False):
        y_index = int(y_coord / self.dy)
        if y_index < 0 or y_index >= self.ny:
            raise ValueError("x_coord is out of bounds")

        x_values = np.arange(self.nx) * self.dx
        vnx = self.device.vnx[:, y_index]

        plt.figure()
        plt.plot(x_values, vnx * 1e2, label="vnx", linestyle="-", color="b")
        plt.xlabel("x Coordinate")
        plt.ylabel("Eletron Velocity (cm/s)")
        plt.title(f"Eletron Velocity at y = {y_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_vp_xcut(self, x_coord=0, save=False):
        x_index = int(x_coord / self.dx)
        if x_index < 0 or x_index >= self.nx:
            raise ValueError("x_coord is out of bounds")

        y_values = np.arange(self.ny) * self.dy
        vpy = self.device.vpx[x_index, :]

        plt.figure()
        plt.plot(y_values, vpy * 1e2, label="vpy", linestyle="-", color="r")
        plt.xlabel("y Coordinate")
        plt.ylabel("Hole Velocity (cm/s)")
        plt.title(f"Hole Velocity at x = {x_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()

    def plot_vp_ycut(self, y_coord=0, save=False):
        y_index = int(y_coord / self.dy)
        if y_index < 0 or y_index >= self.ny:
            raise ValueError("x_coord is out of bounds")

        x_values = np.arange(self.nx) * self.dx
        vpx = self.device.vpx[:, y_index]

        plt.figure()
        plt.plot(x_values, vpx * 1e2, label="vpx", linestyle="-", color="r")
        plt.xlabel("x Coordinate")
        plt.ylabel("Hole Velocity (cm/s)")
        plt.title(f"Hole Velocity at y = {y_coord} m")
        plt.legend()
        plt.grid()
        if save != False:
            if save == True:
                plt.savefig("figure.png")
            else:
                plt.savefig(save)
        plt.show()
