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

from ctypes import memmove
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
from fdint import *


class device:
    def __init__(self, T=300, fermi=True, tolerance=1e-6, L=1):
        self.q = 1.6e-19
        self.T = T
        self.kbT = 1.38e-23 * T
        self.ep0 = 8.854e-12
        self.h = 6.626e-34
        self.hbar = 1.055e-34
        self.fermi = fermi
        self.tolerance = tolerance
        self.L = L
        self.charge_regions = []

    def build_device_structure(
        self, domain_size, spatial_steps, materials, doping_profiles=None
    ):
        """
        Build a 2D array representing the device structure with separate spatial steps for x and y.

        Parameters:
        - domain_size: tuple (width, height) in meters, e.g., (1e-6, 1e-6)
        - spatial_steps: tuple (x_step, y_step) in meters, e.g., (1e-9, 2e-9)
        - materials: list of dictionaries with material and range, e.g.,
          [{"material": "Si", "x": (1e-9, 2e-9), "y": (2e-9, 3e-9)}]
        - doping_profiles: list of dictionaries with doping type, concentration (cm^-3), and range, e.g.,
          [{"type": "n-type", "concentration": 1e18, "x": (1e-9, 2e-9), "y": (2e-9, 3e-9)}]

        Returns:
        - A dictionary containing:
          - "structure": A 2D numpy array representing the device structure materials.
          - "doping": A 2D numpy array representing the doping concentrations (n-type positive, p-type negative).
        """

        # Unpack domain size and spatial steps
        width, height = domain_size
        x_step, y_step = spatial_steps

        # Compute grid size
        self.nx = int(np.round(width / x_step))
        self.ny = int(np.round(height / y_step))
        self.dx = x_step
        self.dy = y_step

        # Initialize arrays for materials and doping concentration
        device_array = np.full((self.nx, self.ny), "Si", dtype=object)
        doping_array = np.zeros((self.nx, self.ny))

        # Populate the material array
        for material in materials:
            mat_name = material["material"]
            x_range = material["x"]
            y_range = material["y"]

            # Convert ranges to grid indices
            x_start = int(np.round(x_range[0] / self.dx))
            x_end = int(np.round(x_range[1] / self.dx))
            y_start = int(np.round(y_range[0] / self.dy))
            y_end = int(np.round(y_range[1] / self.dy))

            if x_start == self.nx:
                x_start = self.nx - 1
            if y_start == self.ny:
                y_start = self.ny - 1

            if x_start == x_end:
                x_end = x_start + 1
            if y_start == y_end:
                y_end = y_start + 1

            # Fill the array with the material name
            device_array[x_start:x_end, y_start:y_end] = mat_name

        # Populate the doping array if doping profiles are provided
        if doping_profiles:
            for doping in doping_profiles:
                dop_type = doping["type"]
                x_range = doping["x"]
                y_range = doping["y"]

                # Convert ranges to grid indices
                x_start = int(np.round(x_range[0] / self.dx))
                x_end = int(np.round(x_range[1] / self.dx))
                y_start = int(np.round(y_range[0] / self.dy))
                y_end = int(np.round(y_range[1] / self.dy))

                if x_start == self.nx:
                    x_start = self.nx - 1
                if y_start == self.ny:
                    y_start = self.ny - 1

                if x_start == x_end:
                    x_end = x_start + 1
                if y_start == y_end:
                    y_end = y_start + 1

                concentration_input = doping["concentration"]
        
                # Check if concentration is a string (formula) or a numeric value
                if isinstance(concentration_input, str):
                    # Create a mesh grid for x and y coordinates
                    x_coords = np.linspace(x_range[0], x_range[1], x_end - x_start)
                    y_coords = np.linspace(y_range[0], y_range[1], y_end - y_start)
                    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
            
                    # Create a safe evaluation environment with allowed variables and functions
                    locals_dict = {'x': X, 'y': Y, 'np': np, 'exp': np.exp, 'sin': np.sin, 'cos': np.cos, 'sqrt': np.sqrt, 'pow': np.power}
            
                    # Evaluate the expression to get a 2D concentration array
                    try:
                        concentration_array = eval(concentration_input, {"__builtins__": {}}, locals_dict)
                    except Exception as e:
                        raise ValueError(f"Error evaluating concentration expression: {e}")
                else:
                    # For constant concentration, create a uniform array
                    concentration_array = np.ones((x_end - x_start, y_end - y_start)) * concentration_input
        
                # Adjust concentration based on doping type
                if dop_type == "n-type":
                    doping_value_array = concentration_array  # Positive for n-type
                elif dop_type == "p-type":
                    doping_value_array = -concentration_array  # Negative for p-type
                else:
                    raise ValueError(f"Unknown doping type: {dop_type}")
        
                # Fill the array with doping information
                doping_array[x_start:x_end, y_start:y_end] += 1e6 * doping_value_array

                # Store the arrays in the instance
                self.structure = device_array
                self.NB = doping_array

    def materialproperties(self, properties):
        self.Nc = 1e6 * properties.get_property(self.structure, "Nc")
        self.Nv = 1e6 * properties.get_property(self.structure, "Nv")
        self.me = self.h**2 / 2 / np.pi / self.kbT * (self.Nc / 2.0) ** (2.0 / 3.0)
        self.mh = self.h**2 / 2 / np.pi / self.kbT * (self.Nv / 2.0) ** (2.0 / 3.0)
        self.epsilon = self.ep0 * properties.get_property(self.structure, "epsilon")
        self.Eg = self.q * properties.get_property(self.structure, "Eg")
        self.xi = self.q * properties.get_property(self.structure, "xi")
        self.un = 1e-4 * properties.get_property(self.structure, "un")
        self.up = 1e-4 * properties.get_property(self.structure, "up")
        self.lambda_n = properties.get_property(self.structure, "lambda_n")
        self.lambda_p = properties.get_property(self.structure, "lambda_p")
        self.vsat_n = 1e-2 * properties.get_property(self.structure, "vsat_n") * (1 + self.lambda_n / self.L)
        self.vsat_p = 1e-2 * properties.get_property(self.structure, "vsat_p") * (1 + self.lambda_p / self.L)
        self.vt_n = 1e-2 * properties.get_property(self.structure, "vt_n")
        self.vt_p = 1e-2 * properties.get_property(self.structure, "vt_p")        
        self.ub_n = 0.5 * self.vt_n * self.L / self.kbT * self.q
        self.ub_p = 0.5 * self.vt_p * self.L / self.kbT * self.q
        self.beta_n = properties.get_property(self.structure, "beta_n")
        self.beta_p = properties.get_property(self.structure, "beta_p")
        self.ua_n = properties.get_property(self.structure, "ua_n")
        self.ua_p = properties.get_property(self.structure, "ua_p")
        self.eu_n = properties.get_property(self.structure, "eu_n")
        self.eu_p = properties.get_property(self.structure, "eu_p")
        self.ud_n = properties.get_property(self.structure, "ud_n")
        self.ud_p = properties.get_property(self.structure, "ud_p")
        self.ucs_n = properties.get_property(self.structure, "ucs_n")
        self.ucs_p = properties.get_property(self.structure, "ucs_p")
        self.nref = 1e6 * properties.get_property(self.structure, "nref")
        self.pref = 1e6 * properties.get_property(self.structure, "pref")
        self.ni = np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg / self.kbT / 2.0)
        self.tau_n = properties.get_property(self.structure, "tau_n")
        self.tau_p = properties.get_property(self.structure, "tau_p")
        self.etrap = self.q * properties.get_property(self.structure, "etrap")

    def assign_boundary_conditions(self, boundary_conditions):
        """
        Assign boundary conditions to the device structure.

        Parameters:
        - boundary_conditions: list of dictionaries specifying boundary conditions, e.g.,
          [{"name": "V1", "contact": "yes", "type": "Dirichlet", "value": 1.0, "barrier_height": 0.5, "x": (0, 1e-9), "y": (0, 2e-9)},
           {"name": "V2", "contact": "no", "type": "Neumann", "x": (1e-9, 2e-9), "y": (0, 1e-9)}]

        Returns:
        - A dictionary containing:
          - "boundary_array": A 2D numpy array with boundary values (without barrier height).
          - "boundary_array_with_barrier": A 2D numpy array with boundary values (including barrier height).
          - "dirichlet_positions": A dictionary with keys like "contact_1", "contact_2", etc., each containing the positions for a specific Dirichlet boundary.
        """

        # Initialize boundary arrays with Neumann
        boundary_array = np.full((self.nx, self.ny), -100.0)
        boundary_array_with_barrier = np.full((self.nx, self.ny), -100.0)
        contact_positions = {}

        # Apply boundary conditions
        contact_counter = 1
        for bc in boundary_conditions:
            contact_name = bc.get(
                "name"
            )  # Retrieve the name of the contact (e.g., "V1", "V2")
            contact = bc.get("contact", "no")  # Default is "no"
            bc_type = bc["type"]
            x_range = bc["x"]
            y_range = bc["y"]

            # Convert ranges to grid indices
            x_start = int(np.round(x_range[0] / self.dx))
            x_end = int(np.round(x_range[1] / self.dx))
            y_start = int(np.round(y_range[0] / self.dy))
            y_end = int(np.round(y_range[1] / self.dy))

            if x_start == self.nx:
                x_start = self.nx - 1
            if y_start == self.ny:
                y_start = self.ny - 1

            if x_start == x_end:
                x_end = x_start + 1
            if y_start == y_end:
                y_end = y_start + 1

            # Handle interface: force Neumann boundary
            if contact == "no":
                boundary_array[x_start:x_end, y_start:y_end] = -100.0
                boundary_array_with_barrier[x_start:x_end, y_start:y_end] = -100.0
                continue

            # Handle contact: allow Dirichlet or Neumann
            if bc_type == "Dirichlet":
                value = bc["value"]
                barrier_height = bc.get(
                    "barrier_height", 0
                )  # Default barrier height is 0

                boundary_array[x_start:x_end, y_start:y_end] = -self.q * value
                boundary_array_with_barrier[x_start:x_end, y_start:y_end] = self.q * (
                    -value + barrier_height
                )

                # Store positions of contact in a contact-specific key, using the contact name
                positions = [
                    (x, y) for x in range(x_start, x_end) for y in range(y_start, y_end)
                ]
                contact_key = f"{contact_name}"  # Unique key using contact name
                contact_positions[contact_key] = positions
                contact_counter += 1

            elif bc_type == "Neumann":
                if contact == "yes":
                    value = bc.get("value", -100.0)
                    boundary_array[x_start:x_end, y_start:y_end] = -self.q * value
                    boundary_array_with_barrier[x_start:x_end, y_start:y_end] = -100.0

                    # Store positions of contact in a contact-specific key
                    positions = [
                        (x, y)
                        for x in range(x_start, x_end)
                        for y in range(y_start, y_end)
                    ]
                    contact_key = f"{contact_name}"  # Unique key using contact name
                    contact_positions[contact_key] = positions
                    contact_counter += 1
                else:
                    boundary_array[x_start:x_end, y_start:y_end] = -100.0
                    boundary_array_with_barrier[x_start:x_end, y_start:y_end] = -100.0

            else:
                raise ValueError(f"Unknown boundary condition type: {bc_type}")

        # Store the arrays in the instance
        self.Ef_BC = boundary_array
        self.BC = boundary_array_with_barrier
        self.contact_positions = contact_positions

    def add_charge_region(self, region):
        name = region["name"]
        x_index = (
            int(np.round(region["x"][0] / self.dx)),
            int(np.round(region["x"][1] / self.dx)),
        )
        y_index = (
            int(np.round(region["y"][0] / self.dy)),
            int(np.round(region["y"][1] / self.dy)),
        )

        region_data = {"name": name, "x": x_index, "y": y_index}
        self.charge_regions.append(region_data)

    def Initialize(self):
        self.Ec = np.zeros((self.nx, self.ny))
        self.Ec[self.NB > 0] = -self.kbT * np.log(
            self.NB[self.NB > 0] / self.Nc[self.NB > 0]
        )
        self.Ec[self.NB == 0] = -self.kbT * np.log(
            self.ni[self.NB == 0] / self.Nc[self.NB == 0]
        )
        self.Ec[self.NB < 0] = self.Eg[self.NB < 0] + self.kbT * np.log(
            -self.NB[self.NB < 0] / self.Nv[self.NB < 0]
        )
        self.Ev = self.Ec - self.Eg
        self.Efn = np.zeros((self.nx, self.ny))
        self.Efp = np.zeros((self.nx, self.ny))
        self.Ef_BC = np.zeros((self.nx, self.ny))
        self.BC = -100.0 * np.ones((self.nx, self.ny))
        self.QCn = np.zeros((self.nx, self.ny))
        self.QCp = np.zeros((self.nx, self.ny))
        self.compute_parameter()
        self.getmatrix()       
        error = 1
        while error > 1e-4:
            self.Poisson()
            error = self.error
            print(f"Error: {error}")
        print("Initialization success!")
        self.n = self.Nc * np.exp((self.Efn - self.Ec) / self.kbT)
        self.p = self.Nv * np.exp((self.Ev - self.Efp) / self.kbT)

    def compute_parameter(self):
        # Define interface values for all material parameters
        # Permitivity
        self.ep_mi, self.ep_pi, self.ep_mj, self.ep_pj = self.compute_interface_values(self.epsilon)

        # Effective mass
        self.me_mi, self.me_pi, self.me_mj, self.me_pj = self.compute_interface_values(self.me)
        self.mh_mi, self.mh_pi, self.mh_mj, self.mh_pj = self.compute_interface_values(self.me)

        # Mobility
        self.un_mi, self.un_pi, self.un_mj, self.un_pj = self.compute_interface_values(self.un)
        self.up_mi, self.up_pi, self.up_mj, self.up_pj = self.compute_interface_values(self.up)
        self.ub_n_mi, self.ub_n_pi, self.ub_n_mj, self.ub_n_pj = self.compute_interface_values(self.ub_n)
        self.ub_p_mi, self.ub_p_pi, self.ub_p_mj, self.ub_p_pj = self.compute_interface_values(self.ub_p)
    
        # Density of states
        self.Nc_mi, self.Nc_pi, self.Nc_mj, self.Nc_pj = self.compute_interface_values(self.Nc)
        self.Nv_mi, self.Nv_pi, self.Nv_mj, self.Nv_pj = self.compute_interface_values(self.Nv)
    
        # Saturation velocities
        self.vsat_n_mi, self.vsat_n_pi, self.vsat_n_mj, self.vsat_n_pj = self.compute_interface_values(self.vsat_n)
        self.vsat_p_mi, self.vsat_p_pi, self.vsat_p_mj, self.vsat_p_pj = self.compute_interface_values(self.vsat_p)
    
        # Beta parameters
        self.beta_n_mi, self.beta_n_pi, self.beta_n_mj, self.beta_n_pj = self.compute_interface_values(self.beta_n)
        self.beta_p_mi, self.beta_p_pi, self.beta_p_mj, self.beta_p_pj = self.compute_interface_values(self.beta_p)
    
        # Mobility degradation parameters
        self.ua_n_mi, self.ua_n_pi, self.ua_n_mj, self.ua_n_pj = self.compute_interface_values(self.ua_n)
        self.eu_n_mi, self.eu_n_pi, self.eu_n_mj, self.eu_n_pj = self.compute_interface_values(self.eu_n)
        self.ud_n_mi, self.ud_n_pi, self.ud_n_mj, self.ud_n_pj = self.compute_interface_values(self.ud_n)
        self.ucs_n_mi, self.ucs_n_pi, self.ucs_n_mj, self.ucs_n_pj = self.compute_interface_values(self.ucs_n)
        self.nref_mi, self.nref_pi, self.nref_mj, self.nref_pj = self.compute_interface_values(self.nref)
        self.ua_p_mi, self.ua_p_pi, self.ua_p_mj, self.ua_p_pj = self.compute_interface_values(self.ua_p)
        self.eu_p_mi, self.eu_p_pi, self.eu_p_mj, self.eu_p_pj = self.compute_interface_values(self.eu_p)
        self.ud_p_mi, self.ud_p_pi, self.ud_p_mj, self.ud_p_pj = self.compute_interface_values(self.ud_p)
        self.ucs_p_mi, self.ucs_p_pi, self.ucs_p_mj, self.ucs_p_pj = self.compute_interface_values(self.ucs_p)
        self.pref_mi, self.pref_pi, self.pref_mj, self.pref_pj = self.compute_interface_values(self.pref)

    def getindex(self, i, j):
        m = j + i * self.ny
        if i < 0:
            m = -1
        elif i > self.nx - 1:
            m = -2
        if j < 0:
            m = -3
        elif j > self.ny - 1:
            m = -4
        return m

    def compute_interface_values(self, arr, pad_mode='edge'):
        """
        Compute interface values for a 2D array along all four directions
    
        Parameters:
            arr: 2D numpy array with shape (nx, ny)
            pad_mode: Padding mode for numpy.pad
    
        Returns:
            mi, pi, mj, pj: Interface values in negative x, positive x, 
                            negative y, and positive y directions
        """
        nx, ny = arr.shape
    
        # Create interface value arrays
        mi = np.zeros((nx, ny))
        pi = np.zeros((nx, ny))
        mj = np.zeros((nx, ny))
        pj = np.zeros((nx, ny))
    
        # For i=0 and i=nx-1, use the node values
        mi[0, :] = arr[0, :]
        pi[nx-1, :] = arr[nx-1, :]
    
        # For j=0 and j=ny-1, use the node values
        mj[:, 0] = arr[:, 0]
        pj[:, ny-1] = arr[:, ny-1]
    
        # For interior nodes, compute the average
        mi[1:, :] = 0.5 * (arr[:-1, :] + arr[1:, :])
        pi[:-1, :] = 0.5 * (arr[1:, :] + arr[:-1, :])
        mj[:, 1:] = 0.5 * (arr[:, :-1] + arr[:, 1:])
        pj[:, :-1] = 0.5 * (arr[:, 1:] + arr[:, :-1])
    
        return mi, pi, mj, pj

    def compute_spatial_gradients(self, arr):
        """
        Compute spatial gradients for a 2D array
    
        Parameters:
            arr: 2D numpy array with shape (nx, ny)
    
        Returns:
            dx_grad, dy_grad: Gradients in x and y directions
        """
        nx, ny = arr.shape
    
        # Initialize gradient arrays
        dx_grad = np.zeros((nx, ny))
        dy_grad = np.zeros((nx, ny))
    
        # Compute interior gradients using central difference
        dx_grad[1:-1, :] = (arr[2:, :] - arr[:-2, :]) / (2 * self.dx)
        dy_grad[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / (2 * self.dy)
    
        # Compute boundary gradients using forward/backward difference
        dx_grad[0, :] = (arr[1, :] - arr[0, :]) / self.dx
        dx_grad[-1, :] = (arr[-1, :] - arr[-2, :]) / self.dx
        dy_grad[:, 0] = (arr[:, 1] - arr[:, 0]) / self.dy
        dy_grad[:, -1] = (arr[:, -1] - arr[:, -2]) / self.dy
    
        return dx_grad, dy_grad

    def getmatrix(self):
        nx, ny = self.nx, self.ny
        self.D2 = lil_matrix((nx * ny, nx * ny))
        self.BD = lil_matrix((nx * ny, 1))
    
        # Create arrays to store interface properties
        ep_mi, ep_pi, ep_mj, ep_pj = self.ep_mi, self.ep_pi, self.ep_mj, self.ep_pj
    
        DEc_mi = np.zeros((nx, ny))
        DEc_pi = np.zeros((nx, ny))
        DEc_mj = np.zeros((nx, ny))
        DEc_pj = np.zeros((nx, ny))

        DEc_mi[0, :] = 0
    
        # Fill interior values (i>0)
        DEc_mi[1:, :] = self.xi[:-1, :] - self.xi[1:, :]
    
        # Fill boundary values (i=nx-1)
        DEc_pi[nx-1, :] = 0
    
        # Fill interior values (i<nx-1)
        DEc_pi[:nx-1, :] = self.xi[:nx-1, :] - self.xi[1:, :]
    
        # Fill boundary values (j=0)
        DEc_mj[:, 0] = 0
    
        # Fill interior values (j>0)
        DEc_mj[:, 1:] = self.xi[:, :-1] - self.xi[:, 1:]
    
        # Fill boundary values (j=ny-1)
        DEc_pj[:, ny-1] = 0
    
        # Fill interior values (j<ny-1)
        DEc_pj[:, :ny-1] = self.xi[:, :ny-1] - self.xi[:, 1:]

        for i in range(self.nx):
            for j in range(self.ny):

                m = self.getindex(i, j)
                self.D2[m, m] = (
                    -(ep_pi[i, j] + ep_mi[i, j]) / self.dx**2 - (ep_pj[i, j] + ep_mj[i, j]) / self.dy**2
                )
                self.BD[m, 0] = (ep_pi[i, j] * DEc_pi[i, j] - ep_mi[i, j] * DEc_mi[i, j]) / self.dx**2 + (
                    ep_pj[i, j] * DEc_pj[i, j] - ep_mj[i, j] * DEc_mj[i, j]
                ) / self.dy**2

                mx = self.getindex(i - 1, j)
                if mx >= 0:
                    self.D2[m, mx] = ep_mi[i, j] / self.dx**2
                elif mx == -1:
                    if self.BC[i, j] == -100.0:
                        self.D2[m, m] = self.D2[m, m] + ep_mi[i, j] / self.dx**2
                    else:
                        self.BD[m, 0] = (
                            self.BD[m, 0] - ep_mi[i, j] * self.BC[i, j] / self.dx**2
                        )

                mx = self.getindex(i + 1, j)
                if mx >= 0:
                    self.D2[m, mx] = ep_pi[i, j] / self.dx**2
                elif mx == -2:
                    if self.BC[i, j] == -100.0:
                        self.D2[m, m] = self.D2[m, m] + ep_pi[i, j] / self.dx**2
                    else:
                        self.BD[m, 0] = (
                            self.BD[m, 0] - ep_pi[i, j] * self.BC[i, j] / self.dx**2
                        )

                my = self.getindex(i, j - 1)
                if my >= 0:
                    self.D2[m, my] = ep_mj[i, j] / self.dy**2
                elif my == -3:
                    if self.BC[i, j] == -100.0:
                        self.D2[m, m] = self.D2[m, m] + ep_mj[i, j] / self.dy**2
                    else:
                        self.BD[m, 0] = (
                            self.BD[m, 0] - ep_mj[i, j] * self.BC[i, j] / self.dy**2
                        )

                my = self.getindex(i, j + 1)
                if my >= 0:
                    self.D2[m, my] = ep_pj[i, j] / self.dy**2
                elif my == -4:
                    if self.BC[i, j] == -100.0:
                        self.D2[m, m] = self.D2[m, m] + ep_pj[i, j] / self.dy**2
                    else:
                        self.BD[m, 0] = (
                            self.BD[m, 0] - ep_pj[i, j] * self.BC[i, j] / self.dy**2
                        )

        self.D2 = self.D2.tocsr()
        self.BD = self.BD.tocsr()

    def Poisson(self, damping=1):
        Ev = self.Ev.reshape(self.nx * self.ny)
        Ec = self.Ec.reshape(self.nx * self.ny)
        QCn = self.QCn.reshape(self.nx * self.ny)
        QCp = self.QCp.reshape(self.nx * self.ny)
        Efn = self.Efn.reshape(self.nx * self.ny)
        Efp = self.Efp.reshape(self.nx * self.ny)
        Nc = self.Nc.reshape(self.nx * self.ny)
        Nv = self.Nv.reshape(self.nx * self.ny)
        NB = self.NB.reshape(self.nx * self.ny, 1)

        if self.fermi == False:
            n = Nc * np.exp((Efn - Ec - QCn) / self.kbT)
            dn = n / self.kbT
            p = Nv * np.exp((Ev + QCp - Efp) / self.kbT)
            dp = p / self.kbT
        else:
            n = Nc * 2 / np.sqrt(np.pi) * fdk(k=0.5, phi=(Efn - Ec - QCn) / self.kbT)
            dn = (
                Nc
                * 2
                / np.sqrt(np.pi)
                * dfdk(k=0.5, phi=(Efn - Ec - QCn) / self.kbT)
                / self.kbT
            )
            p = Nv * 2 / np.sqrt(np.pi) * fdk(k=0.5, phi=(Ev + QCp - Efp) / self.kbT)
            dp = (
                Nv
                * 2
                / np.sqrt(np.pi)
                * dfdk(k=0.5, phi=(Ev + QCp - Efp) / self.kbT)
                / self.kbT
            )

        n_flat = csr_matrix(n.reshape(-1, 1))
        p_flat = csr_matrix(p.reshape(-1, 1))
        dn_diag = csr_matrix(np.diag(dn))
        dp_diag = csr_matrix(np.diag(dp))
        NB_flat = csr_matrix(NB)
        Ec_flat = csr_matrix(Ec.reshape(-1, 1))
        LHS = self.D2 - self.q**2 * (dn_diag + dp_diag)
        RHS = -self.D2 @ Ec_flat + self.BD - self.q**2 * (n_flat - p_flat - NB_flat)
        deltaEc = spsolve(LHS, RHS).real.reshape(-1, 1)
        self.error = np.max(np.abs(deltaEc)) / self.q
        Ec_flat = Ec_flat.toarray()
        Ecnew = Ec_flat + damping * deltaEc
        self.Ec = Ecnew.reshape(self.nx, self.ny)
        self.Ev = self.Ec - self.Eg

    def mobility(self, u0, ub, n, E_ver, E_par, ua, eu, ud, ucs, nref, vsat, beta):
        Eeff = np.abs(E_ver / self.q)
        udd = u0 / (1 + ua * (Eeff) ** eu + ud / (0.5 * (1 + n/nref)) ** ucs)
        ueff = udd * ub / (udd + ub)
        Eeff = np.abs(E_par / self.q)
        u = ueff / (1 + (ueff * Eeff / vsat) ** beta) ** (1 / beta)
        return u

    def recombination(self, i, j):
        SRH = (self.n[i, j] * self.p[i, j] - self.ni[i, j] ** 2) / (
            self.tau_p[i, j]
            * (self.n[i, j] + self.ni[i, j] * np.exp(self.etrap[i, j] / self.kbT))
            + self.tau_n[i, j]
            * (self.p[i, j] + self.ni[i, j] * np.exp(-self.etrap[i, j] / self.kbT))
        )
        return SRH

    def DriftDiffusion(self, recombination=False):
        nx, ny = self.nx, self.ny
        self.nLHS = lil_matrix((nx * ny, nx * ny))
        self.nRHS = lil_matrix((nx * ny, 1))
        self.pLHS = lil_matrix((nx * ny, nx * ny))
        self.pRHS = lil_matrix((nx * ny, 1))

        total_points = nx * ny
    
        # Define interface values for all material parameters
        # Mobility
        un_mi, un_pi, un_mj, un_pj = self.un_mi, self.un_pi, self.un_mj, self.un_pj
        up_mi, up_pi, up_mj, up_pj = self.up_mi, self.up_pi, self.up_mj, self.up_pj 
        ub_n_mi, ub_n_pi, ub_n_mj, ub_n_pj = self.ub_n_mi, self.ub_n_pi, self.ub_n_mj, self.ub_n_pj
        ub_p_mi, ub_p_pi, ub_p_mj, ub_p_pj = self.ub_p_mi, self.ub_p_pi, self.ub_p_mj, self.ub_p_pj 
    
        # Density of states
        Nc_mi, Nc_pi, Nc_mj, Nc_pj = self.Nc_mi, self.Nc_pi, self.Nc_mj, self.Nc_pj
        Nv_mi, Nv_pi, Nv_mj, Nv_pj = self.Nv_mi, self.Nv_pi, self.Nv_mj, self.Nv_pj
          
        # Saturation velocities
        vsat_n_mi, vsat_n_pi, vsat_n_mj, vsat_n_pj = self.vsat_n_mi, self.vsat_n_pi, self.vsat_n_mj, self.vsat_n_pj
        vsat_p_mi, vsat_p_pi, vsat_p_mj, vsat_p_pj = self.vsat_p_mi, self.vsat_p_pi, self.vsat_p_mj, self.vsat_p_pj
    
        # Beta parameters
        beta_n_mi, beta_n_pi, beta_n_mj, beta_n_pj = self.beta_n_mi, self.beta_n_pi, self.beta_n_mj, self.beta_n_pj
        beta_p_mi, beta_p_pi, beta_p_mj, beta_p_pj = self.beta_p_mi, self.beta_p_pi, self.beta_p_mj, self.beta_p_pj
    
        # Mobility degradation parameters
        ua_n_mi, ua_n_pi, ua_n_mj, ua_n_pj = self.ua_n_mi, self.ua_n_pi, self.ua_n_mj, self.ua_n_pj
        eu_n_mi, eu_n_pi, eu_n_mj, eu_n_pj = self.eu_n_mi, self.eu_n_pi, self.eu_n_mj, self.eu_n_pj
        ud_n_mi, ud_n_pi, ud_n_mj, ud_n_pj = self.ud_n_mi, self.ud_n_pi, self.ud_n_mj, self.ud_n_pj
        ucs_n_mi, ucs_n_pi, ucs_n_mj, ucs_n_pj = self.ucs_n_mi, self.ucs_n_pi, self.ucs_n_mj, self.ucs_n_pj
        nref_mi, nref_pi, nref_mj, nref_pj = self.nref_mi, self.nref_pi, self.nref_mj, self.nref_pj
        ua_p_mi, ua_p_pi, ua_p_mj, ua_p_pj = self.ua_p_mi, self.ua_p_pi, self.ua_p_mj, self.ua_p_pj
        eu_p_mi, eu_p_pi, eu_p_mj, eu_p_pj = self.eu_p_mi, self.eu_p_pi, self.eu_p_mj, self.eu_p_pj
        ud_p_mi, ud_p_pi, ud_p_mj, ud_p_pj = self.ud_p_mi, self.ud_p_pi, self.ud_p_mj, self.ud_p_pj
        ucs_p_mi, ucs_p_pi, ucs_p_mj, ucs_p_pj = self.ucs_p_mi, self.ucs_p_pi, self.ucs_p_mj, self.ucs_p_pj
        pref_mi, pref_pi, pref_mj, pref_pj = self.pref_mi, self.pref_pi, self.pref_mj, self.pref_pj  

        # Energy bands
        Ec_mi, Ec_pi, Ec_mj, Ec_pj = self.compute_interface_values(self.Ec+self.QCn)
        Ev_mi, Ev_pi, Ev_mj, Ev_pj = self.compute_interface_values(self.Ev+self.QCp)

        # Quasi-Fermi levels
        Efn_mi, Efn_pi, Efn_mj, Efn_pj = self.compute_interface_values(self.Efn)
        Efp_mi, Efp_pi, Efp_mj, Efp_pj = self.compute_interface_values(self.Efp)

        # midpoint n and p
        if self.fermi == False:
            n_mi = Nc_mi * np.exp((Efn_mi - Ec_mi)/self.kbT)
            n_pi = Nc_pi * np.exp((Efn_pi - Ec_pi)/self.kbT)
            n_mj = Nc_mj * np.exp((Efn_mj - Ec_mj)/self.kbT)
            n_pi = Nc_pj * np.exp((Efn_pj - Ec_pj)/self.kbT)
            p_mi = Nv_mi * np.exp((Ev_mi - Efp_mi)/self.kbT)
            p_pi = Nv_pi * np.exp((Ev_pi - Efp_pi)/self.kbT)
            p_mj = Nv_mj * np.exp((Ev_mj - Efp_mj)/self.kbT)
            p_pi = Nv_pj * np.exp((Ev_pj - Efp_pj)/self.kbT)
        else:                        
            n_mi = fdk(k=0.5, phi=(Efn_mi.reshape(self.nx * self.ny) - Ec_mi.reshape(self.nx * self.ny))/self.kbT)
            n_pi = fdk(k=0.5, phi=(Efn_pi.reshape(self.nx * self.ny) - Ec_pi.reshape(self.nx * self.ny))/self.kbT)
            n_mj = fdk(k=0.5, phi=(Efn_mj.reshape(self.nx * self.ny) - Ec_mj.reshape(self.nx * self.ny))/self.kbT)
            n_pj = fdk(k=0.5, phi=(Efn_pj.reshape(self.nx * self.ny) - Ec_pj.reshape(self.nx * self.ny))/self.kbT)
            p_mi = fdk(k=0.5, phi=(Ev_mi.reshape(self.nx * self.ny) - Efp_mi.reshape(self.nx * self.ny))/self.kbT)
            p_pi = fdk(k=0.5, phi=(Ev_pi.reshape(self.nx * self.ny) - Efp_pi.reshape(self.nx * self.ny))/self.kbT)
            p_mj = fdk(k=0.5, phi=(Ev_mj.reshape(self.nx * self.ny) - Efp_mj.reshape(self.nx * self.ny))/self.kbT)
            p_pj = fdk(k=0.5, phi=(Ev_pj.reshape(self.nx * self.ny) - Efp_pj.reshape(self.nx * self.ny))/self.kbT) 
            
            n_mi = Nc_mi * 2 / np.sqrt(np.pi) * n_mi.reshape(self.nx,self.ny) 
            n_pi = Nc_pi * 2 / np.sqrt(np.pi) * n_pi.reshape(self.nx,self.ny) 
            n_mj = Nc_mj * 2 / np.sqrt(np.pi) * n_mj.reshape(self.nx,self.ny)      
            n_pj = Nc_pj * 2 / np.sqrt(np.pi) * n_pj.reshape(self.nx,self.ny) 
            p_mi = Nv_mi * 2 / np.sqrt(np.pi) * p_mi.reshape(self.nx,self.ny) 
            p_pi = Nv_pi * 2 / np.sqrt(np.pi) * p_pi.reshape(self.nx,self.ny) 
            p_mj = Nv_mj * 2 / np.sqrt(np.pi) * p_mj.reshape(self.nx,self.ny) 
            p_pj = Nv_pj * 2 / np.sqrt(np.pi) * p_pj.reshape(self.nx,self.ny) 

        # Handle boundary conditions for energy bands and quasi-Fermi levels
        # Create masks for boundaries with Dirichlet conditions
        bc_mask = self.BC != -100.0
        ef_bc_mask = self.Ef_BC != -100.0
    
        # Apply boundary conditions to energy bands
        # Left boundary (i=0)
        if np.any(bc_mask[0, :]):
            mask = bc_mask[0, :]
            Ec_mi[0, mask] = 0.5 * (self.Ec[0, mask] + self.QCn[0, mask] + self.BC[0, mask])
            Ev_mi[0, mask] = 0.5 * (self.Ev[0, mask] + self.QCp[0, mask] + self.BC[0, mask] - self.Eg[0, mask])
    
        # Right boundary (i=nx-1)
        if np.any(bc_mask[nx-1, :]):
            mask = bc_mask[nx-1, :]
            Ec_pi[nx-1, mask] = 0.5 * (self.Ec[nx-1, mask] + self.QCn[nx-1, mask] + self.BC[nx-1, mask])
            Ev_pi[nx-1, mask] = 0.5 * (self.Ev[nx-1, mask] + self.QCp[nx-1, mask] + self.BC[nx-1, mask] - self.Eg[nx-1, mask])
    
        # Bottom boundary (j=0)
        if np.any(bc_mask[:, 0]):
            mask = bc_mask[:, 0]
            Ec_mj[mask, 0] = 0.5 * (self.Ec[mask, 0] + self.QCn[mask, 0] + self.BC[mask, 0])
            Ev_mj[mask, 0] = 0.5 * (self.Ev[mask, 0] + self.QCp[mask, 0] + self.BC[mask, 0] - self.Eg[mask, 0])
    
        # Top boundary (j=ny-1)
        if np.any(bc_mask[:, ny-1]):
            mask = bc_mask[:, ny-1]
            Ec_pj[mask, ny-1] = 0.5 * (self.Ec[mask, ny-1] + self.QCn[mask, ny-1] + self.BC[mask, ny-1])
            Ev_pj[mask, ny-1] = 0.5 * (self.Ev[mask, ny-1] + self.QCp[mask, ny-1] + self.BC[mask, ny-1] - self.Eg[mask, ny-1])
    
        # Apply boundary conditions to quasi-Fermi levels
        # Left boundary (i=0)
        if np.any(ef_bc_mask[0, :]):
            mask = ef_bc_mask[0, :]
            Efn_mi[0, mask] = self.Ef_BC[0, mask]
            Efp_mi[0, mask] = self.Ef_BC[0, mask]
    
        # Right boundary (i=nx-1)
        if np.any(ef_bc_mask[nx-1, :]):
            mask = ef_bc_mask[nx-1, :]
            Efn_pi[nx-1, mask] = self.Ef_BC[nx-1, mask]
            Efp_pi[nx-1, mask] = self.Ef_BC[nx-1, mask]
    
        # Bottom boundary (j=0)
        if np.any(ef_bc_mask[:, 0]):
            mask = ef_bc_mask[:, 0]
            Efn_mj[mask, 0] = self.Ef_BC[mask, 0]
            Efp_mj[mask, 0] = self.Ef_BC[mask, 0]
    
        # Top boundary (j=ny-1)
        if np.any(ef_bc_mask[:, ny-1]):
            mask = ef_bc_mask[:, ny-1]
            Efn_pj[mask, ny-1] = self.Ef_BC[mask, ny-1]
            Efp_pj[mask, ny-1] = self.Ef_BC[mask, ny-1]

        # Calculate electric fields at interfaces
        # Compute energy gradients
        dEc_dx, dEc_dy = self.compute_spatial_gradients(self.Ec)
        dEv_dx, dEv_dy = self.compute_spatial_gradients(self.Ev)
    
        # Interface electric fields for mobility calculations
        # Calculate E_ver and E_par for each interface
        # For edge interfaces, average the adjacent gradients
    
        # mi interfaces (normal is x, parallel is y)
        E_ver_mi = np.zeros((nx, ny))
        E_ver_mi[1:, :] = 0.5 * (dEc_dy[:-1, :] + dEc_dy[1:, :])
        En_par_mi = np.zeros((nx, ny))
        En_par_mi[1:, :] = (self.Efn[1:, :] - self.Efn[:-1, :]) / self.dx
        Ep_par_mi = np.zeros((nx, ny))
        Ep_par_mi[1:, :] = (self.Efp[1:, :] - self.Efp[:-1, :]) / self.dx
    
        # Similar calculations for other interfaces
        E_ver_pi = np.zeros((nx, ny))
        E_ver_pi[:-1, :] = 0.5 * (dEc_dy[:-1, :] + dEc_dy[1:, :])
        En_par_pi = np.zeros((nx, ny))
        En_par_pi[:-1, :] = (self.Efn[1:, :] - self.Efn[:-1, :]) / self.dx
        Ep_par_pi = np.zeros((nx, ny))
        Ep_par_pi[:-1, :] = (self.Efp[1:, :] - self.Efp[:-1, :]) / self.dx
    
        E_ver_mj = np.zeros((nx, ny))
        E_ver_mj[:, 1:] = 0.5 * (dEc_dx[:, :-1] + dEc_dx[:, 1:])
        En_par_mj = np.zeros((nx, ny))
        En_par_mj[:, 1:] = (self.Efn[:, 1:] - self.Efn[:, :-1]) / self.dy
        Ep_par_mj = np.zeros((nx, ny))
        Ep_par_mj[:, 1:] = (self.Efp[:, 1:] - self.Efp[:, :-1]) / self.dy
    
        E_ver_pj = np.zeros((nx, ny))
        E_ver_pj[:, :-1] = 0.5 * (dEc_dx[:, :-1] + dEc_dx[:, 1:])
        En_par_pj = np.zeros((nx, ny))
        En_par_pj[:, :-1] = (self.Efn[:, 1:] - self.Efn[:, :-1]) / self.dy
        Ep_par_pj = np.zeros((nx, ny))
        Ep_par_pj[:, :-1] = (self.Efp[:, 1:] - self.Efp[:, :-1]) / self.dy

    
        # Calculate mobility at each interface using vectorized mobility function
        un_mi = self.mobility(un_mi, ub_n_mi, n_mi, E_ver_mi, En_par_mi, ua_n_mi, eu_n_mi, ud_n_mi, ucs_n_mi, nref_mi, vsat_n_mi, beta_n_mi)
        un_pi = self.mobility(un_pi, ub_n_pi, n_pi, E_ver_pi, En_par_pi, ua_n_pi, eu_n_pi, ud_n_pi, ucs_n_pi, nref_pi, vsat_n_pi, beta_n_pi)
        un_mj = self.mobility(un_mj, ub_n_mj, n_mj, E_ver_mj, En_par_mj, ua_n_mj, eu_n_mj, ud_n_mj, ucs_n_mj, nref_mj, vsat_n_mj, beta_n_mj)
        un_pj = self.mobility(un_pj, ub_n_pj, n_pj, E_ver_pj, En_par_pj, ua_n_pj, eu_n_pj, ud_n_pj, ucs_n_pj, nref_pj, vsat_n_pj, beta_n_pj)
    
        # Similar calculations for hole mobility
        up_mi = self.mobility(up_mi, ub_p_mi, p_mi, E_ver_mi, Ep_par_mi, ua_p_mi, eu_p_mi, ud_p_mi, ucs_p_mi, pref_mi, vsat_p_mi, beta_p_mi)
        up_pi = self.mobility(up_pi, ub_p_pi, p_pi, E_ver_pi, Ep_par_pi, ua_p_pi, eu_p_pi, ud_p_pi, ucs_p_pi, pref_pi, vsat_p_pi, beta_p_pi)
        up_mj = self.mobility(up_mj, ub_p_mj, p_mj, E_ver_mj, Ep_par_mj, ua_p_mj, eu_p_mj, ud_p_mj, ucs_p_mj, pref_mj, vsat_p_mj, beta_p_mj)
        up_pj = self.mobility(up_pj, ub_p_pj, p_pj, E_ver_pj, Ep_par_pj, ua_p_pj, eu_p_pj, ud_p_pj, ucs_p_pj, pref_pj, vsat_p_pj, beta_p_pj)
    

        for i in range(self.nx):
            for j in range(self.ny):
                if self.fermi == False:
                    an_mi = self.kbT * un_mi[i,j] * Nc_mi[i,j] * np.exp(-Ec_mi[i,j] / self.kbT)
                    an_pi = self.kbT * un_pi[i,j] * Nc_pi[i,j] * np.exp(-Ec_pi[i,j] / self.kbT)
                    an_mj = self.kbT * un_mj[i,j] * Nc_mj[i,j] * np.exp(-Ec_mj[i,j] / self.kbT)
                    an_pj = self.kbT * un_pj[i,j] * Nc_pj[i,j] * np.exp(-Ec_pj[i,j] / self.kbT)
                    ap_mi = -self.kbT * up_mi[i,j] * Nv_mi[i,j] * np.exp(Ev_mi[i,j] / self.kbT)
                    ap_pi = -self.kbT * up_pi[i,j] * Nv_pi[i,j] * np.exp(Ev_pi[i,j] / self.kbT)
                    ap_mj = -self.kbT * up_mj[i,j] * Nv_mj[i,j] * np.exp(Ev_mj[i,j] / self.kbT)
                    ap_pj = -self.kbT * up_pj[i,j] * Nv_pj[i,j] * np.exp(Ev_pj[i,j] / self.kbT)
                else:
                    an_mi = (
                        self.kbT
                        * un_mi[i,j]
                        * Nc_mi[i,j]
                        * 2
                        / np.sqrt(np.pi)
                        * fdk(k=0.5, phi=(Efn_mi[i,j] - Ec_mi[i,j]) / self.kbT)
                        / np.exp(Efn_mi[i,j] / self.kbT)
                    )
                    an_pi = (
                        self.kbT
                        * un_pi[i,j]
                        * Nc_pi[i,j]
                        * 2
                        / np.sqrt(np.pi)
                        * fdk(k=0.5, phi=(Efn_pi[i,j] - Ec_pi[i,j]) / self.kbT)
                        / np.exp(Efn_pi[i,j] / self.kbT)
                    )
                    an_mj = (
                        self.kbT
                        * un_mj[i,j]
                        * Nc_mj[i,j]
                        * 2
                        / np.sqrt(np.pi)
                        * fdk(k=0.5, phi=(Efn_mj[i,j] - Ec_mj[i,j]) / self.kbT)
                        / np.exp(Efn_mj[i,j] / self.kbT)
                    )
                    an_pj = (
                        self.kbT
                        * un_pj[i,j]
                        * Nc_pj[i,j]
                        * 2
                        / np.sqrt(np.pi)
                        * fdk(k=0.5, phi=(Efn_pj[i,j] - Ec_pj[i,j]) / self.kbT)
                        / np.exp(Efn_pj[i,j] / self.kbT)
                    )
                    ap_mi = (
                        -self.kbT
                        * up_mi[i,j]
                        * Nv_mi[i,j]
                        * 2
                        / np.sqrt(np.pi)
                        * fdk(k=0.5, phi=(Ev_mi[i,j] - Efp_mi[i,j]) / self.kbT)
                        / np.exp(-Efp_mi[i,j] / self.kbT)
                    )
                    ap_pi = (
                        -self.kbT
                        * up_pi[i,j]
                        * Nv_pi[i,j]
                        * 2
                        / np.sqrt(np.pi)
                        * fdk(k=0.5, phi=(Ev_pi[i,j] - Efp_pi[i,j]) / self.kbT)
                        / np.exp(-Efp_pi[i,j] / self.kbT)
                    )
                    ap_mj = (
                        -self.kbT
                        * up_mj[i,j]
                        * Nv_mj[i,j]
                        * 2
                        / np.sqrt(np.pi)
                        * fdk(k=0.5, phi=(Ev_mj[i,j] - Efp_mj[i,j]) / self.kbT)
                        / np.exp(-Efp_mj[i,j] / self.kbT)
                    )
                    ap_pj = (
                        -self.kbT
                        * up_pj[i,j]
                        * Nv_pj[i,j]
                        * 2
                        / np.sqrt(np.pi)
                        * fdk(k=0.5, phi=(Ev_pj[i,j] - Efp_pj[i,j]) / self.kbT)
                        / np.exp(-Efp_pj[i,j] / self.kbT)
                    )

                m = self.getindex(i, j)
                self.nLHS[m, m] = (
                    -(an_mi + an_pi) / self.dx**2 - (an_mj + an_pj) / self.dy**2
                )
                self.pLHS[m, m] = (
                    -(ap_mi + ap_pi) / self.dx**2 - (ap_mj + ap_pj) / self.dy**2
                )

                if recombination == True:
                    R = self.recombination(i, j)
                    self.nRHS[m, 0] = self.q * R
                    self.pRHS[m, 0] = -self.q * R

                mx = self.getindex(i - 1, j)
                if mx >= 0:
                    self.nLHS[m, mx] = an_mi / self.dx**2
                    self.pLHS[m, mx] = ap_mi / self.dx**2
                elif mx == -1:
                    if self.Ef_BC[i, j] == -100.0:
                        self.nLHS[m, m] = self.nLHS[m, m] + an_mi / self.dx**2
                        self.pLHS[m, m] = self.pLHS[m, m] + ap_mi / self.dx**2
                    else:
                        self.nRHS[m, 0] = self.nRHS[
                            m, 0
                        ] - an_mi / self.dx**2 * np.exp(self.Ef_BC[i, j] / self.kbT)
                        self.pRHS[m, 0] = self.pRHS[
                            m, 0
                        ] - ap_mi / self.dx**2 * np.exp(-self.Ef_BC[i, j] / self.kbT)

                mx = self.getindex(i + 1, j)
                if mx >= 0:
                    self.nLHS[m, mx] = an_pi / self.dx**2
                    self.pLHS[m, mx] = ap_pi / self.dx**2
                elif mx == -2:
                    if self.Ef_BC[i, j] == -100.0:
                        self.nLHS[m, m] = self.nLHS[m, m] + an_pi / self.dx**2
                        self.pLHS[m, m] = self.pLHS[m, m] + ap_pi / self.dx**2
                    else:
                        self.nRHS[m, 0] = self.nRHS[
                            m, 0
                        ] - an_pi / self.dx**2 * np.exp(self.Ef_BC[i, j] / self.kbT)
                        self.pRHS[m, 0] = self.pRHS[
                            m, 0
                        ] - ap_pi / self.dx**2 * np.exp(-self.Ef_BC[i, j] / self.kbT)

                my = self.getindex(i, j - 1)
                if my >= 0:
                    self.nLHS[m, my] = an_mj / self.dy**2
                    self.pLHS[m, my] = ap_mj / self.dy**2
                elif my == -3:
                    if self.Ef_BC[i, j] == -100.0:
                        self.nLHS[m, m] = self.nLHS[m, m] + an_mj / self.dy**2
                        self.pLHS[m, m] = self.pLHS[m, m] + ap_mj / self.dy**2
                    else:
                        self.nRHS[m, 0] = self.nRHS[
                            m, 0
                        ] - an_mj / self.dy**2 * np.exp(self.Ef_BC[i, j] / self.kbT)
                        self.pRHS[m, 0] = self.pRHS[
                            m, 0
                        ] - ap_mj / self.dy**2 * np.exp(-self.Ef_BC[i, j] / self.kbT)

                my = self.getindex(i, j + 1)
                if my >= 0:
                    self.nLHS[m, my] = an_pj / self.dy**2
                    self.pLHS[m, my] = ap_pj / self.dy**2
                elif my == -4:
                    if self.Ef_BC[i, j] == -100.0:
                        self.nLHS[m, m] = self.nLHS[m, m] + an_pj / self.dy**2
                        self.pLHS[m, m] = self.pLHS[m, m] + ap_pj / self.dy**2
                    else:
                        self.nRHS[m, 0] = self.nRHS[
                            m, 0
                        ] - an_pj / self.dy**2 * np.exp(self.Ef_BC[i, j] / self.kbT)
                        self.pRHS[m, 0] = self.pRHS[
                            m, 0
                        ] - ap_pj / self.dy**2 * np.exp(-self.Ef_BC[i, j] / self.kbT)

        self.nLHS = self.nLHS.tocsr()
        self.pLHS = self.pLHS.tocsr()
        self.nRHS = self.nRHS.tocsr()
        self.pRHS = self.pRHS.tocsr()

        phi_n = spsolve(self.nLHS, self.nRHS)
        phi_p = spsolve(self.pLHS, self.pRHS)
        phi_n[phi_n <= 0] = (
            self.ni.reshape(self.nx * self.ny)
            / self.Nc.reshape(self.nx * self.ny)
            * np.exp((self.Ec.reshape(self.nx * self.ny) + self.QCn.reshape(self.nx * self.ny)) / self.kbT)
        )[phi_n <= 0]
        phi_p[phi_p <= 0] = (
            self.ni.reshape(self.nx * self.ny)
            / self.Nv.reshape(self.nx * self.ny)
            * np.exp(-(self.Ev.reshape(self.nx * self.ny) + self.QCp.reshape(self.nx * self.ny)) / self.kbT)
        )[phi_p <= 0]
        Efn = self.kbT * np.log(phi_n)
        Efp = -self.kbT * np.log(phi_p)
        if self.fermi == False:
            n = self.Nc.reshape(self.nx * self.ny) * np.exp(
                (Efn - self.Ec.reshape(self.nx * self.ny) - self.QCn.reshape(self.nx * self.ny)) / self.kbT
            )
            p = self.Nv.reshape(self.nx * self.ny) * np.exp(
                (self.Ev.reshape(self.nx * self.ny) + self.QCp.reshape(self.nx * self.ny) - Efp) / self.kbT
            )
        else:
            n = (
                self.Nc.reshape(self.nx * self.ny)
                * 2
                / np.sqrt(np.pi)
                * fdk(k=0.5, phi=(Efn - self.Ec.reshape(self.nx * self.ny) - self.QCn.reshape(self.nx * self.ny)) / self.kbT)
            )
            p = (
                self.Nv.reshape(self.nx * self.ny)
                * 2
                / np.sqrt(np.pi)
                * fdk(k=0.5, phi=(self.Ev.reshape(self.nx * self.ny) + self.QCp.reshape(self.nx * self.ny) - Efp) / self.kbT)
            )

        self.Efn = Efn.reshape(self.nx, self.ny)
        self.Efp = Efp.reshape(self.nx, self.ny)
        self.n = n.reshape(self.nx, self.ny)
        self.p = p.reshape(self.nx, self.ny)

        grad_Ec = np.gradient(self.Ec, self.dx, self.dy)
        grad_Ev = np.gradient(self.Ev, self.dx, self.dy)
        grad_Efn = np.gradient(self.Efn, self.dx, self.dy)
        grad_Efp = np.gradient(self.Efp, self.dx, self.dy)

        unx = self.mobility(
            self.un,
            self.ub_n,
            self.n,
            grad_Ec[1],
            grad_Efn[0],
            self.ua_n,
            self.eu_n,
            self.ud_n,
            self.ucs_n,
            self.nref,
            self.vsat_n,
            self.beta_n
        )
        uny = self.mobility(
            self.un,
            self.ub_n,
            self.n,
            grad_Ec[0],
            grad_Efn[1],
            self.ua_n,
            self.eu_n,
            self.ud_n,
            self.ucs_n,
            self.nref,
            self.vsat_n,
            self.beta_n
        )
        upx = self.mobility(
            self.up,
            self.ub_p,
            self.p,
            grad_Ev[1],
            grad_Efp[0],
            self.ua_p,
            self.eu_p,
            self.ud_p,
            self.ucs_p,
            self.pref,
            self.vsat_p,
            self.beta_p
        )
        upy = self.mobility(
            self.up,
            self.ub_p,
            self.p,
            grad_Ev[0],
            grad_Efp[1],
            self.ua_p,
            self.eu_p,
            self.ud_p,
            self.ucs_p,
            self.pref,
            self.vsat_p,
            self.beta_p
        )

        self.vnx = -unx * grad_Efn[0] / self.q
        self.vny = -uny * grad_Efn[1] / self.q
        self.vpx = upx * grad_Efp[0] / self.q
        self.vpy = upy * grad_Efp[1] / self.q
        self.Jnx = -self.vnx * self.q * self.n
        self.Jny = -self.vny * self.q * self.n
        self.Jpx = self.vpx * self.q * self.p
        self.Jpy = self.vpy * self.q * self.p

    def quantum_getmatrix(self):
        self.He = lil_matrix((self.nx * self.ny, self.nx * self.ny))
        self.Hh = lil_matrix((self.nx * self.ny, self.nx * self.ny))
        self.Qe_RHS = lil_matrix((self.nx * self.ny, 1))
        self.Qe_RHS[:, 0] = 1
        self.Qh_RHS = lil_matrix((self.nx * self.ny, 1))
        self.Qh_RHS[:, 0] = 1

        me_mi, me_pi, me_mj, me_pj = self.me_mi, self.me_pi, self.me_mj, self.me_pj 
        mh_mi, mh_pi, mh_mj, mh_pj = self.mh_mi, self.mh_pi, self.mh_mj, self.mh_pj  

        for i in range(self.nx):
            for j in range(self.ny):

                m = self.getindex(i, j)
                self.He[m, m] = (
                    -(1 / me_mi[i,j] + 1 / me_pi[i,j]) / self.dx**2
                    - (1 / me_mj[i,j] + 1 / me_pj[i,j]) / self.dy**2
                )
                self.Hh[m, m] = (
                    -(1 / mh_mi[i,j] + 1 / mh_pi[i,j]) / self.dx**2
                    - (1 / mh_mj[i,j] + 1 / mh_pj[i,j]) / self.dy**2
                )

                mx = self.getindex(i - 1, j)
                if mx >= 0:
                    self.He[m, mx] = 1 / me_mi[i,j] / self.dx**2
                    self.Hh[m, mx] = 1 / mh_mi[i,j] / self.dx**2
                elif mx == -1:
                    self.He[m, m] = self.He[m, m] + 1 / me_mi[i,j] / self.dx**2
                    self.Hh[m, m] = self.Hh[m, m] + 1 / mh_mi[i,j] / self.dx**2

                mx = self.getindex(i + 1, j)
                if mx >= 0:
                    self.He[m, mx] = 1 / me_pi[i,j] / self.dx**2
                    self.Hh[m, mx] = 1 / mh_pi[i,j] / self.dx**2
                elif mx == -2:
                    self.He[m, m] = self.He[m, m] + 1 / me_pi[i,j] / self.dx**2
                    self.Hh[m, m] = self.Hh[m, m] + 1 / me_pi[i,j] / self.dx**2

                my = self.getindex(i, j - 1)
                if my >= 0:
                    self.He[m, my] = 1 / me_mj[i,j] / self.dy**2
                    self.Hh[m, my] = 1 / mh_mj[i,j] / self.dy**2
                elif my == -3:
                    self.He[m, m] = self.He[m, m] + 1 / me_mj[i,j] / self.dy**2
                    self.Hh[m, m] = self.Hh[m, m] + 1 / mh_mj[i,j] / self.dy**2

                my = self.getindex(i, j + 1)
                if my >= 0:
                    self.He[m, my] = 1 / me_pj[i,j] / self.dy**2
                    self.Hh[m, my] = 1 / mh_pj[i,j] / self.dy**2
                elif my == -4:
                    self.He[m, m] = self.He[m, m] + 1 / me_pj[i,j] / self.dy**2
                    self.Hh[m, m] = self.Hh[m, m] + 1 / mh_pj[i,j] / self.dy**2

        self.He = self.He.tocsr()
        self.Qe_RHS = self.Qe_RHS.tocsr()
        self.Hh = self.Hh.tocsr()
        self.Qh_RHS = self.Qh_RHS.tocsr()

    def equantum(self, Kq=0):
        Vq = self.Ec.copy()
        Vq = Vq.reshape(self.nx * self.ny) 
        Vmin = np.min(Vq) if np.min(Vq)<0 else 0      
        Vq = csr_matrix(np.diag(Vq - Vmin + self.q*Kq))
        self.Qe_LHS = -0.5 * self.hbar**2 * self.He + Vq
        u_Q = spsolve(self.Qe_LHS, self.Qe_RHS)
        self.u_Q = u_Q.reshape(self.nx, self.ny)
        Wc = 1 / self.u_Q + Vmin - self.q*Kq  
        self.QCn = Wc - self.Ec
        self.QCp = self.QCn

    def hquantum(self, Kq=0):
        Vq = -self.Ev.copy()
        Vq = Vq.reshape(self.nx * self.ny) 
        Vmin = np.min(Vq) if np.min(Vq)<0 else 0      
        Vq = csr_matrix(np.diag(Vq - Vmin + self.q*Kq))
        self.Qh_LHS = -0.5 * self.hbar**2 * self.Hh + Vq
        u_Q = spsolve(self.Qh_LHS, self.Qh_RHS)
        self.u_Q = u_Q.reshape(self.nx, self.ny)
        Wv = -1 / self.u_Q - Vmin + self.q*Kq 
        self.QCp = Wv - self.Ev
        self.QCn = self.QCp

    def getcurrent(self):

        current_sums = {}

        for contact_key, positions in self.contact_positions.items():
            # Extract x and y indices
            x_indices = [x for x, _ in positions]
            y_indices = [y for _, y in positions]

            # Check boundary conditions
            x_check = all(x == 0 or x == self.nx - 1 for x in x_indices)
            y_check = all(y == 0 or y == self.ny - 1 for y in y_indices)

            # Calculate currents if conditions are met
            if y_check:
                # Calculate Jny + Jpy for y boundaries
                y_current_sum = sum(
                    (self.Jny[x, y] * self.dx + self.Jpy[x, y] * self.dx)
                    for x, y in positions
                )
                current_sums[contact_key] = y_current_sum

            if x_check:
                # Calculate Jnx + Jpx for x boundaries
                x_current_sum = sum(
                    (self.Jnx[x, y] * self.dy + self.Jpx[x, y] * self.dy)
                    for x, y in positions
                )
                current_sums[contact_key] = x_current_sum

        self.current = current_sums

    def getcharge(self):
        charge_sum = {}
        for region in self.charge_regions:
            name = region["name"]
            x_start, x_end = region["x"]
            y_start, y_end = region["y"]

            # Sum charge values in the region
            total_charge = (
                -self.q
                * np.sum(
                    self.n[x_start:x_end, y_start:y_end]
                    - self.p[x_start:x_end, y_start:y_end]
                    - self.NB[x_start:x_end, y_start:y_end]
                )
                * self.dx
                * self.dy
            )

            # Store the summed charge
            charge_sum[name] = total_charge
        self.charge = charge_sum

    def update_initial_guess(self, Ec, Efn, Efp):
        self.Ec = Ec
        self.Efn = Efn
        self.Efp = Efp
