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

class material:
    def __init__(self):
        # Dictionary to store properties for each material
        self.properties = {
            "Si": {
                "epsilon": 11.7,
                "Eg": 1.12,
                "xi": 4.05,
                "Nc": 2.8e19,
                "Nv": 1.04e19,
                "un": 1400,
                "up": 450,
                "vsat_n": 1e7,
                "beta_n": 2,
                "vsat_p": 1e7,
                "beta_p": 2,
                "vt_n": 1.2e7,
                "vt_p": 1.2e7,
                "lambda_n": 7.65e-9,
                "lambda_p": 7.65e-9,
                "ua_n": 0,
                "eu_n": 1,
                "ud_n": 0,
                "ucs_n": 1,
                "nref": 1e18,
                "ua_p": 0,
                "eu_p": 1,
                "ud_p": 0,
                "ucs_p": 1,
                "pref": 1e18,
                "tau_n": 1,
                "tau_p": 1,
                "etrap": 0,
            },  # Example values
            "SiO2": {
                "epsilon": 3.9,
                "Eg": 8.9,
                "xi": 0.95,
                "Nc": 1e18,
                "Nv": 1e18,
                "un": 1,
                "up": 1,
                "vsat_n": 1e7,
                "beta_n": 2,
                "vsat_p": 1e7,
                "beta_p": 2,
                "vt_n": 1.2e7,
                "vt_p": 1.2e7,
                "lambda_n": 7.65e-9,
                "lambda_p": 7.65e-9,
                "ua_n": 0,
                "eu_n": 1,
                "ud_n": 0,
                "ucs_n": 1,
                "nref": 1e18,
                "ua_p": 0,
                "eu_p": 1,
                "ud_p": 0,
                "ucs_p": 1,
                "pref": 1e18,
                "tau_n": 1,
                "tau_p": 1,
                "etrap": 0,
            },  # Example values
            "Si3N4": {
                "epsilon": 3,
                "Eg": 5.3,
                "xi": 2.15,
                "Nc": 1e18,
                "Nv": 1e18,
                "un": 1,
                "up": 1,
                "vsat_n": 1e7,
                "beta_n": 2,
                "vsat_p": 1e7,
                "beta_p": 2,
                "vt_n": 1.2e7,
                "vt_p": 1.2e7,
                "lambda_n": 7.65e-9,
                "lambda_p": 7.65e-9,
                "ua_n": 0,
                "eu_n": 1,
                "ud_n": 0,
                "ucs_n": 1,
                "nref": 1e18,
                "ua_p": 0,
                "eu_p": 1,
                "ud_p": 0,
                "ucs_p": 1,
                "pref": 1e18,
                "tau_n": 1,
                "tau_p": 1,
                "etrap": 0,
            },  # Example values
            "HfO2": {
                "epsilon": 25,
                "Eg": 5.7,
                "xi": 1.2,
                "Nc": 1e18,
                "Nv": 1e18,
                "un": 1,
                "up": 1,
                "vsat_n": 1e7,
                "beta_n": 2,
                "vsat_p": 1e7,
                "beta_p": 2,
                "vt_n": 1.2e7,
                "vt_p": 1.2e7,
                "lambda_n": 7.65e-9,
                "lambda_p": 7.65e-9,
                "ua_n": 0,
                "eu_n": 1,
                "ud_n": 0,
                "ucs_n": 1,
                "nref": 1e18,
                "ua_p": 0,
                "eu_p": 1,
                "ud_p": 0,
                "ucs_p": 1,
                "pref": 1e18,
                "tau_n": 1,
                "tau_p": 1,
                "etrap": 0,
            },  # Example values
            "Metal": {
                "epsilon": 1,
                "Eg": 0,
                "xi": 4.5,
                "Nc": 1e25,
                "Nv": 1e25,
                "un": 1,
                "up": 1,
                "vsat_n": 1e7,
                "beta_n": 2,
                "vsat_p": 1e7,
                "beta_p": 2,
                "vt_n": 1.2e7,
                "vt_p": 1.2e7,
                "lambda_n": 7.65e-9,
                "lambda_p": 7.65e-9,
                "ua_n": 0,
                "eu_n": 1,
                "ud_n": 0,
                "ucs_n": 1,
                "nref": 1e18,
                "ua_p": 0,
                "eu_p": 1,
                "ud_p": 0,
                "ucs_p": 1,
                "pref": 1e18,
                "tau_n": 1,
                "tau_p": 1,
                "etrap": 0,
            },  # Example values
            "Ge": {
                "epsilon": 16.0,
                "Eg": 0.66,
                "xi": 4.0,
                "Nc": 1.2e19,
                "Nv": 0.6e19,
                "un": 3900,
                "up": 1900,
                "vsat_n": 6e6,
                "beta_n": 2,
                "vsat_p": 6e6,
                "beta_p": 2,
                "vt_n": 3.1e7,
                "vt_p": 1.9e7,
                "lambda_n": 7.65e-9,
                "lambda_p": 7.65e-9,
                "ua_n": 0,
                "eu_n": 1,
                "ud_n": 0,
                "ucs_n": 1,
                "nref": 1e18,
                "ua_p": 0,
                "eu_p": 1,
                "ud_p": 0,
                "ucs_p": 1,
                "pref": 1e18,
                "tau_n": 1,
                "tau_p": 1,
                "etrap": 0,
            },  # Example values
        }

    def get_property(self, materials, property_name):
        """Retrieve a specific property for a given material or array of materials."""
        if isinstance(materials, np.ndarray):
            # Process array element-wise
            return np.array(
                [
                    self.properties.get(element, {}).get(property_name, None)
                    for element in materials.flatten()
                ]
            ).reshape(materials.shape)
        else:
            # Handle single material name
            material_props = self.properties.get(materials, None)
            if material_props:
                return material_props.get(property_name, None)
        return None

    def add_material(
        self, material, epsilon, Eg, xi, Nc, Nv, un, up, vsat_n, beta_n, vsat_p, beta_p, ua_n, eu_n, ud_n, ucs_n, nref, ua_p, eu_p, ud_p, ucs_p, pref, vth_n, vth_p, lambda_n, lambda_p, tau_n, tau_p, etrap 
    ):
        # Add a new material with its properties
        if material in self.properties:
            print(f"Material '{material}' already exists. Overwriting its properties.")

        self.properties[material] = {
            "epsilon": epsilon,  # Permitivity (F/m)
            "Eg": Eg,  # Bandgap energy (eV)
            "xi": xi,  # Electron affinity (eV)
            "Nc": Nc,  # Conduction band effective density of states (cm^-3)
            "Nv": Nv,  # Valence band effective density of states (cm^-3)
            "un": un,  # Drift-diffusion electron mobility (cm^2/V/s)
            "up": up,  # Drif-diffusion hole mobility (cm^2/V/s)
            "vsat_n": vsat_n,  # Electron saturation velocity (cm/s)
            "beta_n": beta_n,  # Electron saturation parameter
            "vsat_p": vsat_p,  # Hole saturation mobility (cm/s)
            "beta_p": beta_p,  # Electron saturation parameter
            "ua_n": ua_n,  # Phonon scattering electron mobility degradation ((m/V)^eu_n)
            "eu_n": eu_n,  # Phonon scattering electron mobility degradation 
            "ud_n": ud_n,  # Columbic scattering electron mobility degradation
            "ucs_n": ucs_n,  # Columbic scattering electron mobility degradation   
            "nref": nref,  # Columbic scattering electron mobility degradation (cm^-3)           
            "ua_p": ua_p,  # Phonon scattering hole mobility degradation ((m/V)^eu_p)
            "eu_p": eu_p,  # Phonon scattering hole mobility degradation  
            "ud_p": ud_p,  # Columbic scattering electron mobility degradation
            "ucs_p": ucs_p,  # Columbic scattering electron mobility degradation  
            "pref": pref,  # Columbic scattering electron mobility degradation (cm^-3)  
            "vth_n": vth_n,  # Electron thermal velocity (cm/s)
            "vth_p": vth_p,  # Hole thermal velocity (cm/s)
            "lambda_n": lambda_n,  # Effective scattering length for electron saturation velocity (m)
            "lambda_p": lambda_p,  # Effective scattering length for hole saturation velocity (m)
            "tau_n": tau_n,  # SRH electron lifetime (s)
            "tau_p": tau_p,  # SRH hole lifetime (s)
            "etrap": etrap,  # Field factor (eV)
        }
        print(f"Added material '{material}' with properties.")

    def update_property(self, material, property_name, value):
        # Update a specific property for an existing material
        if material in self.properties:
            self.properties[material][property_name] = value
            print(f"Updated '{property_name}' for material '{material}' to {value}.")
        else:
            print(
                f"Material '{material}' does not exist. Use add_material() to add it first."
            )
