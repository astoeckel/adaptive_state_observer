#!/usr/bin/env python3

#   C++ Adaptive State Observer
#   Copyright (C) 2019-2020  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""H
Contains the Python binding to the adaptive state observer C++ code. In
particular, the code in this file generates the final version of the C++ code
depending on the given parameters and compiles it.
"""

import dataclasses
import os
import numpy as np
import numpy.ctypeslib

# Import the C types we'll use
from ctypes import POINTER, cast, c_double, c_uint32, c_void_p, Structure
c_double_p = POINTER(c_double)

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))
    import cmodule
else:
    from . import cmodule

###############################################################################
# C++ Code Generation and Dependencies                                        #
###############################################################################


def _get_template_file():
    return os.path.join(os.path.dirname(__file__),
                        os.path.join('cpp', 'adaptive_state_observer.in.cpp'))


def _get_code_from_template(template_file, replacements):
    with open(template_file, 'r') as f:
        template = f.read()
        for key, value in replacements.items():
            template = template.replace("$" + key, str(value))
    return template


def _get_deps():
    return [
        'cpp/dists.cpp',
        'cpp/matrix.cpp',
        'cpp/nef.cpp',
        'cpp/observer.cpp',
    ]


###############################################################################
# Compile-time setup                                                          #
###############################################################################

_neuron_type_map = {
    "lif": "LIFEnsemble",
    "relu": "ReLUEnsemble",
    "rbf": "RBFEnsemble",
}


def _canonical_neuron_type(neuron_type):
    neuron_type = neuron_type.lower()
    if not neuron_type in _neuron_type_map:
        raise RuntimeError(
            "Unknown neuron type \"{}\", valid neuron types are {}".format(
                neuron_type, set(_neuron_type_map.keys())))
    return neuron_type


def _translate_neuron_type(neuron_type):
    return _neuron_type_map[_canonical_neuron_type(neuron_type)]


@dataclasses.dataclass
class Setup:
    """
    Dataclass containing all the static compile-time parameters defining the
    specific AdaptiveStateObserver instance.
    """
    n_state_dim: int = 1
    n_aux_state_dim: int = 0
    n_observation_dim: int = 1
    n_control_dim: int = 0
    n_neurons_g: int = 100
    n_neurons_h: int = 100
    neuron_type: str = "relu"

    @property
    def n_reduced_state_dim(self):
        return self.n_state_dim - self.n_aux_state_dim

    @property
    def n_params_g(self):
        neuron_type = _canonical_neuron_type(self.neuron_type)
        if neuron_type in {"relu", "lif",}:
            return self.N + 2
        elif neuron_type in {"rbf",}:
            return self.N + 1
        else:
            assert False

    @property
    def n_params_h(self):
        neuron_type = _canonical_neuron_type(self.neuron_type)
        if neuron_type in {"relu", "lif",}:
            return self.Nr + 2
        elif neuron_type in {"rbf",}:
            return self.Nr + 1
        else:
            assert False

    @property
    def N(self):
        return self.n_state_dim

    @property
    def Nr(self):
        return self.n_reduced_state_dim

    @property
    def M(self):
        return self.n_observation_dim

    @property
    def U(self):
        return self.n_control_dim

    def to_replacements(self):
        return {
            "N_STATE_DIM": self.n_state_dim,
            "N_AUX_STATE_DIM": self.n_aux_state_dim,
            "N_OBSERVATION_DIM": self.n_observation_dim,
            "N_CONTROL_DIM": self.n_control_dim,
            "N_NEURONS_G": self.n_neurons_g,
            "N_NEURONS_H": self.n_neurons_h,
            "NEURON_TYPE": _translate_neuron_type(self.neuron_type)
        }


###############################################################################
# Run-time parameter structure                                                #
###############################################################################


class Params(Structure):
    _fields_ = [
        ("dt", c_double),
        ("gain", c_double),
        ("eta", c_double),
        ("eta_rel_g", c_double),
        ("eta_rel_h", c_double),
    ]


c_params_p = POINTER(Params)

###############################################################################
# FFI Definition                                                              #
###############################################################################


def _get_array_names_and_shapes(s):
    base_arrs = {
        "G0": ("aso_get_G0",
               (s.N, s.N + s.U),
               (slice(None), slice(None))
              ),
        "g_W": ("aso_get_g_weights",
                (s.Nr, s.n_neurons_g),
                (slice(None), slice(None))
               ),

        "H0": ("aso_get_H0",
               (s.M, s.Nr),
               (slice(None), slice(None))
              ),
        "h_W": ("aso_get_h_weights",
                (s.M, s.n_neurons_h),
                (slice(None), slice(None))
               ),
    }

    neuron_type = _canonical_neuron_type(s.neuron_type)
    if neuron_type in {"relu", "lif",}:
        param_arrs = {
            "g_gain": ("aso_get_g_params",
                    (s.n_neurons_g, s.n_params_g),
                    (slice(None), slice(0, 1))
                   ),
            "g_bias": ("aso_get_g_params",
                    (s.n_neurons_g, s.n_params_g),
                    (slice(None), slice(1, 2))
                   ),
            "g_E": ("aso_get_g_params",
                    (s.n_neurons_g, s.n_params_g),
                    (slice(None), slice(2, None))
                   ),
            "h_gain": ("aso_get_h_params",
                    (s.n_neurons_h, s.n_params_h),
                    (slice(None), slice(0, 1))
                   ),
            "h_bias": ("aso_get_h_params",
                    (s.n_neurons_h, s.n_params_h),
                    (slice(None), slice(1, 2))
                   ),
            "h_E": ("aso_get_h_params",
                    (s.n_neurons_h, s.n_params_h),
                    (slice(None), slice(2, None))
                   ),
        }
    elif neuron_type in {"rbf",}:
        param_arrs = {
            "g_mu": ("aso_get_g_params",
                    (s.n_neurons_g, s.n_params_g),
                    (slice(None), slice(0, s.N))
                   ),
            "g_sigma_sq": ("aso_get_g_params",
                    (s.n_neurons_g, s.n_params_g),
                    (slice(None), slice(s.N, s.N + 1))
                   ),
            "h_mu": ("aso_get_h_params",
                    (s.n_neurons_h, s.n_params_h),
                    (slice(None), slice(0, s.Nr))
                   ),
            "h_sigma_sq": ("aso_get_h_params",
                    (s.n_neurons_h, s.n_params_h),
                    (slice(None), slice(s.Nr, s.Nr + 1))
                   ),
        }
    else:
        assert False

    return {**base_arrs, **param_arrs}


class AdaptiveStateObserverSharedLibrary(cmodule.SharedLibrary):
    def __init__(self, *args, **kwargs):
        # Call the super-constructor
        super().__init__(*args, **kwargs)

        # Instance constructor
        self.aso_create.argtypes = [c_uint32]
        self.aso_create.restype = c_void_p

        # Instance destructor
        self.aso_free.argtypes = [c_void_p]
        self.aso_free.restype = None

        # Step function
        self.aso_step.argtypes = [
            c_void_p, c_double_p, c_double_p, c_double_p, c_params_p,
            c_double_p, c_double_p, c_uint32
        ]
        self.aso_step.restype = None

        # Predict dx function
        self.aso_pred_dx.argtypes = [
            c_void_p, c_double_p, c_double_p, c_double_p, c_uint32
        ]
        self.aso_pred_dx.restype = None

        # Predict z function
        self.aso_pred_z.argtypes = [c_void_p, c_double_p, c_double_p, c_uint32]
        self.aso_pred_z.restype = None

        # Array getters
        arrs = _get_array_names_and_shapes(Setup())
        for fn_name in sorted({x[0] for x in arrs.values()}):
            fn = getattr(self, fn_name)
            fn.argtypes = [c_void_p]
            fn.restype = c_double_p



###############################################################################
# User-facing Python class                                                    #
###############################################################################


def _create_output_array(do_create, shape):
    # If we do not really need the output array, just do nothing
    if not do_create:
        return None, c_double_p()

    # Otherwise create an empty numpy array with the desired shape and return
    # both the array and the FFI pointer
    tar = np.empty(shape, dtype=np.float64, order='C')
    p_tar = tar.ctypes.data_as(c_double_p)
    return tar, p_tar


def _make_adaptive_state_observer_class(setup, soname):
    # Load the shared library and setup the FFI
    lib = AdaptiveStateObserverSharedLibrary(soname)

    # Create the actual AdaptiveStateObserver class
    class AdaptiveStateObserver:
        def __init__(self,
                     rng=np.random,
                     dt=1e-3,
                     gain=10.0,
                     eta=1e-2,
                     eta_rel_g=0.1,
                     eta_rel_h=1.0):
            # Initialize the numpy array map; otherwise calls to __getattr__
            # will cause an infinite recursion
            self._np_arrs = {}

            # Copy the setup
            self.setup = setup

            # Copy the given parameters
            self.dt = dt
            self.gain = gain
            self.eta = eta
            self.eta_rel_g = eta_rel_g
            self.eta_rel_h = eta_rel_h

            # Create the observer instance
            self._p_observer = lib.aso_create(rng.randint(1 << 31))
            assert not self._p_observer is None

            # Fetch a list of available arrays and their shapes; create the
            # corresponding numpy array views
            _arrs = _get_array_names_and_shapes(setup)
            for arr_name, (fn_name, arr_shape, arr_slices) in _arrs.items():
                if np.prod(arr_shape) > 0:
                    fn = getattr(lib, fn_name)
                    arr = numpy.ctypeslib.as_array(fn(self._p_observer),
                                                    shape=arr_shape)
                else:
                    arr = np.zeros(arr_shape)
                self._np_arrs[arr_name] = arr.__getitem__(arr_slices)

        def __del__(self):
            # Destroy the observer; the library may already have been closed at
            # this point; do nothing in this case
            if (not self._p_observer is None) and lib.is_open:
                lib.aso_free(self._p_observer)
            self._p_observer = None

        def step(self, zs, us=None, x0=None, return_xs=True, return_zs=True):
            # Make sure the input arrays are arrays
            zs = np.asarray(zs)
            us = None if us is None else np.asarray(us)
            x0 = None if x0 is None else np.asarray(x0)

            # Initialize x0 to zero if no explicit value is given
            if x0 is None:
                x0 = np.zeros((setup.N,))

            # The first dimension of us and zs should be the number of samples
            if zs.ndim < 2:
                zs = zs.reshape((-1, setup.M))
            n_samples = zs.shape[0]
            if us is None:
                assert setup.U == 0
                us = np.zeros((n_samples, 0))
            elif us.ndim < 2:
                us = us.reshape((-1, setup.U))
            assert us.shape[0] == zs.shape[0] == n_samples

            # Make sure the other dimensions are correct
            assert x0.size == setup.N
            assert zs.size == setup.M * n_samples
            assert us.size == setup.U * n_samples

            # Fetch pointers at the input arrays
            x0 = x0.astype(dtype=np.float64, order='C', copy=False)
            zs = zs.astype(dtype=np.float64, order='C', copy=False)
            us = us.astype(dtype=np.float64, order='C', copy=False)
            p_x0 = x0.ctypes.data_as(c_double_p)
            p_zs = zs.ctypes.data_as(c_double_p)
            p_us = us.ctypes.data_as(c_double_p)

            # Create the output arrays
            xs_tar, p_xs_tar = _create_output_array(return_xs,
                                                    (n_samples, setup.N))
            zs_tar, p_zs_tar = _create_output_array(return_zs,
                                                    (n_samples, setup.M))

            # Assemble the parameter struct
            params = Params(dt=self.dt,
                            gain=self.gain,
                            eta=self.eta,
                            eta_rel_g=self.eta_rel_g,
                            eta_rel_h=self.eta_rel_h)
            p_params = c_params_p(params)

            # Call the step function
            p_obs = self._p_observer
            lib.aso_step(p_obs, p_x0, p_us, p_zs, p_params, p_xs_tar, p_zs_tar,
                         n_samples)

            # Return the array containing the result
            return xs_tar, zs_tar

        def pred_dx(self, xs, us=None):
            # Make sure the input arrays are arrays
            xs = np.asarray(xs)
            us = None if us is None else np.asarray(us)

            # The first dimension of us and xs should be the number of samples
            if xs.ndim < 2:
                xs = xs.reshape((-1, setup.N))
            n_samples = xs.shape[0]
            if us is None:
                assert setup.U == 0
                us = np.zeros((n_samples, 0))
            elif us.ndim < 2:
                us = us.reshape((-1, setup.U))
            assert us.shape[0] == xs.shape[0] == n_samples

            # Make sure the other dimensions are correct
            assert xs.size == setup.N * n_samples
            assert us.size == setup.U * n_samples

            # Fetch pointers at the input arrays
            xs = xs.astype(dtype=np.float64, order='C', copy=False)
            us = us.astype(dtype=np.float64, order='C', copy=False)
            p_xs = xs.ctypes.data_as(c_double_p)
            p_us = us.ctypes.data_as(c_double_p)

            # Create the output array
            tar = np.empty((n_samples, setup.N), dtype=np.float64, order='C')
            p_tar = tar.ctypes.data_as(c_double_p)

            # Call the step function
            p_obs = self._p_observer
            lib.aso_pred_dx(p_obs, p_xs, p_us, p_tar, n_samples)

            # Return the array containing the result
            return tar

        def pred_z(self, xs):
            # Make sure the input arrays are arrays
            xs = np.asarray(xs)

            # The first dimension of us and xs should be the number of samples
            if xs.ndim < 2:
                xs = xs.reshape((-1, setup.N))
            n_samples = xs.shape[0]

            # Make sure the other dimensions are correct
            assert xs.size == setup.N * n_samples

            # Fetch pointers at the input arrays
            xs = xs.astype(dtype=np.float64, order='C', copy=False)
            p_xs = xs.ctypes.data_as(c_double_p)

            # Create the output array
            tar = np.empty((n_samples, setup.M), dtype=np.float64, order='C')
            p_tar = tar.ctypes.data_as(c_double_p)

            # Call the step function
            p_obs = self._p_observer
            lib.aso_pred_z(p_obs, p_xs, p_tar, n_samples)

            # Return the array containing the result
            return tar

        def __getattr__(self, name):
            # Make sure the attribute with the given name exists
            if not name in self._np_arrs:
                raise AttributeError(
                    "'{}' object has no attribute '{}'".format(
                        self.__class__.__name__, name))

            return self._np_arrs[name]

        def __setattr__(self, name, value):
            if ("_np_arrs" in self.__dict__) and (name in self._np_arrs):
                raise AttributeError(
                    "Attribute '{}' of '{}' object is read-only".format(
                        name, self.__class__.__name__))
            return super().__setattr__(name, value)

    # Return the class
    return AdaptiveStateObserver


###############################################################################
# User-facing "compile" function                                              #
###############################################################################


def compile(setup=None,
            include_dirs={'/usr/include/eigen3'},
            debug=False,
            parallel=True):
    # Use default setup if none is given
    if setup is None:
        setup = Setup()

    # Add the "cpp" subdirectory to the search path
    include_dirs.add(os.path.join(os.path.dirname(__file__), 'cpp'))

    # Generate the C++ code
    template_file = _get_template_file()
    replacements = setup.to_replacements()
    code = _get_code_from_template(template_file, replacements)
    deps = _get_deps()

    # Compile the library
    soname = cmodule.compile_cpp_library(
        code=code.encode('utf-8'),
        deps=deps,
        include_dirs=include_dirs,
        debug=debug,
        parallel=parallel,
    )

    return _make_adaptive_state_observer_class(setup, soname)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level="INFO")

    setup = Setup(
        n_state_dim=2,
        n_aux_state_dim=1,
        n_neurons_g=110,
        neuron_type="rbf",
    )
    AdaptiveStateObserver = compile(setup=setup, debug=False)

    observer = AdaptiveStateObserver(eta=1e-1)
    observer.H0[...] = np.zeros((1, 1))
    observer.G0[...] = np.array(((-1.0, 6.0), (-6.0, -1.0)))

    T = 100.0
    ts = np.arange(0, T, observer.dt)
    zs = np.sin(2.0 * np.pi * ts)

    import time
    t0 = time.perf_counter()

    xs, zs_pred = observer.step(zs, return_zs=True, return_xs=True)

    t1 = time.perf_counter()
    print("Execution took {:0.4f}ms".format((t1 - t0) * 1000.0))

    import matplotlib.pyplot as plt
    if zs_pred is None:
        zs_pred = observer.pred_z(xs)
    fig, ax = plt.subplots()
    ax.plot(ts, zs)
    ax.set_prop_cycle(None)
    ax.plot(ts, zs_pred, '--')
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Observation $\\vec z$")

    if not xs is None:
        fig, ax = plt.subplots()
        ax.plot(ts, xs)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("State $\\vec x$")
    plt.show()

