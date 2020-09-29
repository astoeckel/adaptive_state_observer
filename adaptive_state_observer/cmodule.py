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
"""
This file contains helpers for compiling a bunch of C++ code into a library. In
particular, this code caches already compiled C++ modules and recompiles them
if their dependencies change.
"""

import atexit
import ctypes
import hashlib
import multiprocessing
import os
import posix_ipc
import re
import subprocess
import weakref

import logging
logger = logging.getLogger(__name__)

_TMP_DIR = None  # Directory compiled libraries are stored in
_COMPILED_LIBRARY_MAP = {}  # Map storing the DynamicsLearning classes

DEFAULT_FLAGS = [
    '-std=c++14',
    '-march=native',  # Compile for this particular computer
    '-Wall',
    '-Wextra',
    '-Wno-deprecated-copy',
    '-fdiagnostics-color=always',
    '-faligned-new',
    '-fno-partial-inlining', # Does not improve performance and causes a large penalty when -g is active at the same time for some reason
    '-fno-exceptions',  # This code does not use exceptions
    '-fno-rtti',  # We don't need no runtime-type information
    '-ffast-math',  # Be sloppy, we don't have non-finite math
    '-x',  # Specify the language since we're compiling from stdin
    'c++',
    '-fPIC',

]

def _pkg_name():
    return os.path.split(os.path.dirname(os.path.realpath(__file__)))[-1]


def _create_tmp_dir():
    global _TMP_DIR, _COMPILED_LIBRARY_MAP

    # Create a user-accessible temporary directory
    if _TMP_DIR is None or not os.path.exists(_TMP_DIR):
        _COMPILED_LIBRARY_MAP = {}
        _TMP_DIR = os.path.expanduser('~/.cache/{}/'.format(_pkg_name()))
        os.makedirs(_TMP_DIR, exist_ok=True, mode=0O755)

    return _TMP_DIR


def _recursive_get_deps(filename=None,
                        code=None,
                        include_dirs=None,
                        deps=None,
                        exclude_system=True):
    # Make sure we either have a filename or some code that should be compiled
    assert (filename is None) != (code is None)
    deps = set() if deps is None else deps

    # Fetch the module directory which is used as include directory
    if include_dirs is None:
        include_dirs = set()

    # If a filename is given, try to open the corresponding file
    curdir = os.path.dirname(os.path.abspath(__file__))
    if not (filename is None):
        if not os.path.isabs(filename):
            filename = os.path.join(curdir, filename)
        with open(filename, "rb") as f:
            code = f.read()

    # Decode the code to a UTF-8 string
    code = str(code, 'utf-8')

    # Search for all "includes"
    re_include_local = re.compile(r'^\s*#include\s+\"([^\"]+)\"', re.MULTILINE)
    re_include_global = re.compile(r'^\s*#include\s+<([^>]+)>', re.MULTILINE)

    def process_include(include, is_local):
        local_dirs = set()
        if is_local and not (filename is None):
            local_dirs.add(os.path.dirname(filename))
        include_dirs_local = include_dirs | local_dirs
        for dir_ in include_dirs_local:
            if os.path.isfile(os.path.join(dir_, include)):
                dep_path = os.path.realpath(os.path.join(dir_, include))
                if not dep_path in deps:
                    if not dep_path.startswith("/usr/") or not exclude_system:
                        deps.add(dep_path)
                        _recursive_get_deps(dep_path, None, include_dirs, deps)
                        break

    # Search for all "include" statements and add them as a dependency
    for match in re_include_global.finditer(code):
        process_include(match[1], False)
    for match in re_include_local.finditer(code):
        process_include(match[1], True)

    return deps


def _file_hash(filename, hasher):
    """
    Feeds the content of the file with the specified file into the hasher.
    """

    with open(filename, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            hasher.update(data)


def _compute_code_hash(code=None, filename=None, deps=None, extra_args=""):
    # Make sure we either have a filename or some code that should be compiled
    assert (filename is None) != (code is None)
    deps = set() if deps is None else deps

    # Create the hasher
    hasher = hashlib.sha1()

    # Hash the main file itself
    if not code is None:
        hasher.update(code)
    else:
        # Make the filename absolute
        if not os.path.isabs(filename):
            curdir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(curdir, filename)

        # Update the hash
        _file_hash(filename, hasher)

    # Hash all dependencies in order
    for dep in sorted(deps):
        _file_hash(dep, hasher)

    # Add the given extra arguments
    hasher.update(extra_args.encode('utf-8'))

    # Let the hash depend on the current UID in case the files are written to
    # a shared temporary directory
    hasher.update(str(os.getuid()).encode('utf-8'))
    return hasher.hexdigest()


def _compile_cpp_obj(tar_dir,
                     filename=None,
                     code=None,
                     include_dirs=None,
                     flags=DEFAULT_FLAGS,
                     debug=False):
    # Make sure we either have a filename or some code that should be compiled
    assert (filename is None) != (code is None)
    include_dirs = set() if include_dirs is None else include_dirs

    # Make sure that the filename -- if given -- is absolute
    curdir = os.path.dirname(os.path.abspath(__file__))
    if (not filename is None) and (not os.path.isabs(filename)):
        filename = os.path.join(curdir, filename)

    # Fetch all the header files this object is depending on
    deps = _recursive_get_deps(filename=filename,
                               code=code,
                               include_dirs=include_dirs)

    # Compute the compiler command line arguments
    args = list(
        filter(
            lambda x: bool(x),
            [
                'g++',
                '-c',  # Only compiling an object file
                '-g' if debug else '-DNDEBUG',  # Toggle debug code
                '-Og' if debug else '-O3',  # Toggle optimisation
            ] + flags))

    # Fetch the module directory which is used as include directory
    for include_dir in sorted(include_dirs):
        args.append("-I" + include_dir)

    # Compute a hash for the object file
    code_hash = _compute_code_hash(code=code,
                                   filename=filename,
                                   deps=deps,
                                   extra_args=",".join(sorted(args)))

    # Compute the target filename
    tar_filename = _pkg_name()
    if not filename is None:
        tar_filename, _ = os.path.splitext(filename)
    tar_filename = os.path.basename(tar_filename)
    tar = os.path.join(tar_dir, tar_filename + "_" + code_hash[0:8] + ".o")

    # Create a semaphore for the given target file. Only one process should try
    # to compile code at a time
    with posix_ipc.Semaphore(os.path.basename(tar),
                             posix_ipc.O_CREAT,
                             initial_value=1):
        # Someone else compiled the code for us, return!
        if os.path.exists(tar):
            return tar

        # Compile the code
        logger.info("Compiling %s...", (os.path.basename(tar)))
        args += ['-o', tar]

        # Write the source code to a temporary file when debugging
        if debug and filename is None:
            filename = tar + '.cpp'
            with open(filename, 'wb') as f:
                f.write(("// " + " ".join(args) + "\n").encode("utf-8"))
                f.write(code)
        args.append("-" if filename is None else filename)

        # Call the compiler
        process = subprocess.Popen(args,
                                   stdin=None if debug else subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        _, stderr = process.communicate(None if filename else code)
        stderr = str(stderr, 'utf-8')
        if process.returncode != 0:
            raise Exception("Error while compiling the C++ code:\n" + stderr)
        elif len(stderr) > 0:
            print(stderr)

    return tar


def _compile_cpp_obj_mp_wrapper(job):
    return _compile_cpp_obj(
        tar_dir=job["tar_dir"],
        filename=job["filename"],
        code=job["code"],
        include_dirs=job["include_dirs"],
        debug=job["debug"],
    )


def compile_cpp_library(code,
                        deps=[],
                        include_dirs=None,
                        debug=False,
                        parallel=True):
    # Fetch the target directory
    tar_dir = _create_tmp_dir()

    # Process the set of include directories
    curdir = os.path.dirname(os.path.abspath(__file__))
    if include_dirs is None:
        include_dirs = set()
    else:
        include_dirs = set(include_dirs)
    include_dirs |= {
        curdir,
    }

    # Create a list of object files that need to be compiled
    def _mkjob(filename=None, code=None):
        return {
            "tar_dir": tar_dir,
            "filename": filename,
            "code": code,
            "include_dirs": include_dirs,
            "debug": debug,
        }

    jobs = [_mkjob(code=code)] + [_mkjob(filename=f) for f in deps]

    # Compile all the object files using a multiprocessing pool
    if parallel:
        with multiprocessing.Pool() as p:
            objs = p.map(_compile_cpp_obj_mp_wrapper, jobs)
    else:
        objs = list(map(_compile_cpp_obj_mp_wrapper, jobs))

    # Generate a hash from the object basenames
    obj_hash = hashlib.sha1()
    for obj in objs:
        obj_hash.update(os.path.basename(obj).encode('utf-8'))
    obj_hash = obj_hash.hexdigest()
    tar_filename = _pkg_name() + '_' + obj_hash[0:8] + ".so"
    tar = os.path.join(tar_dir, tar_filename)

    # Compile the object files into a single executable
    # Create a semaphore for the given target file. Only one process should try
    # to compile code at a time
    with posix_ipc.Semaphore(os.path.basename(tar),
                             posix_ipc.O_CREAT,
                             initial_value=1) as lock:
        # Someone else compiled the code for us, return!
        if os.path.exists(tar):
            return tar

        # Compile the code
        args = (list(
            filter(lambda x: bool(x), [
                'g++', '-Wall', '-Wextra', '-shared', None if debug else '-s',
                '-fdiagnostics-color=always', '-fPIC',
                '-Wl,--as-needed,-soname,' + tar, '-lm', '-o', tar
            ])) + objs)

        # Call the compiler
        logger.info("Linking %s...", (os.path.basename(tar)))
        process = subprocess.Popen(args,
                                   stdin=None if debug else subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        _, stderr = process.communicate()
        stderr = str(stderr, 'utf-8')
        if process.returncode != 0:
            raise Exception("Error while linking the C++ code:\n" + stderr)
        elif len(stderr) > 0:
            print(stderr)

    # Return the target filename
    return tar

# List of weak references at all SharedLibrary instances
_OPEN_LIB_REFS = set()

# Closes all open libraries
def _close_open_libraries():
    for ref in _OPEN_LIB_REFS:
        lib = ref()
        if not lib is None:
            lib.close()

# Before Python is shutting down, close all open libraries. Otherwise the
# libraries are closed too late in the shutdown process, causing Python to
# throw an exception for some reason
atexit.register(_close_open_libraries)

class SharedLibrary:
    def __init__(self, soname):
        # Open the shared library
        self._lib = ctypes.CDLL(soname)

        # Remember that this "SharedLibrary" instance exists. Close it before
        # Python shuts down.
        _OPEN_LIB_REFS.add(weakref.ref(self))

    def __del__(self):
        self.close()

    def __getattr__(self, attr):
        return getattr(self._lib, attr)

    @property
    def is_open(self):
        return not self._lib is None

    def close(self):
        if self._lib is None:
            return
        try:
            import _ctypes
            if hasattr(_ctypes, 'dlclose'):
                _ctypes.dlclose(self._lib._handle)
            else:
                logger.warning('Cannot close shared library; Python implementation does not provide "_ctypes.dlclose"')
        except Exception as e:
            logger.exception(e)
        finally:
            self._lib = None

if __name__ == "__main__":
    # Enable logging
    logging.basicConfig(level="INFO")

    # Build a shared object
    lib = SharedLibrary(compile_cpp_library(B"""
#include <iostream>

extern "C" {
void run() {
    std::cout << "Hello World from C++" << std::endl;
}
}
"""))

    # Run the function
    lib.run()

    # Close the library
    lib.close()
