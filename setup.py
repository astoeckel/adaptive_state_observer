from setuptools import setup

setup(
    name='adaptive_state_observer',
    packages=['adaptive_state_observer'],
    package_data={
        'adaptive_state_observer':
        [
            'cpp/adaptive_state_observer.in.cpp',
            'cpp/observer.cpp',
            'cpp/observer.hpp',
            'cpp/dists.cpp',
            'cpp/dists.hpp',
            'cpp/nef.cpp',
            'cpp/nef.hpp',
        ]
    },
    version='0.0.1',
    author='Andreas Stöckel',
    description='Adaptive State Observer Implementation',
    url='https://github.com/astoeckel/adaptive_state_observer',
    install_requires=[
        'posix_ipc',
    ],)
