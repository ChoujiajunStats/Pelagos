#!/usr/bin/env python3
"""Compatibility entrypoint: use the native RViz toolbar, with no separate UI."""
import os
from pathlib import Path


def main():
    config = Path(__file__).resolve().parents[1]/'config/pelagos.rviz'
    environment = dict(os.environ, LANG='C.UTF-8', LC_ALL='C.UTF-8', LANGUAGE='en')
    print('The separate task UI was removed. Opening the native RViz toolbar.', flush=True)
    os.execvpe('rviz2', ['rviz2', '-d', str(config), '--ros-args', '-p', 'use_sim_time:=true'], environment)


if __name__ == '__main__':
    main()
