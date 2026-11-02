"""Bind the mapper to the exact camera geometry used by this simulation run."""
import copy
from pathlib import Path


def configure_wavemap(template,geometry,run_dir):
    mapping=copy.deepcopy(template)
    params=mapping['/**']['ros__parameters']
    stride=params['pixel_stride']
    width,height=geometry['width'],geometry['height']
    if type(stride) is not int or stride<1 or width%stride or height%stride:
        raise ValueError('Wavemap pixel_stride must divide both camera dimensions')
    k=geometry['K_left']
    params.update(width=width,height=height,fx=k[0][0],fy=k[1][1],cx=k[0][2],cy=k[1][2],
                  world_frame=geometry['world_frame'],camera_frame=geometry['camera_frame'],
                  map_output_path=str(Path(run_dir)/'map.wvmp'),
                  status_path=str(Path(run_dir)/'mapping-status.json'))
    return mapping
