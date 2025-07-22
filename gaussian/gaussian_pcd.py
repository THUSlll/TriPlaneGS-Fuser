import torch
from dataclasses import dataclass
import plyfile
from plyfile import PlyData, PlyElement
import numpy as np
@dataclass
class Gaussianmodel:
    xyz: torch.Tensor = torch.empty(0)
    opacities: torch.Tensor = torch.empty(0)
    features_dc: torch.Tensor = torch.empty(0)
    features_extra: torch.Tensor = torch.empty(0)
    scales: torch.Tensor = torch.empty(0)
    rots: torch.Tensor = torch.empty(0)

def load_gaussian_ply(path: str) -> Gaussianmodel:

    #Load a Gaussian model from a .ply file.

    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return Gaussianmodel(xyz=torch.from_numpy(xyz).float(),
                            opacities=torch.from_numpy(opacities).float(),
                            features_dc=torch.from_numpy(features_dc).float(),
                            features_extra=torch.from_numpy(features_extra).float(),
                            scales=torch.from_numpy(scales).float(),
                            rots=torch.from_numpy(rots).float())