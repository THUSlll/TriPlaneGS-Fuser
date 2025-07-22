from tgs.utils.graphics_utils import fov2focal, focal2fov
import numpy as np
import os
from gaussian.cameras import Camera
from typing import NamedTuple
from PIL import Image
import torch
import torch.nn.functional as F

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

def JSON_to_camera(camera_entry, image_path=None):
    id = camera_entry['id']
    image_name = camera_entry['img_name']
    width = camera_entry['width']
    height = camera_entry['height']
    position = np.array(camera_entry['position'])
    rotation = np.array(camera_entry['rotation'])
    FovY = focal2fov(camera_entry['fy'], height)
    FovX = focal2fov(camera_entry['fx'], width)
    C2W = np.zeros((4, 4))
    C2W[:3, :3] = rotation
    C2W[:3, 3] = position
    C2W[3, 3] = 1.0
    W2C = np.linalg.inv(C2W)
    R = W2C[:3, :3].transpose()
    T = W2C[:3, 3]
    if image_path is not None:
        if not image_name.endswith('.jpg') and not image_name.endswith('.png'):
            _image_path = os.path.join(image_path, image_name + ".jpg")
    
        else : _image_path = os.path.join(image_path, image_name)
    else:
        _image_path = None

    cam_info = CameraInfo(uid=id, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                              image_path=_image_path, image_name=image_name, width=width, height=height)
    return cam_info

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_path):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_path=is_path))

    return camera_list

def loadCam(args, id, cam_info, resolution_scale, is_path):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    depth_fixed_exist = False
    image_path = cam_info.image_path[:-4]
    if image_path.endswith('png') or image_path.endswith('jpg'):
        image_path = image_path[:-4]
    fixed_path = image_path + '_fixed.npz'

    if cam_info.image_path is not None:
        if not is_path:
            image = Image.open(cam_info.image_path)
            
            resized_image_rgb = PILtoTorch(image, resolution)
        else: resized_image_rgb = None


        if os.path.exists(fixed_path):
            depth_fixed_exist = True
            # depth_fixed_data = np.load(fixed_path)
            # depth_fixed = torch.tensor(depth_fixed_data['depth'], device=args.data_device).float().unsqueeze(0) 
            # np.savez_compressed(cam_info.image_path[:-4] + '_fixed.npz', depth=depth_fixed.squeeze(0).cpu().numpy())
        else:

            try: 

                # 加载原始深度图
                # depth_data = np.load(image_path + '.npy')
                # np.savez_compressed(image_path + '.npz', depth=depth_data)

                depth_data = np.load(image_path + '.npz')
                depth = torch.tensor(depth_data['depth'], device=args.data_device).float().unsqueeze(0) 
                
                # 加载高斯深度图
                # gaussian_depth_data = np.load(image_path + '_gaussian.npy')
                # np.savez_compressed(image_path + '_gaussian.npz', depth=gaussian_depth_data)

                gaussian_depth_data = np.load(image_path + '_gaussian.npz')
                gaussian_depth = torch.tensor(gaussian_depth_data['depth'], device=args.data_device).float() 
                
            except: depth = None
    loaded_mask = None

    if not is_path:
        if resized_image_rgb.shape[0] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]

    if depth_fixed_exist == False and depth != None:

        depth_mask = gaussian_depth == 0
        depth[depth_mask] = 0
        gaussian_depth[depth_mask] = 0
        
        #####################################################
        #单目尺度偏移修正
        valid_mono_depth = depth[~depth_mask].squeeze(0).view(-1)
        valid_GS_depth = gaussian_depth[~depth_mask].squeeze(0).view(-1)
        A = np.vstack([valid_mono_depth, np.ones(len(valid_mono_depth))]).T
        s, t = np.linalg.lstsq(A, valid_GS_depth, rcond=None)[0]
        depth[~depth_mask] = s * depth[~depth_mask] + t
        ######################################################
        depth_fixed = depth
        depth[depth_mask] = 0
         # 保存depth_fixed为npz文件
        np.savez_compressed(fixed_path, depth=depth_fixed.squeeze(0).cpu().numpy())
        # 删除两个深度源文件
        file1 = image_path + '.npz'
        file2 = image_path + '_gaussian.npz'
        for f in [file1, file2]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
    else:
        pass
    fixed_path = image_path + '_gaussian.npy'
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=resized_image_rgb, image_path=cam_info.image_path, resolution=resolution, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, fixed_depth_path=fixed_path)
