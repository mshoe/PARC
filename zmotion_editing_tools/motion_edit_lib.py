import torch
import numpy as np
import pickle
import copy
import math
import random
import util.torch_util as torch_util
import util.geom_util as geom_util
import util.terrain_util as terrain_util
import anim.motion_lib as motion_lib
import anim.kin_char_model as kin_char_model
# import anim.motion_graph as motion_graph
from typing import List
from collections import OrderedDict

# A library of functions that edits motion data
# for simplicity assume everything is a torch tensor (so we can use other torch util functions)

class MotionData:
    def __init__(self, motion_data, device="cpu"):
        self._data = motion_data
        #assert "fps" in self._data
        #assert "frames" in self._data
        #assert "loop_mode" in self._data
        self._device = device
        if "frames" in self._data and not isinstance(self._data["frames"], torch.Tensor):
            self._data["frames"] = torch.tensor(self._data["frames"], dtype=torch.float32, device=device)
            
        if "contacts" in self._data and not isinstance(self._data["contacts"], torch.Tensor):
            self._data["contacts"] = torch.tensor(self._data["contacts"], dtype=torch.float32, device=device)

        if "floor_heights" in self._data and not isinstance(self._data["floor_heights"], torch.Tensor):
            self._data["floor_heights"] = torch.tensor(self._data["floor_heights"], dtype=torch.float32, device=device)

        if "terrain" in self._data:
            self._data["terrain"].update_old()
            self._data["terrain"].to_torch(device)

        if "path_nodes" in self._data:
            self._data["path_nodes"] = self._data["path_nodes"].to(device=device)

        # fix fps
        if "fps" not in self._data:
            self._data["fps"] = 30
        if "loop_mode" not in self._data:
            self._data["loop_mode"] = "CLAMP"
        if self.get_fps() > 29 and self.get_fps() < 31:
            self.set_fps(30)
        return
    
    def set_hf_mask_inds_device(self, device):
        for t in range(len(self._data["hf_mask_inds"])):
            self._data["hf_mask_inds"][t] = self._data["hf_mask_inds"][t].to(device=device)
        return
    
    def set_device(self, device):
        self._device = device
        for key in self._data:
            if key == "fps" or key == "loop_mode" or "target_xy":
                continue

            if key == "hf_mask_inds":
                self.set_hf_mask_inds_device(device)

            elif key == "terrain":
                self._data[key].set_device(device)
            else:
                print(key, device)
                self._data[key] = self._data[key].to(device=device)
        return

    def get_fps(self):
        return self._data["fps"]
    
    def set_fps(self, fps):
        self._data["fps"] = int(fps)
        return

    def get_loop_mode(self):
        return self._data["loop_mode"]
    
    def get_frames(self):
        return self._data["frames"]
    
    def set_frames(self, motion_frames):
        self._data["frames"] = motion_frames
        return
    
    def has_contacts(self):
        return "contacts" in self._data

    def get_contacts(self):
        assert "contacts" in self._data
        return self._data["contacts"]
    
    def set_contacts(self, contacts):
        self._data["contacts"] = contacts
        return
    
    def set_hf_mask_inds(self, hf_mask_inds):
        self._data["hf_mask_inds"] = hf_mask_inds
        return
    
    def has_hf_mask_inds(self):
        return "hf_mask_inds" in self._data
    
    def get_hf_mask_inds(self):
        assert "hf_mask_inds" in self._data
        return self._data["hf_mask_inds"]
    
    # def get_floor_heights(self):
    #     assert "floor_heights" in self._data
    #     return self._data["floor_heights"]
    
    # def set_floor_heights(self, floor_heights):
    #     self._data["floor_heights"] = floor_heights
    #     return

    def has_terrain(self):
        return "terrain" in self._data

    def get_terrain(self) -> terrain_util.SubTerrain:
        assert "terrain" in self._data
        return self._data["terrain"]
    
    def set_terrain(self, terrain):
        self._data["terrain"] = terrain
        return

    def remove_terrain(self):
        del self._data["terrain"]
        return

    def has_opt_body_constraints(self):
        return "opt:body_constraints" in self._data

    def get_opt_body_constraints(self):
        assert "opt:body_constraints" in self._data
        return self._data["opt:body_constraints"]
    
    def set_opt_body_constraints(self, body_constraints):
        self._data["opt:body_constraints"] = body_constraints
        return
    
    def remove_opt_body_constraints(self):
        del self._data["opt:body_constraints"]
        return

    def save_to_file(self, motion_filepath, verbose=True):
        if isinstance(self._data["frames"], torch.Tensor):
            self._data["frames"] = self._data["frames"].cpu().numpy().astype(np.float32)

        if "contacts" in self._data:
            if isinstance(self._data["contacts"], torch.Tensor):
                self._data["contacts"] = self._data["contacts"].cpu().numpy().astype(np.float32)

        if "floor_heights" in self._data:
            if isinstance(self._data["floor_heights"], torch.Tensor):
                self._data["floor_heights"] = self._data["floor_heights"].cpu().numpy().astype(np.float32)

        if "terrain" in self._data:
            if isinstance(self._data["terrain"].hf, torch.Tensor):
                self._data["terrain"] = self._data["terrain"].numpy_copy()

        if "hf_mask_inds" in self._data:
            self.set_hf_mask_inds_device("cpu")
            # these should always be on cpu right?
            # not efficient to put on gpu since we do a lot of small operations on them,
            # then we can pass to gpu

        if self.has_opt_body_constraints():
            body_constraints = self.get_opt_body_constraints()
            for b in range(len(body_constraints)):
                for constraint in body_constraints[b]:
                    constraint.constraint_point = constraint.constraint_point.to(device='cpu')


        with open(motion_filepath, 'wb') as file:
            pickle.dump(self._data, file)
            if verbose:
                print("wrote motion data to", motion_filepath)
        return

def load_motion_file(motion_filepath, device="cpu"):
    import sys
    try:
        import numpy.core
        sys.modules['numpy._core'] = np.core
        if hasattr(np.core, 'multiarray'):
            sys.modules['numpy._core.multiarray'] = np.core.multiarray
    except ImportError:
        pass
    
    with open(motion_filepath, "rb") as filestream:
        motion_data = pickle.load(filestream)
    return MotionData(motion_data, device=device)

def save_motion_data(motion_filepath, motion_frames, contact_frames, 
                     terrain: terrain_util.SubTerrain, fps: int, loop_mode: str,
                     **kwargs):
    data = dict()

    if isinstance(motion_frames, torch.Tensor):
        motion_frames = motion_frames.cpu().numpy().astype(np.float32)

    if motion_frames is not None:
        data["frames"] = motion_frames

    if contact_frames is not None:
        if isinstance(contact_frames, torch.Tensor):
            contact_frames = contact_frames.cpu().numpy().astype(np.float32)
        data["contacts"] = contact_frames

    if terrain is not None:
        if isinstance(terrain.hf, torch.Tensor):
            terrain = terrain.numpy_copy()
        data["terrain"] = terrain

    if fps is not None:
        data["fps"] = fps

    if loop_mode is not None:
        data["loop_mode"] = loop_mode

    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.cpu()
        else:
            data[key] = value

    with open(motion_filepath, "wb") as f:
        pickle.dump(data, f)
        print("wrote motion data to", motion_filepath)
    return

def create_terrain_for_motion(motion_frames: torch.Tensor,
                              char_model: kin_char_model.KinCharModel,
                              char_points,
                              dx=0.1,
                              padding=5.0):
    #motion_frames = motion_data.get_frames()  

    #dx = 0.1
    #padding = 5.0
    num_padding = int(round(padding/dx))

    max_root_x = torch.max(motion_frames[:, 0]).item()
    max_root_y = torch.max(motion_frames[:, 1]).item()
    min_root_x = torch.min(motion_frames[:, 0]).item()
    min_root_y = torch.min(motion_frames[:, 1]).item()
    # first frame root xy is (0, 0)


    num_pos_x = int(abs(round(max_root_x / dx))) + num_padding
    num_neg_x = int(abs(round(min_root_x / dx))) + num_padding
    num_pos_y = int(abs(round(max_root_y / dx))) + num_padding
    num_neg_y = int(abs(round(min_root_y / dx))) + num_padding

    # Now we will deterministically construct a heightfield, so that we can use a heightfield
    # contact function to compute foot contacts and fix foot penetrations
    #hf, hf_mask = terrain_util.hf_from_motion(motion_frames, 
    #                                        min_height=0.0, ground_height=0.0, 
    #                                        dx=dx, char_model=char_model, canon_idx=0,
    #                                     num_neg_x=num_neg_x, num_pos_x=num_pos_x,
    #                                     num_neg_y=num_neg_y, num_pos_y=num_pos_y,
    #                                     floor_heights=platform_heights)
    # # 3x3 maxpool filter
    # #maxpool = torch.nn.MaxPool2d(kernel_size=[3,9], stride=1, padding=[1,4])
    # if maxpool_filter:
    #     character_box_size = 1.0#0.5
    #     kernel_size = int(character_box_size / dx)

    #     maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    #     hf = maxpool(hf.unsqueeze(dim=0)).squeeze(dim=0)
    #     hf = hf.squeeze(0)
    #     #hf_mask = maxpool(hf_mask.float().unsqueeze(0)).squeeze(0).squeeze(0).to(dtype=torch.bool)

    grid_dim_x = num_neg_x + num_pos_x + 1
    grid_dim_y = num_neg_y + num_pos_y + 1
    #low_ind_bound = torch.tensor([0, 0], dtype=torch.int64, device=device)
    #high_ind_bound = torch.tensor([grid_dim_x, grid_dim_y], dtype=torch.int64, device=device) - 1

    hf = torch.zeros(size=(grid_dim_x, grid_dim_y), dtype=torch.float32, device=motion_frames.device)

    min_x = motion_frames[0, 0] - dx * num_neg_x
    min_y = motion_frames[0, 1] - dx * num_neg_y
    min_point = torch.stack([min_x, min_y])

    terrain = terrain_util.SubTerrain(x_dim=hf.shape[0], y_dim=hf.shape[1],
                                      dx=dx, dy=dx, min_x=min_point[0].item(), min_y=min_point[1].item(),
                                      device="cpu")
    terrain.hf = hf
    #terrain.hf_mask = hf_mask

    hf_mask_inds, _ = terrain_util.compute_hf_mask_inds(motion_frames, terrain, char_model, char_points)
    terrain.hf_mask = terrain_util.compute_hf_mask_from_inds(terrain, hf_mask_inds)

    return terrain, hf_mask_inds

def stride_motion(motion_frames, stride_start_idx, stride_end_idx, stride):

    first_frames = motion_frames[:stride_start_idx]
    strided_frames = motion_frames[stride_start_idx:stride_end_idx:stride]
    end_frames = motion_frames[stride_end_idx+1:]

    ret_frames = torch.cat([first_frames, strided_frames, end_frames], dim=0)
    return ret_frames


def cut_motion(motion_frames, cut_start_idx, cut_end_idx):

    first_frames = motion_frames[:cut_start_idx]
    end_frames = motion_frames[cut_end_idx+1:]

    ret_frames = torch.cat([first_frames, end_frames], dim=0)
    return ret_frames

def slice_motion(motion_frames, start_time, end_time, fps, ret_idx_info=False):
    if start_time == None:
        start_frame_idx = 0
    else:
        start_frame_idx = int(start_time * fps)
    if end_time == None:
        end_frame_idx = motion_frames.shape[0]
    else:
        end_frame_idx = int(end_time * fps)

    ret_frames = motion_frames[start_frame_idx:end_frame_idx+1]

    if ret_idx_info:
        return ret_frames, start_frame_idx, end_frame_idx
    else:
        return ret_frames

def stitch_motions(motion_frames_1, start_time_1, end_time_1, 
                   motion_frames_2, start_time_2, end_time_2, fps):
    # Assumes motions have the same fps

    new_motion_frames_1 = slice_motion(motion_frames_1, start_time_1, end_time_1, fps)
    new_motion_frames_2 = slice_motion(motion_frames_2, start_time_2, end_time_2, fps)

    new_motion_frames = torch.cat([new_motion_frames_1, new_motion_frames_2], dim=0)

    return new_motion_frames

def blend_motions(motion_frames_1, start_time_1, end_time_1, 
                  motion_frames_2, start_time_2, end_time_2, fps,
                  num_blend_frames=5, blend_style="linear"):
    assert num_blend_frames >= 2
    device = motion_frames_1.device
    dt = 1.0 / fps

    start_time_2 = start_time_2 - num_blend_frames * dt


    new_motion_frames_1 = slice_motion(motion_frames_1, start_time_1, end_time_1, fps)
    new_motion_frames_2 = slice_motion(motion_frames_2, start_time_2, end_time_2, fps)

    num_dofs = motion_frames_1.shape[1]
    blend_frames_1 = new_motion_frames_1[-num_blend_frames:].clone()
    blend_frames_2 = new_motion_frames_2[:num_blend_frames].clone()

    if blend_style == "linear":
        blend_frames = torch.zeros_like(blend_frames_1)

        for i in range(num_blend_frames):
            a = i / (num_blend_frames - 1.0)
            blend_frames[i] = (1.0 - a) * blend_frames_1[i] + a * blend_frames_2[i]
    else:
        assert False

    new_motion_frames = torch.cat([new_motion_frames_1[:-num_blend_frames], 
                                   blend_frames, 
                                   new_motion_frames_2[num_blend_frames:]], dim=0)

    return new_motion_frames

def blend_motions_2(motion_frames_1, motion_frames_2,
                    num_blend_frames=5, blend_style="linear"):
    assert num_blend_frames >= 2
    blend_frames_1 = motion_frames_1[-num_blend_frames:].clone()
    blend_frames_2 = motion_frames_2[:num_blend_frames].clone()

    if blend_style == "linear":
        blend_frames = torch.zeros_like(blend_frames_1)

        for i in range(num_blend_frames):
            a = i / (num_blend_frames - 1.0)
            blend_frames[i] = (1.0 - a) * blend_frames_1[i] + a * blend_frames_2[i]
    else:
        assert False

    new_motion_frames = torch.cat([motion_frames_1[:-num_blend_frames], 
                                   blend_frames, 
                                   motion_frames_2[num_blend_frames:]], dim=0)

    return new_motion_frames

def blend_motions_linear(root_pos_1, root_rot_1, joint_rot_1,
                         root_pos_2, root_rot_2, joint_rot_2, num_blend_frames=5):

    # pos is 3D
    # rot is 4D quats

    assert num_blend_frames >= 2
    device = root_pos_1.device

    blend_root_pos_1 = root_pos_1[-num_blend_frames:]
    blend_root_rot_1 = root_rot_1[-num_blend_frames:]
    blend_joint_rot_1 = joint_rot_1[-num_blend_frames:]

    blend_root_pos_2 = root_pos_2[:num_blend_frames]
    blend_root_rot_2 = root_rot_2[:num_blend_frames]
    blend_joint_rot_2 = joint_rot_2[:num_blend_frames]

    blend_root_pos = torch.zeros_like(blend_root_pos_1)
    blend_root_rot = torch.zeros_like(blend_root_rot_1)
    blend_joint_rot = torch.zeros_like(blend_joint_rot_1)

    for i in range(num_blend_frames):
        a = i / (num_blend_frames - 1.0)

        blend_root_pos[i] = (1.0 - a) * blend_root_pos_1[i] + a * blend_root_pos_2[i]

        a_th = torch.tensor([a], dtype=torch.float32, device=device)
        blend_root_rot[i] = torch_util.slerp(blend_root_rot_1[i], blend_root_rot_2[i], a_th)
        blend_joint_rot[i] = torch_util.slerp(blend_joint_rot_1[i], blend_joint_rot_2[i], a_th.unsqueeze(0))

    new_root_pos = torch.cat([root_pos_1[:-num_blend_frames], 
                              blend_root_pos, 
                              root_pos_2[num_blend_frames:]], dim=0)
    
    new_root_rot = torch.cat([root_rot_1[:-num_blend_frames], 
                              blend_root_rot, 
                              root_rot_2[num_blend_frames:]], dim=0)
    
    new_joint_rot = torch.cat([joint_rot_1[:-num_blend_frames],
                               blend_joint_rot,
                               joint_rot_2[num_blend_frames:]], dim=0)

    return new_root_pos, new_root_rot, new_joint_rot

def translate_motion(motion_frames, translation):
    ret_frames = motion_frames.clone()
    ret_frames[:, 0:3] += translation
    return ret_frames

def rotate_motion(motion_frames: torch.Tensor, 
                  rot_quat: torch.Tensor, 
                  origin: torch.Tensor):
    assert len(motion_frames.shape) == 2
    num_frames = motion_frames.shape[0]

    root_pos = motion_frames[:, 0:3]
    root_rot = motion_frames[:, 3:6]

    if len(rot_quat.shape) == 1:
        rot_quat = rot_quat.unsqueeze(0)

    local_root_pos = root_pos - origin
    new_local_root_pos = torch_util.quat_rotate(rot_quat, local_root_pos)
    new_root_pos = new_local_root_pos + origin

    root_rot_quat = torch_util.exp_map_to_quat(root_rot)
    new_root_rot_quat = torch_util.quat_multiply(rot_quat, root_rot_quat)
    new_root_rot = torch_util.quat_to_exp_map(new_root_rot_quat)

    new_motion_frames = motion_frames.clone()
    new_motion_frames[:, 0:3] = new_root_pos
    new_motion_frames[:, 3:6] = new_root_rot

    return new_motion_frames

def change_heading_at_frame(motion_frames: torch.Tensor, 
                            new_heading: torch.Tensor, 
                            idx):
    # motion_frames: [num_frames, num_dofs]
    # new_heading: [1]
    # idx: int

    rot_quat = torch_util.exp_map_to_quat(motion_frames[idx, 3:6])
    heading_quat_inv = torch_util.calc_heading_quat_inv(rot_quat)
    

    new_heading_quat = torch_util.heading_to_quat(new_heading)
    
    heading_change_quat = torch_util.quat_multiply(new_heading_quat, heading_quat_inv)

    
    old_rot_quats = torch_util.exp_map_to_quat(motion_frames[:, 3:6])
    new_rot_quats = torch_util.quat_multiply(heading_change_quat, old_rot_quats)
    new_rot_exp_maps = torch_util.quat_to_exp_map(new_rot_quats)

    old_positions = motion_frames[..., 0:3]
    canon_pos = motion_frames[idx, 0:3]
    new_positions = torch_util.quat_rotate(heading_change_quat, old_positions - canon_pos) + canon_pos

    new_motion_frames = motion_frames.clone()
    new_motion_frames[..., 0:3] = new_positions
    new_motion_frames[..., 3:6] = new_rot_exp_maps

    return new_motion_frames

def move_xy_root_to_origin(motion_frames):
    translation = -motion_frames[0, 0:3]
    translation[2] = 0.0
    ret_frames = translate_motion(motion_frames, translation)
    return ret_frames

def flip_rotation_about_XZ_plane(exp_map):
    # reflect vector about XZ plane
    exp_map[1] *= -1.0

    # change direction of rotation
    exp_map[:] *= -1.0
    return

def flip_quat_about_XZ_plane(quat):
    quat[1] *= -1.0
    quat[3] *= -1.0
    return

def flip_motion_about_XZ_plane(motion_frames, char_model: kin_char_model.KinCharModel, contact_frames=None):
    def swap(input_tensor, ind_1, ind_2):
        temp = copy.deepcopy(input_tensor[ind_1])
        input_tensor[ind_1] = input_tensor[ind_2]
        input_tensor[ind_2] = temp
        return
    
    BODY_PELVIS = 0
    BODY_TORSO = 1
    BODY_HEAD = 2
    BODY_RIGHT_UPPER_ARM = 3
    BODY_RIGHT_LOWER_ARM = 4
    BODY_RIGHT_HAND = 5
    BODY_LEFT_UPPER_ARM = 6
    BODY_LEFT_LOWER_ARM = 7
    BODY_LEFT_HAND = 8
    BODY_RIGHT_THIGH = 9
    BODY_RIGHT_SHIN = 10
    BODY_RIGHT_FOOT = 11
    BODY_LEFT_THIGH = 12
    BODY_LEFT_SHIN = 13
    BODY_LEFT_FOOT = 14
    
    new_motion_frames = motion_frames.clone()
    num_frames = len(new_motion_frames)
    for i in range(num_frames):
        curr_motion_frame = new_motion_frames[i]

        root_pos = curr_motion_frame[0:3]
        root_rot = curr_motion_frame[3:6]
        joint_dof = curr_motion_frame[6:]

        # flip root_pos about XZ plane
        root_pos[1] *= -1.0

        # flip root rotation about XZ plane
        flip_rotation_about_XZ_plane(root_rot)

        abdomen = slice(0,3)
        neck = slice(3,6)
        right_shoulder = slice(6,9)
        right_elbow = 9
        left_shoulder = slice(10,13)
        left_elbow = 13
        right_hip = slice(14,17)
        right_knee = 17
        right_ankle = slice(18, 21)
        left_hip = slice(21, 24)
        left_knee = 24
        left_ankle = slice(25, 28)


        # TODO: don't flip joint dof, flip global body rotations
        # flip_rotation_about_XZ_plane(joint_dof[abdomen])
        # flip_rotation_about_XZ_plane(joint_dof[neck])

        # swap(joint_dof, right_shoulder, left_shoulder)
        # swap(joint_dof, right_elbow, left_elbow)
        # swap(joint_dof, right_hip, left_hip)
        # swap(joint_dof, right_knee, left_knee)
        # swap(joint_dof, right_ankle, left_ankle)

        # flip_rotation_about_XZ_plane(joint_dof[right_shoulder])
        # flip_rotation_about_XZ_plane(joint_dof[left_shoulder])
        # flip_rotation_about_XZ_plane(joint_dof[right_hip])
        # flip_rotation_about_XZ_plane(joint_dof[left_hip])
        # flip_rotation_about_XZ_plane(joint_dof[right_ankle])
        # flip_rotation_about_XZ_plane(joint_dof[left_ankle])

        joint_rot = char_model.dof_to_rot(joint_dof)
        for b in range(joint_rot.shape[0]):
            flip_quat_about_XZ_plane(joint_rot[b])

        joint_dof = char_model.rot_to_dof(joint_rot)
        swap(joint_dof, right_shoulder, left_shoulder)
        swap(joint_dof, right_elbow, left_elbow)
        swap(joint_dof, right_hip, left_hip)
        swap(joint_dof, right_knee, left_knee)
        swap(joint_dof, right_ankle, left_ankle)
        curr_motion_frame[6:] = joint_dof
        


    if contact_frames is None:
        return new_motion_frames
    else:
        new_contact_frames = contact_frames.clone()
        for i in range(num_frames):
            curr_contact_frame = new_contact_frames[i]
            swap(curr_contact_frame, BODY_RIGHT_UPPER_ARM, BODY_LEFT_UPPER_ARM)
            swap(curr_contact_frame, BODY_RIGHT_LOWER_ARM, BODY_LEFT_LOWER_ARM)
            swap(curr_contact_frame, BODY_RIGHT_HAND, BODY_LEFT_HAND)
            swap(curr_contact_frame, BODY_RIGHT_THIGH, BODY_LEFT_THIGH)
            swap(curr_contact_frame, BODY_RIGHT_SHIN, BODY_LEFT_SHIN)
            swap(curr_contact_frame, BODY_RIGHT_FOOT, BODY_LEFT_FOOT)

        return new_motion_frames, new_contact_frames
    
def compute_ground_plane_foot_contacts(motion_frames, char_model: kin_char_model.KinCharModel,
                                       contact_eps=0.04):
    device = char_model._device
    num_frames = motion_frames.shape[0]
    num_bodies = len(char_model._body_names)

    contacts = torch.zeros(size=(num_frames, num_bodies), dtype=torch.float32, device=device)

    num_frames = motion_frames.shape[0]
    root_pos, root_rot, joint_dof = motion_lib.extract_pose_data(motion_frames)
    root_rot_quat = torch_util.exp_map_to_quat(root_rot)
    joint_rot = char_model.dof_to_rot(joint_dof)

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot)

    lf_id = char_model.get_body_id("left_foot")
    rf_id = char_model.get_body_id("right_foot")

    key_ids = [lf_id, rf_id]

    for body_id in key_ids:
        key_body_pos = body_pos[:, body_id]
        key_body_rot = body_rot[:, body_id]

        # Assume only 1 geom for feet
        geom = char_model._geoms[body_id][0]
            
        geom_offset = geom._offset
        geom_dims = geom._dims

        box_points = geom_util.get_box_points_batch(key_body_pos, key_body_rot, geom_dims, geom_offset)
        box_points_z = box_points[:, :, 2]
        
        # NOTE: we can extend this to grid heightfields in the future
        # will need to do parallel box/box collision detection
        gplane_contact = box_points_z - contact_eps < 0.0
        gplane_contact = torch.any(gplane_contact, dim=-1)
        
        contacts[:, body_id] = gplane_contact.to(dtype=torch.float32)

    return contacts

def compute_hf_foot_contacts_and_correct_pen(motion_frames, terrain: terrain_util.SubTerrain,
                             char_model: kin_char_model.KinCharModel, contact_eps=0.04):
    
    device = char_model._device
    num_frames = motion_frames.shape[0]
    num_bodies = len(char_model._body_names)

    contacts = torch.zeros(size=(num_frames, num_bodies), dtype=torch.float32, device=device)

    num_frames = motion_frames.shape[0]
    root_pos, root_rot, joint_dof = motion_lib.extract_pose_data(motion_frames)
    root_rot_quat = torch_util.exp_map_to_quat(root_rot)
    joint_rot = char_model.dof_to_rot(joint_dof)

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot)

    lf_id = char_model.get_body_id("left_foot")
    rf_id = char_model.get_body_id("right_foot")

    key_ids = [lf_id, rf_id]

    pen_correction_z = torch.zeros(size=(num_frames,))
    for body_id in key_ids:
        key_body_pos = body_pos[:, body_id]
        key_body_rot = body_rot[:, body_id]

        # Assume only 1 geom for feet
        geom = char_model._geoms[body_id][0]
            
        geom_offset = geom._offset
        geom_dims = geom._dims

        box_points = geom_util.get_box_points_batch(key_body_pos, key_body_rot, geom_dims, geom_offset)
        
        # This method will check the grid indices of each of the box points,
        # then check if the points height is above the height of its grid index.
        # There are edge cases where this will give false positives, so fixing this is
        # a todo with ray box intersections

        grid_inds = terrain.get_grid_index(box_points[..., 0:2])
        hf = terrain.hf.unsqueeze(0).expand(num_frames, -1, -1)
        cell_heights = hf[torch.arange(0, num_frames, 1).unsqueeze(-1), grid_inds[..., 0], grid_inds[..., 1]]
        box_points_z = box_points[..., 2]
        
        floor_contact = torch.any(box_points_z < cell_heights + contact_eps, dim=-1)
        contacts[:, body_id] = floor_contact

        curr_pen_correction_z, _ = torch.min(box_points_z - cell_heights, dim=-1)
        pen_correction_z = torch.min(pen_correction_z, curr_pen_correction_z)

    updated_motion_frames = motion_frames.clone()
    updated_motion_frames[:, 2] -= pen_correction_z
    return updated_motion_frames, contacts

def compute_motion_terrain_hand_contacts(motion_frames, terrain: terrain_util.SubTerrain,
                          char_model: kin_char_model.KinCharModel, contact_eps=0.04):
    
    device = char_model._device
    num_frames = motion_frames.shape[0]
    num_bodies = len(char_model._body_names)

    contacts = torch.zeros(size=(num_frames, num_bodies), dtype=torch.float32, device=device)

    num_frames = motion_frames.shape[0]
    root_pos, root_rot, joint_dof = motion_lib.extract_pose_data(motion_frames)
    root_rot_quat = torch_util.exp_map_to_quat(root_rot)
    joint_rot = char_model.dof_to_rot(joint_dof)

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot)

    lh_id = char_model.get_body_id("left_hand")
    rh_id = char_model.get_body_id("right_hand")

    key_ids = [lh_id, rh_id]

    for body_id in key_ids:
        key_body_pos = body_pos[:, body_id]
        #key_body_rot = body_rot[:, body_id]

        geom = char_model._geoms[body_id][0]
        geom_offset = geom._offset
        geom_dims = geom._dims

        sd = terrain_util.points_hf_sdf(key_body_pos.unsqueeze(0), 
                                        terrain.hf.unsqueeze(0), 
                                        terrain.min_point.unsqueeze(0), 
                                        terrain.dxdy, 
                                        base_z=torch.min(terrain.hf).item() - 10.0,
                                        inverted=False,
                                        radius=geom_dims.item())
        
        contact = (sd[0] < contact_eps).to(dtype=torch.float32)
        contacts[:, body_id] = contact
    return contacts

def correct_foot_ground_pen(motion_frames, char_model: kin_char_model.KinCharModel,
                            ground_height = 0.0):
    num_frames = motion_frames.shape[0]
    root_pos, root_rot, joint_dof = motion_lib.extract_pose_data(motion_frames)
    root_rot_quat = torch_util.exp_map_to_quat(root_rot)
    joint_rot = char_model.dof_to_rot(joint_dof)

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot)

    lf_id = char_model.get_body_id("left_foot")
    rf_id = char_model.get_body_id("right_foot")

    key_ids = [lf_id, rf_id]

    updated_motion_frames = motion_frames.clone()

    # Find the minimum foot box point with z < ground_height for all frames.
    # Then offset the motion by the negative of the minimum foot box points < ground_height

    min_point_z = torch.ones(size=(num_frames,)) * ground_height
    for body_id in key_ids:
        key_body_pos = body_pos[:, body_id]
        key_body_rot = body_rot[:, body_id]

        for geom in char_model._geoms[body_id]:
            
            geom_offset = geom._offset
            geom_dims = geom._dims

            box_points = geom_util.get_box_points_batch(key_body_pos, key_body_rot, geom_dims, geom_offset)

            box_points_z = box_points[:, :, 2]
            curr_geom_min_point_z, _ = torch.min(box_points_z, dim=1)
            min_point_z = torch.min(min_point_z, curr_geom_min_point_z)

    updated_motion_frames[:, 2] -= (min_point_z - ground_height)

    return updated_motion_frames

def create_floor_heights_data(num_frames, dt, times, heights):
    # Given motion frames,
    # a list of times, and a list of heights,
    # return a tensor that is shape (num_frames)
    # that stores what the height below the root pos of the character is.

    # optionally check to make sure the character is not intersecting with
    # the implicitly described heightfield?

    h_info = np.zeros(shape=(num_frames,), dtype=np.float32)
    num_inds = len(times)

    time_inds = np.clip(np.round(times / dt), 0, num_frames-1).astype(np.int64)
    time_inds[-1] = -1 # make sure last time ind is just the last frame

    for i in range(0, num_inds-1):
        curr_ind = time_inds[i]
        next_ind = time_inds[i+1]
        h_info[curr_ind:next_ind] = heights[i]
    return h_info

def compute_floor_height_frame_data(motion_data: MotionData, 
                                    char_model: kin_char_model.KinCharModel,
                                    heights_guess: torch.Tensor,
                                    ground_height=0.0):
    
    frames = motion_data.get_frames()
    num_frames = frames.shape[0]
    dt = 1.0 / motion_data.get_fps()
    device = frames.device

    root_pos = frames[:, 0:3]
    root_rot = torch_util.exp_map_to_quat(frames[:, 3:6])
    joint_rot = char_model.dof_to_rot(frames[:, 6:])

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)

    lf_id = char_model.get_body_id("left_foot")
    rf_id = char_model.get_body_id("right_foot")

    contact_body_ids = [lf_id, rf_id]

    zvt = []

    # First get the min point of the feet along each frame
    for body_id in contact_body_ids:
        key_body_pos = body_pos[:, body_id]
        key_body_rot = body_rot[:, body_id]

        # Assume only 1 geom for feet
        geom = char_model._geoms[body_id][0]
            
        geom_offset = geom._offset
        geom_dims = geom._dims

        box_points = geom_util.get_box_points_batch(key_body_pos, key_body_rot, geom_dims, geom_offset)
        min_foot_z, _ = torch.min(box_points[..., 2], dim=-1)

        zvt.append(min_foot_z)


    local_min_z = []

    for z_plot in zvt:

        # extract only the local min indices

        inds = torch.arange(1, num_frames-1, 1, dtype=torch.int64, device=device)

        local_min_check1 = z_plot[inds] < z_plot[inds+1]
        local_min_check2 = z_plot[inds] < z_plot[inds-1]

        ind_is_local_min = torch.logical_and(local_min_check1, local_min_check2)

        print(ind_is_local_min.shape)

        local_min_inds = inds[ind_is_local_min]
        z = z_plot[local_min_inds]
        local_min_z.append(z)

    # Now fit N lines to minimize a loss function

    def get_line(num_frames, x):
        return torch.ones(size=(num_frames,), dtype=torch.float32, device=device) * x

    def compute_loss(z, x):
        loss = 0.0
        for j in range(len(z)):
            num_frames = z[j].shape[0]
            z_hat = []
            for i in range(len(x)):
                z_hat.append(get_line(num_frames, x[i]))
            z_hat = torch.stack(z_hat, dim=-1)
        
            curr_loss, _ = torch.min(torch.square(z[j].unsqueeze(-1) - z_hat), dim=-1)
            loss += torch.sum(curr_loss)

        return loss

    x = heights_guess.clone()
    x.requires_grad = True
    optim = torch.optim.Adam([x], lr=0.01)

    for i in range(200):
        optim.zero_grad()
        loss = compute_loss(local_min_z, x)
        loss.backward()
        optim.step()
        if i >= 199:
            print("iter:", i, ", loss:", loss.item())
            print(x)

    x = x.detach()

    def min_discrete_line(z, heights):
        ret_z = torch.zeros_like(z[:, 0])
        for i in range(z.shape[0]):

            abs_diff = torch.abs(z[i, :].unsqueeze(-1) - heights.unsqueeze(0))
            abs_diff = torch.sum(abs_diff, dim=0)
            _, ind = torch.min(abs_diff, dim=0)
            ret_z[i] = heights[ind].clone()

        return ret_z

    zvt = torch.stack(zvt, dim=-1)

    # For platform running motions
    min_x, min_ind = torch.min(x, dim=0)
    x[min_ind] = ground_height
    print(x)
    discrete_line = min_discrete_line(zvt, x)#.cpu().numpy()

    return discrete_line

def compute_motion_discrete_heights(motion_data: MotionData, 
                                    char_model: kin_char_model.KinCharModel,
                                    heights_guess: torch.Tensor,
                                    ground_height=0.0):
    
    frames = motion_data.get_frames()
    num_frames = frames.shape[0]
    dt = 1.0 / motion_data.get_fps()
    device = frames.device

    root_pos = frames[:, 0:3]
    root_rot = torch_util.exp_map_to_quat(frames[:, 3:6])
    joint_rot = char_model.dof_to_rot(frames[:, 6:])

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)

    lf_id = char_model.get_body_id("left_foot")
    rf_id = char_model.get_body_id("right_foot")

    contact_body_ids = [lf_id, rf_id]

    zvt = []

    # First get the min point of the feet along each frame
    for body_id in contact_body_ids:
        key_body_pos = body_pos[:, body_id]
        key_body_rot = body_rot[:, body_id]

        # Assume only 1 geom for feet
        geom = char_model._geoms[body_id][0]
            
        geom_offset = geom._offset
        geom_dims = geom._dims

        box_points = geom_util.get_box_points_batch(key_body_pos, key_body_rot, geom_dims, geom_offset)
        min_foot_z, _ = torch.min(box_points[..., 2], dim=-1)

        zvt.append(min_foot_z)


    local_min_z = []

    for z_plot in zvt:

        # extract only the local min indices

        inds = torch.arange(1, num_frames-1, 1, dtype=torch.int64, device=device)

        local_min_check1 = z_plot[inds] < z_plot[inds+1]
        local_min_check2 = z_plot[inds] < z_plot[inds-1]

        ind_is_local_min = torch.logical_and(local_min_check1, local_min_check2)

        print(ind_is_local_min.shape)

        local_min_inds = inds[ind_is_local_min]
        z = z_plot[local_min_inds]
        local_min_z.append(z)

    # Now fit N lines to minimize a loss function

    def get_line(num_frames, x):
        return torch.ones(size=(num_frames,), dtype=torch.float32, device=device) * x

    def compute_loss(z, x):
        loss = 0.0
        for j in range(len(z)):
            num_frames = z[j].shape[0]
            z_hat = []
            for i in range(len(x)):
                z_hat.append(get_line(num_frames, x[i]))
            z_hat = torch.stack(z_hat, dim=-1)
        
            curr_loss, _ = torch.min(torch.square(z[j].unsqueeze(-1) - z_hat), dim=-1)
            loss += torch.sum(curr_loss)

        return loss

    x = heights_guess.clone()
    x.requires_grad = True
    optim = torch.optim.Adam([x], lr=0.01)

    # TODO: replace with K-means algorithm
    for i in range(200):
        optim.zero_grad()
        loss = compute_loss(local_min_z, x)
        loss.backward()
        optim.step()
        if i >= 199:
            print("iter:", i, ", loss:", loss.item())
            print(x)

    x = x.detach()

    # For platform running motions
    min_x, min_ind = torch.min(x, dim=0)
    x[min_ind] = ground_height
    print(x)

    return x


def search_for_matching_motion_frames(mlib_A: motion_lib.MotionLib,
                                      mlib_B: motion_lib.MotionLib,
                                      m_A_start_time,
                                      m_A_end_time,
                                      m_B_start_time,
                                      m_B_end_time):
    assert mlib_A._motion_ids.shape[0] == 1
    assert mlib_B._motion_ids.shape[0] == 1    

    motion_A_fps = mlib_A._motion_fps[0].item()
    motion_B_fps = mlib_B._motion_fps[0].item()
    assert motion_A_fps == motion_B_fps
    assert m_A_start_time < m_A_end_time
    assert m_B_start_time < m_B_end_time

    start_idx_A = int(round(m_A_start_time * motion_A_fps))
    start_idx_B = int(round(m_B_start_time * motion_B_fps))
    end_idx_A = np.clip(int(round(m_A_end_time * motion_A_fps)), 0, mlib_A._motion_num_frames[0].item() - 1)
    end_idx_B = np.clip(int(round(m_B_end_time * motion_B_fps)), 0, mlib_B._motion_num_frames[0].item() - 1)

    ## Get all pose infos across all frames

    motion_times_A = torch.arange(start_idx_A, end_idx_A+1, 1) / motion_A_fps
    motion_ids_A = torch.zeros_like(motion_times_A, dtype=torch.int64)
    mlib_A_frames = mlib_A.calc_motion_frame(motion_ids_A, motion_times_A)
    root_pos_A, root_rot_A, root_vel_A, root_ang_vel_A, joint_rot_A, dof_vel_A = mlib_A_frames[0:6]
    body_pos_A, body_rot_A = mlib_A._kin_char_model.forward_kinematics(root_pos_A, root_rot_A, joint_rot_A)
    # canonicalize body positions
    body_pos_A[:, :, 0:2] -= body_pos_A[:, 0:1, 0:2].clone()
    root_rot_heading_quat_inv_A = torch_util.calc_heading_quat_inv(root_rot_A)
    body_pos_A[:, :, 0:3] = torch_util.quat_rotate(root_rot_heading_quat_inv_A.unsqueeze(1), 
                                                     body_pos_A[:, :, 0:3])


    motion_times_B = torch.arange(start_idx_B, end_idx_B+1, 1) / motion_B_fps
    motion_ids_B = torch.zeros_like(motion_times_B, dtype=torch.int64)
    mlib_B_frames = mlib_B.calc_motion_frame(motion_ids_B, motion_times_B)
    root_pos_B, root_rot_B, root_vel_B, root_ang_vel_B, joint_rot_B, dof_vel_B = mlib_B_frames[0:6]
    body_pos_B, body_rot_B = mlib_B._kin_char_model.forward_kinematics(root_pos_B, root_rot_B, joint_rot_B)
    # canonicalize body positions
    body_pos_B[:, :, 0:2] -= body_pos_B[:, 0:1, 0:2].clone()
    root_rot_heading_quat_inv_B = torch_util.calc_heading_quat_inv(root_rot_B)
    body_pos_B[:, :, 0:3] = torch_util.quat_rotate(root_rot_heading_quat_inv_B.unsqueeze(1), 
                                                     body_pos_B[:, :, 0:3])
    

    # canonicalize root vel and root ang vel
    root_vel_A = torch_util.quat_rotate(root_rot_heading_quat_inv_A, root_vel_A)
    root_vel_B = torch_util.quat_rotate(root_rot_heading_quat_inv_B, root_vel_B)
    
    #root_ang_vel_A = torch_util.quat_rotate(root_rot_heading_quat_inv_A, root_ang_vel_A)
    #root_ang_vel_B = torch_util.quat_rotate(root_rot_heading_quat_inv_B, root_ang_vel_B)
    

    # Compute motion matching errors
    min_motion_match_err = 100000.0
    match_t_A = None
    match_t_B = None
    for i in range(0, end_idx_A-start_idx_A):
        body_pos_diff = torch.norm(body_pos_A[i:i+1, 1:] - body_pos_B[:, 1:], dim=-1)
        body_pos_diff = torch.sum(body_pos_diff, dim=-1)

        
        root_vel_diff = torch.norm(root_vel_A[i] - root_vel_B, dim=-1)
        root_ang_vel_diff = torch.norm(root_ang_vel_A[i] - root_ang_vel_B, dim=-1)


        motion_match_err = body_pos_diff * 0.65 + root_vel_diff * 0.2 + root_ang_vel_diff * 0.15

        for j in range(motion_match_err.shape[0]):
            if motion_match_err[j] < min_motion_match_err:
                min_motion_match_err = motion_match_err[j]
                match_t_A = (start_idx_A + i) / motion_A_fps
                match_t_B = (start_idx_B + j) / motion_B_fps

    print("min motion match err", min_motion_match_err)


    # find rotation and translation of motion B so that it is aligned with motion A at the matching frames
    motion_ids = torch.tensor([0], dtype=torch.int64)
    motion_times_A = torch.tensor([match_t_A], dtype=torch.float32)
    mlib_A_frames = mlib_A.calc_motion_frame(motion_ids, motion_times_A)
    root_pos_A, root_rot_A, root_vel_A, root_ang_vel_A, joint_rot_A, dof_vel_A = mlib_A_frames[0:6]

    motion_times_B = torch.tensor([match_t_B], dtype=torch.float32)
    mlib_B_frames = mlib_B.calc_motion_frame(motion_ids, motion_times_B)
    root_pos_B, root_rot_B, root_vel_B, root_ang_vel_B, joint_rot_B, dof_vel_B = mlib_B_frames[0:6]

    root_pos_A = root_pos_A.squeeze(0)
    root_pos_B = root_pos_B.squeeze(0)

    heading_A = torch_util.calc_heading(root_rot_A)
    heading_B = torch_util.calc_heading(root_rot_B)
    heading_diff = heading_A - heading_B

    z_axis = torch.tensor([0.0, 0.0, 1.0])
    heading_diff_quat = torch_util.axis_angle_to_quat(z_axis, heading_diff).squeeze(0)

    root_pos_B = torch_util.quat_rotate(heading_diff_quat, root_pos_B)
    root_pos_diff = root_pos_A - root_pos_B
    root_pos_diff[2] = 0.0

    return match_t_A, match_t_B, heading_diff, root_pos_diff


def change_motion_fps(src_frames, src_fps, tar_fps, char_model: kin_char_model.KinCharModel):

    device = src_frames.device

    # assumes the first index is the number of motions index
    mlib = motion_lib.MotionLib(src_frames.unsqueeze(0), char_model, device, "motion_frames", loop_mode=motion_lib.LoopMode.CLAMP, fps=src_fps)

    tar_dt = 1.0 / tar_fps
    src_duration = mlib._motion_lengths[0].item()

    new_frames = []

    motion_ids = torch.tensor([0], dtype=torch.int64, device=device)

    t = 0.0
    while t < src_duration:

        t_th = torch.tensor([t], dtype=torch.float32, device=device)

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = mlib.calc_motion_frame(motion_ids, t_th)

        root_rot = torch_util.quat_to_exp_map(root_rot)
        joint_dof = mlib.joint_rot_to_dof(joint_rot)

        curr_frame = torch.cat([root_pos, root_rot, joint_dof], dim=-1).squeeze(0)
        new_frames.append(curr_frame)

        t += tar_dt


    new_frames = torch.stack(new_frames)

    return new_frames


def scale_motion_segment(motion_frames, scale, start_frame_idx, end_frame_idx):
    xy_disp = motion_frames[end_frame_idx, 0:2] - motion_frames[start_frame_idx, 0:2]

    new_xy_disp = scale * xy_disp # random stretch/squish value

    xy_disp_ratio = torch.nan_to_num(new_xy_disp / xy_disp, nan=1.0)
    #xy_disp
    canon_xy = motion_frames[start_frame_idx, 0:2].clone()
    motion_frames[start_frame_idx:end_frame_idx+1, 0:2] -= canon_xy
    motion_frames[start_frame_idx:end_frame_idx+1, 0:2] *= xy_disp_ratio
    motion_frames[start_frame_idx:end_frame_idx+1, 0:2] += canon_xy

    motion_frames[end_frame_idx+1:, 0:2] += new_xy_disp - xy_disp

    return

def spatially_vary_motion(motion_frames: torch.Tensor, 
                          contact_frames:torch.Tensor, 
                          char_model: kin_char_model.KinCharModel,
                          max_scale = 1.3,
                          min_scale = 0.8,
                          max_angle = 30.0):
    device = motion_frames.device
    t_body_names = ["left_foot", "right_foot"]
    t_body_ids = []
    for body_name in t_body_names:
        body_id = char_model.get_body_id(body_name)
        t_body_ids.append(body_id)

    foot_plant_frames = []

    for body_id in t_body_ids:
        contacts = contact_frames[..., body_id]

        contact_inds = (contacts > 0).nonzero().squeeze(-1)

        contact_chain_first_ind, contact_chain_last_ind = motion_graph.find_chains(contact_inds)

        chain_mid_points = (contact_chain_first_ind + contact_chain_last_ind) // 2

        foot_plant_frames.append(chain_mid_points)

    foot_plant_frames = torch.cat(foot_plant_frames)
    foot_plant_frames = torch.sort(foot_plant_frames).values
    foot_plant_frames = torch.unique(foot_plant_frames) # remove duplicates
    
    new_frames = motion_frames.clone()

    # stretch/squish
    for i in range(foot_plant_frames.shape[0] - 1):

        start_frame_idx = foot_plant_frames[i]
        end_frame_idx = foot_plant_frames[i+1]

        if end_frame_idx - start_frame_idx <= 1:
            continue


        # random scale of motion segment
        scale = random.random() * (max_scale - min_scale) + min_scale
        scale_motion_segment(new_frames, scale, start_frame_idx, end_frame_idx)
        
        # random rotation of motion segment
        heading_angle = 2.0* (random.random() - 0.5) * max_angle * (torch.pi/180.0)
        

        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        heading_angle = torch.tensor([heading_angle], dtype=torch.float32, device=device)
        rot_quat = torch_util.axis_angle_to_quat(z_axis, heading_angle)
        new_new_frames = rotate_motion(new_frames[start_frame_idx:], 
                                rot_quat, 
                                new_frames[start_frame_idx, 0:3].clone())
        
        new_frames[start_frame_idx:] = new_new_frames

    return new_frames

def remove_hesitation_frames(motion_frames: torch.Tensor, 
                             contact_frames: torch.Tensor,
                             char_model: kin_char_model.KinCharModel,
                             hesitation_val = 0.15,
                             hesitation_min_seq_len = 4,
                             verbose=False):

    def find_consecutive_groups(int_set):
        if len(int_set) == 0:
            return []
        # Sort the set to process in ascending order
        sorted_list = sorted(int_set)
        
        # Initialize variables
        result = []
        current_group = [sorted_list[0]]
        
        # Iterate through the sorted list
        for i in range(1, len(sorted_list)):
            if sorted_list[i] == sorted_list[i - 1] + 1:
                # Add to the current group if consecutive
                current_group.append(sorted_list[i])
            else:
                # Otherwise, start a new group
                result.append(current_group)
                current_group = [sorted_list[i]]
        
        # Append the last group
        result.append(current_group)
        return result


    root_pos = motion_frames[:, 0:3]
    root_rot = torch_util.exp_map_to_quat(motion_frames[:, 3:6])
    joint_rot = char_model.dof_to_rot(motion_frames[:, 6:])

    # Find sequences of frames where the start and end pose are very similar
    # and there was minimal motion in between

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)

    # for each frame, do a brute force search to find similar future frames.
    num_frames = body_pos.shape[0]
    hesitation_frames = set()
    for i in range(num_frames):
        if i in hesitation_frames:
            continue
        for j in range(i+1, num_frames):
            body_pos_diff = body_pos[j] - body_pos[i]
            body_pos_dist = torch.linalg.norm(body_pos_diff)

            if body_pos_dist < hesitation_val:
                hesitation_frames.add(j)
    hesitation_groups = find_consecutive_groups(hesitation_frames)

    new_hesitation_groups = []
    hesitation_frames = []
    for group in hesitation_groups:
        if len(group) < hesitation_min_seq_len:
            continue
        new_hesitation_groups.append(group)
        hesitation_frames.extend(group)

    if verbose:
        print("HESITATION FRAMES")
        print(new_hesitation_groups)

    new_motion_frames = []
    new_contact_frames = []
    for i in range(num_frames):
        if i in hesitation_frames:
            continue
        new_motion_frames.append(motion_frames[i])
        new_contact_frames.append(contact_frames[i]) # NOTE: might need to do something special here?

    new_motion_frames = torch.stack(new_motion_frames, dim=0)
    new_contact_frames = torch.stack(new_contact_frames, dim=0)

    return new_motion_frames, new_contact_frames