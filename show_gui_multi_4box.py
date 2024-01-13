import torch
from opt import get_opts
import numpy as np
from einops import rearrange
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import time

from datasets import dataset_dict
from datasets.ray_utils import get_ray_directions, get_rays
from models.networks import NGP
from models.rendering import render
from train import depth2img
from utils import load_ckpt

import warnings; warnings.filterwarnings("ignore")

import pdb


class OrbitCamera:
    def __init__(self, K, img_wh, r):
        self.K = K
        self.W, self.H = img_wh
        self.radius = r
        self.center = np.zeros(3)
        self.rot = np.eye(3)

        # --- for replica room0 initial pose --- #
        # self.radius = 0.1302
        # self.center = np.array([-0.190185  , -0.0569687 , -0.01437071])
        # self.rot = np.array([[-0.27136618, -0.16452724,  0.94830965],
        #                     [-0.9619266 ,  0.07965296, -0.26144336],
        #                     [-0.03252111, -0.98315116, -0.17987824]])

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(0.05 * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                   R.from_rotvec(rotvec_x).as_matrix() @ \
                   self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-4 * self.rot @ np.array([dx, dy, dz])


class NGPGUI:
    def __init__(self, hparams, K, img_wh, radius=2.5):
        self.hparams = hparams
        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        # network_kwargs = {
        #     'semantic_flag' : self.hparams.semantic_flag,
        #     'semantic_dim': self.hparams.semantic_dim,
        # }
        # self.model = NGP(scale=hparams.scale, rgb_act=rgb_act, **network_kwargs).cuda()
        self.model = NGP(scale=hparams.scale, rgb_act=rgb_act,
                         semantic_flag=hparams.semantic_flag).cuda()
        load_ckpt(self.model, hparams.ckpt_path)
        self.model.requires_grad_(False)

        self.cam = OrbitCamera(K, img_wh, r=radius)
        self.W, self.H = img_wh
        self.render_buffer = np.ones((self.W*2, self.H*2, 3), dtype=np.float32)

        # placeholders
        self.dt = 0
        self.mean_samples = 0
        self.img_mode = 0
        self.clickmode_button = False
        self.clickmode_multi_button = False     # choose multiple object 
        self.new_click = False
        self.prompt_num = 0
        self.new_click_xy = []
        self.clear_edit = False
        self.binary_threshold_button = False

        # self.pca_proj_mat = np.loadtxt("results/mvsnet/0712_twg_2_svr6/results/pca_proj_mat/pca_proj_mat.txt")
        # self.pca_proj_mat = np.loadtxt("results/replica/0806_room0_1/results/pca_proj_mat/pca_proj_mat.txt")
        # self.pca_proj_mat = np.loadtxt("results/replica/0806_room0_2_no_order/results/pca_proj_mat/pca_proj_mat.txt")
        
        # self.pca_proj_mat = np.loadtxt("results/replica/0826_room0_8_cc_debug_Hi_decre10/results/pca_proj_mat/pca_proj_mat.txt")

        # self.pca_proj_mat = np.loadtxt("results/colmap/360_sam/1122_counter/results/pca_proj_mat/pca_proj_mat.txt")
        # self.pca_proj_mat = np.loadtxt("results/replica/0826_room0_8_cc_debug_Hi_decre10/results/pca_proj_mat/pca_proj_mat.txt")
        # self.pca_proj_mat = torch.FloatTensor(self.pca_proj_mat).cuda()

        # pdb.set_trace()
        self.register_dpg()

    def render_cam(self, cam):
        t = time.time()
        directions = get_ray_directions(cam.H, cam.W, cam.K, device='cuda')
        rays_o, rays_d = get_rays(directions, torch.cuda.FloatTensor(cam.pose))

        # TODO: set these attributes by gui
        if self.hparams.dataset_name in ['colmap', 'nerfpp']:
            exp_step_factor = 1/256
        else: exp_step_factor = 0

        # self.model.aabb_min[0, 0] = dpg.get_value('_Xmin')
        # self.model.aabb_max[0, 0] = dpg.get_value('_Xmax')
        # self.model.aabb_min[0, 1] = dpg.get_value('_Ymin')
        # self.model.aabb_max[0, 1] = dpg.get_value('_Ymax')
        # self.model.aabb_min[0, 2] = dpg.get_value('_Zmin')
        # self.model.aabb_max[0, 2] = dpg.get_value('_Zmax')

        results = render(self.model, rays_o, rays_d,
                         **{'test_time': True,
                            'to_cpu': False, 'to_numpy': False,
                            'T_threshold': 1e-2,
                            'exposure': torch.cuda.FloatTensor([0.2]), # torch.cuda.FloatTensor([dpg.get_value('_exposure')]),
                            'max_samples': 100,
                            'exp_step_factor': exp_step_factor,
                            'semantic_flag': self.hparams.semantic_flag,
                            'semantic_dim': self.hparams.semantic_dim})

        rgb = rearrange(results["rgb"], "(h w) c -> h w c", h=self.H)
        depth = rearrange(results["depth"], "(h w) -> h w", h=self.H)
        rgb_score = rgb.clone()
        depth_score = depth.clone()

        if self.hparams.semantic_flag:
            sam_img_raw = results["semantic"]

            # sam_img = sam_img_raw @ self.pca_proj_mat

            # # --- opt1. for closed scene (like Replica) --- #
            # sam_img = (sam_img - sam_img.min())/(sam_img.max() - sam_img.min())     # range: (0, 1)
            # # sam_img = torch.clip(sam_img*0.5+0.5, 0, 1)
            # sam_img = rearrange(sam_img, "(h w) c -> h w c", h=self.H)

            sam_img = rgb.clone()


            W_XYZ = dpg.get_value('_XYZThres')
            # if W_XYZ > 0:
            xyz_map = (rays_o[:, :3] + rays_d[:, :3] * depth.reshape(-1, 1)).reshape(self.H, self.W, 3)
            

            if self.clear_edit:
                self.new_click_xy = []
                self.clear_edit = False
                self.prompt_num = 0

            if len(self.new_click_xy) > 0:

                featmap = sam_img_raw.reshape(self.H, self.W, -1)
                
                if self.new_click:
                    xy = self.new_click_xy
                    new_feat = featmap[int(xy[1])%self.H, int(xy[0])%self.W, :].reshape(featmap.shape[-1], -1)
                    new_xyz = xyz_map[int(xy[1])%self.H, int(xy[0])%self.W, :].reshape(1, 3)
                    if (self.prompt_num == 0) or (self.clickmode_multi_button == False):
                        self.chosen_feature = new_feat
                        self.chosen_xyz = new_xyz
                    else:
                        self.chosen_feature = torch.cat([self.chosen_feature, new_feat], dim=-1)    # extend to get more prompt features
                        self.chosen_xyz = torch.cat([self.chosen_xyz, new_xyz], dim=0)
                    self.prompt_num += 1
                    self.new_click = False
                
                score_map = featmap @ self.chosen_feature
                if W_XYZ > 0: score_map *= torch.cdist(xyz_map, self.chosen_xyz[None, :, :]).mul_(-W_XYZ).exp_()

                score_map = (score_map + 1.0) / 2
                score_binary = score_map > dpg.get_value('_ScoreThres')

                # score_norm = (score_map - score_map.min())/(score_map.max() - score_map.min())     # range: (0, 1)
                
                score_map[~score_binary] = 0.0
                score_map = torch.max(score_map, dim=-1).values
                score_norm = (score_map - dpg.get_value('_ScoreThres')) / (1 - dpg.get_value('_ScoreThres'))
                # score_norm /= score_norm.max()
                # score_norm = torch.clip(score_map - dpg.get_value('_ScoreThres'), 0, 1)


                if self.binary_threshold_button:
                    rgb_score = rgb * torch.max(score_binary, dim=-1, keepdim=True).values    # option: binary
                else:
                    rgb_score = rgb * score_norm[:, :, None]
                    # rgb_score = rgb * score_map[:, :, None]
                depth_score = 1 - torch.clip(score_norm, 0, 1)





        torch.cuda.synchronize()
        self.dt = time.time()-t
        self.mean_samples = results['total_samples']/len(rays_o)

        if self.img_mode == 0:
            # return rgb_score.cpu().numpy()
            rgb = rgb.cpu().numpy()
            rgb_score = rgb_score.cpu().numpy()
            depth = depth2img(depth.cpu().numpy()).astype(np.float32)/255.0
            depth_score = depth2img(depth_score.cpu().numpy()).astype(np.float32)/255.0
            row_1 = np.concatenate([rgb, sam_img.cpu().numpy()], axis=1)
            row_2 = np.concatenate([rgb_score, depth_score], axis=1)
            return np.concatenate([row_1, row_2], axis=0)
        
        if self.img_mode == 0:
            return rgb_score.cpu().numpy()
        elif self.img_mode == 1:
            return depth2img(depth_score.cpu().numpy()).astype(np.float32)/255.0
        elif self.img_mode == 2:
            return rgb.cpu().numpy()
            # return sam_img.cpu().numpy()


    def register_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title="ngp_pl", width=self.W*2+200, height=self.H*2, resizable=False)

        ## register texture ##
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W*2,
                self.H*2,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture")

        ## register window ##
        with dpg.window(tag="_primary_window", width=self.W*2, height=self.H*2):
            dpg.add_image("_texture")
        dpg.set_primary_window("_primary_window", True)

        def callback_depth(sender, app_data):
            if self.hparams.semantic_flag:
                self.img_mode = (self.img_mode + 1) % 3
            else:
                self.img_mode = 1-self.img_mode
        

        def change_pos(sender, app_data):
            xy = dpg.get_mouse_pos(local=False)
            dpg.set_value("pos_item", f"Mouse position = ({xy[0]}, {xy[1]})")
            if self.clickmode_button and app_data == 1:     # in the click mode and right click
                print(xy)
                self.new_click_xy = np.array(xy)
                self.new_click = True

        def clickmode_callback(sender):
            # --- mode switch --- #
            self.clickmode_button = 1 - self.clickmode_button

        def clickmode_multi_callback(sender):
            # --- mode switch --- #
            self.clickmode_multi_button = dpg.get_value(sender)
            print("clickmode_multi_button = ", self.clickmode_multi_button)
        
        def binary_threshold_callback(sender):
            # --- mode switch --- #
            self.binary_threshold_button = dpg.get_value(sender)
            print("binary_threshold_button = ", self.binary_threshold_button)

        def clear_edit():
            self.clear_edit = True

        ## control window ##
        with dpg.window(label="Control", tag="_control_window", width=200, height=150):
            # dpg.add_slider_float(label="exposure", default_value=0.2,
            #                      min_value=1/60, max_value=32, tag="_exposure")
            dpg.add_slider_float(label="ScoreThres", default_value=0.0,
                                 min_value=0.0, max_value=1.0, tag="_ScoreThres")
            dpg.add_button(label="show depth", tag="_button_depth",
                            callback=callback_depth)
            dpg.add_text("Mouse position: click anywhere to start. ", tag="pos_item")
            dpg.add_checkbox(label="clickmode", callback=clickmode_callback, user_data="Some Data")
            dpg.add_checkbox(label="multi-clickmode", callback=clickmode_multi_callback, user_data="Some Data")
            dpg.add_checkbox(label="binary_threshold", callback=binary_threshold_callback, user_data="Some Data")
            dpg.add_button(label="clear_edit", callback=clear_edit, user_data="Some Data")
            dpg.add_separator()
            dpg.add_text('no data', tag="_log_time")
            dpg.add_text('no data', tag="_samples_per_ray")

            # aabb_min = self.model.aabb_min.cpu().numpy()
            # aabb_max = self.model.aabb_max.cpu().numpy()
            # dpg.add_slider_float(label="Xmin", default_value=aabb_min[0, 0],
            #                      min_value=aabb_min[0, 0], max_value=aabb_max[0, 0], tag="_Xmin")
            # dpg.add_slider_float(label="Xmax", default_value=aabb_max[0, 0],
            #                      min_value=aabb_max[0, 0], max_value=aabb_min[0, 0], tag="_Xmax")

            # dpg.add_slider_float(label="Ymin", default_value=aabb_min[0, 1],
            #                      min_value=aabb_min[0, 1], max_value=aabb_max[0, 1], tag="_Ymin")
            # dpg.add_slider_float(label="Ymax", default_value=aabb_max[0, 1],
            #                      min_value=aabb_max[0, 1], max_value=aabb_min[0, 1], tag="_Ymax")

            # dpg.add_slider_float(label="Zmin", default_value=aabb_min[0, 2],
            #                      min_value=aabb_min[0, 2], max_value=aabb_max[0, 2], tag="_Zmin")
            # dpg.add_slider_float(label="Zmax", default_value=aabb_max[0, 2],
            #                      min_value=aabb_max[0, 2], max_value=aabb_min[0, 2], tag="_Zmax")

            dpg.add_slider_float(label="XYZThres", default_value=0.0,
                                 min_value=0.0, max_value=2.0, tag="_XYZThres")

        ## register camera handler ##
        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.orbit(app_data[1], app_data[2])

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.scale(app_data)

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.pan(app_data[1], app_data[2])

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )
            dpg.add_mouse_click_handler(callback=change_pos)

        ## Avoid scroll bar in the window ##
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        ## Launch the gui ##
        dpg.setup_dearpygui()
        dpg.set_viewport_small_icon("assets/icon.png")
        dpg.set_viewport_large_icon("assets/icon.png")
        dpg.show_viewport()

    def render(self):
        while dpg.is_dearpygui_running():
            dpg.set_value("_texture", self.render_cam(self.cam))
            dpg.set_value("_log_time", f'Render time: {1000*self.dt:.2f} ms')
            dpg.set_value("_samples_per_ray", f'Samples/ray: {self.mean_samples:.2f}')
            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    hparams = get_opts()
    kwargs = {'root_dir': hparams.root_dir,
              'downsample': hparams.downsample,
              'read_meta': False}
    dataset = dataset_dict[hparams.dataset_name](**kwargs)

    NGPGUI(hparams, dataset.K, dataset.img_wh).render()
    dpg.destroy_context()
