import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import open3d as o3d
import viser
import viser.transforms as viser_tf
from termcolor import colored
from typing import Dict, List

from src.vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from src.vggt.utils.load_fn import load_and_preprocess_images
from src.vggt.utils.pose_enc import pose_encoding_to_extri_intri

from src.da3_slam.loop_closure import ImageRetrieval
from src.da3_slam.frame_overlap import FrameTracker
from src.da3_slam.map import GraphMap
from src.da3_slam.submap import Submap
from src.da3_slam.h_solve import ransac_projective
from src.da3_slam.gradio_viewer import TrimeshViewer

def color_point_cloud_by_confidence(pcd, confidence, cmap='viridis'):
    """
    Color a point cloud based on per-point confidence values.
    """
    assert len(confidence) == len(pcd.points), "Confidence length must match number of points"
    confidence_normalized = (confidence - np.min(confidence)) / (np.ptp(confidence) + 1e-8)
    colormap = plt.get_cmap(cmap)
    colors = colormap(confidence_normalized)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

class Viewer:
    def __init__(self, port: int = 8080):
        print(f"Starting viser server on port {port}")

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # --- [新增] 1. 全局控制：显示/隐藏动态物体 ---
        self.gui_show_dynamic = self.server.gui.add_checkbox(
            "Show Dynamic Objects",
            initial_value=True,
        )
        self.gui_show_dynamic.on_update(self._on_update_show_dynamic)

        # Global toggle for all frames and frustums
        self.gui_show_frames = self.server.gui.add_checkbox(
            "Show Cameras",
            initial_value=True,
        )
        self.gui_show_frames.on_update(self._on_update_show_frames)

        # Store frames and frustums by submap
        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}
        
        # --- [新增] 2. 存储动态点云的 Handle，以便统一控制显隐 ---
        self.dynamic_pcd_handles: List[viser.PointCloudHandle] = []

        num_rand_colors = 250
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int, image_scale: float=0.5) -> None:
        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if submap_id not in self.submap_frames:
            self.submap_frames[submap_id] = []
            self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            # Add the coordinate frame
            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            # Convert image and add frustum
            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            img_resized = cv2.resize(
                img,
                (int(img.shape[1] * image_scale), int(img.shape[0] * image_scale)),
                interpolation=cv2.INTER_AREA
            )

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img_resized,
                line_width=3.0,
                color=self.random_colors[submap_id]
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)
    
    # --- [新增] 3. 封装添加点云的逻辑，自动分离静态和动态 ---
    def add_split_point_cloud(self, name: str, points: np.ndarray, colors: np.ndarray, point_size: float):
        """
        根据颜色（纯红色 [255, 0, 0]）自动分离静态和动态点云，并添加到 Viser。
        """
        # 检测红色点 (Dynamic)
        # colors 是 float [0,1] 还是 uint8 [0,255]? 根据 submap.py 来看通常是 float, 
        # 但在 solver.add_points 里转成了 uint8。这里为了稳健，先转为 uint8 判断。
        
        colors_uint8 = colors
        if colors.max() <= 1.01:
             colors_uint8 = (colors * 255).astype(np.uint8)
        else:
             colors_uint8 = colors.astype(np.uint8)

        # 定义红色掩码: R > 200, G < 50, B < 50 (给予少量容差，或者严格匹配 [255, 0, 0])
        is_dynamic = (colors_uint8[:, 0] >= 250) & (colors_uint8[:, 1] == 0) & (colors_uint8[:, 2] == 0)
        
        # 1. 添加静态点 (Static)
        pts_static = points[~is_dynamic]
        col_static = colors[~is_dynamic]
        
        if len(pts_static) > 0:
            self.server.scene.add_point_cloud(
                name=f"pcd_{name}_static",
                points=pts_static,
                colors=col_static,
                point_size=point_size,
                point_shape="circle",
            )

        # 2. 添加动态点 (Dynamic)
        pts_dynamic = points[is_dynamic]
        col_dynamic = colors[is_dynamic]

        if len(pts_dynamic) > 0:
            handle = self.server.scene.add_point_cloud(
                name=f"pcd_{name}_dynamic",
                points=pts_dynamic,
                colors=col_dynamic,
                point_size=point_size,
                point_shape="circle",
            )
            # 存入列表管理
            self.dynamic_pcd_handles.append(handle)
            # 设置初始可见性
            handle.visible = self.gui_show_dynamic.value

    def _on_update_show_frames(self, _) -> None:
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible

    # --- [新增] 4. 更新回调：点击按钮时，批量修改动态点云可见性 ---
    def _on_update_show_dynamic(self, _) -> None:
        visible = self.gui_show_dynamic.value
        # 遍历所有已添加的动态点云 Handle，设置可见性
        # 清理失效的 handle (如果 submap 被删除了) - Viser 通常会自动处理，但简单的 try-except 可以防崩
        valid_handles = []
        for handle in self.dynamic_pcd_handles:
            try:
                handle.visible = visible
                valid_handles.append(handle)
            except:
                pass # Handle 可能已经失效
        self.dynamic_pcd_handles = valid_handles


class Solver:
    def __init__(self,
        init_conf_threshold: float,
        use_point_map: bool = False,
        visualize_global_map: bool = False,
        use_sim3: bool = False,
        gradio_mode: bool = False,
        vis_stride: int = 1,
        vis_point_size: float = 0.001):
        
        self.init_conf_threshold = init_conf_threshold
        self.use_point_map = use_point_map
        self.gradio_mode = gradio_mode

        if self.gradio_mode:
            self.viewer = TrimeshViewer()
        else:
            self.viewer = Viewer()

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.use_sim3 = use_sim3
        if self.use_sim3:
            from src.da3_slam.graph_se3 import PoseGraph
        else:
            from src.da3_slam.graph import PoseGraph
        self.graph = PoseGraph()

        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None
        self.first_edge = True
        self.T_w_kf_minus = None
        self.prior_pcd = None
        self.prior_conf = None
        self.vis_stride = vis_stride
        self.vis_point_size = vis_point_size

        print("Starting viser server...")

    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.gradio_mode:
            self.viewer.add_point_cloud(points_in_world_frame, points_colors)
        else:
            # --- [修改] 调用 Viewer 的智能分离方法，而不是直接调用 server ---
            self.viewer.add_split_point_cloud(
                name=name,
                points=points_in_world_frame,
                colors=points_colors,
                point_size=point_size
            )

    def set_submap_point_cloud(self, submap):
        points_in_world_frame = submap.get_points_in_world_frame(stride = self.vis_stride)
        points_colors = submap.get_points_colors(stride = self.vis_stride)
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, self.vis_point_size)

    def set_submap_poses(self, submap):
        extrinsics = submap.get_all_poses_world()
        if self.gradio_mode:
            for i in range(extrinsics.shape[0]):
                self.viewer.add_camera_pose(extrinsics[i])
        else:
            images = submap.get_all_frames()
            self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def export_3d_scene(self, output_path="output.glb"):
        return self.viewer.export(output_path)

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    def add_points(self, pred_dict):
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "mask": (S, H, W)        - Segmentation mask (0/1),
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        images = pred_dict["images"]  # (S, 3, H, W)
        mask = pred_dict["mask"]      # (S, H, W)
        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
        detected_loops = pred_dict["detected_loops"]

        if self.use_point_map:
            world_points_map = pred_dict["world_points"]
            conf = pred_dict["world_points_conf"]
            world_points = world_points_map
        else:
            depth_map = pred_dict["depth"]
            conf = pred_dict["depth_conf"]
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)

        # Apply mask to turn dynamic objects red
        is_dynamic = mask > 0.5
        colors[is_dynamic] = [255, 0, 0]

        cam_to_world = closed_form_inverse_se3(extrinsics_cam)
        points_in_first_cam = world_points[0,...]
        h, w = points_in_first_cam.shape[0:2]

        new_pcd_num = self.current_working_submap.get_id()
        if self.first_edge:
            self.first_edge = False
            self.prior_pcd = world_points[-1,...].reshape(-1, 3)
            self.prior_conf = conf[-1,...].reshape(-1)

            H_w_submap = np.eye(4)
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_prior_factor(new_pcd_num, H_w_submap, self.graph.anchor_noise)
        else:
            prior_pcd_num = self.map.get_largest_key()
            prior_submap = self.map.get_submap(prior_pcd_num)
            current_pts = world_points[0,...].reshape(-1, 3)
            good_mask = self.prior_conf > prior_submap.get_conf_threshold() * (conf[0,...,:].reshape(-1) > prior_submap.get_conf_threshold())
            
            if self.use_sim3:
                R_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,0:3]
                t_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,3]
                T_temp = np.eye(4)
                T_temp[0:3,0:3] = R_temp
                T_temp[0:3,3] = t_temp
                T_temp = np.linalg.inv(T_temp)
                scale_factor = np.mean(np.linalg.norm((T_temp[0:3,0:3] @ self.prior_pcd[good_mask].T).T + T_temp[0:3,3], axis=1) / np.linalg.norm(current_pts[good_mask], axis=1))
                print(colored("scale factor", 'green'), scale_factor)
                H_relative = np.eye(4)
                H_relative[0:3,0:3] = R_temp
                H_relative[0:3,3] = t_temp

                world_points *= scale_factor
                cam_to_world[:, 0:3, 3] *= scale_factor
            else:
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])
            
            H_w_submap = prior_submap.get_reference_homography() @ H_relative
            non_lc_frame = self.current_working_submap.get_last_non_loop_frame_index()
            pts_cam0_camn = world_points[non_lc_frame,...].reshape(-1, 3)
            self.prior_pcd = pts_cam0_camn
            self.prior_conf = conf[non_lc_frame,...].reshape(-1)

            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)
            print("added between factor", prior_pcd_num, new_pcd_num, H_relative)

        self.current_working_submap.set_reference_homography(H_w_submap)
        self.current_working_submap.add_all_poses(cam_to_world)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, intrinsics_cam)
        self.current_working_submap.set_conf_masks(conf)

        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()
            loop_index = self.current_working_submap.get_last_non_loop_frame_index() + index + 1

            if self.use_sim3:
                pose_world_detected = self.map.get_submap(loop.detected_submap_id).get_pose_subframe(loop.detected_submap_frame)
                pose_world_query = self.current_working_submap.get_pose_subframe(loop_index)
                pose_world_detected = gtsam.Pose3(pose_world_detected)
                pose_world_query = gtsam.Pose3(pose_world_query)
                H_relative_lc = pose_world_detected.between(pose_world_query).matrix()
            else:
                points_world_detected = self.map.get_submap(loop.detected_submap_id).get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)
                points_world_query = self.current_working_submap.get_frame_pointcloud(loop_index).reshape(-1, 3)
                H_relative_lc = ransac_projective(points_world_query, points_world_detected)

            self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)
            self.graph.increment_loop_closure()

            print("added loop closure factor", loop.detected_submap_id, loop.query_submap_id, H_relative_lc)
            print("homography between nodes estimated to be", np.linalg.inv(self.map.get_submap(loop.detected_submap_id).get_reference_homography()) @ H_w_submap)

        self.map.add_submap(self.current_working_submap)

    def sample_pixel_coordinates(self, H, W, n):
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    def run_predictions(self, image_names, model, max_loops):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = load_and_preprocess_images(image_names).to(device)
        print(f"Preprocessed images shape: {images.shape}")

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        new_pcd_num = self.map.get_largest_key() + 1
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(new_submap))

        detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops)
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
        retrieved_frames = self.map.get_frames_from_loops(detected_loops)

        num_loop_frames = len(retrieved_frames)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        if num_loop_frames > 0:
            image_tensor = torch.stack(retrieved_frames)
            images = torch.cat([images, image_tensor], dim=0)
            new_submap.add_all_frames(images)

        self.current_working_submap = new_submap

        S, C, H, W = images.shape
        imgs_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        input_images_list = [img for img in imgs_np]

        print(f"Running DA3 inference on {S} frames...")
        with torch.no_grad():
            da3_out = model.inference(input_images_list)
        
        device = images.device
        
        depth_raw = torch.from_numpy(da3_out.depth).to(device)
        conf_raw = torch.from_numpy(da3_out.conf).to(device)
        
        if hasattr(da3_out, 'motion_seg_mask') and da3_out.motion_seg_mask is not None:
             mask_raw = da3_out.motion_seg_mask.to(device).float()
        else:
             print("Warning: No motion_seg_mask found, using zeros.")
             mask_raw = torch.zeros_like(depth_raw)

        if depth_raw.shape[0] == S and depth_raw.dim() == 3:
            _, H_out, W_out = depth_raw.shape
        else:
            H_out, W_out = depth_raw.shape[-2], depth_raw.shape[-1]

        print(f"Input: {H}x{W}, Output: {H_out}x{W_out}")

        depth = torch.nn.functional.interpolate(
            depth_raw.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(1)
        
        conf = torch.nn.functional.interpolate(
            conf_raw.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(1)

        if mask_raw.dim() == 3:
             mask_in = mask_raw.unsqueeze(1)
        else:
             mask_in = mask_raw

        mask = torch.nn.functional.interpolate(
            mask_in, size=(H, W), mode='nearest'
        ).squeeze(1)

        extrinsics = torch.from_numpy(da3_out.extrinsics).to(device).view(S, 3, 4)
        intrinsics_raw = torch.from_numpy(da3_out.intrinsics).to(device).view(S, 3, 3)

        scale_x = W / W_out
        scale_y = H / H_out
        
        intrinsics = intrinsics_raw.clone()
        intrinsics[:, 0, 0] *= scale_x
        intrinsics[:, 0, 2] *= scale_x
        intrinsics[:, 1, 1] *= scale_y
        intrinsics[:, 1, 2] *= scale_y

        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        pixels = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1).float()
        pixels = pixels.unsqueeze(0).expand(S, -1, -1, -1)
        pixels_flat = pixels.reshape(S, H*W, 3)

        K_inv = torch.inverse(intrinsics)
        cam_points = torch.bmm(K_inv, pixels_flat.transpose(1, 2)).transpose(1, 2)
        cam_points = cam_points * depth.reshape(S, H*W, 1) 

        bottom_row = torch.tensor([0,0,0,1], device=device, dtype=torch.float32).view(1,1,4).expand(S, -1, -1)
        E_4x4 = torch.cat([extrinsics, bottom_row], dim=1)
        E_inv = torch.inverse(E_4x4)
        
        cam_points_homo = torch.cat([cam_points, torch.ones((S, H*W, 1), device=device)], dim=-1)
        world_points_flat = torch.bmm(E_inv, cam_points_homo.transpose(1, 2)).transpose(1, 2)
        world_points = world_points_flat[..., :3].view(S, H, W, 3)

        predictions = {
            "pose_enc": None,         
            "depth": depth.unsqueeze(-1),
            "depth_conf": conf,
            "mask": mask,
            "world_points": world_points,
            "world_points_conf": conf,
            "images": images,
            "extrinsic": extrinsics,
            "intrinsic": intrinsics,
            "detected_loops": detected_loops
        }

        for key in predictions.keys():
             if isinstance(predictions[key], torch.Tensor):
                 predictions[key] = predictions[key].cpu().numpy()
                 if predictions[key].shape[0] == 1 and S == 1: 
                     predictions[key] = predictions[key].squeeze(0)

        return predictions