#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import vdbfusion
import trimesh
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import normal_from_depth_image, depth2point


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "median_depth")
    normal_from_depth_image_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_from_depth")
    normal_from_gs_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_from_gs")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(normal_from_depth_image_path, exist_ok=True)
    makedirs(normal_from_gs_path, exist_ok=True)

    vdb_volume = vdbfusion.VDBVolume(voxel_size=0.004, sdf_trunc=0.02, space_carving=False) # For Scene

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        median_depth = results["median_depth"]
        median_depth_image = median_depth / (median_depth.max() + 1e-5)
        normal_from_depth = normal_from_depth_image(median_depth[0], view.intrinsics.cuda(), 
                                                    view.extrinsics.cuda())[0].permute(2, 0, 1)
        normal_from_gs = results["normal_from_gs"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(median_depth_image, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normal_from_depth, os.path.join(normal_from_depth_image_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normal_from_gs, os.path.join(normal_from_gs_path, '{0:05d}'.format(idx) + ".png"))
        if True:
            rendered_pcd_cam, rendered_pcd_world = depth2point(median_depth[0], view.intrinsics.to(median_depth.device), 
                                                                view.extrinsics.to(median_depth.device))
            P = view.extrinsics
            P_inv = P.inverse()
            cam_center = P_inv[:3, 3]
            invalid_mask = median_depth[0] > 6
            median_depth[0][invalid_mask] = 0
            rendered_pcd_cam, rendered_pcd_world = depth2point(median_depth[0], view.intrinsics.to(median_depth.device), 
                                                    view.extrinsics.to(median_depth.device))
            rendered_pcd_world = rendered_pcd_world[~invalid_mask]
            vdb_volume.integrate(rendered_pcd_world.double().cpu().numpy(), extrinsic=cam_center.double().cpu().numpy())

    vertices, faces = vdb_volume.extract_triangle_mesh(min_weight=5)
    geo_mesh = trimesh.Trimesh(vertices, faces)
    geo_mesh.export(os.path.join(model_path, 'fused_mesh.ply'))
    print("Fused mesh saved to {}".format(os.path.join(model_path, 'fused_mesh.ply')))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)