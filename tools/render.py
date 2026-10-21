"""Render recorded evaluation trajectories (supermimic/evaluate.py --record) with MuJoCo (EGL).

The simulated robot is drawn solid; the reference motion is a translucent ghost anchored to the
robot's xy, as the tracking reward does.
Usage: MUJOCO_GL=egl python tools/render.py traj.npz out.mp4 --clips dance2_subject1 jumps1_subject1 --seconds 12
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supermimic.robot import G1_JOINT_NAMES, MJCF_PATH


def two_robot_model(width: int, height: int) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    ghost = mujoco.MjSpec.from_file(MJCF_PATH)
    ghost.add_material(name="ghost", rgba=[0.35, 0.65, 1.0, 0.25])
    for g in ghost.geoms:
        g.material = "ghost"
    for s in (spec, ghost):
        for g in s.geoms:
            g.contype, g.conaffinity = 0, 0
    for sens in list(ghost.sensors):
        ghost.delete(sens)
    spec.worldbody.add_frame().attach_body(ghost.body("pelvis"), "ref/", "")
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.92, 0.93, 0.94], rgb2=[0.80, 0.82, 0.85], width=512, height=512)  # fmt: skip
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.95, 0.96, 0.98], rgb2=[0.75, 0.8, 0.88], width=256, height=256)  # fmt: skip
    mat = spec.add_material(name="grid", texrepeat=[8, 8], reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid",
                            contype=0, conaffinity=0)  # fmt: skip
    spec.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.8, 0.8, 0.8])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    spec.visual.quality.shadowsize = 4096
    return spec.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--title", type=str, default="")
    a = ap.parse_args()

    d = dict(np.load(a.traj))
    names = list(d["names"])
    model = two_robot_model(a.width, a.height)
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    data = mujoco.MjData(model)
    root = model.joint("floating_base_joint").qposadr[0]
    ref_root = model.joint("ref/floating_base_joint").qposadr[0]
    qadr = np.array([model.joint(n).qposadr[0] for n in G1_JOINT_NAMES])
    ref_qadr = np.array([model.joint("ref/" + n).qposadr[0] for n in G1_JOINT_NAMES])
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.6, 135.0, -12.0
    font = ImageFont.load_default(size=22)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}", "-r", str(a.fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    stride = int(round(50 / a.fps))
    for clip in a.clips or names:
        c = names.index(clip)
        T = d["root_p"].shape[0]
        alive = d["alive"][:, c]
        t0 = int(a.start * 50)
        t1 = min(T, t0 + int(a.seconds * 50))
        fell_at = None
        for t in range(t0, t1, stride):
            if not alive[t] and fell_at is None:
                fell_at = t
            tt = t if alive[t] else (fell_at if fell_at is not None else t)
            f = int(d["f"][tt, c])
            data.qpos[root : root + 3] = d["root_p"][tt, c]
            data.qpos[root + 3 : root + 7] = d["root_quat"][tt, c]
            data.qpos[qadr] = d["q"][tt, c]
            data.qpos[ref_root : ref_root + 2] = d["root_p"][tt, c][:2]
            data.qpos[ref_root + 2] = d["ref_root_p"][f][2]
            mujoco.mju_mat2Quat(data.qpos[ref_root + 3 : ref_root + 7], d["ref_root_R"][f].reshape(-1).astype(np.float64))
            data.qpos[ref_qadr] = d["ref_q"][f]
            mujoco.mj_kinematics(model, data)
            cam.lookat[:] = [data.qpos[root], data.qpos[root + 1], 0.75]
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            label = f"{a.title}  {clip}  t={(f - int(d['clip_start'][c])) * 0.02:5.1f}s"
            if fell_at is not None:
                label += "  [terminated]"
            for xy, txt in (((16, 12), label), ((16, a.height - 38), "solid: simulated G1 policy (mjlab, MuJoCo-Warp)   blue ghost: reference")):
                bb = draw.textbbox(xy, txt, font=font)
                draw.rectangle((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), fill=(255, 255, 255))
                draw.text(xy, txt, fill=(20, 20, 20), font=font)
            ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
