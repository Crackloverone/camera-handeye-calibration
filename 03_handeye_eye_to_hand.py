#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_handeye_eye_to_hand.py —— 眼在手外（Eye-to-Hand）手眼标定
==============================================================

对应课程实验：实验 9 第 5 步 眼在手外-手眼标定

构型说明
--------
    相机固定在外部支架上（不动），标定板装在机械臂末端（随臂运动）。
    求：X = T_base←cam，即相机在机器人基座坐标系下的固定位姿。

核心方程
--------
    物理模型： T_base←board = T_base←grip(i) · Y = X · T_cam←board(i)
               Y = T_grip←board 固定不变
    令  a_i = T_grip←base(i) = T_base←grip(i)⁻¹    ← 注意！这里要对正解取逆
        b_i = T_cam←board(i)
    则  a_i · X · b_i = 常量  ⇒  A·X = X·B ,
        A = a_j⁻¹ a_i ,  B = b_j b_i⁻¹

与 eye-in-hand 的唯一差别
-------------------------
    eye-in-hand ： a_i = T_base←grip(i)  → 解出 X = T_grip←cam
    eye-to-hand ： a_i = T_grip←base(i)  → 解出 X = T_base←cam
    其余完全一样。实测（见 test_handeye_synthetic.py【B】组）：
    忘记取逆会造成旋转误差 180°、平移误差 380+ mm —— 这就是工程上
    「结果完全离谱」的头号原因。

平面作业的简化路线
------------------
    若工件都在同一工作平面（固定 Z 高度），不必做完整 SE(3) 手眼标定，
    用「九点标定法」拟合 2D 仿射矩阵即可，见 04_nine_point_affine.py。
    实验 9 采用的就是这条简化路线（Z=0 假设 + 仿射变换）。

用法
----
python 03_handeye_eye_to_hand.py --mode capture --camera 0 \
       --intrinsics calib_out/intrinsics.json --dataset dataset_e2h
python 03_handeye_eye_to_hand.py --mode solve --dataset dataset_e2h
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

import calib_utils as cu
import handeye_capture as hc


def load_intrinsics(path):
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    return (np.array(res["K"], dtype=np.float64),
            np.array(res["dist"], dtype=np.float64).ravel())


def mode_capture(args):
    K, dist = load_intrinsics(args.intrinsics)
    board = {
        "type": args.board_type,
        "inner_cols": args.cols,
        "inner_rows": args.rows,
        "square_mm": args.square,
    }
    if args.board_type == "charuco":
        board.update({"squares_x": args.squares_x, "squares_y": args.squares_y,
                      "marker_mm": args.marker})
    ds = hc.HandEyeDataset(args.dataset, K, dist, board, eye_in_hand=False,
                           notes="eye-to-hand, 实验9 手眼标定（标定板装末端）")
    hc.capture_console(ds, cam_index=args.camera,
                       width=args.width, height=args.height)
    return 0


def mode_solve(args):
    print("=" * 74)
    print("  Eye-to-Hand 手眼标定求解  →  求 X = T_base←cam")
    print("=" * 74)
    g_list, c_list, K, dist, meta = hc.HandEyeDataset.load(args.dataset)
    n = len(g_list)
    print(f"  载入 {n} 个位姿")
    if n < 3:
        print("[错误] 位姿数不足 3 个。")
        return 1

    cu.motion_informativeness(g_list)

    print("  按 5 种算法独立求解（注意 eye_in_hand=False 会自动对机器人位姿取逆）：")
    sols = cu.cross_check_methods(g_list, c_list, eye_in_hand=False)
    X = sols.get("PARK", list(sols.values())[0])

    A, B = cu.build_AB(g_list, c_list, eye_in_hand=False)
    res = cu.axxb_residual(A, B, X)
    print("  【第一层验证】AX=XB 残差")
    print(f"    旋转均值 {res['rot_mean_deg']:.4f}°  平移均值 "
          f"{res['trans_mean_mm']:.4f} mm")

    # 几何自洽性：T_base←board = G_i · Y = X · C_i，用首帧定 Y 再逐帧比对
    Y = cu.inv_T(g_list[0]) @ X @ c_list[0]      # T_grip←board（应恒定）
    spread_r, spread_t = 0.0, 0.0
    for i in range(n):
        pred = X @ c_list[i]
        ref = g_list[i] @ Y
        a_, b_ = cu.pose_diff(ref, pred)
        spread_r = max(spread_r, a_)
        spread_t = max(spread_t, b_)
    print("\n  【第二层验证】G_i·Y 应恒等于 X·C_i")
    print(f"    离散度：{spread_r:.4f}° / {spread_t:.4f} mm  "
          f"{'通过' if spread_r < 0.5 and spread_t < 5 else '偏大，建议补采'}")

    print("\n" + "=" * 74)
    print("  最终手眼矩阵 X = T_base←cam")
    print("=" * 74)
    print(np.array2string(X, precision=6, suppress_small=True))
    rvec, tvec = cu.T_to_rvec_tvec(X)
    print(f"\n  平移向量 (m)  : {np.array2string(tvec, precision=6)}")
    print(f"  相机距基座 (mm): {np.linalg.norm(tvec)*1000:.2f}")
    print(f"  旋转角 (deg)  : {cu.rot_angle_deg(X[:3, :3]):.3f}")

    out = {
        "config": "eye_to_hand",
        "X_base2cam": X.tolist(),
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "rot_deg": cu.rot_angle_deg(X[:3, :3]),
        "axxb_residual": res,
        "const_spread": {"rot_deg": spread_r, "trans_mm": spread_t},
        "all_methods": {k: v.tolist() for k, v in sols.items()},
        "K": K.tolist(),
        "dist": dist.tolist(),
    }
    path = os.path.join(args.dataset, "handeye_result.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n[保存] {path}")

    print("\n  应用示例：像素点 → 机器人基座坐标系（射线与工作平面求交）")
    u, v = args.pixel_u, args.pixel_v
    if u is not None and v is not None:
        # 步骤 1：像素点去畸变，并取出相机系下的归一化射线方向
        norm = cv2.undistortPoints(
            np.array([[[float(u), float(v)]]], dtype=np.float64),
            K, dist).reshape(2)
        ray_cam = np.array([norm[0], norm[1], 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        # 步骤 2：把射线方向与光心变换到机器人基座系
        R = X[:3, :3]
        cam_origin_base = tvec.reshape(3)
        ray_base = R @ ray_cam

        # 步骤 3：与水平工作平面 Z = plane_z 求交
        z = args.plane_z
        denom = ray_base[2]
        if abs(denom) < 1e-9:
            print("    [警告] 射线与工作平面近乎平行，无法求交。")
        else:
            s = (z - cam_origin_base[2]) / denom
            pt_base = cam_origin_base + s * ray_base
            print(f"    像素 ({u:.1f}, {v:.1f}) 在工作平面 Z={z:.3f} m 上")
            print(f"    ⇒ 机器人坐标 ≈ ({pt_base[0]*1000:.2f}, "
                  f"{pt_base[1]*1000:.2f}, {pt_base[2]*1000:.2f}) mm")
            print("    这是「像素 → 基座」的完整解法：去畸变 → 归一化射线 →")
            print("    用 X 把射线转到基座系 → 与已知高度平面求交。")
            print("    若工作平面在基座系里不是 Z=常量，请改用一般平面方程 ax+by+cz=d。")
    print("\n  注意：易错点排查清单")
    print("    [ ] 机器人位姿是否统一为 T_base←grip 方向？")
    print("    [ ] solvePnP 输出的 T 是 T_cam←board，不要多求一次逆？")
    print("    [ ] 单位是否统一为米？（示教器常显示 mm）")
    print("    [ ] 旋转用旋转向量还是欧拉角？顺序是否一致？")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="Eye-to-Hand 手眼标定教学脚本")
    p.add_argument("--mode", required=True, choices=["capture", "solve"])
    p.add_argument("--dataset", default="dataset_e2h")
    p.add_argument("--intrinsics",
                   default=os.path.join("calib_out", "intrinsics.json"))
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--board_type", default="charuco",
                   choices=["chessboard", "charuco"])
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--square", type=float, default=25.0)
    p.add_argument("--squares_x", type=int, default=7)
    p.add_argument("--squares_y", type=int, default=5)
    p.add_argument("--marker", type=float, default=22.0)
    p.add_argument("--pixel_u", type=float, default=None)
    p.add_argument("--pixel_v", type=float, default=None)
    p.add_argument("--plane_z", type=float, default=0.02)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit({"capture": mode_capture,
              "solve": mode_solve}[args.mode](args) or 0)
