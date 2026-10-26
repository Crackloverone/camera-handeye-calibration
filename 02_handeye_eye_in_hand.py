#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_handeye_eye_in_hand.py —— 眼在手上（Eye-in-Hand）手眼标定
=============================================================

对应课程实验：实验 8「Aruco 码机械臂随动」第 3 步 手眼标定

构型说明
--------
    相机固定在机械臂末端法兰上，标定板固定在工作台（不动）。
    求：X = T_grip←cam，即相机相对末端法兰的固定位姿。

核心方程
--------
    物理模型： T_base←board = T_base←grip(i) · X · T_cam←board(i) = 常量
    令  a_i = T_base←grip(i)（机器人正解）， b_i = T_cam←board(i)（PnP 外参）
    则  a_i · X · b_i = a_j · X · b_j
    ⇒   A · X = X · B ,  A = a_j⁻¹ a_i ,  B = b_j b_i⁻¹

用法
----
# 1) 采集（需先完成 01 的内参标定）
python 02_handeye_eye_in_hand.py --mode capture --camera 0 \
       --intrinsics calib_out/intrinsics.json --dataset dataset_eih

# 2) 求解 + 交叉验证 + 精度评估
python 02_handeye_eye_in_hand.py --mode solve --dataset dataset_eih

# 3) 合成自检（不用真实设备也能验证流程与约定，课堂演示推荐）
python 02_handeye_eye_in_hand.py --mode selftest
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

import calib_utils as cu
import handeye_capture as hc


# --------------------------------------------------------------------------
def load_intrinsics(path):
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    return (np.array(res["K"], dtype=np.float64),
            np.array(res["dist"], dtype=np.float64).ravel())


# --------------------------------------------------------------------------
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

    ds = hc.HandEyeDataset(args.dataset, K, dist, board,
                           eye_in_hand=True,
                           notes="eye-in-hand, 实验8 手眼标定")
    hc.capture_console(ds, cam_index=args.camera,
                       width=args.width, height=args.height)
    return 0


# --------------------------------------------------------------------------
def mode_solve(args):
    print("=" * 74)
    print("  Eye-in-Hand 手眼标定求解")
    print("=" * 74)
    g_list, c_list, K, dist, meta = hc.HandEyeDataset.load(args.dataset)
    n = len(g_list)
    print(f"  载入 {n} 个位姿（图像 {meta['n_frames']} 帧）")
    if n < 3:
        print("[错误] 位姿数不足 3 个，无法求解。")
        return 1

    # ---- 第一步：运动激励诊断（不达标直接劝退，避免浪费时间）----
    cu.motion_informativeness(g_list)

    # ---- 第二步：五算法交叉求解 ----
    print("  按 5 种算法独立求解并交叉验证（原理互为独立，一致才可信）：")
    sols = cu.cross_check_methods(g_list, c_list, eye_in_hand=True)
    X = sols.get("PARK", list(sols.values())[0])

    # ---- 第三步：AX=XB 残差（第一层验证）----
    A, B = cu.build_AB(g_list, c_list, eye_in_hand=True)
    res = cu.axxb_residual(A, B, X)
    print("  【第一层验证】AX=XB 残差")
    print(f"    旋转残差：均值 {res['rot_mean_deg']:.4f}°  最大 {res['rot_max_deg']:.4f}°")
    print(f"    平移残差：均值 {res['trans_mean_mm']:.4f} mm  最大 {res['trans_max_mm']:.4f} mm")
    print("    注：残差小只说明解与方程自洽，不能证明 X 正确（坐标系搞反时残差同样小）。")

    # ---- 第四步：几何自洽性（第二层验证）----
    consts = [g_list[i] @ X @ c_list[i] for i in range(n)]
    spr = max(cu.pose_diff(consts[0], c)[0] for c in consts)
    spt = max(cu.pose_diff(consts[0], c)[1] for c in consts)
    print("\n  【第二层验证】T_base←board 应为常量")
    print(f"    离散度：{spr:.4f}° / {spt:.4f} mm  "
          f"{'通过' if spr < 0.5 and spt < 5 else '偏大，建议补采'}")

    # ---- 输出结果 ----
    print("\n" + "=" * 74)
    print("  最终手眼矩阵 X = T_grip←cam")
    print("=" * 74)
    print(np.array2string(X, precision=6, suppress_small=True))
    rvec, tvec = cu.T_to_rvec_tvec(X)
    print(f"\n  旋转向量 (rad) : {np.array2string(rvec, precision=6)}")
    print(f"  平移向量 (m)   : {np.array2string(tvec, precision=6)}")
    print(f"  平移模长 (mm)  : {np.linalg.norm(tvec)*1000:.2f}")
    print(f"  旋转角 (deg)   : {cu.rot_angle_deg(X[:3, :3]):.3f}")

    # 校验：旋转矩阵是否正交
    R = X[:3, :3]
    orth = np.linalg.norm(R.T @ R - np.eye(3))
    det = np.linalg.det(R)
    print(f"  正交性 ‖RᵀR−I‖ = {orth:.2e}    det(R) = {det:.10f}")
    if orth > 1e-6 or abs(det - 1) > 1e-6:
        print("  [警告] 旋转矩阵不满足 SE(3) 约束，结果不可用！")

    out = {
        "config": "eye_in_hand",
        "X_gripper2cam": X.tolist(),
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "tvec_mm": (tvec * 1000).tolist(),
        "rot_deg": cu.rot_angle_deg(X[:3, :3]),
        "axxb_residual": res,
        "const_spread": {"rot_deg": spr, "trans_mm": spt},
        "all_methods": {k: v.tolist() for k, v in sols.items()},
        "K": K.tolist(),
        "dist": dist.tolist(),
    }
    path = os.path.join(args.dataset, "handeye_result.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n[保存] {path}")

    # ---- 第五步：任务级验证提示 ----
    print("\n  【第三层验证】任务级实测（必做，别跳过）：")
    print("    1) 把一枚尖针装在末端，在标定板上选一个已知角点，")
    print("       用 X 反算该角点在末端系下的坐标，控制机械臂去触碰，测实际偏差；")
    print("    2) 或直接执行一次抓取，量末端实际落点误差。")
    print("    3) 把角度误差折算到工作距离处（这一步最容易被忽略，却决定验收结论）：")
    for d in (0.3, 0.5, 0.8):
        e = cu.rotation_error_to_linear(cu.rot_angle_deg(X[:3, :3]), d)
        e03 = cu.rotation_error_to_linear(0.3, d)
        print(f"       工作距离 {d:.1f} m：本次标定角度误差 "
              f"{cu.rot_angle_deg(X[:3, :3]):.3f}° ⇒ {e*1000:6.2f} mm；"
              f"若为 0.3° ⇒ {e03*1000:5.2f} mm")
    elapsed = [cu.rotation_error_to_linear(0.3, d) * 1000 for d in (0.3, 0.5, 0.8)]
    print(f"    4) 误差预算示例（0.3° 旋转误差 + 1.5 mm 平移误差 + "
          f"0.1 mm 机器人绝对精度 + 0.05 mm 板制造误差）：")
    for d, e in zip((0.3, 0.5, 0.8), elapsed):
        budget = cu.task_level_error(0.3, 1.5, d)
        print(f"       工作距离 {d:.1f} m ⇒ 合成任务级误差 RSS ≈ "
              f"{budget['rss_total_mm']:.2f} mm")
    return 0


# --------------------------------------------------------------------------
def mode_selftest(args):
    """
    无设备合成自检：构造已知真值 → 求解 → 对比。
    课堂用途：让学生在不碰硬件的情况下建立「什么是对的」的直觉。
    """
    print("=" * 74)
    print("  合成数据自检（工厂模式：已知真值）")
    print("=" * 74)
    rng = np.random.default_rng(2026)

    X_true = cu.make_T(cv2.Rodrigues(
        np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
        * np.radians(40))[0], [0.05, -0.03, 0.10])
    T_base2board = cu.make_T(np.eye(3), [0.35, 0.0, 0.02])

    for n_poses in (3, 5, 8, 12, 20, 30):
        errs_r, errs_t = [], []
        for _ in range(30):
            G, C = [], []
            for _ in range(n_poses):
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                g = cu.make_T(cv2.Rodrigues(axis * np.radians(
                    rng.uniform(20, 45)))[0], rng.uniform(-0.2, 0.2, 3))
                c = cu.inv_T(X_true) @ cu.inv_T(g) @ T_base2board
                # 注入 0.3° / 1 mm 位姿噪声（模拟真实机器人 + PnP 误差）
                for arr, tgt in ((g, G), (c, C)):
                    ax = rng.normal(size=3)
                    ax /= np.linalg.norm(ax)
                    tgt.append(arr @ cu.make_T(
                        cv2.Rodrigues(ax * np.radians(rng.normal(0, 0.3)))[0],
                        rng.normal(0, 0.001, 3)))
            Xs = cu.solve_hand_eye(G, C, True, "PARK")
            a, b = cu.pose_diff(X_true, Xs)
            errs_r.append(a)
            errs_t.append(b)
        print(f"  位姿数 {n_poses:>3d}  →  旋转误差均值 {np.mean(errs_r):7.4f}°   "
              f"平移误差均值 {np.mean(errs_t):8.4f} mm")
    print("\n  结论：位姿数 ≥ 8 后进入误差平台期，12~20 是性价比最优区间；")
    print("        继续堆到 30 个位姿，误差仅再降一点点，而采集成本线性上升。")
    return 0


# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Eye-in-Hand 手眼标定教学脚本")
    p.add_argument("--mode", required=True,
                   choices=["capture", "solve", "selftest"])
    p.add_argument("--dataset", default="dataset_eih")
    p.add_argument("--intrinsics", default=os.path.join("calib_out",
                                                        "intrinsics.json"))
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
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit({"capture": mode_capture, "solve": mode_solve,
              "selftest": mode_selftest}[args.mode](args) or 0)
