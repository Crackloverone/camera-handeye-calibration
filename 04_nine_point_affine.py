#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_nine_point_affine.py —— 九点标定法（工业平面抓取的事实标准）
================================================================

对应课程实验：实验 9 第 5 步「眼在手外-手眼标定」
（指导书中：把机械臂移到标定板上两点对准，记录 point1/point2 的 (x,y)，
  再由 4x4 棋盘格的九个内角点用仿射矩阵完成像素→机器人的坐标变换）

为什么工业界爱用九点法？
------------------------
    大量抓取场景中工件都在同一工作平面（传送带、上料台），高度固定。
    此时 Z 方向的 6 自由度外参完全是浪费——把问题降为
    「像素平面 → 机器人平面」的 2D 仿射变换，只需 6 个参数、
    最少 3 对点即可求闭式解，9 点构成超定系统抑制噪声。

数学形式
--------
    ┌ x_r ┐   ┌ a  b  c ┐ ┌ u ┐
    │ y_r │ = │ d  e  f │ │ v │
    └  1  ┘   └ 0  0  1 ┘ └ 1 ┘

    M = [a b c; d e f] 为 2x3 仿射矩阵，对应 OpenCV 的 cv2.estimateAffine2D()。
    自由度分布：c/f = 平移；a/b/d/e = 旋转 + 缩放 + 错切。

适用边界（务必讲清楚，否则学生乱用）
------------------------------------
    ✓ 工件共面、高度固定         ✗ 需要三维定位 / 姿态估计
    ✓ 相机内参已标定或畸变可忽略  ✗ 大畸变未校正（边缘会被拉偏）
    ✓ 工作距离远 > 视野尺寸       ✗ 相机光轴严重倾斜

用法
----
# 交互式采集 9 点（相机看标定板，机械臂依次对准每个内角点）
python 04_nine_point_affine.py --mode collect --camera 0 \
       --intrinsics calib_out/intrinsics.json --out nine_point.json

# 求解 + 验证
python 04_nine_point_affine.py --mode solve --data nine_point.json

# 合成演示（课堂无设备演示，同时给出误差量级直觉）
python 04_nine_point_affine.py --mode demo
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

import calib_utils as cu


# --------------------------------------------------------------------------
def mode_collect(args):
    """
    采集流程（严格对应实验 9 的操作方式）：
      1) 相机对准工作台上的标定板，检测内角点
      2) 用示教器把机械臂末端（或标定针）依次对准 9 个内角点
      3) 每对准一个点，回车记录「该点的像素坐标 + 示教器显示坐标」
    """
    with open(args.intrinsics, encoding="utf-8") as fh:
        intrin = json.load(fh)
    K = np.array(intrin["K"], dtype=np.float64)
    dist = np.array(intrin["dist"], dtype=np.float64).ravel()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[错误] 打不开相机 {args.camera}")
        return 1

    px_pts, rb_pts = [], []
    print("=" * 74)
    print("  九点标定 —— 数据采集")
    print("=" * 74)
    print("  步骤：")
    print("    1) 固定标定板，保证九个内角点全部清晰可见、不反光")
    print("    2) 用示教器把末端对准第 1 个内角点（建议从左上开始按行推进）")
    print("    3) 回车 → 输入示教器显示的 X Y（单位 mm，与实验 9 一致）")
    print("    4) 重复直到 9 点采完。建议再采 2~3 个「验证点」不参与拟合")
    print("  操作：回车=记录当前点  u=撤销  q=结束并保存")
    print("=" * 74)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cu.detect_chessboard(gray, (args.cols, args.rows))
        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, (args.cols, args.rows), corners, True)
            # 标出未被选中的下一个候选点
            if len(px_pts) < len(corners):
                p = corners[len(px_pts)].reshape(2)
                cv2.circle(vis, (int(p[0]), int(p[1])), 12, (0, 255, 255), 3)
                cv2.putText(vis, "next candidate", (int(p[0]) + 15,
                                                    int(p[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, f"points: {len(px_pts)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
        cv2.imshow("nine-point collect", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('u') and px_pts:
            px_pts.pop()
            rb_pts.pop()
            print("  [撤销] 已删除最后一个点")
            continue
        if key in (13, 10):
            if not found:
                print("  [x] 当前未检出角点，无法记录。")
                continue
            # 默认取下一个未使用的角点；也允许手动改
            auto = corners[min(len(px_pts), len(corners) - 1)].reshape(2)
            print(f"\n  第 {len(px_pts)+1} 点")
            print(f"    自动选取的像素坐标：({auto[0]:.1f}, {auto[1]:.1f})")
            print("    如需手动指定请直接输入 u v，回车则用自动值：")
            s = input("    > ").strip()
            if s:
                try:
                    uu, vv = [float(x) for x in s.replace(",", " ").split()]
                except ValueError:
                    print("    [x] 格式错误，跳过。")
                    continue
            else:
                uu, vv = float(auto[0]), float(auto[1])

            print("    请输入示教器上显示的机械臂 X Y 坐标（mm）：")
            sr = input("    > ").strip()
            try:
                xr, yr = [float(x) for x in sr.replace(",", " ").split()]
            except ValueError:
                print("    [x] 格式错误，跳过。")
                continue
            px_pts.append([uu, vv])
            rb_pts.append([xr, yr])
            print(f"    [v] 记录：像素({uu:.1f},{vv:.1f}) ↔ 机器人({xr:.1f},{yr:.1f}) mm")

    cap.release()
    cv2.destroyAllWindows()

    if len(px_pts) < 3:
        print("[错误] 点数不足 3，无法拟合。")
        return 1

    M, info = cu.nine_point_affine(px_pts, rb_pts)
    print("\n" + "=" * 74)
    print("  标定结果")
    print("=" * 74)
    print(f"  仿射矩阵 M =\n{np.array2string(np.asarray(M), precision=6)}")
    print(f"\n  参与拟合 {info['n_points']} 点：平均残差 "
          f"{info['mean_err_mm']:.3f} mm，最大残差 {info['max_err_mm']:.3f} mm")
    for i, (e, p) in enumerate(zip(info["per_point_err_mm"], px_pts)):
        flag = "  <-- 残差偏大，检查该点是否对准" if e > max(
            0.5, 3 * info["mean_err_mm"]) else ""
        print(f"    第 {i+1:>2d} 点  ({p[0]:8.1f},{p[1]:8.1f}) px  残差 {e:7.3f} mm{flag}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"M": np.asarray(M).tolist(),
                   "px_points": px_pts, "robot_points": rb_pts,
                   "info": info}, fh, ensure_ascii=False, indent=2)
    print(f"\n[保存] {os.path.abspath(args.out)}")
    print("\n  验收建议：平均残差 < 0.3 mm 优秀，< 1 mm 可用，> 2 mm 重采。")
    return 0


# --------------------------------------------------------------------------
def mode_solve(args):
    with open(args.data, encoding="utf-8") as fh:
        d = json.load(fh)
    M = np.array(d["M"], dtype=np.float64)
    px = np.array(d["px_points"], dtype=np.float64)
    rb = np.array(d["robot_points"], dtype=np.float64)
    info = d["info"]

    print("=" * 74)
    print("  九点标定结果校验")
    print("=" * 74)
    print(f"  M =\n{np.array2string(M, precision=6)}")
    print(f"  平均残差 {info['mean_err_mm']:.3f} mm / 最大 {info['max_err_mm']:.3f} mm")

    # 留出验证：轮流剔除 1 点重解，看稳定性
    print("\n  留一交叉验证（每次剔除 1 点重解，检验结果是否被单点带偏）：")
    errs = []
    n = len(px)
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        if len(idx) < 3:
            break
        Mk, _ = cu.nine_point_affine(px[idx], rb[idx])
        pred = cv2.transform(px[k].reshape(1, 1, 2).astype(np.float64),
                             Mk).reshape(2)
        e = float(np.linalg.norm(pred - rb[k]))
        errs.append(e)
        print(f"    剔除第 {k+1:>2d} 点 → 该点预测残差 {e:7.3f} mm")
    if errs:
        print(f"  留一交叉验证平均 {np.mean(errs):.3f} mm，最大 "
              f"{np.max(errs):.3f} mm")
        if np.max(errs) > 3 * np.mean(errs) + 0.2:
            print("  [警告] 存在明显的坏点（某点剔除后残差远大于其他），"
                  "建议重采该点。")

    print("\n  仿射矩阵的物理含义解读：")
    scale = np.sqrt(abs(np.linalg.det(M[:, :2])))
    rot = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    print(f"    尺度因子 ≈ {scale:.5f} mm/px  （即 1 像素对应实际多少毫米）")
    print(f"    旋转分量 ≈ {rot:.3f}°")
    print(f"    平移分量 = ({M[0, 2]:.2f}, {M[1, 2]:.2f}) mm")
    print("    提示：若 |旋转分量| 明显偏离实际安装角，说明像素/机器人点位次序对错了。")
    return 0


# --------------------------------------------------------------------------
def mode_demo(args):
    """
    合成演示：制造一个已知的「像素→机器人」映射，加噪声后反解，
    量化「点数」与「噪声」对精度的影响 —— 用来讲清为什么是 9 点而不是 3 点。
    """
    print("=" * 74)
    print("  九点标定法 合成实验（讲清「为什么用 9 点」）")
    print("=" * 74)
    rng = np.random.default_rng(2026)

    mm_per_px = 0.12                       # 1 像素 = 0.12 mm
    theta = np.radians(0.8)                # 相机相对机器人转了 0.8°
    Rm = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]]) * mm_per_px
    M_true = np.hstack([Rm, np.array([[120.0], [85.0]])])   # 2x3

    # 工作区上的 3x3 均匀点位（像素坐标）
    us = np.linspace(300, 1700, 3)
    vs = np.linspace(250, 1150, 3)
    grid = np.array([[u, v] for v in vs for u in us], dtype=np.float64)
    robot_true = cv2.transform(grid.reshape(-1, 1, 2), M_true).reshape(-1, 2)

    print(f"  真值：尺度 {mm_per_px} mm/px，旋转 0.8°，平移 (120, 85) mm")
    print(f"  工作区 3x3 九点，覆盖 {(us[-1]-us[0])*mm_per_px:.0f} x "
          f"{(vs[-1]-vs[0])*mm_per_px:.0f} mm\n")
    print(f"  {'点数':>6}{'噪声(px)':>12}{'平均误差(mm)':>16}{'最大误差(mm)':>16}")
    print("  " + "-" * 52)

    for noise_px in (0.5, 1.0):
        for n_pts in (3, 4, 5, 9):
            errs = []
            for _ in range(200):
                idx = rng.choice(len(grid), size=n_pts, replace=False)
                obs_px = grid[idx] + rng.normal(0, noise_px, (n_pts, 2))
                obs_rb = robot_true[idx] + rng.normal(0, 0.05, (n_pts, 2))
                Mk, _ = cu.nine_point_affine(obs_px, obs_rb)
                pred = cv2.transform(grid.reshape(-1, 1, 2), Mk).reshape(-1, 2)
                e = np.linalg.norm(pred - robot_true, axis=1)
                errs.append(e)
            errs = np.array(errs)
            print(f"  {n_pts:>6}{noise_px:>12.1f}"
                  f"{errs.mean():>16.4f}{errs.max():>16.4f}")

    print("\n  结论（可直接作为课堂提问的答案）：")
    print("    1) 3 点虽然数学上够解，但完全没有冗余，一个坏点就能毁掉整个标定；")
    print("       9 点构成超定系统，用最小二乘把随机噪声平均掉。")
    print("    2) 点数从 3 增到 9，误差下降明显；再往 12、16 点走收益迅速递减。")
    print("    3) 点位必须覆盖工作区四角+中心。若九点挤在一小块区域，")
    print("       边缘外推时误差会被放大数倍 —— 这是工业现场最常见的隐性误差源。")
    return 0


# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="九点标定法教学脚本")
    p.add_argument("--mode", required=True, choices=["collect", "solve", "demo"])
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--intrinsics",
                   default=os.path.join("calib_out", "intrinsics.json"))
    p.add_argument("--data", default="nine_point.json")
    p.add_argument("--out", default="nine_point.json")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--rows", type=int, default=4)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit({"collect": mode_collect, "solve": mode_solve,
              "demo": mode_demo}[args.mode](args) or 0)
