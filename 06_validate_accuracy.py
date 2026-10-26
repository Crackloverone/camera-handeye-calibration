#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_validate_accuracy.py —— 标定精度验证与误差预算
===================================================

为什么需要单独一个「验证」脚本？
--------------------------------
    标定跑完不报错，不等于标定正确。工程上必须**分层验证**，
    单一指标通过不能作为验收依据：

    第一层  AX=XB 残差        → 解与方程是否自洽（对系统性错误不敏感）
    第二层  留出位姿验证      → 几何链是否自洽、是否过拟合
    第三层  任务级实测        → 探针触碰已知点 / 实际抓取，量末端落点误差

    三层的严格程度递增，只有第三层能回答「这套标定能不能干活」。

本脚本还实现「误差预算」：把各误差源折算到工作距离处，
回答「精度天花板在哪里、还能优化什么」。

用法
----
# 完整验证（需要 handeye_result.json + 数据集）
python 06_validate_accuracy.py --mode validate \
       --dataset dataset_eih --handeye dataset_eih/handeye_result.json

# 任务级实测数据录入与分析
python 06_validate_accuracy.py --mode probe \
       --dataset dataset_eih --handeye dataset_eih/handeye_result.json

# 误差预算计算器（可独立运行，课堂讨论用）
python 06_validate_accuracy.py --mode budget --rot_deg 0.3 --trans_mm 1.5 \
       --distance 0.5
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

import calib_utils as cu
import handeye_capture as hc


def load_handeye(path):
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    eye_in_hand = res["config"] == "eye_in_hand"
    X = np.array(res.get("X_gripper2cam") or res["X_base2cam"], dtype=np.float64)
    K = np.array(res["K"], dtype=np.float64)
    dist = np.array(res["dist"], dtype=np.float64).ravel()
    return X, eye_in_hand, K, dist, res


# ==========================================================================
# 第一层 + 第二层验证
# ==========================================================================
def mode_validate(args):
    X, eye_in_hand, K, dist, hres = load_handeye(args.handeye)
    g_list, c_list, _, _, meta = hc.HandEyeDataset.load(args.dataset)
    n = len(g_list)

    print("=" * 78)
    print("  手眼标定分层验证报告")
    print("=" * 78)
    print(f"  构型          : {'Eye-in-Hand' if eye_in_hand else 'Eye-to-Hand'}")
    print(f"  样本数        : {n} 个位姿")
    print(f"  内参重投影 RMS: 见 01 脚本报告（手眼标定不会修复错误内参）")
    print("-" * 78)

    # ---------- 第一层 ----------
    A, B = cu.build_AB(g_list, c_list, eye_in_hand=eye_in_hand)
    res = cu.axxb_residual(A, B, X)
    print("\n  【第一层】AX = XB 残差")
    print(f"    旋转残差：均值 {res['rot_mean_deg']:.5f}°  "
          f"最大 {res['rot_max_deg']:.5f}°")
    print(f"    平移残差：均值 {res['trans_mean_mm']:.5f} mm  "
          f"最大 {res['trans_max_mm']:.5f} mm")
    print("    解读：残差反映「解与方程组」的拟合程度。")
    print("          ⚠ 坐标系方向搞反时残差同样很小 —— 残差小 ≠ 标定正确！")

    # ---------- 第二层：留一交叉验证 ----------
    print("\n  【第二层 A】留一交叉验证（每次剔除 1 个位姿重解）")
    loo_r, loo_t = [], []
    if n >= 8:
        for k in range(n):
            idx = [i for i in range(n) if i != k]
            gk = [g_list[i] for i in idx]
            ck = [c_list[i] for i in idx]
            try:
                Xk = cu.solve_hand_eye(gk, ck, eye_in_hand, "PARK")
                da, dt = cu.pose_diff(X, Xk)
                loo_r.append(da)
                loo_t.append(dt)
            except Exception:                     # noqa: BLE001
                continue
        if loo_r:
            print(f"    剔除任一帧后结果变化：旋转 {np.mean(loo_r):.4f}° "
                  f"(最大 {np.max(loo_r):.4f}°)，平移 {np.mean(loo_t):.4f} mm "
                  f"(最大 {np.max(loo_t):.4f} mm)")
            worst = int(np.argmax(loo_t))
            if np.max(loo_t) > 3 * np.mean(loo_t) + 0.5:
                print(f"    [!] 剔除第 {worst+1} 帧后结果变化最大 → 该帧可能是坏数据")
            else:
                print("    [OK] 结果对单帧不敏感，未见过拟合")
    else:
        print(f"    样本仅 {n} 个，跳过（建议 ≥ 8 个后再做此项）。")

    # ---------- 第二层：几何自洽性 ----------
    print("\n  【第二层 B】几何链自洽性（常量不变性检查）")
    consts = [g_list[i] @ X @ c_list[i] for i in range(n)] if eye_in_hand else None
    if eye_in_hand:
        spr = max(cu.pose_diff(consts[0], c)[0] for c in consts)
        spt = max(cu.pose_diff(consts[0], c)[1] for c in consts)
        print(f"    eye-in-hand：T_base←board = G_i·X·C_i 应恒定")
    else:
        Y = cu.inv_T(g_list[0]) @ X @ c_list[0]
        spr, spt = 0.0, 0.0
        for i in range(n):
            da, dt = cu.pose_diff(g_list[i] @ Y, X @ c_list[i])
            spr = max(spr, da)
            spt = max(spt, dt)
        print(f"    eye-to-hand：G_i·Y 应恒等于 X·C_i")
    print(f"    离散度：{spr:.4f}° / {spt:.4f} mm  "
          f"{'通过' if spr < 0.5 and spt < 5 else '偏大，建议补采或剔除坏帧'}")

    # ---------- 误差折算 ----------
    print("\n  【误差折算】把角度误差折算到工作距离处（最常被忽略的一步）")
    rot_err = res["rot_mean_deg"]
    trans_err = res["trans_mean_mm"]
    print(f"    {'工作距离':>10}{'0.3°折算':>14}{'本次角度折算':>16}{'合成任务级RSS':>18}")
    for d in (0.3, 0.5, 0.8, 1.0):
        e03 = cu.rotation_error_to_linear(0.3, d) * 1000
        e_cur = cu.rotation_error_to_linear(rot_err, d) * 1000
        bud = cu.task_level_error(rot_err, trans_err, d)
        print(f"    {d:>8.1f} m{e03:>12.2f} mm{e_cur:>14.2f} mm"
              f"{bud['rss_total_mm']:>16.2f} mm")

    # ---------- 结论 ----------
    print("\n" + "=" * 78)
    print("  验收判定")
    print("=" * 78)
    ok1 = res["rot_mean_deg"] < 0.5 and res["trans_mean_mm"] < 5.0
    ok2 = spr < 0.5 and spt < 5.0
    print(f"    第一层（残差）      : {'通过' if ok1 else '不通过'}")
    print(f"    第二层（几何自洽）  : {'通过' if ok2 else '不通过'}")
    print(f"    第三层（任务级实测）: 待做 —— 运行 --mode probe")
    print("\n    ⚠ 提醒：重投影误差小只说明「图像检测与相机模型自洽」，")
    print("      不能单独作为手眼标定正确的证据。必须做任务级实测。")
    return 0


# ==========================================================================
# 第三层：任务级实测（探针法）
# ==========================================================================
def mode_probe(args):
    X, eye_in_hand, K, dist, hres = load_handeye(args.handeye)

    print("=" * 78)
    print("  第三层验证：任务级实测（探针法）")
    print("=" * 78)
    print("""  操作步骤：
    1) 在末端装一根尖针（或直接用夹爪尖端），把当前末端位置作为 TCP
    2) 在工作台上选 5~8 个「已知机器人坐标」的位置（例如标定板角点
       在基座系下的坐标，可由标定板位姿推算，或用尺子量出网格点）
    3) 依次用示教器让针尖对准这些点，记录示教器读数
    4) 计算 实测坐标 与 名义坐标 之差，输入本脚本
  """)
    print("  请输入实测数据，每行一组： 名义x 名义y 名义z 实测x 实测y 实测z (mm)")
    print("  直接回车结束输入。")

    rows = []
    while True:
        s = input("  > ").strip()
        if not s:
            break
        try:
            v = [float(x) for x in s.replace(",", " ").split()]
            if len(v) != 6:
                raise ValueError
            rows.append(v)
        except ValueError:
            print("    格式错误，需 6 个数字。")

    if not rows:
        print("  [i] 未输入数据，跳过。")
        return 0

    arr = np.array(rows)
    diff = arr[:, 3:] - arr[:, :3]
    err = np.linalg.norm(diff, axis=1)
    print("\n  " + "-" * 62)
    print(f"  {'#':>3}{'名义(mm)':>26}{'误差(mm)':>12}")
    for i, (r, e) in enumerate(zip(arr, err)):
        print(f"  {i+1:>3}  ({r[0]:7.1f},{r[1]:7.1f},{r[2]:7.1f})"
              f"{e:>12.3f}")
    print("  " + "-" * 62)
    print(f"  平均误差 {err.mean():.3f} mm   最大 {err.max():.3f} mm   "
          f"标准差 {err.std():.3f} mm")

    # 恒定偏移 vs 随位姿变化 —— 工程上诊断价值极高
    mean_off = diff.mean(axis=0)
    print(f"\n  平均偏移向量 = ({mean_off[0]:.3f}, {mean_off[1]:.3f}, "
          f"{mean_off[2]:.3f}) mm")
    if err.std() < 0.3 and np.linalg.norm(mean_off) > np.maximum(err.mean(), 0.5):
        print("  诊断：偏差基本是**恒定偏移** → 优先检查 TCP 设定、")
        print("        X 的平移分量、以及标定板原点定义。")
    else:
        print("  诊断：偏差**随位姿变化** → 更可能是机器人绝对精度或")
        print("        世界/基座标定问题，而不是手眼矩阵本身。")
    print("\n  参考量级：认真执行的眼在手上系统可做到 1~2 mm / 0.3° 量级。")
    return 0


# ==========================================================================
# 误差预算计算器
# ==========================================================================
def mode_budget(args):
    print("=" * 78)
    print("  误差预算计算器 —— 精度天花板在哪里？")
    print("=" * 78)
    b = cu.task_level_error(args.rot_deg, args.trans_mm, args.distance,
                            args.robot_abs, args.board_err)
    total = b["rss_total_mm"]
    print(f"\n  输入：手眼旋转误差 {args.rot_deg:.3f}°，平移误差 {args.trans_mm:.3f} mm")
    print(f"        工作距离 {args.distance:.2f} m，机器人绝对精度 "
          f"{args.robot_abs:.3f} mm，标定板制造误差 {args.board_err:.3f} mm\n")
    print(f"  {'误差源':<26}{'折算值(mm)':>14}{'占总方差':>12}")
    print("  " + "-" * 54)
    items = [
        (f"旋转误差 {args.rot_deg:.2f}° × {args.distance:.2f} m",
         b["rotation_to_linear_mm"]),
        ("手眼平移误差", b["handeye_trans_mm"]),
        ("机器人绝对精度", b["robot_abs_mm"]),
        ("标定板制造/打印误差", b["board_mm"]),
    ]
    var = np.array([max(v, 0.0) ** 2 for _, v in items])
    share = var / (var.sum() + 1e-12)
    for (name, val), sh in zip(items, share):
        bar = "#" * int(round(sh * 30))
        print(f"  {name:<26}{val:>14.3f}{sh*100:>11.1f}%  {bar}")
    print("  " + "-" * 54)
    print(f"  {'合成（RSS）':<26}{total:>14.3f}")
    print("\n  关键结论（课堂要点）：")
    print("    1) 旋转误差的影响随工作距离**线性放大**：")
    print(f"       {args.rot_deg:.2f}° 在 {args.distance:.2f} m 处 ⇒ "
          f"{b['rotation_to_linear_mm']:.2f} mm")
    print("       所以评估手眼标定必须折算到实际工作距离，不能只看旋转角。")
    big = items[int(np.argmax(share))]
    print(f"    2) 当前最大误差源是「{big[0]}」（占 {share.max()*100:.0f}%），")
    print("       优化应优先攻这一项，而不是笼统地「重标一次」。")
    print("    3) 若某项已接近其物理下限（如机器人绝对精度厂商给不出更小值），")
    print("       那它就是这套系统的精度天花板 —— 承认它，别再空耗。")
    return 0


# ==========================================================================
def build_parser():
    p = argparse.ArgumentParser(description="标定精度验证与误差预算")
    p.add_argument("--mode", required=True,
                   choices=["validate", "probe", "budget"])
    p.add_argument("--dataset", default="dataset_eih")
    p.add_argument("--handeye",
                   default=os.path.join("dataset_eih", "handeye_result.json"))
    p.add_argument("--rot_deg", type=float, default=0.3)
    p.add_argument("--trans_mm", type=float, default=1.5)
    p.add_argument("--distance", type=float, default=0.5)
    p.add_argument("--robot_abs", type=float, default=0.10,
                   help="机器人绝对精度 mm（工业臂常见 0.05~0.3）")
    p.add_argument("--board_err", type=float, default=0.05)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit({"validate": mode_validate, "probe": mode_probe,
              "budget": mode_budget}[args.mode](args) or 0)
