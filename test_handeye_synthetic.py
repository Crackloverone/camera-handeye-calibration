# -*- coding: utf-8 -*-
"""
test_handeye_synthetic.py —— 手眼标定「坐标系约定」合成数据自检
==============================================================

为什么必须有这个脚本？
----------------------
手眼标定的绝大多数失败不是算法问题，而是「变换方向搞反 / 该求逆的没求逆」。
本脚本用**已知真值**的合成数据，把 OpenCV calibrateHandEye 的输入约定、
两种构型的差别、以及「约定错了会怎样」全部量化出来。

它同时也是课堂演示实验：
    学生先跑这个脚本看到「真值 → 解 → 误差 0.0000°」，
    再自己动手改一行求逆位置，立刻看到误差跳到几十毫米，
    比讲十遍公式都有效。

运行：
    python test_handeye_synthetic.py
输出：
    终端表格 + handeye_convention_result.txt
"""

import os
import numpy as np
import cv2

import calib_utils as cu

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


def axis_angle_pose(axis, deg, t):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    R = cv2.Rodrigues(axis * np.radians(deg))[0]
    return cu.make_T(R, np.asarray(t, dtype=float))


def random_pose(rng, min_rot_deg=15.0, max_rot_deg=50.0, trans_range=0.20):
    """随机刚体变换。旋转角下界保证方程不退化，上界避免接近 180° 的奇异。"""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.radians(rng.uniform(min_rot_deg, max_rot_deg))
    R = cv2.Rodrigues(axis * ang)[0]
    t = rng.uniform(-trans_range, trans_range, size=3)
    return cu.make_T(R, t)


def pose_noise(rng, rot_deg, trans_mm):
    R = np.eye(3)
    if rot_deg > 0:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        R = cv2.Rodrigues(axis * np.radians(rng.normal(0, rot_deg)))[0]
    t = rng.normal(0, trans_mm / 1000.0, size=3) if trans_mm > 0 else np.zeros(3)
    return cu.make_T(R, t)


# ==========================================================================
# 【A】Eye-in-Hand：相机装在末端法兰，标定板固定在工作台
#   物理模型： T_base←board = T_base←grip(i) · X · T_cam←board(i) = 常量
#   待求    ： X = T_grip←cam    （OpenCV 文档记作 cam2gripper）
#   推导    ： 令 a_i = T_base←grip(i)，b_i = T_cam←board(i)
#              a_i X b_i = a_j X b_j
#              ⇒ A·X = X·B，  A = a_j⁻¹a_i，B = b_j b_i⁻¹
# ==========================================================================
def experiment_eye_in_hand(rng, n_poses=15,
                           noise_rot_deg=0.0, noise_trans_mm=0.0):
    X_true = axis_angle_pose([0.3, -0.5, 0.8], 40.0, [0.05, -0.03, 0.10])
    T_base2board = cu.make_T(np.eye(3), [0.35, 0.0, 0.02])   # 板固定在基座系

    G_list, C_list = [], []
    for _ in range(n_poses):
        G = random_pose(rng)
        # C = X⁻¹ · G⁻¹ · T_base←board
        C = cu.inv_T(X_true) @ cu.inv_T(G) @ T_base2board
        if noise_rot_deg > 0 or noise_trans_mm > 0:
            G = G @ pose_noise(rng, noise_rot_deg, noise_trans_mm)
            C = C @ pose_noise(rng, noise_rot_deg, noise_trans_mm)
        G_list.append(G)
        C_list.append(C)

    X_solved = cu.solve_hand_eye(G_list, C_list, eye_in_hand=True, method="PARK")
    A, B = cu.build_AB(G_list, C_list, eye_in_hand=True)

    # ---- 几何自洽性检查：G_i X C_i 是否恒为常量 ----
    consts = [G_list[i] @ X_solved @ C_list[i] for i in range(n_poses)]
    spread_r = max(cu.pose_diff(consts[0], c)[0] for c in consts)
    spread_t = max(cu.pose_diff(consts[0], c)[1] for c in consts)
    return X_true, X_solved, (A, B), (spread_r, spread_t)


# ==========================================================================
# 【B】Eye-to-Hand：相机固定在外部支架，标定板装在末端法兰
#   物理模型： T_base←board = T_base←grip(i) · Y = X · T_cam←board(i)
#   待求    ： X = T_base←cam    （OpenCV 文档记作 cam2base）
#   关键    ： 与 eye-in-hand 的唯一差别是 a_i 取 T_grip←base(i)（对正解取逆）
# ==========================================================================
def experiment_eye_to_hand(rng, n_poses=15):
    X_true = axis_angle_pose([0.1, 0.2, 0.97], 35.0, [0.40, 0.10, 0.55])
    Y_true = cu.make_T(np.eye(3), [0.02, 0.03, 0.08])         # 末端←标定板

    G_list, C_list = [], []
    for _ in range(n_poses):
        G = random_pose(rng)
        C = cu.inv_T(X_true) @ G @ Y_true
        G_list.append(G)
        C_list.append(C)

    # 正确约定
    X_good = cu.solve_hand_eye(G_list, C_list, eye_in_hand=False, method="PARK")
    # 错误约定：忘记对机器人位姿取逆（等价于当成 eye-in-hand 处理）
    X_bad = cu.solve_hand_eye(G_list, C_list, eye_in_hand=True, method="PARK")
    return X_true, X_good, X_bad


def main():
    rng = np.random.default_rng(20260920)

    log("=" * 76)
    log("  手眼标定坐标系约定自检（合成真值验证）")
    log(f"  OpenCV {cv2.__version__} / numpy {np.__version__}")
    log("=" * 76)

    # ---------------- 【A】Eye-in-Hand ----------------
    log("\n【A】Eye-in-Hand —— 待求真值 X_true = T_grip←cam")
    log("-" * 76)
    X_true, X_solved, (A, B), (spr, spt) = experiment_eye_in_hand(rng, 15)
    da, dt = cu.pose_diff(X_true, X_solved)
    log(f"  解 vs 真值：旋转误差 {da:.6f}°   平移误差 {dt:.6f} mm")
    res = cu.axxb_residual(A, B, X_solved)
    log(f"  AX=XB 残差：旋转均值 {res['rot_mean_deg']:.3e}°  "
        f"平移均值 {res['trans_mean_mm']:.3e} mm")
    log(f"  几何自洽性 G_i·X·C_i 离散度：{spr:.3e}° / {spt:.3e} mm")
    log(f"  ⇒ {'通过' if da < 1e-3 and dt < 1e-3 else '未通过'}")

    # ---------------- 【B】Eye-to-Hand ----------------
    log("\n【B】Eye-to-Hand —— 待求真值 X_true = T_base←cam")
    log("-" * 76)
    X_et, X_good, X_bad = experiment_eye_to_hand(rng, 15)
    dg = cu.pose_diff(X_et, X_good)
    db = cu.pose_diff(X_et, X_bad)
    log(f"  正确约定（机器人位姿取逆）：旋转误差 {dg[0]:.6f}°  平移误差 {dg[1]:.6f} mm")
    log(f"  错误约定（忘记取逆）      ：旋转误差 {db[0]:.6f}°  平移误差 {db[1]:.6f} mm")
    log(f"  ⇒ 搞错求逆位置，平移误差放大约 "
        f"{(db[1] / max(dg[1], 1e-9)) if dg[1] > 1e-9 else float('inf'):.0f} 倍")

    # ---------------- 【C】噪声敏感性 ----------------
    log("\n【C】噪声敏感性（eye-in-hand，15 个位姿，每档 30 次重复）")
    log("-" * 76)
    log(f"  {'旋转噪声(°)':>12}{'平移噪声(mm)':>14}"
        f"{'旋转误差均值(°)':>18}{'平移误差均值(mm)':>18}")
    for n_rot, n_tr in [(0.0, 0.0), (0.3, 1.0), (1.0, 1.0), (2.0, 2.0)]:
        er, et = [], []
        for _ in range(30):
            Xt, Xs, _, _ = experiment_eye_in_hand(
                rng, 15, noise_rot_deg=n_rot, noise_trans_mm=n_tr)
            a, b = cu.pose_diff(Xt, Xs)
            er.append(a)
            et.append(b)
        log(f"  {n_rot:>12.1f}{n_tr:>14.1f}"
            f"{np.mean(er):>18.4f}{np.mean(et):>18.4f}")

    # ---------------- 【D】运动激励诊断 ----------------
    log("\n【D】运动激励诊断 —— 手眼标定失败的头号原因")
    log("-" * 76)
    log("  D1. 只绕 Z 轴旋转的 12 个位姿（典型错误做法）：")
    only_z = [axis_angle_pose([0, 0, 1], 10 * k, [0.02 * k, 0, 0])
              for k in range(12)]
    cu.motion_informativeness(only_z)

    log("  D2. 多轴大角度运动的 12 个位姿（推荐做法）：")
    good = [random_pose(rng) for _ in range(12)]
    cu.motion_informativeness(good)

    # 用一个「只绕单轴」的数据集实际解一次，看误差
    log("  D3. 实测对比（同为 12 个位姿，真值相同）：")
    X_true = axis_angle_pose([0.3, -0.5, 0.8], 40.0, [0.05, -0.03, 0.10])
    T_base2board = cu.make_T(np.eye(3), [0.35, 0.0, 0.02])
    for tag, poses in [("只绕 Z 轴", only_z), ("多轴大角度", good)]:
        G_list = [p @ pose_noise(rng, 0.3, 1.0) for p in poses]
        C_list = [cu.inv_T(X_true) @ cu.inv_T(p) @ T_base2board
                  @ pose_noise(rng, 0.3, 1.0) for p in poses]
        try:
            Xs = cu.solve_hand_eye(G_list, C_list, True, "PARK")
            a, b = cu.pose_diff(X_true, Xs)
            log(f"    {tag:<14s} 旋转误差 {a:8.3f}°   平移误差 {b:10.3f} mm")
        except Exception as exc:               # noqa: BLE001
            log(f"    {tag:<14s} 求解失败：{exc}")
    log("")
    log("  结论：同样的噪声水平下，只绕单轴运动会让平移分量不可观，")
    log("        误差放大 1~2 个数量级。所以「数据质量 > 算法选择」。")

    log("\n" + "=" * 76)
    log("  全部自检完成。")
    log("=" * 76)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "handeye_convention_result.txt"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    log("\n结果已写入 handeye_convention_result.txt")


if __name__ == "__main__":
    main()
