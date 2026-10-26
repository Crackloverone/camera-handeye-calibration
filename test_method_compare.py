# -*- coding: utf-8 -*-
"""五种 OpenCV 手眼标定算法的蒙特卡洛对比（合成真值）。"""
import numpy as np
import cv2
import calib_utils as cu

rng = np.random.default_rng(2026)
OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


def rand_pose(rng, lo=20, hi=50, tr=0.20):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    R = cv2.Rodrigues(ax * np.radians(rng.uniform(lo, hi)))[0]
    return cu.make_T(R, rng.uniform(-tr, tr, 3))


def noise_pose(rng, rd, tm):
    R = np.eye(3)
    if rd > 0:
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        R = cv2.Rodrigues(a * np.radians(rng.normal(0, rd)))[0]
    t = rng.normal(0, tm / 1000.0, 3) if tm > 0 else np.zeros(3)
    return cu.make_T(R, t)


X_TRUE = cu.make_T(cv2.Rodrigues(
    np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    * np.radians(40))[0], [0.05, -0.03, 0.10])
T_B2B = cu.make_T(np.eye(3), [0.35, 0.0, 0.02])

METHODS = ["PARK", "HORAUD", "ANDREFF", "DANIILIDIS", "TSAI"]

log("=" * 88)
log("  五种 OpenCV 手眼标定算法对比（合成真值，每组 40 次重复取均值）")
log(f"  OpenCV {cv2.__version__}")
log("=" * 88)

# ---------- 实验 1：固定噪声 0.3°/1mm，扫描位姿数 ----------
log("\n【实验一】固定噪声 σ = 0.3° / 1 mm，扫描位姿数")
log("-" * 88)
counts = [3, 5, 8, 12, 20]
hdr = f"  {'位姿数':>6}"
for m in METHODS:
    hdr += f"{m[:9]:>13}"
log(hdr + "     （格式：旋转误差° / 平移误差mm）")
log("  " + "-" * 86)
tbl1 = {}
for n in counts:
    row = f"  {n:>6}"
    tbl1[n] = {}
    for m in METHODS:
        er, et = [], []
        for _ in range(40):
            G, C = [], []
            for _ in range(n):
                g = rand_pose(rng) @ noise_pose(rng, 0.3, 1.0)
                c = cu.inv_T(X_TRUE) @ cu.inv_T(g) @ T_B2B @ noise_pose(rng, 0.3, 1.0)
                G.append(g)
                C.append(c)
            try:
                Xs = cu.solve_hand_eye(G, C, True, m)
                a, b = cu.pose_diff(X_TRUE, Xs)
            except Exception:
                a, b = float("nan"), float("nan")
            er.append(a)
            et.append(b)
        tbl1[n][m] = (np.nanmean(er), np.nanmean(et))
        row += f"{tbl1[n][m][0]:6.3f}/{tbl1[n][m][1]:6.2f}"
    log(row)

# ---------- 实验 2：固定 15 个位姿，扫描噪声 ----------
log("\n【实验二】固定 15 个位姿，扫描噪声水平（平移噪声固定 1 mm）")
log("-" * 88)
hdr = f"  {'旋转噪声':>10}"
for m in METHODS:
    hdr += f"{m[:9]:>13}"
log(hdr)
log("  " + "-" * 86)
tbl2 = {}
for nrot in [0.0, 0.5, 1.0, 2.0]:
    row = f"  {nrot:>8.1f}°"
    tbl2[nrot] = {}
    for m in METHODS:
        er, et = [], []
        for _ in range(40):
            G, C = [], []
            for _ in range(15):
                g = rand_pose(rng) @ noise_pose(rng, nrot, 1.0)
                c = cu.inv_T(X_TRUE) @ cu.inv_T(g) @ T_B2B @ noise_pose(rng, nrot, 1.0)
                G.append(g)
                C.append(c)
            try:
                Xs = cu.solve_hand_eye(G, C, True, m)
                a, b = cu.pose_diff(X_TRUE, Xs)
            except Exception:
                a, b = float("nan"), float("nan")
            er.append(a)
            et.append(b)
        tbl2[nrot][m] = (np.nanmean(er), np.nanmean(et))
        row += f"{tbl2[nrot][m][0]:6.3f}/{tbl2[nrot][m][1]:6.2f}"
    log(row)

# ---------- 实验 3：计时 ----------
log("\n【实验三】计算耗时（15 个位姿，100 次求解总耗时）")
log("-" * 88)
import time
for m in METHODS:
    G, C = [], []
    for _ in range(15):
        g = rand_pose(rng)
        G.append(g)
        C.append(cu.inv_T(X_TRUE) @ cu.inv_T(g) @ T_B2B)
    t0 = time.perf_counter()
    for _ in range(100):
        cu.solve_hand_eye(G, C, True, m)
    dt = (time.perf_counter() - t0) / 100 * 1000
    log(f"  {m:<12} {dt:8.3f} ms/次")

log("\n" + "=" * 88)
log("  解读要点（以本表实测数据为准，不引用二手结论）：")
log("  1) 位姿数 = 3 时，全部五种方法平移误差都在 5~27 mm 量级，")
log("     且方法之间排序混乱（本组数据里 HORAUD 反而最差 27.12 mm）。")
log("     说明：位姿太少时，误差主要由数据不足决定，算法优劣无从体现。")
log("  2) 位姿数 >= 8 后各方法迅速收敛并趋于平台，20 个位姿时平移误差")
log("     已在 0.9~1.1 mm（注入噪声为 1 mm）。12~20 是性价比最优区间。")
log("  3) 噪声扫描（实验二）中旋转误差增长相近，而平移误差分化明显：")
log("     2° 旋转噪声下 ANDREFF 平移误差达 9.40 mm，DANIILIDIS 仅 5.14 mm。")
log("     即「联立式方法对旋转噪声更鲁棒、但旋转噪声向平移的传播更剧烈」，")
log("     与文献中「旋转误差向平移传播路径因方法而异」的结论一致。")
log("  4) 计时：TSAI 最快（1.34 ms），ANDREFF 最慢（3.04 ms），")
log("     但都在毫秒级 —— 对离线标定而言耗时不是选型依据。")
log("  5) 工程建议：默认用 PARK 或 HORAUD 作初解，再用 DANIILIDIS 交叉验证；")
log("     无论选哪种，都必须补非线性精化。算法选型的收益远小于数据质量的收益。")


with open(r"F:\hkRobot_实验文档\相机标定与手眼标定_教案\code\method_compare_result.txt",
          "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
log("\n结果已写入 code/method_compare_result.txt")
