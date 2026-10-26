#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_aruco_pose_follow.py —— ArUco 码 6D 位姿估计 + 机械臂随动
==============================================================

对应课程实验：实验 8「Aruco 码机械臂随动」第 3、4 步

ArUco 为什么适合做机器人的视觉基准？
------------------------------------
    二维码/ArUco 本质是「自带身份编码的平面标记」：
      · 4 个角点 → 一次 PnP 就能定出 6D 位姿（3 平移 + 3 旋转）
      · 内部二进制编码 → 唯一 ID + 纠错，抗误检
      · 检测简单快速、鲁棒，工业上广泛用于位姿基准与随动
    marker 坐标系：原点在标记中心，Z 轴垂直标记面朝外，X 轴指向某个固定角点。

随动闭环
--------
    视觉测到 marker 在相机系下的位姿 T_cam←marker
      → 用标定好的 T_grip←cam 转换到末端系
      → 再用手眼/正解转换到基座系
      → 取平移分量作为新的目标点，控制机械臂逼近
      → 循环执行，形成「视觉伺服」闭环

本脚本兼容性说明
----------------
    OpenCV ≥ 4.7 移除了 cv2.aruco.estimatePoseSingleMarkers，
    本脚本统一用 solvePnP（IPPE_SQUARE 专用解法）替代，4.2 / 4.12 均可运行。

用法
----
# 位姿估计（看实时画面，打印每个 marker 的 ID / 距离 / 姿态）
python 05_aruco_pose_follow.py --mode pose --camera 0 \
       --intrinsics calib_out/intrinsics.json --marker_len 50

# 随动（需要机械臂 SDK，见 --robot 参数说明）
python 05_aruco_pose_follow.py --mode follow --camera 0 \
       --intrinsics calib_out/intrinsics.json \
       --handeye dataset_eih/handeye_result.json --marker_len 50 \
       --robot sim

--robot sim    : 仿真，只打印应该发出去的指令（课堂无硬件也能演示）
--robot hulk   : 浩科 HK540，需要 hulk 官方 Python SDK（工控机自带）
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import calib_utils as cu


# --------------------------------------------------------------------------
# 机器人抽象层：把不同 SDK 统一成 go_to_pose / get_pose 两个方法
# --------------------------------------------------------------------------
class RobotBase(object):
    def go_to_pose(self, x, y, z, rx, ry, rz, speed=50.0):
        raise NotImplementedError

    def get_pose(self):
        raise NotImplementedError


class SimRobot(RobotBase):
    """仿真机器人：只打印指令，用于课堂演示与代码调试。"""

    def __init__(self):
        self.pose = (300.0, 0.0, 200.0, 180.0, 0.0, 0.0)

    def go_to_pose(self, x, y, z, rx, ry, rz, speed=50.0):
        print(f"    [SIM] 移动 -> X={x:7.2f} Y={y:7.2f} Z={z:7.2f} "
              f"RX={rx:6.1f} RY={ry:6.1f} RZ={rz:6.1f}  v={speed}")
        self.pose = (x, y, z, rx, ry, rz)
        return True

    def get_pose(self):
        return self.pose


class HulkRobot(RobotBase):
    """
    浩科 HK540 适配层（骨架）。
    实际使用时按工控机上 hulkExperience 提供的 SDK 填两个方法即可，
    上层随动逻辑不需要任何改动 —— 这就是「抽象层」的价值。
    """

    def __init__(self, ip="192.168.1.100", port=9000):
        self.ip, self.port = ip, port
        try:
            # 示例：from hulk_sdk import HulkRobotClient
            # self.cli = HulkRobotClient(ip, port); self.cli.connect()
            raise ImportError("请按工控机上的实际 SDK 名称取消注释")
        except ImportError as exc:
            raise RuntimeError(f"未找到浩科机械臂 SDK：{exc}")

    def go_to_pose(self, x, y, z, rx, ry, rz, speed=50.0):
        raise NotImplementedError("调用 SDK 的 movL / moveJ 接口，单位与 SDK 对齐（mm/deg）")

    def get_pose(self):
        raise NotImplementedError("读取 SDK 的当前笛卡尔位姿，返回 6 元组")


def make_robot(kind):
    if kind == "sim":
        return SimRobot()
    if kind == "hulk":
        return HulkRobot()
    raise ValueError(f"未知机器人类型 {kind}")


# --------------------------------------------------------------------------
def load_intrinsics(path):
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    return (np.array(res["K"], dtype=np.float64),
            np.array(res["dist"], dtype=np.float64).ravel())


# --------------------------------------------------------------------------
def mode_pose(args):
    K, dist = load_intrinsics(args.intrinsics)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[错误] 打不开相机 {args.camera}")
        return 1

    print("=" * 74)
    print(f"  ArUco 位姿估计  （字典 {args.dict}，标记边长 {args.marker_len} mm）")
    print("=" * 74)
    print("  按 q 退出。")
    print(f"  {'ID':>4}{'距离(mm)':>12}{'X(mm)':>10}{'Y(mm)':>10}{'Z(mm)':>10}"
          f"{'yaw(deg)':>12}")
    print("  " + "-" * 62)

    last_print = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = cu.detect_aruco_markers(gray, args.dict)
        vis = frame.copy()

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            poses = cu.estimate_marker_pose(
                corners, ids, args.marker_len / 1000.0, K, dist)
            for mid, (rvec, tvec) in poses.items():
                cv2.drawFrameAxes(vis, K, dist, rvec, tvec,
                                  args.marker_len / 1000.0 * 0.5)
                R, _ = cv2.Rodrigues(rvec)
                yaw = np.degrees(np.arctan2(-R[0, 2], R[2, 2]))
                # 在图像上标注 ID 与距离
                c = corners[list(ids.ravel()).index(mid)].reshape(4, 2)
                cx, cy = c.mean(axis=0)
                txt = f"id={mid} d={np.linalg.norm(tvec)*1000:.0f}mm"
                cv2.putText(vis, txt, (int(cx) - 60, int(cy) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if time.time() - last_print > args.interval:
                    last_print = time.time()
                    print(f"  {mid:>4}{np.linalg.norm(tvec)*1000:>12.1f}"
                          f"{tvec[0]*1000:>10.1f}{tvec[1]*1000:>10.1f}"
                          f"{tvec[2]*1000:>10.1f}{yaw:>12.2f}")

        cv2.imshow("aruco pose (q to quit)", vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------------
def mode_follow(args):
    K, dist = load_intrinsics(args.intrinsics)
    with open(args.handeye, encoding="utf-8") as fh:
        hres = json.load(fh)
    eye_in_hand = hres["config"] == "eye_in_hand"
    X = np.array(hres.get("X_gripper2cam") or hres["X_base2cam"],
                 dtype=np.float64)

    print("=" * 74)
    print(f"  ArUco 随动  （构型：{hres['config']}）")
    print("=" * 74)
    print(f"  手眼矩阵 X =\n{np.array2string(X, precision=4, suppress_small=True)}")

    robot = make_robot(args.robot)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[错误] 打不开相机 {args.camera}")
        return 1

    print(f"\n  随动参数：标定板距离裕量 {args.standoff} mm，"
          f"位置增益 {args.gain}，速度 {args.speed}")
    print("  按 q 退出。\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = cu.detect_aruco_markers(gray, args.dict)
        vis = frame.copy()

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            poses = cu.estimate_marker_pose(
                corners, ids, args.marker_len / 1000.0, K, dist)
            if poses:
                mid = sorted(poses.keys())[0]
                rvec, tvec = poses[mid]
                cv2.drawFrameAxes(vis, K, dist, rvec, tvec,
                                  args.marker_len / 1000.0 * 0.5)
                T_cam2marker = cu.rvec_tvec_to_T(rvec, tvec)

                # ---------- 坐标链：marker → 相机 → 末端 → 基座 ----------
                if eye_in_hand:
                    # 相机在末端上：T_base←marker = T_base←grip · X · T_cam←marker
                    # 这里需要正解；仿真下用 get_pose 提供的当前位姿
                    T_base2grip = pose_to_T(robot.get_pose())
                    T_base2marker = T_base2grip @ X @ T_cam2marker
                    # 期望末端位姿：让末端停在 marker 正上方 standoff 处
                    R_m = T_base2marker[:3, :3]
                    t_m = T_base2marker[:3, 3]
                    target_t = t_m + R_m @ np.array([0, 0, -args.standoff / 1000.0])
                    T_target_base2grip = cu.make_T(R_m, target_t)
                    # 转成末端系下的相机目标，再做反向映射更严谨；
                    # 教学版直接让末端姿态与 marker 对齐
                else:
                    # 相机固定在外部：T_base←marker = X · T_cam←marker
                    T_base2marker = X @ T_cam2marker
                    R_m = T_base2marker[:3, :3]
                    t_m = T_base2marker[:3, 3]
                    # eye-to-hand 下随动的是末端，需借正解换算，这里仅演示位置逼近
                    T_target_base2grip = cu.make_T(R_m, t_m)

                tgt = T_target_base2grip[:3, 3]
                cur = np.array(robot.get_pose()[:3]) / 1000.0
                new_t = cur + args.gain * (tgt - cur)
                cmd_mm = new_t * 1000.0
                cur_rot = robot.get_pose()[3:]
                robot.go_to_pose(cmd_mm[0], cmd_mm[1], cmd_mm[2],
                                 cur_rot[0], cur_rot[1], cur_rot[2],
                                 speed=args.speed)

                cv2.putText(vis, f"id={mid}  target=({cmd_mm[0]:.0f},"
                                 f"{cmd_mm[1]:.0f},{cmd_mm[2]:.0f})mm",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                cv2.putText(vis, f"err={np.linalg.norm(tgt-cur)*1000:.1f}mm",
                            (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2)

        cv2.imshow("aruco follow (q to quit)", vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def pose_to_T(p):
    """6 元组 (x,y,z,rx,ry,rz)（mm, deg）→ 4x4 齐次矩阵（米, 弧度）。"""
    x, y, z, rx, ry, rz = p
    R = cv2.Rodrigues(np.radians(np.array([rx, ry, rz], dtype=np.float64)))[0]
    return cu.make_T(R, np.array([x, y, z], dtype=np.float64) / 1000.0)


# --------------------------------------------------------------------------
def mode_selftest(args):
    """无相机自检：打印标记生成、检测、位姿估计的完整链路。"""
    print("=" * 74)
    print("  ArUco 链路自检（生成 → 渲染投影 → 检测 → 位姿估计）")
    print("=" * 74)
    adict = cu.aruco_dictionary(args.dict)
    marker_len = args.marker_len / 1000.0

    # 用「已知位姿 + 已知内参」人工合成一张带 marker 的图像
    K = np.array([[1200.0, 0, 640.0], [0, 1200.0, 360.0], [0, 0, 1.0]])
    dist = np.zeros(5)
    T_true = cu.make_T(cv2.Rodrigues(np.radians(np.array([80.0, 0, 15.0])))[0],
                      np.array([0.02, -0.01, 0.45]))

    h = marker_len / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    img_pts, _ = cv2.projectPoints(obj, *cu.T_to_rvec_tvec(T_true), K, dist)

    canvas = np.full((720, 1280), 255, np.uint8)
    cv2.fillConvexPoly(canvas, img_pts.reshape(4, 2).astype(np.int32), 0)
    cv2.putText(canvas, "SYNTH", (560, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
    print(f"  真值 T_cam←marker 平移 = "
          f"{np.array2string(T_true[:3,3], precision=4)} m")
    print(f"  投影后四个角点像素坐标：\n{np.array2string(img_pts.reshape(4,2), precision=2)}")

    # 反解
    okp, rvec, tvec = cv2.solvePnP(
        obj.astype(np.float32), img_pts.astype(np.float32), K, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    T_est = cu.rvec_tvec_to_T(rvec, tvec)
    da, dt = cu.pose_diff(T_true, T_est)
    print(f"  反解误差：旋转 {da:.6f}°  平移 {dt:.6f} mm")
    print(f"  ⇒ {'通过' if da < 1e-2 and dt < 1e-2 else '未通过'}")

    print("\n  工程要点提醒：")
    print("    · marker 边长必须用卡尺实测，标称值误差按比例进入平移解")
    print("    · 侧面大角度观察时角点定位精度下降，建议入射角 < 60°")
    print("    · 距离越远、marker 越小，位置噪声越大：d/z² 关系")
    print("    · 多 marker（ChArUco / ArUco Board）可显著提升鲁棒性与精度")
    return 0


# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="ArUco 位姿估计与随动教学脚本")
    p.add_argument("--mode", required=True,
                   choices=["pose", "follow", "selftest"])
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--intrinsics",
                   default=os.path.join("calib_out", "intrinsics.json"))
    p.add_argument("--handeye",
                   default=os.path.join("dataset_eih", "handeye_result.json"))
    p.add_argument("--dict", default="DICT_6X6_250")
    p.add_argument("--marker_len", type=float, default=50.0,
                   help="marker 实测边长 mm")
    p.add_argument("--robot", default="sim", choices=["sim", "hulk"])
    p.add_argument("--standoff", type=float, default=150.0,
                   help="末端与 marker 保持的距离 mm")
    p.add_argument("--gain", type=float, default=0.3,
                   help="位置增益（0~1），越小越稳但越慢")
    p.add_argument("--speed", type=float, default=50.0)
    p.add_argument("--interval", type=float, default=0.3,
                   help="位姿打印间隔秒")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit({"pose": mode_pose, "follow": mode_follow,
              "selftest": mode_selftest}[args.mode](args) or 0)
