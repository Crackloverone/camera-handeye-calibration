# -*- coding: utf-8 -*-
"""
handeye_capture.py —— 手眼标定数据采集公共模块
================================================

手眼标定的精度上限在采集环节就决定了。本模块把采集规范固化成代码，
避免学生「拍完照才发现数据不能用」。

核心约束（全部在代码里强制检查）：
  1. 【静态采集】机械臂完全停稳后再曝光（动态采集时间戳不同步，误差可达 10+ mm）
  2. 【位姿多样】12~20 个位姿，相邻旋转 > 30°，三轴均有旋转分量
  3. 【板不入出视野】标定板完整可见，占视野 1/4 以上
  4. 【留档可复现】位姿、图像、内参、板尺寸、设备信息全部落盘

目录结构：
    dataset/
      ├── rgb/          img_0001.jpg ...       图像
      ├── pose/         img_0001.npy ...       机器人位姿 4x4（T_base←grip）
      ├── extrinsics.json                      每帧标定板外参 T_cam←board
      └── session.json                         采集元信息
"""

import os
import json
import time

import cv2
import numpy as np

import calib_utils as cu


class HandEyeDataset(object):
    """一次手眼标定采集会话的落盘容器。"""

    def __init__(self, root, K, dist, board, eye_in_hand=True, notes=""):
        self.root = root
        self.rgb_dir = os.path.join(root, "rgb")
        self.pose_dir = os.path.join(root, "pose")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.pose_dir, exist_ok=True)

        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()
        self.board = board
        self.eye_in_hand = bool(eye_in_hand)
        self.notes = notes
        self.frames = []          # [{"img":..,"pose":..,"extrinsic":..}]

        # ChArUco / 棋盘格目标点
        self.is_charuco = board.get("type") == "charuco"
        if self.is_charuco:
            self._board = cu.make_charuco_board(
                board["squares_x"], board["squares_y"],
                board["square_mm"], board["marker_mm"],
                board.get("dict_name", "DICT_5X5_100"))
        else:
            self._objp = cu.build_chessboard_points(
                board["inner_cols"], board["inner_rows"], board["square_mm"])
            self._pattern = (board["inner_cols"], board["inner_rows"])

    # ---------------- 加一帧 ----------------
    def add_frame(self, bgr, T_base2grip):
        """
        bgr         : 采集到的图像
        T_base2grip : 该时刻机器人正解 4x4（T_base←grip，单位米）
        返回 True 表示成功收录（含标定板外参解算）。
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.is_charuco:
            ok, corners, ids = cu.detect_charuco(gray, self._board)
            objp = cu.charuco_object_points(self._board, ids) if ok else None
        else:
            ok, corners = cu.detect_chessboard(gray, self._pattern)
            objp = self._objp if ok else None

        if not ok:
            print("  [x] 标定板角点未检出，本帧丢弃。"
                  "请检查：光照/反光、板是否完整入镜、格子尺寸参数是否匹配。")
            return False

        okp, rvec, tvec = cv2.solvePnP(
            np.asarray(objp, dtype=np.float64),
            np.asarray(corners, dtype=np.float64), self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not okp:
            print("  [x] solvePnP 失败，本帧丢弃。")
            return False

        # 重投影误差把关：单帧误差过大说明角点或曝光有问题
        proj, _ = cv2.projectPoints(objp, rvec, tvec, self.K, self.dist)
        err = np.linalg.norm(proj.reshape(-1, 2)
                             - corners.reshape(-1, 2), axis=1)
        if float(err.mean()) > 1.5:
            print(f"  [x] 单帧重投影误差 {err.mean():.3f} px > 1.5 px，本帧丢弃。")
            return False

        idx = len(self.frames) + 1
        img_name = f"img_{idx:04d}.jpg"
        pose_name = f"img_{idx:04d}.npy"
        cv2.imwrite(os.path.join(self.rgb_dir, img_name), bgr)
        np.save(os.path.join(self.pose_dir, pose_name),
                np.asarray(T_base2grip, dtype=np.float64))

        # T_target2cam = T_cam←board（solvePnP 输出即为该含义）
        T_target2cam = cu.rvec_tvec_to_T(rvec, tvec)

        self.frames.append({
            "img": img_name,
            "pose": pose_name,
            "reproj_px": float(err.mean()),
            "n_points": int(len(objp)),
            "T_target2cam": T_target2cam.tolist(),
        })
        print(f"  [v] 收录第 {idx} 帧  重投影 {err.mean():.3f} px  "
              f"角点 {len(objp)} 个")
        return True

    # ---------------- 保存 ----------------
    def save(self):
        meta = {
            "eye_in_hand": self.eye_in_hand,
            "board": self.board,
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "image_size": None,
            "n_frames": len(self.frames),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": self.notes,
            "cv_version": cv2.__version__,
        }
        if self.frames:
            meta["image_size"] = list(
                cv2.imread(os.path.join(self.rgb_dir,
                                        self.frames[0]["img"])).shape[1::-1])
        with open(os.path.join(self.root, "session.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        ext = {f["img"]: {"T_target2cam": f["T_target2cam"],
                          "reproj_px": f["reproj_px"]}
               for f in self.frames}
        with open(os.path.join(self.root, "extrinsics.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(ext, fh, ensure_ascii=False, indent=2)

        print(f"\n[保存] 采集会话已落盘：{os.path.abspath(self.root)}")
        print(f"       图像 {len(self.frames)} 帧 + 位姿 npy + 外参 json + 会话元信息")
        print("       留档价值：任何一项缺失都会让标定结果无法复现或追溯。")
        return self.frames

    # ---------------- 读取（供求解脚本使用） ----------------
    @staticmethod
    def load(root):
        """从已落盘目录恢复数据，返回 (gripper2base, target2cam, K, dist, meta)。"""
        with open(os.path.join(root, "session.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        with open(os.path.join(root, "extrinsics.json"), encoding="utf-8") as fh:
            ext = json.load(fh)

        g_list, c_list = [], []
        for img_name in sorted(ext.keys()):
            pose_file = os.path.join(root, "pose",
                                     img_name.replace(".jpg", ".npy"))
            if not os.path.exists(pose_file):
                continue
            g_list.append(np.load(pose_file))
            c_list.append(np.array(ext[img_name]["T_target2cam"],
                                   dtype=np.float64))
        return (g_list, c_list,
                np.array(meta["K"], dtype=np.float64),
                np.array(meta["dist"], dtype=np.float64).ravel(), meta)


def capture_console(dataset, cam_index=0, width=0, height=0):
    """
    交互式采集：终端输入机器人位姿 → 回车采图。
    适用于「示教器手动摆位 + 上位机录入坐标」的课堂场景
    （对应实验 8 第 3 步 / 实验 9 第 5 步的操作方式）。

    快捷键：
        回车  = 采一帧（会先提示输入该位姿的 XYZ / 或直接读示教器显示）
        d     = 删掉最后一帧
        q     = 结束采集
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[错误] 打不开相机 {cam_index}")
        return
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("\n" + "=" * 66)
    print("  手眼标定数据采集（静态采集模式）")
    print("=" * 66)
    print("  操作顺序（每帧都要严格照做）：")
    print("    1) 用示教器把机械臂移到新位姿")
    print("    2) 等机械臂完全停稳（不抖、无残余振动）")
    print("    3) 确认标定板完整可见且占视野 1/4 以上")
    print("    4) 回到本终端按回车采图")
    print("  采集目标：12~20 帧，相邻位姿旋转 > 30°，X/Y/Z 三轴都要转")
    print("  命令：回车=采集  d=删除上一帧  q=结束")
    print("=" * 66)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[错误] 读帧失败")
            break
        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if dataset.is_charuco:
            found, corners, _ = cu.detect_charuco(gray, dataset._board)
        else:
            found, corners = cu.detect_chessboard(gray, dataset._pattern)
        if found:
            cv2.drawChessboardCorners(
                vis, dataset._pattern if not dataset.is_charuco
                else (dataset.board["squares_x"] - 1,
                      dataset.board["squares_y"] - 1),
                corners, True)
        cv2.putText(vis, f"frames: {len(dataset.frames)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
        cv2.putText(vis, "DETECTED" if found else "NOT FOUND", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 200, 0) if found else (0, 0, 255), 2)
        cv2.imshow("hand-eye capture", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d') and dataset.frames:
            last = dataset.frames.pop()
            for d, nm in [(dataset.rgb_dir, last["img"]),
                          (dataset.pose_dir, last["pose"])]:
                p = os.path.join(d, nm)
                if os.path.exists(p):
                    os.remove(p)
            print(f"  [撤销] 已删除 {last['img']}")
            continue
        if key in (13, 10):            # 回车
            print("\n请输入该位姿的机器人坐标（单位米）：")
            print("  格式： tx ty tz rx ry rz   （笛卡尔 X Y Z + 旋转向量，空格分隔）")
            print("  或直接回车，表示稍后从机器人 SDK 自动读取")
            s = input("  > ").strip()
            if s:
                try:
                    v = [float(x) for x in s.replace(",", " ").split()]
                    if len(v) != 6:
                        raise ValueError
                    T = cu.rvec_tvec_to_T(np.array(v[3:]), np.array(v[:3]))
                except ValueError:
                    print("  [x] 格式错误，本帧跳过。")
                    continue
            else:
                print("  [i] 未输入坐标。实际项目中应改为从机器人 SDK 直接读正解，"
                      "避免人工录入误差。")
                print("      本课示例用单位矩阵占位，仅用于演示流程。")
                T = np.eye(4)
            dataset.add_frame(frame, T)

    cap.release()
    cv2.destroyAllWindows()
    if dataset.frames:
        dataset.save()
        # 采集后立刻做运动激励诊断，不达标就当场补采
        gaps = [f["T_target2cam"] for f in dataset.frames]
        cu.motion_informativeness(
            [np.load(os.path.join(dataset.pose_dir, f["pose"]))
             for f in dataset.frames])
        print("  提示：若上面出现 [!] 警告，请立即补采，"
              "不要等到解算失败再回头重来。")
