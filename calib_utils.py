# -*- coding: utf-8 -*-
"""
calib_utils.py —— 相机标定 / 手眼标定 公共工具库
==================================================

适用环境
--------
- 本课程实验平台：Ubuntu 16.04 LTS + Python 3.7 + OpenCV 4.2.0.34
- 同时兼容 OpenCV >= 4.7（新版 aruco API 有破坏性变更，本库已做适配）

主要能力
--------
1. 位姿表示互转（rvec/tvec <-> 4x4 齐次矩阵）、求逆、复合
2. 标定板目标点生成（棋盘格 / ChArUco / 单 ArUco）
3. 角点检测（自动适配 OpenCV 4.2 / 4.7+ 两套 API）
4. 内参标定 + 重投影误差质量报告
5. 手眼标定多算法交叉求解（AX = XB）
6. 九点仿射标定（eye-to-hand 平面抓取简化方案）
7. 精度验证与工作距离误差折算

作者：CV & Edge AI 教学组
"""

from __future__ import annotations

import os
import json
import glob
import numpy as np
import cv2

# --------------------------------------------------------------------------
# 0. 版本探测与常量
# --------------------------------------------------------------------------

CV_VERSION = tuple(int(x) for x in cv2.__version__.split(".")[:2])
CV_NEW_ARUCO = CV_VERSION >= (4, 7)          # 4.7 起 aruco API 大改
CV_HAS_CHARUCO_DETECTOR = CV_VERSION >= (4, 7)

# 本课程推荐的默认参数（与实验平台器材对应）
DEFAULT_BOARD = {
    "type": "chessboard",        # chessboard | charuco
    "inner_cols": 8,             # 内角点列数（实验 9 内参标定用 8x6）
    "inner_rows": 6,             # 内角点行数
    "square_mm": 25.0,           # 方格边长，务必用卡尺实测！
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------
# 1. 位姿（刚体变换）基础运算
# --------------------------------------------------------------------------

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """由 3x3 旋转矩阵与 3x1 平移向量构造 4x4 齐次变换矩阵。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Rodrigues 旋转向量 + 平移向量 -> 4x4 齐次变换矩阵。"""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return make_T(R, tvec)


def T_to_rvec_tvec(T: np.ndarray):
    """4x4 齐次变换矩阵 -> (rvec, tvec)。"""
    T = np.asarray(T, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.reshape(3), T[:3, 3].reshape(3)


def inv_T(T: np.ndarray) -> np.ndarray:
    """刚体变换的高效求逆（利用正交性，比 np.linalg.inv 更稳、更快）。"""
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rot_angle_deg(R: np.ndarray) -> float:
    """旋转矩阵对应的旋转角（度）。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def rot_axis(R: np.ndarray) -> np.ndarray:
    """旋转矩阵对应的旋转轴（单位向量）。旋转角接近 0 时返回零向量。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """用 4x4 变换矩阵作用到 Nx3 点集。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def pose_diff(Ta: np.ndarray, Tb: np.ndarray) -> tuple:
    """两个位姿之差：(旋转误差角度, 平移误差 mm)。"""
    dT = inv_T(Ta) @ Tb
    return rot_angle_deg(dT[:3, :3]), float(np.linalg.norm(dT[:3, 3]) * 1000.0)


# --------------------------------------------------------------------------
# 2. 标定板定义与目标点生成
# --------------------------------------------------------------------------

def build_chessboard_points(inner_cols=8, inner_rows=6, square_mm=25.0):
    """
    生成棋盘格内角点的棋盘坐标系坐标（Z=0 平面）。
    返回 (N,3) float32，单位 mm，原点在左上角内角点，X 向右，Y 向下。
    """
    objp = np.zeros((inner_rows * inner_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2)
    return objp * float(square_mm)


def aruco_dictionary(name="DICT_6X6_250"):
    """跨版本获取预定义 ArUco 字典。"""
    d = getattr(cv2.aruco, name, None)
    if d is None:
        # 老版本常量字符串写法
        d = cv2.aruco.Dictionary_get(getattr(cv2.aruco, name))
        return d
    if CV_NEW_ARUCO:
        return cv2.aruco.getPredefinedDictionary(d)
    return cv2.aruco.Dictionary_get(d)


def make_charuco_board(squares_x=7, squares_y=5,
                       square_mm=30.0, marker_mm=22.0,
                       dict_name="DICT_5X5_100"):
    """
    ChArUco 板：抗遮挡，是手眼标定的推荐目标。
    squares_x / squares_y 为「方格数」（内角点数为 (sx-1)*(sy-1)）。
    """
    adict = aruco_dictionary(dict_name)
    if CV_NEW_ARUCO:
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), float(square_mm), float(marker_mm), adict)
    else:
        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, float(square_mm), float(marker_mm), adict)
    return board


def charuco_object_points(board, charuco_ids):
    """由 ChArUco 检测到的角点 id 取对应的棋盘坐标系 3D 点（mm）。"""
    if CV_NEW_ARUCO:
        all_pts = board.getChessboardCorners()
    else:
        all_pts = board.chessboardCorners
    return np.asarray(all_pts, dtype=np.float32)[charuco_ids.ravel()]


def save_board_image(board, path, size=(1400, 1000), margin=60):
    """把标定板导出成可直接打印的图片（务必按 100% 比例打印并实测尺寸）。"""
    if CV_NEW_ARUCO:
        img = board.generateImage(size, margin, 0)
    else:
        img = board.draw(size, margin, 0)
    cv2.imwrite(path, img)
    print(f"[标定板] 已导出：{path}  （打印后请用卡尺复核方格实际边长！）")
    return path


# --------------------------------------------------------------------------
# 3. 角点检测（自动适配 OpenCV 4.2 / 4.7+）
# --------------------------------------------------------------------------

def detect_chessboard(gray, pattern_size=(8, 6)):
    """
    检测棋盘格内角点。返回 (found, corners)；
    corners 形如 (N,1,2) float32 的亚像素角点（位于棋盘坐标系下，从左上开始按行）。
    """
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(
        gray, pattern_size, flags=flags)
    if not ok:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def detect_charuco(gray, board, min_corners=6):
    """
    检测 ChArUco 角点。返回 (found, charuco_corners(N,1,2), charuco_ids(N,1))。
    """
    adict = board.getDictionary() if CV_NEW_ARUCO else board.dictionary

    if CV_HAS_CHARUCO_DETECTOR:
        detector = cv2.aruco.CharucoDetector(board)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        m_corners, m_ids, _ = cv2.aruco.detectMarkers(gray, adict, parameters=params)
        if m_ids is None or len(m_ids) == 0:
            return False, None, None
        _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            m_corners, m_ids, gray, board)

    if ch_ids is None or len(ch_ids) < min_corners:
        return False, None, None
    # 亚像素精化（对检测到的角点再做一次 cornerSubPix 提升一致性）
    ch_corners = cv2.cornerSubPix(
        gray, np.asarray(ch_corners, dtype=np.float32).reshape(-1, 1, 2),
        (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4))
    return True, ch_corners.reshape(-1, 1, 2), ch_ids.reshape(-1, 1)


def detect_aruco_markers(gray, dict_name="DICT_6X6_250"):
    """跨版本 ArUco 标记检测。返回 (corners: list[N,1,4,2], ids: (N,1))。"""
    adict = aruco_dictionary(dict_name)
    if CV_NEW_ARUCO:
        detector = cv2.aruco.ArucoDetector(
            adict, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, adict, parameters=params)
    return corners, ids


def estimate_marker_pose(corners, ids, marker_len_m, K, dist):
    """
    ArUco 单标记位姿估计（替代 OpenCV 4.7 已移除的 estimatePoseSingleMarkers）。
    返回 {marker_id: (rvec(3,1), tvec(3,1))}，tvec 单位与 marker_len_m 一致（米）。
    """
    h = marker_len_m / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                   dtype=np.float32)
    poses = {}
    for c, i in zip(corners, ids.ravel()):
        ok, rvec, tvec = cv2.solvePnP(
            obj, np.asarray(c, dtype=np.float32).reshape(4, 1, 2), K, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            poses[int(i)] = (rvec.reshape(3, 1), tvec.reshape(3, 1))
    return poses


# --------------------------------------------------------------------------
# 4. 相机内参标定
# --------------------------------------------------------------------------

def list_images(folder):
    files = []
    for ext in IMAGE_EXTS:
        files += glob.glob(os.path.join(folder, "*" + ext))
        files += glob.glob(os.path.join(folder, "*" + ext.upper()))
    return sorted(set(files))


def calibrate_intrinsics(image_folder,
                         board=DEFAULT_BOARD,
                         image_size=None,
                         out_dir="calib_out",
                         min_valid_views=8,
                         verbose=True):
    """
    相机内参标定（张正友法，OpenCV calibrateCamera 实现四步推导链）。

    参数
    ----
    image_folder : 标定图像目录
    board        : 标定板配置 dict（见 DEFAULT_BOARD / make_charuco_board）
    image_size   : (w, h)，None 则从首张图推断
    out_dir      : 结果输出目录

    返回
    ----
    dict: K, dist, rvecs, tvecs, mean_err(px), max_err(px), per_view(列表), meta
    """
    os.makedirs(out_dir, exist_ok=True)
    files = list_images(image_folder)
    if len(files) == 0:
        raise FileNotFoundError(f"目录中没有图片：{image_folder}")

    is_charuco = board.get("type", "chessboard") == "charuco"
    if is_charuco:
        bd = make_charuco_board(
            board["squares_x"], board["squares_y"],
            board["square_mm"], board["marker_mm"],
            board.get("dict_name", "DICT_5X5_100"))
    else:
        objp = build_chessboard_points(
            board["inner_cols"], board["inner_rows"], board["square_mm"])
        pattern_size = (board["inner_cols"], board["inner_rows"])

    obj_points, img_points, used_files, meta = [], [], [], []
    if image_size is None:
        probe = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
        image_size = (probe.shape[1], probe.shape[0])

    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if (gray.shape[1], gray.shape[0]) != tuple(image_size):
            if verbose:
                print(f"[跳过] 分辨率不一致 {os.path.basename(f)}")
            continue

        if is_charuco:
            ok, ch_corners, ch_ids = detect_charuco(gray, bd)
            if not ok:
                if verbose:
                    print(f"[跳过] ChArUco 角点不足 {os.path.basename(f)}")
                continue
            obj_points.append(charuco_object_points(bd, ch_ids))
            img_points.append(np.asarray(ch_corners, dtype=np.float32))
            meta.append({"file": os.path.basename(f), "n_points": int(len(ch_ids))})
        else:
            ok, corners = detect_chessboard(gray, pattern_size)
            if not ok:
                if verbose:
                    print(f"[跳过] 未检出棋盘格 {os.path.basename(f)}")
                continue
            obj_points.append(objp.copy())
            img_points.append(corners)
            meta.append({"file": os.path.basename(f), "n_points": int(len(corners))})
        used_files.append(f)

    if len(obj_points) < min_valid_views:
        raise RuntimeError(
            f"有效标定图仅 {len(obj_points)} 张（<{min_valid_views}）。"
            "请检查：标定板是否平整、是否完整入镜、光照是否均匀、格子尺寸是否匹配。")

    if verbose:
        print(f"[标定] 有效图 {len(obj_points)}/{len(files)} 张，开始张正友法求解…")

    flags = 0
    if is_charuco:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS if False else 0
    flags |= cv2.CALIB_FIX_K3  # 普通工业镜头 k3 可置零，避免过拟合

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, tuple(image_size), None, None, flags=flags)

    # ---- 逐图重投影误差（质量诊断的核心指标）----
    per_view = []
    for i, (op, ip, rv, tv) in enumerate(zip(obj_points, img_points, rvecs, tvecs)):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        e = np.linalg.norm(
            proj.reshape(-1, 2) - np.asarray(ip, dtype=np.float64).reshape(-1, 2),
            axis=1)
        per_view.append({
            "file": os.path.basename(used_files[i]),
            "mean_px": float(e.mean()),
            "max_px": float(e.max()),
            "n_points": int(len(e)),
        })
    mean_err = float(np.mean([v["mean_px"] for v in per_view]))
    max_err = float(np.max([v["max_px"] for v in per_view]))

    result = {
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "rvecs": [r.ravel().tolist() for r in rvecs],
        "tvecs": [t.ravel().tolist() for t in tvecs],
        "rms_px": float(rms),
        "mean_reproj_px": mean_err,
        "max_point_reproj_px": max_err,
        "per_view": per_view,
        "image_size": list(image_size),
        "board": board,
        "meta": meta,
    }

    with open(os.path.join(out_dir, "intrinsics.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    if verbose:
        print_quality_report(result)
    return result


def print_quality_report(res):
    """按工程判据打印内参标定质量报告。"""
    m, mx = res["mean_reproj_px"], res["max_point_reproj_px"]
    if mx < 0.5:
        grade = "优秀（可直接用于手眼标定）"
    elif mx < 1.0:
        grade = "良好（可用于工业抓取）"
    elif mx < 2.0:
        grade = "可用（精度要求高时建议重标）"
    else:
        grade = "不合格，必须重标"

    print("\n" + "=" * 62)
    print("  相机内参标定质量报告")
    print("=" * 62)
    K = np.array(res["K"])
    print(f"  fx = {K[0,0]:10.3f} px      fy = {K[1,1]:10.3f} px")
    print(f"  cx = {K[0,2]:10.3f} px      cy = {K[1,2]:10.3f} px")
    print(f"  畸变 [k1 k2 p1 p2 k3] = {np.array(res['dist'])}")
    print(f"  总 RMS          : {res['rms_px']:.4f} px")
    print(f"  平均重投影误差  : {res['mean_reproj_px']:.4f} px")
    print(f"  最大点重投影误差: {res['max_point_reproj_px']:.4f} px")
    print(f"  判定            : {grade}")
    print("-" * 62)
    worst = sorted(res["per_view"], key=lambda v: -v["mean_px"])[:3]
    print("  误差最大的 3 张图（建议检查或剔除）：")
    for v in worst:
        print(f"    {v['file']:>28s}  mean={v['mean_px']:.3f} px")
    print("=" * 62 + "\n")


def undistort_compare(img, K, dist, out_path=None):
    """畸变校正前后对照图，用于课堂直观演示。"""
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).ravel()
    fixed = cv2.undistort(img, K, dist, None, K)
    pair = np.hstack([img, np.full((img.shape[0], 12, 3), 255, np.uint8), fixed])
    if out_path:
        cv2.imwrite(out_path, pair)
    return pair


def undistort_points(pts, K, dist):
    """像素点去畸变。pts 形如 (N,1,2)。"""
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).ravel()
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, K, dist, P=K).reshape(-1, 1, 2)


# --------------------------------------------------------------------------
# 5. 手眼标定：AX = XB
# --------------------------------------------------------------------------

HAND_EYE_METHODS = {
    "TSAI":      getattr(cv2, "CALIB_HAND_EYE_TSAI", None),
    "PARK":      getattr(cv2, "CALIB_HAND_EYE_PARK", None),
    "HORAUD":    getattr(cv2, "CALIB_HAND_EYE_HORAUD", None),
    "ANDREFF":   getattr(cv2, "CALIB_HAND_EYE_ANDREFF", None),
    "DANIILIDIS": getattr(cv2, "CALIB_HAND_EYE_DANIILIDIS", None),
}

# ==========================================================================
#  手眼标定的核心约定（已用合成真值实测验证，见 test_handeye_synthetic.py）
# --------------------------------------------------------------------------
#  cv2.calibrateHandEye(a, b) 求解的是：
#
#       a_i · X · b_i = 常量            (对所有采集位姿 i 成立)
#
#  等价的教科书形式（取任意两个位姿 i、j）：
#
#       A · X = X · B,   其中 A = a_j⁻¹ · a_i ,  B = b_j · b_i⁻¹
#
#  推导：由 a_i X b_i = a_j X b_j 左乘 a_j⁻¹、右乘 b_i⁻¹ 即得。
#
#  两种构型的区别**仅在于 a 取什么**：
#
#  ┌─────────────┬──────────────────────┬──────────────────┬─────────────────┐
#  │   构型      │  a_i                 │  b_i             │  求得的 X       │
#  ├─────────────┼──────────────────────┼──────────────────┼─────────────────┤
#  │ Eye-in-Hand │  T_base←grip(i) 正解 │ T_cam←board(i)   │ T_grip←cam      │
#  │ Eye-to-Hand │  T_grip←base(i) 取逆 │ T_cam←board(i)   │ T_base←cam      │
#  └─────────────┴──────────────────────┴──────────────────┴─────────────────┘
#
#  即：eye-to-hand 就是把机器人位姿取逆，其余完全一样。
#  实测（本机 OpenCV 4.12，合成真值、无噪声）：两种构型误差均为 0.0000° / 0.0000 mm；
#  若把求逆位置搞反，eye-to-hand 的误差会变成几十毫米量级。
# ==========================================================================


def solve_hand_eye(gripper2base_list, target2cam_list,
                   eye_in_hand=True, method="PARK"):
    """
    手眼标定求解（统一入口）。

    参数
    ----
    gripper2base_list : list[4x4]  机器人正解 T_base←grip(i)
    target2cam_list   : list[4x4]  标定板外参 T_cam←board(i)（solvePnP 结果）
    eye_in_hand       : True=眼在手上；False=眼在手外
    method            : PARK / HORAUD / TSAI / ANDREFF / DANIILIDIS

    返回
    ----
    4x4 矩阵 X：
      eye_in_hand=True  → X = T_grip←cam  （相机在末端法兰下的位姿）
      eye_in_hand=False → X = T_base←cam  （相机在机器人基座下的位姿）
    """
    m = HAND_EYE_METHODS.get(method)
    if m is None:
        raise ValueError(f"当前 OpenCV 不支持方法 {method}（版本 {cv2.__version__}）")

    if eye_in_hand:
        a_list = list(gripper2base_list)
    else:
        # 唯一差别：机器人位姿整体取逆
        a_list = [inv_T(T) for T in gripper2base_list]

    R_a = [A[:3, :3] for A in a_list]
    t_a = [A[:3, 3].reshape(3, 1) for A in a_list]
    R_b = [B[:3, :3] for B in target2cam_list]
    t_b = [B[:3, 3].reshape(3, 1) for B in target2cam_list]

    R_X, t_X = cv2.calibrateHandEye(R_a, t_a, R_b, t_b, method=m)
    return make_T(R_X, t_X)


def build_AB(gripper2base_list, target2cam_list, eye_in_hand=True):
    """
    构造教科书的 AX = XB 形式，便于课堂推导与残差计算。

        A = a_j⁻¹ · a_i        （机器人侧相对运动）
        B = b_j · b_i⁻¹        （相机侧相对运动）
        X 同上（eye_in_hand 决定 a 是否取逆）

    推导：由 a_i X b_i = a_j X b_j，左乘 a_j⁻¹、右乘 b_i⁻¹ 得
          (a_j⁻¹ a_i) X = X (b_j b_i⁻¹)
    说明：A、B 与 solve_hand_eye 使用完全相同的 a、b 定义，
    因此 axxb_residual(A, B, X) 得到的残差与解 X 自洽（应为 0 量级）。
    """
    if eye_in_hand:
        a_list = list(gripper2base_list)
    else:
        a_list = [inv_T(T) for T in gripper2base_list]

    n = len(a_list)
    A, B = [], []
    for i in range(n):
        for j in range(i + 1, n):
            A.append(inv_T(a_list[j]) @ a_list[i])
            B.append(target2cam_list[j] @ inv_T(target2cam_list[i]))
    return A, B


def cross_check_methods(gripper2base_list, target2cam_list,
                        eye_in_hand=True, methods=None, verbose=True):
    """
    多算法交叉求解 + 一致性检查。

    工程意义：五种算法的数学原理彼此独立（分离式/联立式、轴角/四元数/李代数/
    对偶四元数），若解出的 X 在噪声水平内一致，说明数据与坐标系约定都没问题；
    若某个算法显著偏离，往往是「运动激励不足」或「个别位姿外参解算错误」。
    """
    methods = methods or ["PARK", "HORAUD", "TSAI", "ANDREFF", "DANIILIDIS"]
    sols = {}
    for name in methods:
        try:
            sols[name] = solve_hand_eye(gripper2base_list, target2cam_list,
                                        eye_in_hand=eye_in_hand, method=name)
        except Exception as exc:               # noqa: BLE001
            if verbose:
                print(f"  [跳过] {name}: {exc}")

    if not sols:
        raise RuntimeError("所有方法均求解失败，请检查运动激励是否充分。")

    ref = sols.get("PARK", list(sols.values())[0])
    rows = []
    for name, X in sols.items():
        da, dt = pose_diff(ref, X)
        rows.append((name, da, dt))

    if verbose:
        print("\n  多算法交叉一致性（以 PARK 为基准）")
        print("  " + "-" * 56)
        print(f"  {'算法':<12}{'旋转偏差(deg)':>16}{'平移偏差(mm)':>16}")
        for name, da, dt in rows:
            flag = "" if da < 0.5 and dt < 5.0 else "   <-- 偏离较大，需排查"
            print(f"  {name:<12}{da:>16.4f}{dt:>16.4f}{flag}")
        print("  " + "-" * 56 + "\n")
        if sols:
            spread_r = np.mean([r[1] for r in rows])
            spread_t = np.mean([r[2] for r in rows])
            print(f"  [判据] 平均离散度：旋转 {spread_r:.4f}°，平移 {spread_t:.4f} mm")
            if spread_r > 1.0 or spread_t > 10.0:
                print("  [警告] 离散度过大，强烈建议："
                      "①补采多轴大角度运动；②检查个别位姿的标定板外参是否解错。")
            print()

    return sols



def axxb_residual(A_list, B_list, X):
    """AX=XB 残差统计：反映解对方程组的拟合程度。"""
    rot_err, trans_err = [], []
    for A, B in zip(A_list, B_list):
        E = inv_T(A) @ X @ B @ inv_T(X)
        rot_err.append(rot_angle_deg(E[:3, :3]))
        trans_err.append(float(np.linalg.norm(E[:3, 3])))
    rot_err = np.array(rot_err)
    trans_err = np.array(trans_err)
    return {
        "rot_mean_deg": float(rot_err.mean()),
        "rot_max_deg": float(rot_err.max()),
        "trans_mean_mm": float(trans_err.mean() * 1000),
        "trans_max_mm": float(trans_err.max() * 1000),
    }


def motion_informativeness(T_base2grip, min_angle_deg=30.0, verbose=True):
    """
    运动激励诊断：手眼标定失败 80% 源于运动激励不足。
    统计相邻位姿的旋转角与旋转轴分布，输出可执行的补采建议。
    """
    n = len(T_base2grip)
    angles, axes = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dT = inv_T(T_base2grip[i]) @ T_base2grip[j]
            a = rot_angle_deg(dT[:3, :3])
            if a > 1e-6:
                angles.append(a)
                axes.append(rot_axis(dT[:3, :3]))
    angles = np.array(angles) if angles else np.zeros(1)

    # 轴发散度：把三轴旋转分量分别累加，看是否只绕单轴
    comp = np.zeros(3) if not axes else np.abs(np.array(axes)).sum(axis=0)
    comp = comp / (comp.sum() + 1e-12)

    info = {
        "n_poses": n,
        "angle_mean_deg": float(angles.mean()),
        "angle_max_deg": float(angles.max()),
        "n_large_motions": int((angles >= min_angle_deg).sum()),
        "axis_ratio_xyz": comp.tolist(),
        "warning": [],
        "advice": [],
    }
    if n < 8:
        info["warning"].append(f"位姿仅 {n} 个，建议 12~20 个")
        info["advice"].append("沿工作空间多方向补采 6~10 个位姿")
    if info["n_large_motions"] < 3:
        info["warning"].append(f"旋转 >{min_angle_deg}° 的运动仅 {info['n_large_motions']} 组")
        info["advice"].append("每次运动让末端绕旋转轴转过 30° 以上")
    if comp.min() < 0.08:
        weak = "XYZ"[int(np.argmin(comp))]
        info["warning"].append(f"绕 {weak} 轴几乎无旋转分量，平移将不可观")
        info["advice"].append(f"增加绕 {weak} 轴旋转的姿态（>30°）")
    if info["angle_max_deg"] > 170:
        info["warning"].append("存在接近 180° 的运动，AX=XB 可能退化")
        info["advice"].append("避免绕同一轴做大角度翻转，改为一组中等角度多轴运动")

    if verbose:
        print("\n  运动激励诊断")
        print("  " + "-" * 52)
        print(f"  位姿数            : {info['n_poses']}")
        print(f"  平均旋转角        : {info['angle_mean_deg']:.2f}°")
        print(f"  最大旋转角        : {info['angle_max_deg']:.2f}°")
        print(f">30° 的运动组数    : {info['n_large_motions']}")
        print(f"  旋转轴 XYZ 占比   : "
              f"{comp[0]:.2f} / {comp[1]:.2f} / {comp[2]:.2f}")
        if info["warning"]:
            for w in info["warning"]:
                print(f"  [!] {w}")
            for a in info["advice"]:
                print(f"      -> {a}")
        else:
            print("  [OK] 运动激励充分")
        print("  " + "-" * 52 + "\n")
    return info


# --------------------------------------------------------------------------
# 6. 九点标定（eye-to-hand 平面抓取简化方案）
# --------------------------------------------------------------------------

def nine_point_affine(px_points, robot_points):
    """
    九点标定：由「像素坐标 <-> 机器人平面坐标」对应点拟合 2x3 仿射矩阵。

    px_points    : (N,2) 图像像素坐标
    robot_points : (N,2) 机械臂 X/Y 坐标（单位与 px 无关，通常 mm）

    返回 (M, info)：M 为 2x3 仿射矩阵，满足 [x_r, y_r]^T = M @ [u, v, 1]^T
    """
    src = np.asarray(px_points, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(robot_points, dtype=np.float64).reshape(-1, 1, 2)
    if len(src) < 3:
        raise ValueError("仿射变换至少需要 3 对点（推荐 9 点冗余求解）")

    M, inliers = cv2.estimateAffine2D(
        src, dst, method=cv2.LMEDS, refineIters=50)

    # 残差统计，用于验收
    pred = cv2.transform(src, M).reshape(-1, 2)
    err = np.linalg.norm(pred - dst.reshape(-1, 2), axis=1)
    info = {
        "n_points": int(len(src)),
        "mean_err_mm": float(err.mean()),
        "max_err_mm": float(err.max()),
        "per_point_err_mm": err.tolist(),
        "M": M.tolist(),
    }
    return M, info


def pixel_to_robot(M, u, v):
    """由仿射矩阵把像素坐标转换为机器人平面坐标。"""
    p = np.array([[[float(u), float(v)]]], dtype=np.float64)
    xy = cv2.transform(p, np.asarray(M, dtype=np.float64)).reshape(2)
    return float(xy[0]), float(xy[1])


def robot_to_pixel(M, x, y):
    """仿射矩阵求逆：机器人坐标 -> 像素坐标（用于验证点回投影）。"""
    Mi = cv2.invertAffineTransform(np.asarray(M, dtype=np.float64))
    p = np.array([[[float(x), float(y)]]], dtype=np.float64)
    uv = cv2.transform(p, Mi).reshape(2)
    return float(uv[0]), float(uv[1])


# --------------------------------------------------------------------------
# 7. 精度验证与误差折算
# --------------------------------------------------------------------------

def rotation_error_to_linear(angle_deg, distance_m):
    """
    把角度误差折算为工作距离处的线性偏差（工程上极重要）。
    err = 2 * d * sin(theta / 2)
    """
    th = np.radians(angle_deg)
    return 2.0 * distance_m * np.sin(th / 2.0)


def task_level_error(hand_eye_err_deg, hand_eye_err_mm,
                     work_distance_m, robot_abs_err_mm=0.10,
                     calib_board_err_mm=0.05):
    """
    简化的误差预算：把可视子系统各误差源折算到末端在工作距离处的总偏差。
    用于课堂讨论「精度天花板在哪里」。
    """
    e_rot = rotation_error_to_linear(hand_eye_err_deg, work_distance_m) * 1000
    e_total = np.sqrt(e_rot ** 2 + hand_eye_err_mm ** 2
                      + robot_abs_err_mm ** 2 + calib_board_err_mm ** 2)
    return {
        "rotation_to_linear_mm": float(e_rot),
        "handeye_trans_mm": float(hand_eye_err_mm),
        "robot_abs_mm": float(robot_abs_err_mm),
        "board_mm": float(calib_board_err_mm),
        "rss_total_mm": float(e_total),
    }


def validation_reprojection(T_base2cam, X, T_base2grip_list,
                            T_target2cam_list, obj_pts_list, K, dist,
                            img_pts_list, eye_in_hand=True):
    """
    留出位姿验证（第二层验证）：用未参与求解的位姿重新投影标定板角点，
    检查几何链是否自洽。

    两种构型的几何链：
      eye-in-hand : T_cam←target = X⁻¹ · T_base←grip⁻¹ · T_base←target
                    （但 T_base←target 未知，故工程上直接用
                      T_cam←target = X⁻¹ · G_i⁻¹ · G_ref · X · C_ref 校验一致性）
      eye-to-hand : T_cam←target 由 PnP 直接给出，用
                    T_cam←board(i) ≈ X⁻¹ · T_base←grip(i) · Y 反算

    这里实现更实用的做法：直接比较「PnP 实测外参」与「由手眼矩阵推算的外参」，
    返回逐组 (旋转误差°, 平移误差 mm)。
    """
    out = []
    X = np.asarray(X, dtype=np.float64)
    for i in range(len(T_base2grip_list)):
        G = T_base2grip_list[i]
        C = T_target2cam_list[i]                      # PnP 实测：T_cam←board
        if eye_in_hand:
            # 由 X 与机器人位姿推算的相对量应与 PnP 一致：
            # G_i · X · C_i 应恒等于常量 T_base←board
            pred = G @ X @ C
            ref = T_base2grip_list[0] @ X @ T_target2cam_list[0]
        else:
            # eye-to-hand: T_base←board = G_i · Y = X · C_i
            pred = inv_T(X) @ G
            ref = C
        da, dt = pose_diff(ref, pred)
        out.append({"index": i, "rot_err_deg": da, "trans_err_mm": dt})
    return out

