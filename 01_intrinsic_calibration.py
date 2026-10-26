#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_intrinsic_calibration.py —— 相机内参标定（张正友法完整实现）
================================================================

对应课程实验：实验 7「单目相机标定、RGBD 相机对齐及 3D 点云可视化」
              实验 8「第 2 步 执行相机内参标定任务」
              实验 9「第 4 步 相机内参标定」

本脚本把张正友法的四步推导链完整走一遍，并在每一步打印中间量，
方便课堂对照「单应矩阵 → 内参闭式解 → 外参分解 → 非线性精化」的推导。

用法
----
# 1) 生成标定板图片（打印用）
python 01_intrinsic_calibration.py --mode make_board --out board_8x6.png

# 2) 采图（USB 相机 / 海康工业相机 / D435，用 -d 指定设备号）
python 01_intrinsic_calibration.py --mode capture --device 0 --outdir images

# 3) 标定 + 质量报告
python 01_intrinsic_calibration.py --mode calibrate --imagedir images \
        --cols 8 --rows 6 --square 25.0

# 4) 畸变校正前后对照（课堂演示）
python 01_intrinsic_calibration.py --mode undistort --imagedir images --image images/img001.jpg

依赖：numpy, opencv-contrib-python (>=4.2)
"""

import argparse
import os
import sys
import glob

import cv2
import numpy as np

import calib_utils as cu


# --------------------------------------------------------------------------
# 模式 1：生成可打印标定板
# --------------------------------------------------------------------------
def mode_make_board(args):
    if args.board_type == "charuco":
        board = cu.make_charuco_board(
            args.squares_x, args.squares_y, args.square, args.marker)
        cu.save_board_image(board, args.out)
    else:
        # 棋盘格：直接绘制，保证是矢量级清晰度
        cols, rows = args.cols, args.rows
        mm = args.square
        dpi = 300
        px_per_mm = dpi / 25.4
        sq_px = int(round(mm * px_per_mm))
        margin = sq_px
        W = (cols + 1) * sq_px + 2 * margin
        H = (rows + 1) * sq_px + 2 * margin
        img = np.full((H, W), 255, np.uint8)
        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:
                    y0 = margin + r * sq_px
                    x0 = margin + c * sq_px
                    img[y0:y0 + sq_px, x0:x0 + sq_px] = 0
        cv2.imwrite(args.out, img)
        actual = 0
        text = (f"棋盘格 {cols}x{rows} 内角点, 方格标称 {mm} mm, DPI={dpi}\n"
                f"打印后请用卡尺实测方格边长并把 --square 改成实测值！\n"
                f"当前图纸尺寸约 {(H/px_per_mm):.1f} x {(W/px_per_mm):.1f} mm")
        print(text)
    print(f"[输出] {os.path.abspath(args.out)}")
    print("=" * 66)
    print("打印与制作规范（决定精度上限，务必执行）：")
    print("  1) 100% 比例打印，不要选「适应页面」")
    print("  2) 贴在玻璃板或铝蜂窝板上保证平整，翘曲会直接污染标定")
    print("  3) 用卡尺实测 10 个方格取平均，作为 --square 输入")
    print("  4) 哑光相纸/背胶纸，避免反光")
    print("=" * 66)


# --------------------------------------------------------------------------
# 模式 2：交互式采图
# --------------------------------------------------------------------------
def mode_capture(args):
    os.makedirs(args.outdir, exist_ok=True)
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"[错误] 无法打开相机设备 {args.device}")
        return 1

    # 关键：锁死分辨率与对焦，标定过程中不得改变（改了就等于换了台相机）
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"[采图] 分辨率 {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("  按 s 保存当前帧  |  按 q 退出")
    print("  采集要求：15~25 张；覆盖视野四角+中心；远近与俯仰/偏航角多样")

    n = 0
    existing = cu.list_images(args.outdir)
    n = len(existing)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        okc, corners = cu.detect_chessboard(gray, (args.cols, args.rows))
        if okc:
            cv2.drawChessboardCorners(vis, (args.cols, args.rows), corners, okc)
            cv2.putText(vis, "DETECTED - press s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        else:
            cv2.putText(vis, "NOT FOUND", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(vis, f"saved: {n}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.imshow("capture (s=save, q=quit)", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            n += 1
            path = os.path.join(args.outdir, f"img{n:04d}.jpg")
            cv2.imwrite(path, frame)
            print(f"  [保存] {path}  (第 {n} 张)")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[采图] 共 {n} 张，保存在 {os.path.abspath(args.outdir)}")
    return 0


# --------------------------------------------------------------------------
# 模式 3：标定 + 质量报告 + 可视化
# --------------------------------------------------------------------------
def mode_calibrate(args):
    board = {
        "type": args.board_type,
        "inner_cols": args.cols,
        "inner_rows": args.rows,
        "square_mm": args.square,
    }
    if args.board_type == "charuco":
        board.update({"squares_x": args.squares_x, "squares_y": args.squares_y,
                      "marker_mm": args.marker})

    print("=" * 66)
    print("  相机内参标定 —— 张正友法四步推导链")
    print("=" * 66)
    print(f"  标定板类型 : {args.board_type}")
    if args.board_type == "charuco":
        print(f"  方格/标记  : {args.square} mm / {args.marker} mm")
    else:
        print(f"  内角点     : {args.cols} x {args.rows}")
        print(f"  方格实测   : {args.square} mm")
    print(f"  图像目录   : {os.path.abspath(args.imagedir)}")
    print("-" * 66)
    print("  Step1 求解单应矩阵 H：板面 Z=0 ⇒ 2D-2D 单应，SVD 超定求解")
    print("  Step2 由单应约束内参：r1ᵀr2=0, |r1|=|r2| ⇒ B=K⁻ᵀK⁻¹，≥3 姿态")
    print("  Step3 分解外参：r3=r1×r2，逐帧恢复板位姿态")
    print("  Step4 LM 非线性精化：最小化全量重投影误差（含畸变系数）")
    print("-" * 66)

    res = cu.calibrate_intrinsics(
        args.imagedir, board=board, out_dir=args.outdir,
        min_valid_views=args.min_views)

    K = np.array(res["K"], dtype=np.float64)
    dist = np.array(res["dist"], dtype=np.float64).ravel()

    # ---- 课堂可视化：角点覆盖热力图 + 重投影误差分布 ----
    if not args.no_plot:
        make_diagnostics_figure(args.imagedir, board, K, dist, args.outdir)

    print(f"\n[结果] 内参矩阵 K =\n{K}\n")
    print(f"[结果] 畸变系数 = {dist}")
    print(f"[保存] 参数：{os.path.join(args.outdir, 'intrinsics.json')}")
    print("\n下一步（重要）：把 fx,fy,cx,cy 与畸变系数填进手眼标定脚本，")
    print("             并牢记：手眼标定不会修复错误的内参。")
    return 0


def make_diagnostics_figure(imagedir, board, K, dist, outdir):
    """
    生成两张课堂用的诊断图（纯 OpenCV 绘制，不依赖 matplotlib）：
      1) 所有标定图角点的覆盖分布 —— 检查是否覆盖视野四角与中心
      2) 逐图平均重投影误差柱状图 —— 检查误差分布是否均匀、有无离群帧
    """
    files = cu.list_images(imagedir)
    is_charuco = board.get("type") == "charuco"
    if is_charuco:
        bd = cu.make_charuco_board(board["squares_x"], board["squares_y"],
                                   board["square_mm"], board["marker_mm"])

    w = h = None
    per_view = []
    first = True
    overlay = None

    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if w is None:
            h, w = gray.shape[:2]
            overlay = np.zeros((h, w), np.uint8)

        if is_charuco:
            ok, corners, ids = cu.detect_charuco(gray, bd)
            objp = cu.charuco_object_points(bd, ids) if ok else None
        else:
            ok, corners = cu.detect_chessboard(gray, (board["inner_cols"],
                                                      board["inner_rows"]))
            objp = cu.build_chessboard_points(
                board["inner_cols"], board["inner_rows"],
                board["square_mm"]) if ok else None
        if not ok:
            continue

        for p in corners.reshape(-1, 2):
            cv2.circle(overlay, (int(p[0]), int(p[1])), 4, 255, -1)

        # 单帧 PnP 求外参 -> 重投影误差
        okp, rvec, tvec = cv2.solvePnP(
            np.asarray(objp, dtype=np.float64),
            np.asarray(corners, dtype=np.float64), K, dist)
        if okp:
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            e = np.linalg.norm(proj.reshape(-1, 2)
                               - corners.reshape(-1, 2), axis=1)
            per_view.append((os.path.basename(f), float(e.mean())))

    if overlay is None:
        return
    overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
    cv2.putText(overlay, "Corner coverage (should fill the whole frame)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(outdir, "diag_coverage.png"), overlay)

    # 误差柱状图
    if per_view:
        names = [p[0] for p in per_view]
        errs = [p[1] for p in per_view]
        W_, H_ = 1000, 420
        canvas = np.full((H_, W_, 3), 255, np.uint8)
        mx = max(errs + [1e-6]) * 1.2
        n = len(errs)
        bw = max(4, int((W_ - 120) / max(n, 1)) - 6)
        for i, e in enumerate(errs):
            x0 = 90 + i * (bw + 6)
            y0 = int(H_ - 60 - (e / mx) * (H_ - 140))
            color = (60, 60, 200) if e < 0.5 else (
                (40, 180, 220) if e < 1.0 else (40, 40, 220))
            cv2.rectangle(canvas, (x0, y0), (x0 + bw, H_ - 60), color, -1)
        cv2.line(canvas, (80, H_ - 60), (W_ - 30, H_ - 60), (0, 0, 0), 2)
        # 0.5px / 1.0px 判据线
        for thr, lbl in [(0.5, "0.5px excellent"), (1.0, "1.0px good")]:
            y = int(H_ - 60 - (thr / mx) * (H_ - 140))
            cv2.line(canvas, (80, y), (W_ - 30, y), (0, 160, 0), 2)
            cv2.putText(canvas, lbl, (W_ - 250, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 0), 2)
        cv2.putText(canvas, "Per-view mean reprojection error (px)",
                    (90, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.putText(canvas, "green line = 0.5px", (90, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1)
        cv2.imwrite(os.path.join(outdir, "diag_reproj_hist.png"), canvas)
    print(f"[诊断图] {os.path.join(outdir, 'diag_coverage.png')}")
    print(f"[诊断图] {os.path.join(outdir, 'diag_reproj_hist.png')}")


# --------------------------------------------------------------------------
# 模式 4：畸变校正对照
# --------------------------------------------------------------------------
def mode_undistort(args):
    json_path = os.path.join(args.outdir, "intrinsics.json")
    import json
    with open(json_path, encoding="utf-8") as fh:
        res = json.load(fh)
    K = np.array(res["K"], dtype=np.float64)
    dist = np.array(res["dist"], dtype=np.float64).ravel()

    img = cv2.imread(args.image)
    if img is None:
        print(f"[错误] 读不到图片 {args.image}")
        return 1
    pair = cu.undistort_compare(img, K, dist,
                                os.path.join(args.outdir, "undistort_compare.jpg"))
    # 叠加直线参考网格，直观展示桶形/枕形畸变的消失
    cv2.putText(pair, "RAW (distorted)", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(pair, "UNDISTORTED", (img.shape[1] + 30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 0), 3)
    cv2.imwrite(os.path.join(args.outdir, "undistort_compare.jpg"), pair)
    print(f"[输出] {os.path.join(args.outdir, 'undistort_compare.jpg')}")
    return 0


# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="相机内参标定（张正友法）教学脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True,
                   choices=["make_board", "capture", "calibrate", "undistort"])
    p.add_argument("--board_type", default="chessboard",
                   choices=["chessboard", "charuco"])
    p.add_argument("--cols", type=int, default=8, help="棋盘格内角点列数")
    p.add_argument("--rows", type=int, default=6, help="棋盘格内角点行数")
    p.add_argument("--square", type=float, default=25.0,
                   help="方格实测边长 mm（务必卡尺实测）")
    p.add_argument("--squares_x", type=int, default=7, help="ChArUco 方格列数")
    p.add_argument("--squares_y", type=int, default=5, help="ChArUco 方格行数")
    p.add_argument("--marker", type=float, default=22.0, help="ChArUco 标记边长 mm")
    p.add_argument("--imagedir", default="images")
    p.add_argument("--image", default=None)
    p.add_argument("--outdir", default="calib_out")
    p.add_argument("--out", default="board.png")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--min_views", type=int, default=8)
    p.add_argument("--no_plot", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dispatch = {
        "make_board": mode_make_board,
        "capture": mode_capture,
        "calibrate": mode_calibrate,
        "undistort": mode_undistort,
    }
    sys.exit(dispatch[args.mode](args) or 0)
