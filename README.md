# 相机标定与手眼标定 · 实践代码包

配套教案《相机标定与手眼标定》的完整可运行代码。

## 环境要求

| 项目 | 本课程实验平台（推荐） | 已验证的较新环境 |
|---|---|---|
| 操作系统 | Ubuntu 16.04 LTS | Windows / Linux 均可 |
| Python | 3.7 | 3.13 |
| OpenCV | 4.2.0.34（opencv-contrib-python） | 4.12.0.88 ✓ 已实测 |
| NumPy | 1.21.6 | 2.2.6 ✓ 已实测 |

```bash
# 本课程工控机（已预装，通常无需再装）
pip install opencv-contrib-python numpy
```

> **兼容性说明**：本代码库同时兼容 OpenCV 4.2 与 4.7+ 两套 aruco API。
> 4.7 起 `estimatePoseSingleMarkers` 被移除、`CharucoBoard` 构造方式改变，
> 代码中已统一封装，学生不必关心版本差异。

## 文件清单

| 文件 | 用途 | 对应实验 |
|---|---|---|
| `calib_utils.py` | 核心工具库：位姿运算、标定板、角点检测、内参标定、手眼求解、误差折算 | 全部 |
| `handeye_capture.py` | 手眼标定数据采集与落盘（含采集规范强制检查） | 实验 8 / 9 |
| `test_handeye_synthetic.py` | **坐标系约定自检**（合成真值验证，强烈建议先跑） | 实验 8 |
| `test_method_compare.py` | 五种 OpenCV 算法蒙特卡洛对比 + 计时 | 原理教学 |
| `01_intrinsic_calibration.py` | 相机内参标定 + 质量报告 + 诊断图 | 实验 7 / 8 / 9 |
| `02_handeye_eye_in_hand.py` | 眼在手上手眼标定 | 实验 8 |
| `03_handeye_eye_to_hand.py` | 眼在手外手眼标定 + 像素→基座坐标解算 | 实验 9 |
| `04_nine_point_affine.py` | 九点标定法（工业平面抓取方案） | 实验 9 |
| `05_aruco_pose_follow.py` | ArUco 6D 位姿估计 + 机械臂随动 | 实验 8 |
| `06_validate_accuracy.py` | 三层精度验证 + 误差预算计算器 | 全部 |

## 快速开始（15 分钟跑通全流程）

```bash
# 第 0 步（重要）：先验证坐标系约定，建立「什么是对的」的直觉
python test_handeye_synthetic.py
#   → 期望看到：Eye-in-Hand 误差 0.000000°、Eye-to-Hand 误差 0.000000°
#     若约定写错，误差会显示 180°/380+ mm

# 第 1 步：生成并打印标定板
python 01_intrinsic_calibration.py --mode make_board --out board_8x6.png
#   → 100% 比例打印 + 贴平板 + 卡尺实测方格边长

# 第 2 步：采图（相机对准标定板，多姿态，按 s 保存）
python 01_intrinsic_calibration.py --mode capture --device 0 --outdir images

# 第 3 步：内参标定（把 --square 换成实测值）
python 01_intrinsic_calibration.py --mode calibrate --imagedir images \
       --cols 8 --rows 6 --square 24.92
#   → 看质量报告：重投影误差 < 0.5 px 为优秀
#   → 看 calib_out/diag_coverage.png（角点是否覆盖全视野）
#   → 看 calib_out/diag_reproj_hist.png（误差是否均匀、有无离群帧）

# 第 4 步：手眼标定（眼在手上）
python 02_handeye_eye_in_hand.py --mode capture --camera 0 \
       --intrinsics calib_out/intrinsics.json --dataset dataset_eih
python 02_handeye_eye_in_hand.py --mode solve --dataset dataset_eih

# 第 5 步：分层验证
python 06_validate_accuracy.py --mode validate \
       --dataset dataset_eih --handeye dataset_eih/handeye_result.json
python 06_validate_accuracy.py --mode probe \
       --dataset dataset_eih --handeye dataset_eih/handeye_result.json

# 无硬件时也能跑的三组演示
python 02_handeye_eye_in_hand.py --mode selftest   # 位姿数量的影响
python 04_nine_point_affine.py  --mode demo        # 为什么用 9 点而不是 3 点
python 06_validate_accuracy.py  --mode budget --rot_deg 0.3 --trans_mm 1.5 --distance 0.5
```

## 已实测数据（可直接用于课堂）

### 1. 手眼标定：坐标系约定错了会怎样

| 构型 / 约定 | 旋转误差 | 平移误差 |
|---|---|---|
| Eye-in-Hand，正确约定 | 0.000000° | 0.000000 mm |
| Eye-to-Hand，正确约定 | 0.000000° | 0.000000 mm |
| Eye-to-Hand，**忘记对机器人位姿取逆** | 180.000000° | 383.94 mm |

### 2. 手眼标定：运动激励的影响（12 个位姿，噪声 0.3°/1 mm）

| 运动方式 | 旋转误差 | 平移误差 |
|---|---|---|
| 只绕 Z 轴旋转 | 174.559° | 2753.54 mm |
| 多轴大角度旋转 | 0.367° | 2.02 mm |

**结论：数据质量 > 算法选择。单轴运动会让平移分量完全不可观。**

### 3. 手眼标定：位姿数量的影响（噪声 0.3°/1 mm，30 次重复）

| 位姿数 | 3 | 5 | 8 | 12 | 20 | 30 |
|---|---|---|---|---|---|---|
| 旋转误差(°) | 1.0149 | 0.4760 | 0.3065 | 0.2848 | 0.1838 | 0.1504 |
| 平移误差(mm) | 10.64 | 4.19 | 2.67 | 2.09 | 1.29 | 1.00 |

**结论：≥8 个位姿后进入误差平台期，12~20 是性价比最优区间。**

### 4. 九点标定：为什么是 9 点而不是 3 点（噪声 0.5 px）

| 点数 | 3 | 4 | 5 | 9 |
|---|---|---|---|---|
| 平均误差(mm) | 34.05 | 3.98 | 1.68 | **0.064** |
| 最大误差(mm) | 2764.44 | 293.91 | 288.43 | **0.377** |

### 5. 误差预算：0.5 m 工作距离下的误差构成

| 误差源 | 折算值 (mm) | 占总方差 |
|---|---|---|
| 旋转误差 0.30° × 0.50 m | 2.618 | **75.2%** |
| 手眼平移误差 | 1.500 | 24.7% |
| 机器人绝对精度 | 0.100 | 0.1% |
| 标定板制造误差 | 0.050 | 0.0% |
| **合成（RSS）** | **3.019** | 100% |

**结论：工作距离 0.5 m 时，角度误差完全主导总精度。优化要优先攻旋转精度，
而不是换更高像素的相机。**

### 6. 五算法对比（位姿噪声 0.3°/1 mm，每组 40 次重复，平移误差 mm）

| 位姿数 | PARK | HORAUD | ANDREFF | DANIILIDIS | TSAI |
|---|---|---|---|---|---|
| 3 | 8.94 | **27.12** | 13.33 | 8.39 | **5.30** |
| 5 | 2.53 | 2.43 | 2.64 | 2.72 | 2.25 |
| 8 | 1.73 | 1.81 | 1.76 | 1.54 | 1.81 |
| 12 | 1.39 | 1.56 | 1.49 | 1.21 | 1.23 |
| 20 | 1.05 | 0.98 | 1.01 | 0.90 | 1.10 |

**注意 3 个位姿处的反直觉现象：此时 TSAI 反而最稳（5.30 mm），HORAUD 最差（27.12 mm）。
位姿太少时误差主要由「数据不足」决定，算法优劣无从体现。**

旋转噪声扫描（15 位姿，平移噪声固定 1 mm，表中为平移误差 mm）：

| 旋转噪声 | PARK | HORAUD | ANDREFF | DANIILIDIS | TSAI |
|---|---|---|---|---|---|
| 0.0° | 0.81 | 0.82 | 0.89 | 0.88 | 0.92 |
| 0.5° | 1.57 | 1.61 | 1.45 | 1.51 | 1.64 |
| 1.0° | 2.62 | 2.71 | 3.41 | 2.96 | 2.81 |
| 2.0° | 5.70 | 5.63 | **9.40** | **5.14** | 5.39 |

计时（15 位姿）：TSAI 1.34 ms、HORAUD 1.80 ms、DANIILIDIS 2.16 ms、
PARK 2.42 ms、ANDREFF 3.04 ms —— 均在毫秒级，离线标定中不是选型依据。

**选型建议：默认 PARK/HORAUD 作初解，DANIILIDIS 作交叉验证，最后补非线性精化。**

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `Not enough informative motions` | 旋转量不足 / 只绕单轴 | 补采 >30° 的三轴旋转，增至 15+ 位姿 |
| 内参重投影 > 2 px | 板不平、格子尺寸写错、离群帧 | 检查 `diag_reproj_hist.png`，剔除坏帧重标 |
| 手眼结果量级离谱（米级） | 求逆位置搞反 / mm 与 m 混用 | 跑 `test_handeye_synthetic.py` 对照约定 |
| 平移误差大但旋转正常 | 运动仅绕固定轴，平移不可观 | 增加多轴运动，检查采集是否静态 |
| `charuco` 相关报错 | OpenCV 版本 API 差异 | 本库已适配 4.2 / 4.7+，确认用 `calib_utils` 的封装 |
| 标定板原点跳动 | 棋盘格行列同奇偶导致 180° 歧义 | 改用奇×偶行列或 ChArUco 板 |

## 注意事项

1. **先标内参，再标手眼。** 手眼标定不会修复错误的内参。
2. **全程锁死焦距、分辨率、对焦。** 任何一项改变都等于换了一台相机，必须重标。
3. **静态采集。** 机械臂完全停稳后再曝光；动态采集因时间戳不同步可引入 10+ mm 误差。
4. **标定完成后不要移动相机。** 移动即作废，需重新标定。
5. **所有实测尺寸用卡尺，别信打印标称值。**
