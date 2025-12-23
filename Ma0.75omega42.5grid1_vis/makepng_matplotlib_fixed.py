#!/usr/bin/env python3
"""
使用 matplotlib 和 fluidfoam 库可视化 OpenFOAM 压力场数据
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from fluidfoam import readmesh, readscalar
import warnings
warnings.filterwarnings('ignore')

def read_boundary_faces(mesh_dir='constant/polyMesh', boundary_name='aerofoil'):
    """读取指定边界的面和点坐标"""
    # 读取 boundary 文件获取边界信息
    boundary_file = os.path.join(mesh_dir, 'boundary')
    with open(boundary_file, 'r') as f:
        lines = f.readlines()

    # 查找指定边界的起始面和面数
    start_face = None
    n_faces = None
    in_boundary = False

    for i, line in enumerate(lines):
        line = line.strip()
        if line == boundary_name:
            in_boundary = True
        elif in_boundary:
            if 'nFaces' in line:
                n_faces = int(line.split()[1].rstrip(';'))
            elif 'startFace' in line:
                start_face = int(line.split()[1].rstrip(';'))
                break

    if start_face is None or n_faces is None:
        return None, None

    # 读取 faces 文件
    faces_file = os.path.join(mesh_dir, 'faces')
    with open(faces_file, 'r') as f:
        lines = f.readlines()

    # 解析面数据
    faces = []
    in_data = False
    face_count = 0

    for line in lines:
        line = line.strip()
        if line == '(':
            in_data = True
            continue
        elif line == ')':
            break
        elif in_data:
            face_count += 1
            if face_count > start_face and face_count <= start_face + n_faces:
                # 解析面，格式如: 4(0 1 2 3)
                line = line.replace('(', ' ').replace(')', '')
                parts = line.split()
                if len(parts) > 1:
                    try:
                        n_points = int(parts[0])
                        face_points = [int(x) for x in parts[1:n_points+1]]
                        faces.append(face_points)
                    except ValueError:
                        pass

    # 读取点坐标
    points_file = os.path.join(mesh_dir, 'points')
    with open(points_file, 'r') as f:
        lines = f.readlines()

    points = []
    in_data = False

    for line in lines:
        line = line.strip()
        if line == '(':
            in_data = True
            continue
        elif line == ')':
            break
        elif in_data:
            line = line.replace('(', '').replace(')', '')
            try:
                coords = [float(x) for x in line.split()]
                if len(coords) == 3:
                    points.append(coords)
            except ValueError:
                pass

    points = np.array(points)

    # 提取边界点坐标
    boundary_points = []
    for face in faces:
        for pt_idx in face:
            if pt_idx < len(points):
                boundary_points.append(points[pt_idx])

    return np.array(boundary_points), faces

def getTimeStepList(directory='.'):
    """扫描目录中的所有数值时间步目录，按时间顺序排列"""
    listDirs = next(os.walk(directory))[1]
    timeStepList = [float(d) for d in listDirs if d.replace('.', '', 1).replace('-', '', 1).isdigit()]
    timeStepList.sort()
    return timeStepList

# 主程序
if __name__ == '__main__':
    # 获取时间步列表
    timeStepList = getTimeStepList()
    print(f"找到 {len(timeStepList)} 个时间步")
    print(f"时间步范围: {timeStepList[0]} 到 {timeStepList[-1]}")

    # 创建输出目录
    output_dir = 'matplotlib_output'
    os.makedirs(output_dir, exist_ok=True)

    # 读取网格信息（仅需读取一次）
    print("\n正在读取网格信息...")
    try:
        x, y, z = readmesh('.', structured=False)
        print(f"  ✓ 读取了 {len(x)} 个单元中心坐标")
        print(f"  X 范围: {x.min():.3f} 到 {x.max():.3f}")
        print(f"  Y 范围: {y.min():.3f} 到 {y.max():.3f}")
        print(f"  Z 范围: {z.min():.3f} 到 {z.max():.3f}")

        # 检查是否为 2D 问题
        if np.std(z) < 1e-6:
            print("  检测到 2D 问题（z 方向无变化，使用 x-y 平面）")
            plot_x, plot_y = x, y
            xlabel, ylabel = 'X (m)', 'Y (m)'
        elif np.std(y) < 1e-6:
            print("  检测到 2D 问题（y 方向无变化，使用 x-z 平面）")
            plot_x, plot_y = x, z
            xlabel, ylabel = 'X (m)', 'Z (m)'
        elif np.std(x) < 1e-6:
            print("  检测到 2D 问题（x 方向无变化，使用 y-z 平面）")
            plot_x, plot_y = y, z
            xlabel, ylabel = 'Y (m)', 'Z (m)'
        else:
            print("  检测到 3D 问题，使用 x-y 平面投影")
            plot_x, plot_y = x, y
            xlabel, ylabel = 'X (m)', 'Y (m)'
    except Exception as e:
        print(f"  ✗ 读取网格失败: {e}")
        exit(1)

    # 读取翼型边界
    print("\n正在读取翼型边界...")
    try:
        aerofoil_points, aerofoil_faces = read_boundary_faces()
        if aerofoil_points is not None:
            print(f"  ✓ 读取了翼型边界: {len(aerofoil_points)} 个点, {len(aerofoil_faces)} 个面")
            # 提取翼型的 x-z 坐标（因为是 y 方向无变化的 2D 问题）
            aerofoil_x = aerofoil_points[:, 0]
            aerofoil_z = aerofoil_points[:, 2]
        else:
            print("  ✗ 未能读取翼型边界")
            aerofoil_x, aerofoil_z = None, None
    except Exception as e:
        print(f"  ✗ 读取翼型边界失败: {e}")
        aerofoil_x, aerofoil_z = None, None

    # 设置压力范围（与 ParaView 脚本一致）
    p_min = 80000.0
    p_max = 120000.0

    print("\n开始生成图像...")

    # 主循环：为每个时间步生成图像
    for idx, ts in enumerate(timeStepList):
        print(f"\n[{idx+1}/{len(timeStepList)}] 处理时间步: {ts}")

        # 读取压力场数据
        try:
            p_data = readscalar('.', str(ts), 'p')
            print(f"  ✓ 读取压力数据: {len(p_data)} 个单元")
            print(f"  压力范围: {p_data.min():.2f} 到 {p_data.max():.2f} Pa")
        except Exception as e:
            print(f"  ✗ 读取压力数据失败: {e}")
            continue

        # 创建图形
        fig, ax = plt.subplots(figsize=(12.8, 8.0), dpi=100)

        # 使用三角剖分进行可视化
        # 创建 Delaunay 三角剖分
        # 添加微小随机扰动以避免共线点问题
        px_jittered = plot_x + np.random.normal(0, 1e-10, size=plot_x.shape)
        py_jittered = plot_y + np.random.normal(0, 1e-10, size=plot_y.shape)

        try:
            triang = tri.Triangulation(px_jittered, py_jittered)
        except RuntimeError:
            # 如果三角剖分失败，使用散点图
            print(f"  警告: 三角剖分失败，使用散点图")
            scatter = ax.scatter(plot_x, plot_y, c=p_data, cmap='coolwarm', s=5,
                               vmin=p_min, vmax=p_max, edgecolors='none')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'Pressure Field (t = {ts:.6f} s)', fontsize=14, fontweight='bold')
            ax.set_aspect('equal', adjustable='box')

            # 绘制翼型轮廓
            if aerofoil_x is not None and aerofoil_z is not None:
                ax.fill(aerofoil_x, aerofoil_z, color='gray', edgecolor='black',
                       linewidth=1.5, zorder=10, alpha=0.9)

            # 设置视图范围
            ax.set_xlim(-1.5, 2.5)
            ax.set_ylim(-1.5, 1.5)

            cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                              pad=0.1, aspect=30, shrink=0.5)
            cbar.set_label('p (Pa)', fontsize=11)
            cbar.ax.tick_params(labelsize=10)

            filename = os.path.join(output_dir, f'p_deg{str(ts).replace(".", "")}.png')
            plt.tight_layout()
            plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor='white')
            print(f"  ✓ 已保存: {filename}")
            plt.close(fig)
            continue

        # 绘制填充等值线图
        levels = np.linspace(p_min, p_max, 100)
        contourf = ax.tricontourf(triang, p_data, levels=levels, cmap='coolwarm',
                                  vmin=p_min, vmax=p_max, extend='both')

        # 可选：添加等值线
        contour = ax.tricontour(triang, p_data, levels=10, colors='black',
                               linewidths=0.5, alpha=0.3)

        # 设置坐标轴
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Pressure Field (t = {ts:.6f} s)', fontsize=14, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')

        # 绘制翼型轮廓
        if aerofoil_x is not None and aerofoil_z is not None:
            # 绘制翼型为填充的灰色区域
            ax.fill(aerofoil_x, aerofoil_z, color='gray', edgecolor='black',
                   linewidth=1.5, zorder=10, alpha=0.9)

        # 设置视图范围（类似 ParaView 的相机设置）
        # 焦点在 (0.425, -0.05, 0.006)，显示翼型周围区域
        ax.set_xlim(-1.5, 2.5)  # X 范围：翼型大约在 -0.5 到 1.5
        ax.set_ylim(-1.5, 1.5)   # Z 范围：对称分布

        # 添加颜色条（水平方向）
        cbar = plt.colorbar(contourf, ax=ax, orientation='horizontal',
                          pad=0.1, aspect=30, shrink=0.5)
        cbar.set_label('p (Pa)', fontsize=11)
        cbar.ax.tick_params(labelsize=10)

        # 保存图像
        filename = os.path.join(output_dir, f'p_deg{str(ts).replace(".", "")}.png')
        try:
            plt.tight_layout()
            plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor='white')
            print(f"  ✓ 已保存: {filename}")
        except Exception as e:
            print(f"  ✗ 保存失败: {e}")
        finally:
            plt.close(fig)

    print(f"\n" + "="*60)
    print(f"完成！共生成 {len(timeStepList)} 张图像")
    print(f"图像保存在: {output_dir}/ 目录")
    print("="*60)
