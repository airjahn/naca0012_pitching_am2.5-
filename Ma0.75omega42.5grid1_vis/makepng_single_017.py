#!/usr/bin/env python3
"""
使用 matplotlib 和 fluidfoam 库可视化 OpenFOAM 压力场数据 (单个时间步 t≈0.17)
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
    boundary_file = os.path.join(mesh_dir, 'boundary')
    with open(boundary_file, 'r') as f:
        lines = f.readlines()

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

    faces_file = os.path.join(mesh_dir, 'faces')
    with open(faces_file, 'r') as f:
        lines = f.readlines()

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
                line = line.replace('(', ' ').replace(')', '')
                parts = line.split()
                if len(parts) > 1:
                    try:
                        n_points = int(parts[0])
                        face_points = [int(x) for x in parts[1:n_points+1]]
                        faces.append(face_points)
                    except ValueError:
                        pass

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

    boundary_points = []
    for face in faces:
        for pt_idx in face:
            if pt_idx < len(points):
                boundary_points.append(points[pt_idx])

    return np.array(boundary_points), faces

# 主程序
if __name__ == '__main__':
    # 输出目录
    output_dir = '/home/air/projects/ppnn-main/gnn_unstructured_work/visualization_output'
    os.makedirs(output_dir, exist_ok=True)

    # 时间步
    ts = '0.170001'

    # 读取网格信息
    print("正在读取网格信息...")
    try:
        x, y, z = readmesh('.', structured=False)
        print(f"  读取了 {len(x)} 个单元中心坐标")
        print(f"  X 范围: {x.min():.3f} 到 {x.max():.3f}")
        print(f"  Y 范围: {y.min():.3f} 到 {y.max():.3f}")
        print(f"  Z 范围: {z.min():.3f} 到 {z.max():.3f}")

        # 检查是否为 2D 问题
        if np.std(z) < 1e-6:
            print("  2D 问题（x-y 平面）")
            plot_x, plot_y = x, y
            xlabel, ylabel = 'X (m)', 'Y (m)'
        elif np.std(y) < 1e-6:
            print("  2D 问题（x-z 平面）")
            plot_x, plot_y = x, z
            xlabel, ylabel = 'X (m)', 'Z (m)'
        else:
            plot_x, plot_y = x, y
            xlabel, ylabel = 'X (m)', 'Y (m)'
    except Exception as e:
        print(f"  读取网格失败: {e}")
        exit(1)

    # 读取翼型边界
    print("\n正在读取翼型边界...")
    try:
        aerofoil_points, aerofoil_faces = read_boundary_faces()
        if aerofoil_points is not None:
            print(f"  读取了翼型边界: {len(aerofoil_points)} 个点, {len(aerofoil_faces)} 个面")
            aerofoil_x = aerofoil_points[:, 0]
            aerofoil_z = aerofoil_points[:, 2]
        else:
            aerofoil_x, aerofoil_z = None, None
    except Exception as e:
        print(f"  读取翼型边界失败: {e}")
        aerofoil_x, aerofoil_z = None, None

    # 设置压力范围
    p_min = 80000.0
    p_max = 120000.0

    print(f"\n处理时间步: {ts}")

    # 读取压力场数据
    try:
        p_data = readscalar('.', ts, 'p')
        print(f"  压力范围: {p_data.min():.2f} 到 {p_data.max():.2f} Pa")
    except Exception as e:
        print(f"  读取压力数据失败: {e}")
        exit(1)

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)

    # 三角剖分
    px_jittered = plot_x + np.random.normal(0, 1e-10, size=plot_x.shape)
    py_jittered = plot_y + np.random.normal(0, 1e-10, size=plot_y.shape)
    triang = tri.Triangulation(px_jittered, py_jittered)

    # 绘制填充等值线图
    levels = np.linspace(p_min, p_max, 100)
    contourf = ax.tricontourf(triang, p_data, levels=levels, cmap='coolwarm',
                              vmin=p_min, vmax=p_max, extend='both')
    ax.tricontour(triang, p_data, levels=10, colors='black', linewidths=0.5, alpha=0.3)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Pressure Field - Ma0.75omega42.5grid1_vis (t = {ts} s)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    # 绘制翼型轮廓
    if aerofoil_x is not None and aerofoil_z is not None:
        ax.fill(aerofoil_x, aerofoil_z, color='gray', edgecolor='black',
               linewidth=1.5, zorder=10, alpha=0.9)

    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.5, 1.5)

    cbar = plt.colorbar(contourf, ax=ax, orientation='horizontal',
                      pad=0.1, aspect=30, shrink=0.5)
    cbar.set_label('p (Pa)', fontsize=11)

    # 保存图像
    filename = os.path.join(output_dir, 'p_fixed_0_17.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"  已保存: {filename}")
    plt.close(fig)

    print("\n完成！")
