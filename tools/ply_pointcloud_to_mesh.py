"""
将纯点云 PLY 转为带 face 的 mesh PLY，使 VS Code 3D Viewer 插件可以正常查看。
方法: 对点云做 Ball Pivoting 或简单的最近邻三角化。
如果点太多会先下采样。

Usage:
    python tools/ply_pointcloud_to_mesh.py <input.ply> [output.ply] [--max-points 100000]
"""
import argparse
import sys
import struct
import numpy as np


def parse_ascii_ply(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    header_end = 0
    vertex_count = 0
    props = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        elif line.startswith('property'):
            props.append(line.split()[-1])
        elif line == 'end_header':
            header_end = i + 1
            break
    data = []
    for i in range(header_end, header_end + vertex_count):
        vals = lines[i].strip().split()
        data.append([float(v) for v in vals])
    data = np.array(data)
    verts = data[:, :3]
    colors = data[:, 3:6].astype(np.uint8) if data.shape[1] >= 6 else None
    return verts, colors


def parse_binary_ply(path):
    with open(path, 'rb') as f:
        header = b''
        while True:
            line = f.readline()
            header += line
            if b'end_header' in line:
                break
        header_str = header.decode('ascii')
        lines = header_str.strip().split('\n')
        vertex_count = 0
        props = []
        fmt = 'little'
        for line in lines:
            line = line.strip()
            if line.startswith('format'):
                if 'big' in line:
                    fmt = 'big'
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                props.append((parts[1], parts[2]))
        # build struct format
        type_map = {
            'float': 'f', 'double': 'd',
            'uchar': 'B', 'uint8': 'B',
            'char': 'b', 'int8': 'b',
            'short': 'h', 'ushort': 'H',
            'int': 'i', 'uint': 'I',
            'int32': 'i', 'uint32': 'I',
        }
        endian = '<' if fmt == 'little' else '>'
        struct_fmt = endian + ''.join(type_map.get(p[0], 'f') for p in props)
        row_size = struct.calcsize(struct_fmt)
        raw = f.read(row_size * vertex_count)
    verts = np.zeros((vertex_count, 3), dtype=np.float32)
    colors = None
    # find xyz and rgb indices
    xyz_idx = []
    rgb_idx = []
    for i, (ptype, pname) in enumerate(props):
        if pname in ('x', 'y', 'z'):
            xyz_idx.append(i)
        if pname in ('red', 'green', 'blue'):
            rgb_idx.append(i)
    if rgb_idx:
        colors = np.zeros((vertex_count, 3), dtype=np.uint8)
    for vi in range(vertex_count):
        vals = struct.unpack_from(struct_fmt, raw, vi * row_size)
        for j, idx in enumerate(xyz_idx):
            verts[vi, j] = vals[idx]
        if rgb_idx:
            for j, idx in enumerate(rgb_idx):
                colors[vi, j] = int(vals[idx])
    return verts, colors


def read_ply(path):
    with open(path, 'rb') as f:
        first_lines = f.read(256).decode('ascii', errors='ignore')
    if 'format ascii' in first_lines:
        return parse_ascii_ply(path)
    else:
        return parse_binary_ply(path)


def write_mesh_ply(path, verts, colors, faces):
    """Write binary little-endian PLY with faces."""
    with open(path, 'wb') as f:
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {len(verts)}\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        if colors is not None:
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += f"element face {len(faces)}\n"
        header += "property list uchar int vertex_indices\n"
        header += "end_header\n"
        f.write(header.encode('ascii'))
        for i in range(len(verts)):
            f.write(struct.pack('<fff', *verts[i]))
            if colors is not None:
                f.write(struct.pack('<BBB', *colors[i]))
        for face in faces:
            f.write(struct.pack('<B', 3))
            f.write(struct.pack('<iii', *face))


def pointcloud_to_mesh_delaunay(verts, colors, max_points=100000):
    """Use scipy Delaunay to create a surface mesh from point cloud."""
    from scipy.spatial import Delaunay

    if len(verts) > max_points:
        idx = np.random.choice(len(verts), max_points, replace=False)
        idx.sort()
        verts = verts[idx]
        if colors is not None:
            colors = colors[idx]
        print(f"  下采样到 {max_points} 个点")

    print(f"  正在计算 Delaunay 三角化 ({len(verts)} 点)...")
    tri = Delaunay(verts)
    # Extract surface faces from tetrahedra
    tets = tri.simplices  # (N, 4)
    # Each tet has 4 faces
    face_set = set()
    for tet in tets:
        for combo in [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]:
            face = tuple(sorted([tet[combo[0]], tet[combo[1]], tet[combo[2]]]))
            if face in face_set:
                face_set.discard(face)  # interior face
            else:
                face_set.add(face)
    faces = np.array(list(face_set), dtype=np.int32)
    print(f"  生成 {len(faces)} 个三角面")
    return verts, colors, faces


def main():
    parser = argparse.ArgumentParser(description='Convert point cloud PLY to mesh PLY')
    parser.add_argument('input', help='Input PLY file (point cloud)')
    parser.add_argument('output', nargs='?', default=None, help='Output PLY file (mesh)')
    parser.add_argument('--max-points', type=int, default=100000, help='Max points before downsampling')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace('.ply', '_mesh.ply')

    print(f"读取: {args.input}")
    verts, colors = read_ply(args.input)
    print(f"  顶点数: {len(verts)}, 有颜色: {colors is not None}")

    verts, colors, faces = pointcloud_to_mesh_delaunay(verts, colors, args.max_points)

    print(f"写入: {args.output}")
    write_mesh_ply(args.output, verts, colors, faces)
    print("完成!")


if __name__ == '__main__':
    main()
