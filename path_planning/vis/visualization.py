"""
visualization.py — 地图 + AGV 可视化

功能：
    1. draw_map         绘制 GraphMap 静态拓扑图
    2. draw_agv         在地图上绘制单台 AGV（矩形车身 + 车头标记）
    3. compute_path_segments  将节点路径转换为"转向/直行"交替的运动片段
    4. animate_agv_path  播放 AGV 沿路径运动的动画（原地转向 + 梯形曲线直行）

依赖接口：
    graph_map.py : GraphMap, NodeType, GRID_SPACING
    agv.py       : AGV, AGVParams
        agv.get_corners()         当前姿态下车身四角坐标
        agv.get_front_point()     车头标记点坐标
        agv.heading_from_vector() 根据位移向量算朝向角度
        agv.turn_time()           转向耗时
        agv.travel_time()         直行耗时（梯形速度曲线）
        agv.pose_during_turn()    转向过程中 t 时刻的 heading
        agv.pose_during_travel()  直行过程中 t 时刻的 (x,y)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from math import sqrt
import sys
import os

# 将 path_planning 根目录加入路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from maps.graph_map import GraphMap, NodeType, GRID_SPACING
from agvs.agv import AGV, AGVParams

# Windows 中文字体设置
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False


# ── 地图可视化配置 ────────────────────────────────────────────

NODE_STYLE = {
    NodeType.INTERSECTION: dict(color="#5B8DB8", zorder=3, s=120),
    NodeType.WORKSTATION:  dict(color="#2EAA6E", zorder=4, s=220),
    NodeType.DEPOT:        dict(color="#E07B3A", zorder=4, s=180),
}

EDGE_STYLE = dict(color="#AAAAAA", linewidth=1.2, zorder=1)

LABEL_STYLE = {
    NodeType.INTERSECTION: dict(fontsize=7,  color="#3A3A3A", ha="center", va="bottom"),
    NodeType.WORKSTATION:  dict(fontsize=8,  color="#1A6644", ha="center", va="bottom"),
    NodeType.DEPOT:        dict(fontsize=8,  color="#9B4E10", ha="center", va="top"),
}

LABEL_OFFSET = {
    NodeType.INTERSECTION: (0,  0.3),
    NodeType.WORKSTATION:  (0,  0.3),
    NodeType.DEPOT:        (0, -0.3),
}


# ──────────────────────────────────────────────────────────────
# 地图静态绘制
# ──────────────────────────────────────────────────────────────

def draw_map(gmap, ax=None, show=True, title="工业车间拓扑图"):
    """
    可视化 GraphMap。

    参数：
        gmap   GraphMap 实例（已调用 build_workshop_layout）
        ax     传入已有 Axes，为 None 时自动创建
        show   True 时保存 png 并 plt.show()；
               动画场景应传 False，由调用方自行 show()
        title  图标题

    返回：
        fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = ax.get_figure()

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, pad=12)
    ax.axis("off")

    # ── 画边 ───────────────────────────────────────────
    drawn_edges = set()
    for (fid, tid), edge in gmap.edges.items():
        key = (min(fid, tid), max(fid, tid))
        if key in drawn_edges:
            continue
        drawn_edges.add(key)
        fn = gmap.nodes[fid]
        tn = gmap.nodes[tid]
        ax.plot([fn.x, tn.x], [fn.y, tn.y], **EDGE_STYLE)

    # ── 画节点 ─────────────────────────────────────────
    for node in gmap.nodes.values():
        style = NODE_STYLE.get(node.node_type,
                               NODE_STYLE[NodeType.INTERSECTION])
        ax.scatter(node.x, node.y,
                   c=style["color"],
                   s=style["s"],
                   zorder=style["zorder"],
                   edgecolors="white",
                   linewidths=0.8)

    # ── 画标签 ─────────────────────────────────────────
    for node in gmap.nodes.values():
        lstyle = LABEL_STYLE.get(node.node_type,
                                 LABEL_STYLE[NodeType.INTERSECTION])
        ox, oy = LABEL_OFFSET.get(node.node_type, (0, 0.3))
        ax.text(node.x + ox, node.y + oy,
                f"{node.label}\n({node.node_id})",
                **lstyle)

    # ── 图例 ───────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=NODE_STYLE[NodeType.INTERSECTION]["color"],
                       label="交叉节点 intersection"),
        mpatches.Patch(color=NODE_STYLE[NodeType.WORKSTATION]["color"],
                       label="工位 workstation"),
        mpatches.Patch(color=NODE_STYLE[NodeType.DEPOT]["color"],
                       label="停车区 depot"),
        Line2D([0], [0], color=EDGE_STYLE["color"],
               linewidth=1.5, label="通道边"),
    ]
    ax.legend(handles=legend_items, loc="upper right",
              fontsize=8, framealpha=0.9)

    # ── 坐标轴范围 ─────────────────────────────────────
    xs = [n.x for n in gmap.nodes.values()]
    ys = [n.y for n in gmap.nodes.values()]
    margin = 2.5
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    # ── 行列参考线 ─────────────────────────────────────
    S = GRID_SPACING
    for col in range(7):
        ax.axvline(col * S, color="#CCCCCC", linewidth=0.5,
                   zorder=0, linestyle="--")

    if show:
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 "workshop_map.png"),
                    dpi=150, bbox_inches="tight")
        print("[visualization] 已保存为 workshop_map.png")
        plt.show()

    return fig, ax


# ──────────────────────────────────────────────────────────────
# AGV 静态绘制
# ──────────────────────────────────────────────────────────────

def draw_agv(ax, agv, zorder=5):
    """
    在 ax 上绘制一台 AGV：矩形车身 + 车头标记点。

    返回 (body_patch, front_marker)，供动画逐帧更新使用：
        body_patch.set_xy(agv.get_corners())
        front_marker.set_data([fx], [fy])
    """
    body_patch = mpatches.Polygon(
        agv.get_corners(), closed=True,
        facecolor=agv.color, edgecolor="black",
        linewidth=1.0, alpha=0.9, zorder=zorder
    )
    ax.add_patch(body_patch)

    fx, fy = agv.get_front_point()
    front_marker, = ax.plot([fx], [fy], marker="o",
                            color="red", markersize=6,
                            zorder=zorder + 1)

    return body_patch, front_marker


# ──────────────────────────────────────────────────────────────
# 路径 → 运动片段
# ──────────────────────────────────────────────────────────────

def compute_path_segments(gmap, agv, path_nodes, initial_heading=None):
    """
    将节点路径 [n0, n1, n2, ...] 转换为运动片段列表。

    方向相同的连续边会被合并成一段连续直行（单个梯形速度曲线），
    只有朝向真正改变时才会减速、原地转弯、再重新加速。
    这样经过中间节点（方向不变）时 AGV 不会停顿。

    返回片段列表，每个片段是字典：
        type='turn'   : duration, start_heading, target_heading, pos
        type='travel' : duration, profile, start_xy, end_xy, heading

    initial_heading 为 None 时，AGV 初始朝向取第一条边的方向
    （即出发时不需要先转向）。
    """
    segments = []
    heading = initial_heading
    pos = (gmap.nodes[path_nodes[0]].x, gmap.nodes[path_nodes[0]].y)

    n = len(path_nodes)
    i = 0
    while i < n - 1:
        a = gmap.nodes[path_nodes[i]]
        b = gmap.nodes[path_nodes[i + 1]]
        dx, dy = b.x - a.x, b.y - a.y
        target_heading = AGV.heading_from_vector(dx, dy)

        if heading is None:
            heading = target_heading

        # ── 转向片段（朝向变化才插入）──────────────────
        turn_duration, diff = agv.turn_time(target_heading, from_heading=heading)
        if abs(diff) > 1e-6:
            segments.append({
                "type": "turn",
                "duration": turn_duration,
                "start_heading": heading,
                "target_heading": target_heading,
                "pos": pos,
            })
            heading = target_heading

        # ── 合并方向相同的连续边为一段直行 ───────────────
        total_distance = sqrt(dx * dx + dy * dy)
        end_pos = (b.x, b.y)
        j = i + 1
        while j < n - 1:
            c = gmap.nodes[path_nodes[j]]
            d = gmap.nodes[path_nodes[j + 1]]
            dx2, dy2 = d.x - c.x, d.y - c.y
            next_heading = AGV.heading_from_vector(dx2, dy2)
            if abs(AGV._normalize_angle(next_heading - heading)) > 1e-6:
                break
            total_distance += sqrt(dx2 * dx2 + dy2 * dy2)
            end_pos = (d.x, d.y)
            j += 1

        profile = agv.travel_time(total_distance)
        segments.append({
            "type": "travel",
            "duration": profile["t_total"],
            "profile": profile,
            "start_xy": pos,
            "end_xy": end_pos,
            "heading": heading,
        })
        pos = end_pos
        i = j

    return segments


def pose_at_time(agv, segments, t):
    """
    根据运动片段列表，返回全局时刻 t 对应的 (x, y, heading)。
    t 超出总时长时，保持在终点姿态。
    """
    elapsed = 0.0
    for seg in segments:
        if t <= elapsed + seg["duration"]:
            local_t = t - elapsed
            if seg["type"] == "turn":
                h = agv.pose_during_turn(
                    local_t, seg["start_heading"],
                    seg["target_heading"], seg["duration"]
                )
                return seg["pos"][0], seg["pos"][1], h
            else:
                x, y = agv.pose_during_travel(
                    local_t, seg["start_xy"], seg["end_xy"], seg["profile"]
                )
                return x, y, seg["heading"]
        elapsed += seg["duration"]

    # t 超出总时长，返回终点姿态
    last = segments[-1]
    if last["type"] == "turn":
        return last["pos"][0], last["pos"][1], last["target_heading"]
    else:
        return last["end_xy"][0], last["end_xy"][1], last["heading"]


# ──────────────────────────────────────────────────────────────
# 动画播放
# ──────────────────────────────────────────────────────────────

def animate_agv_path(gmap, agv, path_nodes, initial_heading=None,
                     fps=20, title="AGV 运动演示"):
    """
    播放 AGV 沿 path_nodes 运动的动画（转向 + 梯形曲线直行交替）。

    返回 fig, ax, anim, segments
    """
    fig, ax = draw_map(gmap, show=False, title=title)

    segments = compute_path_segments(gmap, agv, path_nodes, initial_heading)
    total_time = sum(s["duration"] for s in segments)
    print(f"[animate_agv_path] 共 {len(segments)} 个运动片段，"
          f"总时长 {total_time:.2f}s")
    for idx, seg in enumerate(segments):
        if seg["type"] == "turn":
            print(f"  [{idx}] 转向: {seg['start_heading']:.0f}° → "
                  f"{seg['target_heading']:.0f}°, 耗时 {seg['duration']:.2f}s")
        else:
            print(f"  [{idx}] 直行: {seg['start_xy']} → {seg['end_xy']}, "
                  f"耗时 {seg['duration']:.2f}s")

    # 初始化为第一个片段起点的姿态
    agv.x, agv.y, agv.heading = pose_at_time(agv, segments, 0.0)
    body_patch, front_marker = draw_agv(ax, agv)

    n_frames = int(total_time * fps) + int(fps * 0.5)  # 末尾多留0.5s停留

    def update(frame):
        t = frame / fps
        x, y, h = pose_at_time(agv, segments, t)
        agv.x, agv.y, agv.heading = x, y, h

        body_patch.set_xy(agv.get_corners())
        fx, fy = agv.get_front_point()
        front_marker.set_data([fx], [fy])
        return body_patch, front_marker

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 / fps, blit=True, repeat=False)

    return fig, ax, anim, segments


# ──────────────────────────────────────────────────────────────
# 快速验证
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gmap = GraphMap()
    gmap.build_workshop_layout()

    params = AGVParams()

    # 路径：C0(14) → B0(7) → B1(8)
    #   C0→B0 方向朝北(90°)，B0→B1 方向朝东(0°)
    #   AGV 初始朝向设为北(90°)，到 B0 时原地转向到东(0°)
    # path = [14, 7, 8, 9, 10]
    path = [24, 14, 7]
    agv = AGV(agv_id=1, params=params,
             x=gmap.nodes[14].x, y=gmap.nodes[14].y,
             heading=90.0, current_node=14, color="#3477EB")

    fig, ax, anim, segments = animate_agv_path(
        gmap, agv, path, initial_heading=0.0,
        title="AGV 运动演示：直行 → 原地转向 → 直行"
    )

    plt.show()