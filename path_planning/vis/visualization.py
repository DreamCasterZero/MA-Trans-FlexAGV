"""
visualization.py — 地图静态可视化

当前阶段：只可视化 GraphMap，不涉及 AGV 运动。
后续扩展：传入 AGV 列表和路径，叠加动画。

依赖接口（来自 graph_map.py）：
    gmap.nodes                   Dict[int, Node]
    gmap.edges                   Dict[(int,int), Edge]
    gmap.get_nodes_by_type(t)    List[Node]
    NodeType.INTERSECTION / WORKSTATION / DEPOT
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import sys
import os

# 将 path_planning 根目录加入路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from maps.graph_map import GraphMap, NodeType, GRID_SPACING

# Windows 中文字体设置
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# ── 可视化配置 ────────────────────────────────────────────────

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

# depot 标签偏移向下，工位标签偏移向上，交叉节点偏移向上
LABEL_OFFSET = {
    NodeType.INTERSECTION: (0,  0.3),
    NodeType.WORKSTATION:  (0,  0.3),
    NodeType.DEPOT:        (0, -0.3),
}


def draw_map(gmap, ax=None, show=True, title="工业车间拓扑图"):
    """
    可视化 GraphMap。

    参数：
        gmap   GraphMap 实例（已调用 build_workshop_layout）
        ax     传入已有 Axes，为 None 时自动创建
        show   是否调用 plt.show()
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

    # ── 画边（先画，节点覆盖在上面）────────────────────────
    drawn_edges = set()
    for (fid, tid), edge in gmap.edges.items():
        key = (min(fid, tid), max(fid, tid))
        if key in drawn_edges:
            continue
        drawn_edges.add(key)
        fn = gmap.nodes[fid]
        tn = gmap.nodes[tid]
        ax.plot([fn.x, tn.x], [fn.y, tn.y], **EDGE_STYLE)

    # ── 画节点 ───────────────────────────────────────────
    for node in gmap.nodes.values():
        style = NODE_STYLE.get(node.node_type,
                               NODE_STYLE[NodeType.INTERSECTION])
        ax.scatter(node.x, node.y,
                   c=style["color"],
                   s=style["s"],
                   zorder=style["zorder"],
                   edgecolors="white",
                   linewidths=0.8)

    # ── 画标签 ───────────────────────────────────────────
    for node in gmap.nodes.values():
        lstyle = LABEL_STYLE.get(node.node_type,
                                 LABEL_STYLE[NodeType.INTERSECTION])
        ox, oy = LABEL_OFFSET.get(node.node_type, (0, 0.3))
        ax.text(node.x + ox, node.y + oy,
                f"{node.label}\n({node.node_id})",
                **lstyle)

    # ── 图例 ─────────────────────────────────────────────
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

    # ── 坐标轴范围留边距 ─────────────────────────────────
    xs = [n.x for n in gmap.nodes.values()]
    ys = [n.y for n in gmap.nodes.values()]
    margin = 2.5
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    # ── 行列参考线（虚线，辅助对齐）────────────────────────
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


if __name__ == "__main__":
    gmap = GraphMap()
    gmap.build_workshop_layout()
    draw_map(gmap)