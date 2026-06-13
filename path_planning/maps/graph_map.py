"""
graph_map.py — 工业车间拓扑图

布局（3×7主干道 + 3个工位 + 6个depot）：

        0     1     2     3     4     5     6
              M1          M2          M3
              |           |           |
  A:  [0] – [1] – [2] – [3] – [4] – [5] – [6]
       |     |     |     |     |     |     |
  B:  [7] – [8] – [9] –[10] –[11] –[12] –[13]
       |     |     |     |     |     |     |
  C:  [14]– [15]– [16]– [17]– [18]– [19]– [20]
       |                                   |
   D1 D2 D3                           D4 D5 D6
  [21][22][23]                        [24][25][26]

节点编号：
  主干道A : 0–6
  主干道B : 7–13
  主干道C : 14–20
  工位    : 21(M1), 22(M2), 23(M3)   挂在A1/A3/A5上方
  depot   : 24(D1),25(D2),26(D3)     并联接C0(14)
            27(D4),28(D5),29(D6)     并联接C6(20)

共 30 个节点，适配 6 辆 AGV。

visualization.py 使用约定：
  gmap.nodes        → Dict[int, Node]，含 x/y/node_type/label
  gmap.edges        → Dict[(int,int), Edge]，含 from_id/to_id/length
  gmap.get_nodes_by_type(t) → List[Node]
  NodeType          → 节点类型常量（用于可视化着色）
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from math import sqrt


# ── 物理参数常量 ─────────────────────────────────────────────

GRID_SPACING  = 4.0   # 主干道节点间距（米）
SPUR_LENGTH   = 3.0   # 工位支线长度（米）
DEPOT_LENGTH  = 2.0   # depot 到主干道C的边长（米）


# ── 节点类型 ─────────────────────────────────────────────────

class NodeType:
    INTERSECTION = "intersection"   # 主干道交叉节点
    WORKSTATION  = "workstation"    # 机器停靠工位
    DEPOT        = "depot"          # AGV 初始停车位


# ── 数据结构 ─────────────────────────────────────────────────

@dataclass
class Node:
    node_id:   int
    x:         float
    y:         float
    node_type: str = NodeType.INTERSECTION
    label:     str = ""


@dataclass
class Edge:
    from_id: int
    to_id:   int
    length:  float


@dataclass
class TimeInterval:
    t_start: float
    t_end:   float
    agv_id:  int


# ── 预留表 ───────────────────────────────────────────────────

class ReservationTable:
    """
    连续时间窗预留表。

    节点和边分开记录，条目按 t_start 升序维护。
    算法通过 map_query.py 的统一接口读写，不直接操作本类。
    """

    def __init__(self):
        self._nodes: Dict[int, List[TimeInterval]] = {}
        self._edges: Dict[Tuple[int, int], List[TimeInterval]] = {}

    def register_node(self, node_id):
        self._nodes.setdefault(node_id, [])

    def register_edge(self, from_id, to_id):
        self._edges.setdefault((from_id, to_id), [])

    # ── 查询 ────────────────────────────────────────────────

    def is_node_free(self, node_id, t_start, t_end, exclude_agv=None):
        for iv in self._nodes.get(node_id, []):
            if exclude_agv is not None and iv.agv_id == exclude_agv:
                continue
            if t_start < iv.t_end and t_end > iv.t_start:
                return False
        return True

    def is_edge_free(self, from_id, to_id, t_start, t_end, exclude_agv=None):
        for iv in self._edges.get((from_id, to_id), []):
            if exclude_agv is not None and iv.agv_id == exclude_agv:
                continue
            if t_start < iv.t_end and t_end > iv.t_start:
                return False
        return True

    def earliest_free_node(self, node_id, earliest, duration):
        """返回节点可被占用 duration 秒的最早开始时刻（>= earliest）"""
        t = earliest
        for iv in sorted(self._nodes.get(node_id, []), key=lambda x: x.t_start):
            if iv.t_end <= t:
                continue
            if iv.t_start >= t + duration:
                break
            t = iv.t_end
        return t

    def earliest_free_edge(self, from_id, to_id, earliest, duration):
        t = earliest
        for iv in sorted(self._edges.get((from_id, to_id), []), key=lambda x: x.t_start):
            if iv.t_end <= t:
                continue
            if iv.t_start >= t + duration:
                break
            t = iv.t_end
        return t

    # ── 预留 ────────────────────────────────────────────────

    def reserve_node(self, node_id, t_start, t_end, agv_id):
        iv = TimeInterval(t_start, t_end, agv_id)
        lst = self._nodes.setdefault(node_id, [])
        lst.append(iv)
        lst.sort(key=lambda x: x.t_start)

    def reserve_edge(self, from_id, to_id, t_start, t_end, agv_id):
        iv = TimeInterval(t_start, t_end, agv_id)
        lst = self._edges.setdefault((from_id, to_id), [])
        lst.append(iv)
        lst.sort(key=lambda x: x.t_start)

    # ── 释放 ────────────────────────────────────────────────

    def release_agv(self, agv_id):
        """清除某辆 AGV 的全部预留，供重调度使用"""
        for lst in self._nodes.values():
            lst[:] = [iv for iv in lst if iv.agv_id != agv_id]
        for lst in self._edges.values():
            lst[:] = [iv for iv in lst if iv.agv_id != agv_id]

    def reset(self):
        for lst in self._nodes.values():
            lst.clear()
        for lst in self._edges.values():
            lst.clear()

    # ── 只读访问（供 map_query 使用）────────────────────────

    def get_node_intervals(self, node_id):
        return list(self._nodes.get(node_id, []))

    def get_edge_intervals(self, from_id, to_id):
        return list(self._edges.get((from_id, to_id), []))


# ── 地图主类 ─────────────────────────────────────────────────

class GraphMap:
    """
    工业车间拓扑图。

    职责：
        - 存储节点、边、邻接表
        - 持有 ReservationTable
        - 构建预置车间布局

    不包含任何算法逻辑。

    visualization.py 所需接口：
        gmap.nodes                      所有节点字典
        gmap.edges                      所有有向边字典
        gmap.get_nodes_by_type(t)       按类型筛选节点
        gmap.get_neighbors(node_id)     邻居节点列表
        gmap.get_edge_length(f, t)      边长度
        NodeType.INTERSECTION 等常量    用于可视化着色
    """

    def __init__(self):
        self.nodes:     Dict[int, Node]             = {}
        self.edges:     Dict[Tuple[int, int], Edge] = {}
        self.adjacency: Dict[int, List[int]]        = {}
        self.reservation = ReservationTable()

    # ── 内部注册 ─────────────────────────────────────────────

    def _add_node(self, node):
        self.nodes[node.node_id] = node
        self.adjacency.setdefault(node.node_id, [])
        self.reservation.register_node(node.node_id)

    def _add_edge(self, from_id, to_id, length, bidirectional=True):
        self.edges[(from_id, to_id)] = Edge(from_id, to_id, length)
        self.adjacency.setdefault(from_id, [])
        if to_id not in self.adjacency[from_id]:
            self.adjacency[from_id].append(to_id)
        self.reservation.register_edge(from_id, to_id)

        if bidirectional:
            self.edges[(to_id, from_id)] = Edge(to_id, from_id, length)
            self.adjacency.setdefault(to_id, [])
            if from_id not in self.adjacency[to_id]:
                self.adjacency[to_id].append(from_id)
            self.reservation.register_edge(to_id, from_id)

    # ── 对外查询接口 ─────────────────────────────────────────

    def get_node(self, node_id):
        return self.nodes[node_id]

    def get_neighbors(self, node_id):
        return list(self.adjacency.get(node_id, []))

    def get_edge_length(self, from_id, to_id):
        return self.edges[(from_id, to_id)].length

    def has_edge(self, from_id, to_id):
        return (from_id, to_id) in self.edges

    def get_nodes_by_type(self, node_type):
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def euclidean_dist(self, from_id, to_id):
        """两节点欧氏距离，供 A* 启发函数使用"""
        a = self.nodes[from_id]
        b = self.nodes[to_id]
        return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def node_count(self):
        return len(self.nodes)

    def edge_count(self):
        return len(self.edges)

    # ── 预置车间布局 ─────────────────────────────────────────

    def build_workshop_layout(self):
        """
        构建预置工业车间布局，共 30 个节点。

        坐标系：x 向右，y 向上。
            主干道A : y = 2S = 8m
            主干道B : y =  S = 4m
            主干道C : y =  0
            工位    : y = 2S + SPUR_LENGTH
            depot   : y = -DEPOT_LENGTH
        """
        S  = GRID_SPACING
        SL = SPUR_LENGTH
        DL = DEPOT_LENGTH

        y_A    =  2 * S
        y_B    =  S
        y_C    =  0.0
        y_ws   =  y_A + SL
        y_dep  = -DL

        # ── 主干道 A（节点 0–6）──────────────────────────────
        for col in range(7):
            self._add_node(Node(col, col * S, y_A,
                                NodeType.INTERSECTION, f"A{col}"))

        # ── 主干道 B（节点 7–13）─────────────────────────────
        for col in range(7):
            self._add_node(Node(7 + col, col * S, y_B,
                                NodeType.INTERSECTION, f"B{col}"))

        # ── 主干道 C（节点 14–20）────────────────────────────
        for col in range(7):
            self._add_node(Node(14 + col, col * S, y_C,
                                NodeType.INTERSECTION, f"C{col}"))

        # ── 工位（节点 21–23，挂在 A1/A3/A5 上方）───────────
        ws_configs = [
            (21, 1 * S, y_ws, "M1"),   # 挂在 A1(节点1) 上方
            (22, 3 * S, y_ws, "M2"),   # 挂在 A3(节点3) 上方
            (23, 5 * S, y_ws, "M3"),   # 挂在 A5(节点5) 上方
        ]
        for nid, x, y, lbl in ws_configs:
            self._add_node(Node(nid, x, y, NodeType.WORKSTATION, lbl))

        # ── depot 左侧（节点 24–26，并联接 C0=14）───────────
        # 三个 depot 水平排列在 C0 下方，x 间距 DL
        depot_left = [
            (24, 0 * S - DL, y_dep, "D1"),
            (25, 0 * S,      y_dep, "D2"),
            (26, 0 * S + DL, y_dep, "D3"),
        ]
        for nid, x, y, lbl in depot_left:
            self._add_node(Node(nid, x, y, NodeType.DEPOT, lbl))

        # ── depot 右侧（节点 27–29，并联接 C6=20）───────────
        depot_right = [
            (27, 6 * S - DL, y_dep, "D4"),
            (28, 6 * S,      y_dep, "D5"),
            (29, 6 * S + DL, y_dep, "D6"),
        ]
        for nid, x, y, lbl in depot_right:
            self._add_node(Node(nid, x, y, NodeType.DEPOT, lbl))

        # ════════════════════════════════════════════════════
        # 边
        # ════════════════════════════════════════════════════

        # ── 主干道横向边 ──────────────────────────────────
        for col in range(6):
            self._add_edge(col,      col + 1,      S)   # A
            self._add_edge(7 + col,  7 + col + 1,  S)   # B
            self._add_edge(14 + col, 14 + col + 1, S)   # C

        # ── 纵向通道（A↔B↔C，7列全部贯通）──────────────────
        for col in range(7):
            self._add_edge(col,      7 + col,  S)   # A↔B
            self._add_edge(7 + col,  14 + col, S)   # B↔C

        # ── 工位支线（单向：A→工位 双向，AGV 可进可出）──────
        self._add_edge(1, 21, SL)   # A1 ↔ M1
        self._add_edge(3, 22, SL)   # A3 ↔ M2
        self._add_edge(5, 23, SL)   # A5 ↔ M3

        # ── depot 并联（各自独立接入 C0 或 C6）──────────────
        c0 = self.nodes[14]
        for nid in [24, 25, 26]:
            n = self.nodes[nid]
            dist = sqrt((n.x - c0.x) ** 2 + (n.y - c0.y) ** 2)
            self._add_edge(14, nid, dist)

        c6 = self.nodes[20]
        for nid in [27, 28, 29]:
            n = self.nodes[nid]
            dist = sqrt((n.x - c6.x) ** 2 + (n.y - c6.y) ** 2)
            self._add_edge(20, nid, dist)

        print(f"[GraphMap] 构建完成：{self.node_count()} 节点，"
                f"{self.edge_count()} 条有向边")

    # ── 调试工具 ─────────────────────────────────────────────

    def print_summary(self):
        type_count = {}
        for n in self.nodes.values():
            type_count[n.node_type] = type_count.get(n.node_type, 0) + 1
        print("── GraphMap ──────────────────────────────────")
        print(f"  节点总数 : {self.node_count()}")
        print(f"  边总数   : {self.edge_count()} (含反向)")
        for t, c in sorted(type_count.items()):
            print(f"  {t:15s}: {c}")
        print("──────────────────────────────────────────────")

    def print_node(self, node_id):
        n  = self.nodes[node_id]
        nb = self.get_neighbors(node_id)
        print(f"  节点{node_id:2d} [{n.label:6s}] "
                f"({n.x:5.1f}, {n.y:5.1f})  邻居: {nb}")


# ── 快速验证 ─────────────────────────────────────────────────

if __name__ == "__main__":
    gmap = GraphMap()
    gmap.build_workshop_layout()
    gmap.print_summary()

    print("\n── 关键节点验证 ──")
    for nid in [0, 1, 3, 5, 7, 14, 20, 21, 22, 23, 24, 25, 26, 27]:
        gmap.print_node(nid)

    print("\n── 工位列表 ──")
    for n in gmap.get_nodes_by_type(NodeType.WORKSTATION):
        print(f"  {n.node_id}: {n.label}  ({n.x:.1f}, {n.y:.1f})")

    print("\n── depot 列表 ──")
    for n in gmap.get_nodes_by_type(NodeType.DEPOT):
        print(f"  {n.node_id}: {n.label}  ({n.x:.1f}, {n.y:.1f})")

    print("\n── 边长验证 ──")
    print(f"  A0→A1  : {gmap.get_edge_length(0,  1 ):.2f}m")
    print(f"  A1→M1  : {gmap.get_edge_length(1,  21):.2f}m")
    print(f"  A1→B1  : {gmap.get_edge_length(1,  8 ):.2f}m")
    print(f"  C0→D1  : {gmap.get_edge_length(14, 24):.2f}m")
    print(f"  C0→D2  : {gmap.get_edge_length(14, 25):.2f}m")
    print(f"  C0→D3  : {gmap.get_edge_length(14, 26):.2f}m")