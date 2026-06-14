"""
agv.py — AGV 运动学模型

AGV 建模为长方体（俯视图为矩形），两轮差速底盘，转弯为原地旋转。

核心概念：
    - 位置 (x, y)：AGV 几何中心的物理坐标（米）
    - heading：车头朝向角度（度），0°=朝+x方向(东)，90°=朝+y方向(北)，
               逆时针为正，与数学坐标系一致
    - 车头/车尾：沿 heading 方向，车头在前，车尾在后

运动学：
    - 直线行驶：梯形速度曲线（加速→匀速→减速，距离不够时退化为三角形曲线）
    - 转弯：原地旋转，恒定角速度，不前进

本文件不依赖 graph_map，只提供：
    - 运动时间计算（供 ctw_planner 计算时间窗）
    - 几何外形计算（供 visualization 画矩形车身）

不包含任何路径规划或地图相关逻辑。
"""

from dataclasses import dataclass, field
from math import sqrt, cos, sin, radians, atan2, degrees


# ──────────────────────────────────────────────────────────────
# AGV 参数（同构 AGV，所有车辆共用一套参数）
# ──────────────────────────────────────────────────────────────

@dataclass
class AGVParams:
    length:     float = 1.2     # 车长（米），沿 heading 方向
    width:      float = 0.8     # 车宽（米），垂直 heading 方向
    max_speed:  float = 1.5     # 最大直线速度（米/秒）
    max_accel:  float = 0.5     # 加速度（米/秒²）
    max_decel:  float = 0.5     # 减速度（米/秒²）
    turn_speed: float = 90.0    # 原地转向角速度（度/秒）
    load_time:  float = 5.0     # 工位装卸停留时间（秒）


# ──────────────────────────────────────────────────────────────
# AGV 实体
# ──────────────────────────────────────────────────────────────

class AGV:
    """
    单台 AGV 的状态 + 运动学计算。

    使用示例：
        params = AGVParams()
        agv = AGV(agv_id=1, params=params, x=0.0, y=0.0,
                  heading=0.0, current_node=14)

        # 计算走4米需要多久
        profile = agv.travel_time(4.0)
        print(profile["t_total"])

        # 计算从朝东转到朝北需要多久
        t_turn, diff = agv.turn_time(90.0)

        # 可视化用：获取车身四个角点和车头标记点
        corners = agv.get_corners()
        front   = agv.get_front_point()
    """

    def __init__(self, agv_id, params=None, x=0.0, y=0.0,
                 heading=0.0, current_node=None, color="#3477EB"):
        self.agv_id = agv_id
        self.params = params if params is not None else AGVParams()
        self.x = x
        self.y = y
        self.heading = heading % 360       # 归一化到 [0, 360)
        self.current_node = current_node
        self.color = color

    # ════════════════════════════════════════════════════════
    # 直线运动：梯形速度曲线
    # ════════════════════════════════════════════════════════

    def travel_time(self, distance):
        """
        计算匀加速→匀速→匀减速通过 distance 所需时间。

        假设起止速度均为 0（每段边都从静止出发、到达后停下，
        和"原地转弯"的假设一致）。

        若距离不足以达到最大速度，退化为三角形速度曲线
        （只有加速段和减速段，没有匀速段）。

        返回字典：
            t_acc    加速段时间
            t_const  匀速段时间
            t_dec    减速段时间
            t_total  总时间
            v_peak   实际达到的峰值速度
        """
        p = self.params
        v_max, a_acc, a_dec = p.max_speed, p.max_accel, p.max_decel

        # 加速到 v_max 所需距离 + 减速到 0 所需距离
        d_acc_full = v_max ** 2 / (2 * a_acc)
        d_dec_full = v_max ** 2 / (2 * a_dec)

        if d_acc_full + d_dec_full <= distance:
            # ── 梯形曲线：能达到最大速度 ──
            t_acc   = v_max / a_acc
            t_dec   = v_max / a_dec
            d_const = distance - d_acc_full - d_dec_full
            t_const = d_const / v_max
            v_peak  = v_max
        else:
            # ── 三角形曲线：距离太短，达不到 v_max ──
            # 加减速距离之和 = distance 时求峰值速度 v_peak
            # d = v_peak^2/(2*a_acc) + v_peak^2/(2*a_dec)
            v_peak = sqrt(2 * distance * a_acc * a_dec / (a_acc + a_dec))
            t_acc   = v_peak / a_acc
            t_dec   = v_peak / a_dec
            t_const = 0.0

        return {
            "t_acc":   t_acc,
            "t_const": t_const,
            "t_dec":   t_dec,
            "t_total": t_acc + t_const + t_dec,
            "v_peak":  v_peak,
        }

    def distance_at_time(self, t, profile):
        """
        根据梯形/三角形速度曲线 profile，返回 t 时刻已行驶的距离。

        对速度曲线分段积分：
            加速段: d = 0.5 * a_acc * t²
            匀速段: d = d_acc + v_peak * (t - t_acc)
            减速段: d = d_acc + d_const + (v_peak*td - 0.5*a_dec*td²)
        """
        p = self.params
        a_acc, a_dec = p.max_accel, p.max_decel
        t_acc, t_const, t_dec = profile["t_acc"], profile["t_const"], profile["t_dec"]
        v_peak, t_total = profile["v_peak"], profile["t_total"]

        if t <= 0:
            return 0.0

        d_acc   = 0.5 * a_acc * t_acc ** 2
        d_const = v_peak * t_const

        if t >= t_total:
            d_dec = v_peak * t_dec - 0.5 * a_dec * t_dec ** 2
            return d_acc + d_const + d_dec

        if t < t_acc:
            return 0.5 * a_acc * t ** 2

        if t < t_acc + t_const:
            return d_acc + v_peak * (t - t_acc)

        # 减速段
        td = t - t_acc - t_const
        d_dec = v_peak * td - 0.5 * a_dec * td ** 2
        return d_acc + d_const + d_dec

    def pose_during_travel(self, t, start_xy, end_xy, profile):
        """
        返回 t 时刻沿直线从 start_xy 行驶到 end_xy 的 (x, y)。

        基于 distance_at_time 算出已行驶距离，再按比例
        在直线上线性插值出当前坐标。t 超出 t_total 时返回终点。
        """
        sx, sy = start_xy
        ex, ey = end_xy
        total_dist = sqrt((ex - sx) ** 2 + (ey - sy) ** 2)

        if total_dist == 0:
            return (sx, sy)

        d = self.distance_at_time(t, profile)
        ratio = min(d / total_dist, 1.0)

        x = sx + ratio * (ex - sx)
        y = sy + ratio * (ey - sy)
        return (x, y)

    # ════════════════════════════════════════════════════════
    # 原地转弯
    # ════════════════════════════════════════════════════════

    def turn_time(self, target_heading, from_heading=None):
        """
        计算从 from_heading 原地转向到 target_heading 所需时间
        （恒定角速度）。

        from_heading 为 None 时使用 self.heading（AGV当前姿态）。
        在路径规划场景中应显式传入“当前追踪的朝向”，避免依赖
        AGV对象的状态（self.heading 可能与路径推进的朝向不同步）。

        返回 (turn_duration, angle_diff)：
            turn_duration  转向耗时（秒）
            angle_diff     需要转过的角度，范围 [-180, 180]，
                           正值表示逆时针，负值表示顺时针
        """
        base = self.heading if from_heading is None else from_heading
        diff = self._normalize_angle(target_heading - base)
        duration = abs(diff) / self.params.turn_speed
        return duration, diff

    def pose_during_turn(self, t, start_heading, target_heading, turn_duration):
        """
        返回 t 时刻原地转弯到一半的朝向角度（恒定角速度插值）。

        t <= 0 时返回 start_heading，t >= turn_duration 时返回 target_heading。
        转弯期间位置 (x,y) 不变，只有 heading 变化。
        """
        diff = self._normalize_angle(target_heading - start_heading)

        if turn_duration <= 0:
            return target_heading % 360

        ratio = min(max(t / turn_duration, 0.0), 1.0)
        return (start_heading + diff * ratio) % 360

    def apply_turn(self, target_heading):
        """转向到 target_heading（直接更新 heading，不前进）"""
        self.heading = target_heading % 360

    @staticmethod
    def _normalize_angle(angle):
        """将角度归一化到 [-180, 180]"""
        return (angle + 180) % 360 - 180

    # ════════════════════════════════════════════════════════
    # 朝向与运动方向换算
    # ════════════════════════════════════════════════════════

    @staticmethod
    def heading_from_vector(dx, dy):
        """根据位移向量 (dx, dy) 计算朝向角度（度，[0,360)）"""
        return degrees(atan2(dy, dx)) % 360

    # ════════════════════════════════════════════════════════
    # 几何外形（供 visualization 使用）
    # ════════════════════════════════════════════════════════

    def get_corners(self, x=None, y=None, heading=None):
        """
        返回车身矩形的四个角点 [(x,y), ...]，按逆时针顺序：
            右前、左前、左后、右后

        可传入 x/y/heading 覆盖当前状态（用于绘制轨迹中间帧）。
        """
        cx = self.x if x is None else x
        cy = self.y if y is None else y
        h  = self.heading if heading is None else heading

        L, W = self.params.length, self.params.width
        hl, hw = L / 2.0, W / 2.0   # half-length, half-width
        rad = radians(h)
        cos_h, sin_h = cos(rad), sin(rad)

        # 矩形局部坐标系下的四角（局部x沿车头方向，局部y沿车宽方向）
        local_corners = [
            ( hl,  hw),   # 右前
            ( hl, -hw),   # 左前
            (-hl, -hw),   # 左后
            (-hl,  hw),   # 右后
        ]

        world_corners = []
        for lx, ly in local_corners:
            wx = cx + lx * cos_h - ly * sin_h
            wy = cy + lx * sin_h + ly * cos_h
            world_corners.append((wx, wy))
        return world_corners

    def get_front_point(self, x=None, y=None, heading=None):
        """返回车头中心点坐标，用于可视化标记朝向"""
        cx = self.x if x is None else x
        cy = self.y if y is None else y
        h  = self.heading if heading is None else heading

        hl = self.params.length / 2.0
        rad = radians(h)
        return (cx + hl * cos(rad), cy + hl * sin(rad))

    def get_rear_point(self, x=None, y=None, heading=None):
        """返回车尾中心点坐标"""
        cx = self.x if x is None else x
        cy = self.y if y is None else y
        h  = self.heading if heading is None else heading

        hl = self.params.length / 2.0
        rad = radians(h)
        return (cx - hl * cos(rad), cy - hl * sin(rad))

    # ════════════════════════════════════════════════════════
    # 状态更新
    # ════════════════════════════════════════════════════════

    def set_pose(self, x, y, heading=None, current_node=None):
        """更新位置/朝向/所在节点"""
        self.x = x
        self.y = y
        if heading is not None:
            self.heading = heading % 360
        if current_node is not None:
            self.current_node = current_node

    def __repr__(self):
        return (f"AGV(id={self.agv_id}, pos=({self.x:.2f},{self.y:.2f}), "
                f"heading={self.heading:.1f}°, node={self.current_node})")


# ──────────────────────────────────────────────────────────────
# 快速验证
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    params = AGVParams()
    agv = AGV(agv_id=1, params=params, x=0.0, y=0.0, heading=0.0, current_node=14)

    print("── 直线行驶时间计算 ──")
    for dist in [1.0, 2.0, 4.0, 2.83]:
        profile = agv.travel_time(dist)
        print(f"  距离 {dist:5.2f}m → "
              f"t_acc={profile['t_acc']:.2f}s, "
              f"t_const={profile['t_const']:.2f}s, "
              f"t_dec={profile['t_dec']:.2f}s, "
              f"总计={profile['t_total']:.2f}s, "
              f"v_peak={profile['v_peak']:.2f}m/s")

    print("\n── 原地转向时间计算 ──")
    agv.heading = 0.0  # 朝东
    for target in [90.0, 180.0, 270.0, -90.0]:
        t, diff = agv.turn_time(target)
        print(f"  从 {agv.heading:5.1f}° 转到 {target % 360:5.1f}° "
              f"→ 转角差={diff:6.1f}°, 耗时={t:.2f}s")

    print("\n── 几何外形（位于原点，朝东）──")
    agv.heading = 0.0
    agv.x, agv.y = 0.0, 0.0
    print(f"  四角坐标: {[(round(x,2), round(y,2)) for x,y in agv.get_corners()]}")
    print(f"  车头点  : {tuple(round(v,2) for v in agv.get_front_point())}")
    print(f"  车尾点  : {tuple(round(v,2) for v in agv.get_rear_point())}")

    print("\n── 几何外形（位于原点，朝北90°）──")
    agv.heading = 90.0
    print(f"  四角坐标: {[(round(x,2), round(y,2)) for x,y in agv.get_corners()]}")
    print(f"  车头点  : {tuple(round(v,2) for v in agv.get_front_point())}")

    print("\n── 连续轨迹插值：从 (0,0) 走到 (4,0)，朝东 ──")
    agv.heading = 0.0
    start_xy, end_xy = (0.0, 0.0), (4.0, 0.0)
    profile = agv.travel_time(4.0)
    print(f"  总耗时 t_total={profile['t_total']:.2f}s")
    for t in [0.0, 0.7, 1.4, 2.0, 2.83, 4.0, 5.66, 6.0]:
        x, y = agv.pose_during_travel(t, start_xy, end_xy, profile)
        d = agv.distance_at_time(t, profile)
        print(f"  t={t:5.2f}s → 已行驶 {d:.2f}m, 位置=({x:.2f}, {y:.2f})")

    print("\n── 原地转弯插值：从朝东(0°)转到朝北(90°) ──")
    turn_duration, diff = agv.turn_time(90.0)
    print(f"  转角差={diff:.1f}°, 总耗时={turn_duration:.2f}s")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        h = agv.pose_during_turn(t, 0.0, 90.0, turn_duration)
        print(f"  t={t:5.2f}s → heading={h:.1f}°")