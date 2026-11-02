"""
漏水引流器 —— build123d 参数化建模脚本
=========================================

把原来用 SolidWorks 手工做的「引流」零件改成全部用代码描述：
可复现、可编辑、可 diff、可参数化出 V1.0 / V1.1 两个版本。

坐标系约定（与原始 SW 模型一致，单位 mm）：

        Z  ┌ 竖直，Z=0 为底面
           │
           │        ┌──── 卡水管（C 形卡箍，开口朝 +Y）
           │   ┌────┴────┐
           │   │ 挡板    │
           └───┴─────────┴────────────────────── X  沿水管轴线
          Y 横向

零件由三个实体合并而成：

1. 引水槽（channel）  长条形开口槽，截面对称，轴线在 XY 面内相对 X 轴倾斜 -10°，
                      起点被 X=0 平面切平，末端垂直轴线切断；
                      V1.1 起末端沿 R50 圆弧下弯 45°，形成水龙头式出水口。
2. 挡板（plate）      × 右端立板，15 × 32 × 10，顶/底 8 条边 R2 圆角。
3. 卡水管（clamp）    C 形卡箍，内径 26 / 外径 32，开口朝 +Y 略偏 -Z（约 -3.8°）。

README.md 里记录的实测值与本脚本默认参数（V1.0，复刻现有 STL）：

    卡水管  设计 26（模型 26，实物 25.6）→ clamp_inner_dia = 26
    卡水管  壁厚 3                        → clamp_wall = 3
    引水槽  设计 24，实际 25.71（实物外径 25.3）→ channel_outer_dia = 25.71

其余尺寸是从原 STL 反推的（截面积分 + 圆拟合）：

    卡箍开口  实测 -57.7°..+50.2°          → clamp_open_angle = 107.9，mouth_angle = -3.8
    挡板      Y 35.93..50.93，Z 0..32，X 0..10，顶/底 8 条边 R2
    引水槽    轴线在 X=0 处 Y=49.11、Z=15.525，相对 X 轴倾斜 -10°，沿轴长 210.4

校核（V1.0 重建 vs 原 STL）：包围盒 208.022×74.738×32.000 vs 208.073×74.652×32.000，
体积 24.84 vs 24.59 cm³（+0.91%，原 STL 三角面很粗，圆弧按弦近似本身偏小）。

V1.1（README.md 的修改 1/2/3 + 末端出水口 + 加长）：

    修改 1：卡水管内径 26 → 24（26 太松，扣不住 24mm 水管）
    修改 2：卡水管壁厚 3 → 2
    修改 3：引水槽外径 25.71 → 24（README.md 里写了"设计引水直径 24"但约束错了）
    修改 4：末端出水口下弯 45°（R50 圆弧 + 30mm 直口），斜向前下方，像水龙头
    修改 5：引水槽总长 210.4 → 250（沿轴线的总长，含弯折段和直口）

用法（在本项目根目录下执行）：

    python codes/引流模型.py               # 生成 V1.0（复刻现在的模型）
    python codes/引流模型.py --ver v11     # 生成 V1.1（含上面全部修改）
    python codes/引流模型.py --no-export   # 只打印尺寸/体积，不写文件
    python codes/引流模型.py --ver v11 --out-dir 其他目录

输出默认写到 output/V1.0/ 或 output/V1.1/。
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass, replace
from math import cos, radians, sin
from pathlib import Path

from build123d import (
    Align,
    Axis,
    Box,
    Circle,
    Part,
    Plane,
    Polygon,
    Pos,
    Vector,
    export_step,
    export_stl,
    extrude,
    fillet,
    revolve,
)


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    """全部尺寸参数（mm）。改这里就能改模型，不需要动建模逻辑。"""

    # ---- 卡水管：端部 C 形卡箍 ----
    clamp_inner_dia: float = 26.0        # 卡水管直径（夹持 24mm 水管）
    clamp_wall: float = 3.0              # 卡箍壁厚
    clamp_open_angle: float = 107.9      # 开口角（原模型实测 -57.7°..+50.2°）
    clamp_mouth_angle: float = -3.8      # 开口轴线相对 +Y 的偏角（草图内逆时针为正）
    clamp_center_y: float = 64.4         # 卡箍圆心 Y
    clamp_center_z: float = 16.0         # 卡箍圆心 Z（= 外径/2）

    # ---- 挡板：左端立板 ----
    plate_y: float = 35.93               # 挡板起始 Y
    plate_len_y: float = 15.0            # 挡板 Y 方向长度
    plate_len_z: float = 32.0            # 挡板 Z 方向高度
    plate_thickness: float = 10.0        # 挡板 X 方向厚度
    plate_fillet: float = 2.0            # 挡板顶/底 8 条边的圆角

    # ---- 引水槽：主长条 ----
    channel_outer_dia: float = 25.71     # 槽外径
    channel_wall: float = 2.0            # 槽壁厚
    channel_open_angle: float = 137.0    # 槽口张角（开口朝轴线局部 +Y）
    channel_tilt_deg: float = -10.0      # 轴线相对 X 轴的倾角（俯视逆时针为正）
    channel_center_y: float = 49.11      # 轴线在 X=0 处的 Y
    channel_center_z: float = 15.525     # 轴线高度（恒定）
    channel_length: float = 210.4        # 沿轴线的总长度（含弯折段，末端垂直切断）

    # ---- 引水槽末端出水口：向下弯折（水龙头式） ----
    channel_bend_angle: float = 0.0      # 下弯角度（0 = 直槽；正角度向 -Z 弯）
    channel_bend_radius: float = 50.0    # 弯折半径（越大越平缓）
    channel_tip_len: float = 12.0        # 弯完之后留的直段长度

    @property
    def clamp_outer_dia(self) -> float:
        return self.clamp_inner_dia + 2 * self.clamp_wall

    @property
    def channel_inner_dia(self) -> float:
        return self.channel_outer_dia - 2 * self.channel_wall


V10 = Params()

# readme.txt 里的 V1.1 三处修改 + 末端出水口下弯
V11 = replace(
    V10,
    clamp_inner_dia=24.0,
    clamp_wall=2.0,
    channel_outer_dia=24.0,
    channel_length=250.0,
    channel_bend_angle=45.0,
    channel_bend_radius=50.0,
    channel_tip_len=30.0,
)


# --------------------------------------------------------------------------- #
# 几何
# --------------------------------------------------------------------------- #
def annulus_sector(
    outer_r: float, inner_r: float, open_angle: float, mouth_angle: float = 0.0
) -> object:
    """XY 平面内、圆心在原点、开口朝 +X 的环形扇区（2D 草图）。

    mouth_angle 让开口轴线在草图内旋转（逆时针为正）。
    """
    half = radians(open_angle) / 2
    a = radians(mouth_angle)
    far = 3 * outer_r  # 楔形裁切范围要盖过整个环
    mouth = Polygon(
        (0, 0),
        (far * cos(a + half), far * sin(a + half)),
        (far * cos(a - half), far * sin(a - half)),
    )
    return Circle(outer_r) - Circle(inner_r) - mouth


def build_channel(p: Params) -> Part:
    """引水槽：倾斜轴线的环形扇区棱柱，起点被 X=0 平面切平。

    channel_bend_angle > 0 时，末端沿圆弧向下弯一段，再留一小段直口，
    像水龙头出水口一样斜向前下方（弯折轴是水平的开口方向）。
    """
    tilt = radians(p.channel_tilt_deg)
    axis_dir = Vector(cos(tilt), sin(tilt), 0.0)     # 轴线方向
    mouth_dir = Vector(-sin(tilt), cos(tilt), 0.0)   # 局部 +Y，即开口朝向
    up_dir = Vector(0.0, 0.0, 1.0)

    def section_at(origin: Vector, axis: Vector) -> object:
        plane = Plane(origin=origin, x_dir=mouth_dir, z_dir=axis)
        return plane * annulus_sector(
            p.channel_outer_dia / 2, p.channel_inner_dia / 2, p.channel_open_angle
        )

    start = Vector(0.0, p.channel_center_y, p.channel_center_z)

    if p.channel_bend_angle <= 0:
        body = extrude(section_at(start, axis_dir), amount=p.channel_length)
    else:
        theta = radians(p.channel_bend_angle)
        arc_len = p.channel_bend_radius * theta
        straight_len = p.channel_length - arc_len - p.channel_tip_len
        body = extrude(section_at(start, axis_dir), amount=straight_len)

        # 弯折段：绕水平开口方向转，正角度使轴线朝 -Z 下沉
        bend_start = start + axis_dir * straight_len
        bend_center = bend_start - up_dir * p.channel_bend_radius
        body += revolve(
            section_at(bend_start, axis_dir),
            axis=Axis(origin=bend_center, direction=mouth_dir),
            revolution_arc=p.channel_bend_angle,
        )

        # 弯折后的直口
        axis_out = axis_dir * cos(theta) - up_dir * sin(theta)
        tip_start = bend_center + (
            up_dir * cos(theta) + axis_dir * sin(theta)
        ) * p.channel_bend_radius
        body += extrude(section_at(tip_start, axis_out), amount=p.channel_tip_len)

    # 起点端用 X=0 平面切平（末端保持垂直于轴线的斜切口）
    trim = Box(400, 400, 400, align=(Align.MIN, Align.CENTER, Align.CENTER))
    return body & trim


def build_plate(p: Params) -> Part:
    """挡板：15 × 32 × 10 立板，顶/底共 8 条边 R2 圆角（竖边保持直角），X 从 0 起。"""
    plate = Pos(0, p.plate_y, 0) * Box(
        p.plate_thickness,
        p.plate_len_y,
        p.plate_len_z,
        align=(Align.MIN, Align.MIN, Align.MIN),
    )
    edges = plate.edges().filter_by(Axis.X) + plate.edges().filter_by(Axis.Y)
    return fillet(edges, radius=p.plate_fillet)


def build_clamp(p: Params) -> Part:
    """卡水管：C 形卡箍，开口朝 +Y（略偏 -Z），沿 X 拉伸与挡板同厚。"""
    section = Plane.YZ * (
        Pos(p.clamp_center_y, p.clamp_center_z)
        * annulus_sector(
            p.clamp_outer_dia / 2,
            p.clamp_inner_dia / 2,
            p.clamp_open_angle,
            p.clamp_mouth_angle,
        )
    )
    return extrude(section, amount=p.plate_thickness)


def build(p: Params = V10) -> Part:
    """生成整个零件。"""
    part = build_channel(p) + build_plate(p) + build_clamp(p)
    part.label = "引流"
    if len(part.solids()) != 1:
        raise RuntimeError(
            f"合并后得到 {len(part.solids())} 个实体，说明卡箍/挡板/引水槽没有连上，"
            "请检查 clamp_center_y 与 plate_y + plate_len_y 是否有重叠"
        )
    return part


# --------------------------------------------------------------------------- #
# 校验与导出
# --------------------------------------------------------------------------- #
def report(part: Part, p: Params, name: str) -> None:
    bb = part.bounding_box()
    print(f"\n[{name}]")
    print(f"  包围盒   {bb.size.X:.3f} × {bb.size.Y:.3f} × {bb.size.Z:.3f} mm")
    print(f"  X        {bb.min.X:.3f} .. {bb.max.X:.3f}")
    print(f"  Y        {bb.min.Y:.3f} .. {bb.max.Y:.3f}")
    print(f"  Z        {bb.min.Z:.3f} .. {bb.max.Z:.3f}")
    print(f"  体积     {part.volume / 1000:.2f} cm³")
    print(f"  实体数   {len(part.solids())}   面数 {len(part.faces())}")
    print(
        f"  卡水管   内径 {p.clamp_inner_dia:.1f}  外径 {p.clamp_outer_dia:.1f}"
        f"  壁厚 {p.clamp_wall:.1f}  开口 {p.clamp_open_angle:.1f}°"
        f"  开口偏角 {p.clamp_mouth_angle:.1f}°"
    )
    print(
        f"  引水槽   内径 {p.channel_inner_dia:.2f}  外径 {p.channel_outer_dia:.2f}"
        f"  壁厚 {p.channel_wall:.2f}  开口 {p.channel_open_angle:.1f}°"
        f"  倾角 {p.channel_tilt_deg:.1f}°"
    )


def export_files(part: Part, out: Path, stem: str) -> tuple[Path, Path]:
    """导出 STL + STEP。

    STL 用 OCC 的 StlAPI 直接写工作区目录时会静默失败（STEP 正常），
    所以先写到临时目录再复制过去，避免用户以为导出成功。
    """
    step_path = out / f"{stem}.step"
    stl_path = out / f"{stem}.stl"

    export_step(part, str(step_path))
    with tempfile.TemporaryDirectory() as td:
        tmp_stl = Path(td) / "part.stl"
        export_stl(part, str(tmp_stl), tolerance=0.02, angular_tolerance=0.2)
        shutil.copyfile(tmp_stl, stl_path)

    for f in (stl_path, step_path):
        if not f.exists() or f.stat().st_size == 0:
            raise RuntimeError(f"导出失败：{f}")
    return stl_path, step_path


def main() -> None:
    ap = argparse.ArgumentParser(description="漏水引流器参数化建模")
    ap.add_argument("--ver", choices=("v10", "v11"), default="v10", help="版本")
    ap.add_argument("--out-dir", default=None, help="输出目录，默认 output/V1.0 或 output/V1.1")
    ap.add_argument("--no-export", action="store_true", help="只打印，不导出文件")
    args = ap.parse_args()

    params = V10 if args.ver == "v10" else V11
    name = "引流V1.0" if args.ver == "v10" else "引流V1.1"
    part = build(params)
    report(part, params, name)

    if args.no_export:
        return
    # 脚本放在 codes/ 下，输出统一放到项目根目录的 output/<版本>/
    root = Path(__file__).resolve().parent.parent
    out = Path(args.out_dir) if args.out_dir else root / "output" / name.replace("引流", "")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{name}-build123d"
    stl_path, step_path = export_files(part, out, stem)
    print(f"\n  已导出   {stl_path}")
    print(f"           {step_path}")


if __name__ == "__main__":
    main()
