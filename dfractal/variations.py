import taichi as ti
import taichi.math as tm


@ti.func
def sphere(p: tm.vec2) -> tm.vec2:
    r2 = p.x**2 + p.y**2
    return p / r2


@ti.func
def swirl(p: tm.vec2) -> tm.vec2:
    r2 = p.x**2 + p.y**2
    return tm.vec2(
        [p.x * tm.sin(r2) - p.y * tm.cos(r2), p.x * tm.cos(r2) + p.y * tm.sin(r2)]
    )


@ti.func
def horshoe(p: tm.vec2) -> tm.vec2:
    r = tm.sqrt(p.x**2 + p.y**2)
    return tm.vec2([(p.x - p.y) * (p.x + p.y), 2.0 * p.x * p.y]) * r**-1


@ti.func
def polar(p: tm.vec2) -> tm.vec2:
    theta = tm.atan2(p.x, p.y)
    r = tm.sqrt(p.x**2 + p.y**2)
    return tm.vec2([theta / tm.pi, r - 1.0])


@ti.func
def hankerchief(p: tm.vec2) -> tm.vec2:
    theta = tm.atan2(p.x, p.y)
    r = tm.sqrt(p.x**2 + p.y**2)
    return tm.vec2([tm.sin(theta + r), tm.cos(theta - r)])


@ti.func
def heart(p: tm.vec2) -> tm.vec2:
    theta = tm.atan2(p.x, p.y)
    r = tm.sqrt(p.x**2 + p.y**2)
    return tm.vec2([tm.sin(theta * r), -tm.cos(theta * r)]) * r


@ti.func
def disc(p: tm.vec2) -> tm.vec2:
    theta = tm.atan2(p.x, p.y)
    r = tm.sqrt(p.x**2 + p.y**2)
    return tm.vec2([tm.sin(tm.pi * r), tm.cos(tm.pi * r)]) * theta / tm.pi


@ti.func
def spiral(p: tm.vec2) -> tm.vec2:
    theta = tm.atan2(p.x, p.y)
    r = tm.sqrt(p.x**2 + p.y**2)
    return tm.vec2([tm.cos(theta) + tm.sin(r), tm.sin(theta) - tm.cos(r)]) * r**-1


@ti.func
def hyperbolic(p: tm.vec2) -> tm.vec2:
    theta = tm.atan2(p.x, p.y)
    r = tm.sqrt(p.x**2 + p.y**2)
    return tm.vec2([tm.sin(theta) / r, r * tm.cos(theta)])


@ti.func
def diamond(p: tm.vec2) -> tm.vec2:
    theta = tm.atan2(p.x, p.y)
    r = tm.sqrt(p.x**2 + p.y**2)
    return tm.vec2([tm.sin(theta) * tm.cos(r), tm.cos(theta) * tm.sin(r)])


@ti.func
def fisheye(p: tm.vec2) -> tm.vec2:
    r = tm.sqrt(p.x**2 + p.y**2)
    return tm.vec2([p.y, p.x]) * 2.0 / (r + 1.0)


@ti.func
def exponential(p: tm.vec2) -> tm.vec2:
    return tm.vec2([tm.cos(tm.pi * p.y), tm.sin(tm.pi * p.y)]) * tm.exp(p.x - 1)


@ti.func
def power(p: tm.vec2) -> tm.vec2:
    theta = tm.atan2(p.x, p.y)
    r = tm.sqrt(p.x**2 + p.y**2)
    return tm.vec2([tm.cos(theta), tm.sin(theta)]) * r ** tm.sin(theta)


@ti.func
def eyefish(p: tm.vec2) -> tm.vec2:
    r = tm.sqrt(p.x**2 + p.y**2)
    return p * 2.0 / (r + 1.0)


@ti.func
def bubble(p: tm.vec2) -> tm.vec2:
    r2 = p.x**2 + p.y**2
    return p * 4.0 / (r2 + 4.0)
