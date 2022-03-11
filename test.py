import taichi as ti

ti.init()

F = ti.Matrix.field(3, 3, float, ())
x = ti.Vector.field(3, float, ())
w = ti.Vector.field(3, float, ())

x[None] = ti.Vector((1,2,3))
w[None] = ti.Vector((1,2,3))

F = x[None] @ w[None].transpose()
print(F)

c = x[None].transpose() @ w[None]
print(c[0,0])