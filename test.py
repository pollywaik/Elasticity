import taichi as ti
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
import math
ti.init()


R0 = np.array([1, 0, 0])
R1 = np.array([0, 1, 0])
R2 = np.array([0, 0, 1])

F0 = np.array([0.078438, 0.078438, 0.078438])
F1 = np.array([0.078438, 0.078438, 0.078438])
F2 = np.array([0.078438, 0.078438, 0.078438])

R = ti.Matrix.field(3, 3, float, 3)
F = ti.Matrix.field(3, 3, float, 3)
R_adv = ti.Matrix.field(3,3,float,())

R_adv[None] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
R[0] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
F[0] = ti.Matrix([[0.1, 0.07, 0.07], [0.1, 0.07, 0.07], [0.1, 0.07, 0.07]])

print(type(R[0]))
@ti.func
def matrix_cross(A, B):
    v1 = A[1, 0]*B[2, 0]-A[2, 0]*B[1, 0] + A[1, 1]*B[2, 1]-A[2, 1]*B[1, 1] + A[1, 2]*B[2, 2]-A[2, 2]*B[1, 2]
    v2 = A[2, 0]*B[0, 0]-A[0, 0]*B[2, 0] + A[2, 1]*B[0, 1]-A[0, 1]*B[2, 1] + A[2, 2]*B[0, 2]-A[0, 2]*B[2, 2]
    v3 = A[0, 0]*B[1, 0]-A[1, 0]*B[0, 0] + A[0, 1]*B[1, 1]-A[1, 1]*B[0, 1] + A[0, 2]*B[1, 2]-A[1, 2]*B[0, 2]
    return [v1, v2, v3]

@ti.func
def matrix_dot(A, B):
    res = (A[0, 0] * B[0, 0] + A[1, 0] * B[1, 0] + A[2, 0] * B[2, 0]) + \
          (A[0, 1] * B[0, 1] + A[1, 1] * B[1, 1] + A[2, 1] * B[2, 1]) + \
          (A[0, 2] * B[0, 2] + A[1, 2] * B[1, 2] + A[2, 2] * B[2, 2])
    return res

@ti.func
def rotvet_to_matrix(axis, angle, R_adv:ti.template()):
    _cos = 1-ti.cos(angle)
    R_adv[0, 0] = ti.cos(angle) + _cos * axis.x ** 2
    R_adv[0, 1] = -ti.sin(angle) * axis.z + _cos * axis.x * axis.y
    R_adv[0, 2] = ti.sin(angle) * axis.y + _cos * axis.x * axis.z

    R_adv[1, 0] = ti.sin(angle) * axis.z + _cos * axis.x * axis.y
    R_adv[1, 1] = ti.cos(angle) + _cos * axis.y ** 2
    R_adv[1, 2] = -ti.sin(angle) * axis.x + _cos * axis.y * axis.z

    R_adv[2, 0] = -ti.sin(angle) * axis.y + _cos * axis.x * axis.z
    R_adv[2, 1] = ti.sin(angle) * axis.x + _cos * axis.y * axis.z
    R_adv[2, 2] = ti.cos(angle) + _cos * axis.z ** 2



@ti.kernel
def extractRotation(R:ti.template(), F:ti.template(), iter:ti.template()):
    for i in range(1):
        omega = ti.Vector(matrix_cross(R, F)) * (1.0 / abs(matrix_dot(R, F) + 1.0e-9))
        w = omega.norm()  # angle
        rotvet_to_matrix(omega/w, w, R_adv[None])

        # # if w < 1.0e-9:
        # #     break

        # r = Rotation.from_rotvec(omega)
        # R_adv.from_numpy(r.as_matrix())
        res = R @ R_adv[None]
        print(res)
        R = res

        print(R)

        # R = R_adv[None] @ R
# print(np.array(matrix_cross(R[0], F[0])))
extractRotation(R[0],F[0],10)

# [[ 0.98441565 -0.12434998 -0.12434998]
#  [ 0.12434998  0.99220783 -0.00779218]
#  [ 0.12434998 -0.00779218  0.99220783]]


# [[-0.333333, 0.666667, 0.666666], [0.666667, -0.333333, 0.666667], [0.666667, 0.666667, -0.333333]]
for i in range(1):
    omega = (np.cross(R0, F0) + np.cross(R1, F1) + np.cross(R2, F2)) *\
            (1.0 / abs(np.dot(R0, F0) + np.dot(R1, F1) + np.dot(R2, F2) + 1.0e-9))
    w = np.linalg.norm(omega)
    if w < 1.0e-9:
        break
    print(np.cross(R0, F0))
    print(np.cross(R1, F1))
    print(np.cross(R2, F2))
    r = R.from_rotvec(omega)
    r.as_quat()
    print(r.as_quat())

# import pysplishsplash as sph
#
# def main():
#     base = sph.Exec.SimulatorBase()
#     base.init(useGui=False)
#     base.setValueFloat(base.STOP_AT, 10.0) # Important to have the dot to denote a float
#     base.run()
#
# if __name__ == "__main__":
#     main()