from sph import *
from scipy.spatial.transform import Rotation as Rotation
################################################# obj ############################################

@ti.kernel
def init_values(obj: ti.template()):
    obj.init_pos0()
    for i in range(obj.part_num[None]):
        obj.R[i] = ti.Matrix.identity(float, 3)
        # print(obj.pos[i])
        # print(obj.pos0[i])
        # print(obj.R[i])

    # print(self.pos0)
    # print(self.mass)
    # print(self.rest_volume)

@ti.kernel
def init_neighbors(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]

                        xij0 = obj.pos0[i] - nobj.pos0[neighb_pid]
                        r = xij0.norm()
                        if 0 < r < config.kernel_h[1]:
                            obj.neighbors_initial[i, obj.neighbors_num[i]] = neighb_pid
                            # print("npid", i, obj.neighbors_num[i], neighb_pid, obj.neighbors_initial[i, obj.neighbors_num[i]])
                            obj.neighbors_num[i] += 1
    # for i in range(obj.neighbors_num[1]):
    #     print(obj.neighbors_initial[1, i])


################################################# tool ############################################
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
def rotvet_to_matrix(axis, angle, R_adv):
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

    return R_adv


@ti.func
def multi_6v_with_3v(M, v):  # use 6vec to represent the matrix
    v0 = M[0] * v[0] + M[3] * v[1] + M[4] * v[2]
    v1 = M[3] * v[0] + M[1] * v[1] + M[5] * v[2]
    v2 = M[4] * v[0] + M[5] * v[1] + M[2] * v[2]
    return [v0, v1, v2]


@ti.kernel
def elasticity_cfl_condition(obj: ti.template(), config: ti.template()):
    config.dt[None] = config.part_size[1] / config.cs[None]
    for i in range(obj.part_num[None]):
        v_norm = obj.vel_adv[i].norm()  # v_elastic
        if v_norm > 1e-4:
            atomic_min(config.dt[None], config.part_size[1] / v_norm * config.cfl_factor[None])

@ti.kernel
def elasticity_clean_value(obj: ti.template(), config: ti.template()):
    dim = ti.static(config.gravity.n)
    for i in range(obj.part_num[None]):
        for j in ti.static(range(dim)):
            obj.elastic_force[i][j] = 0
            obj.elastic_force_adv[i][j] = 0
            obj.r[i][j] = 0
            obj.p[i][j] = 0
            obj.Ap[i][j] = 0

        for j in ti.static(range(6)):
            obj.stress[i][j] = 0
            obj.stress_adv[i][j] = 0

        for x in ti.static(range(dim)):
            for y in ti.static(range(dim)):
                obj.F[i][x, y] = 0
                obj.grad_u[i][x, y] = 0

        # obj.grad_u[i] = ti.Matrix.zero(float, 3, 3)
        # obj.F[i] = ti.Matrix.zero(float, 3, 3)

    for j in ti.static(range(6)):
        obj.strain[None][j] = 0

################################################# func ############################################
@ti.kernel
def elasticity_compute_W(obj: ti.template(), nobj: ti.template(), config: ti.template()):  # L W -- united
    for i in range(obj.part_num[None]):
        for n in range(obj.neighbors_num[i]):
            j = obj.neighbors_initial[i, n]
            xij_0 = obj.pos0[i] - nobj.pos0[j]
            xji_0 = -xij_0
            r = xij_0.norm()
            obj.L[i] += nobj.rest_volume[j] * (W_grad(r, config) * xij_0 / r) @ xji_0.transpose()  # Matrix W,  (xij_0 / r)--A unit vector
            # if i == 1:
            #     print(obj.neighbors_num[i], j)
            #     print(xij_0, xji_0, r)
            #     print(nobj.rest_volume[j], W_grad(r, config), (W_grad(r, config) * xij_0 / r), obj.L[i])
        # add 1 to z - component. otherwise we get a singular matrix in 2D
        if config.dim[None] == 2:
            obj.L[i][2, 2] = 1.0

        for n in range(obj.neighbors_num[i]):
            j = obj.neighbors_initial[i, n]
            xij_0 = obj.pos0[i] - nobj.pos0[j]
            r = xij_0.norm()
            if obj.L[i].determinant() != 0:
                obj.corrected_W[i] = obj.L[i].inverse() @ (W_grad(r, config) * xij_0 / r)  # corrected_W[i] todo
        if i == 0:
            print("W", obj.corrected_W[i])


@ti.func
def extract_rotation(R:ti.template(), F, iter):
    for i in range(iter):
        omega = ti.Vector(matrix_cross(R, F)) * (1.0 / abs(matrix_dot(R, F) + 1.0e-9))
        w = omega.norm()
        if w < 1.0e-9:
            break
        R = rotvet_to_matrix(omega / w, w, R) @ R


@ti.kernel
def elasticity_compute_rotations(obj: ti.template(), nobj: ti.template(), config: ti.template()):  # R RL -- united
    for i in range(obj.part_num[None]):
        F = ti.Matrix.zero(float, 3, 3)
        R_adv = ti.Matrix.identity(float, 3)
        for n in range(obj.neighbors_num[i]):
        # for n in ti.static(range(obj.neighbors_num[i])):
            j = obj.neighbors_initial[i, n]
            xji = nobj.pos[j] - obj.pos[i]
            F += nobj.rest_volume[j] * xji @ obj.corrected_W[i].transpose()

        if config.dim[None] == 2:
            F[2, 2] = 1.0

        extract_rotation(R_adv, F, 10)  # todo
        obj.R[i] = R_adv
        U, sig, V = ti.svd(F)
        temp = U @ V.transpose()

        print(F, obj.R[i], temp)

################################################# solve rhs ############################################
@ti.kernel
def elasticity_compute_stress(obj: ti.template(), nobj: ti.template(), config: ti.template()):  # F(xt, x0)  strain(xt, x0) stress(xt, x0)
    for i in range(obj.part_num[None]):
        # for n in ti.static(range(obj.neighbors_num[i])):
        for n in range(obj.neighbors_num[i]):
            j = obj.neighbors_initial[i, n]
            xji = nobj.pos[j] - obj.pos[i]
            # xji_0 = nobj.pos0[j] - obj.pos0[i]  # initial neighbor
            RLW = obj.R[i]@obj.corrected_W[i]
            # obj.F[i] += nobj.rest_volume[j] * (xji - obj.R[i] @ xji_0) @ RLW.transpose() # Matrix W_grad(r)
            obj.F[i] += nobj.rest_volume[j] * xji @ RLW.transpose()  # Matrix W_grad(r) simplify todo

        # obj.F[i] += ti.Matrix.identity(float, 3)
        if config.dim[None] == 2:
            obj.F[i][2, 2] = 1.0

        # ε = (F+FT)/2 -I symmetric
        obj.strain[None][0] = obj.F[i][0, 0] - 1.0
        obj.strain[None][1] = obj.F[i][1, 1] - 1.0
        obj.strain[None][2] = obj.F[i][2, 2] - 1.0
        obj.strain[None][3] = 0.5 * (obj.F[i][0, 1] + obj.F[i][1, 0])
        obj.strain[None][4] = 0.5 * (obj.F[i][0, 2] + obj.F[i][2, 0])
        obj.strain[None][5] = 0.5 * (obj.F[i][1, 2] + obj.F[i][2, 1])

        # P = 2μ*strain + (K-2μ/3)*trace(strain)*I
        # K = λ+2μ/3
        obj.stress[i] = obj.strain[None] * 2.0 * config.elasticity_mu[None]
        ltrace = config.elasticity_lambda[None]*(obj.strain[None][0] + obj.strain[None][1] + obj.strain[None][2])
        obj.stress[i][0] += ltrace
        obj.stress[i][1] += ltrace
        obj.stress[i][2] += ltrace
        print("F stress", obj.F[i], obj.stress[i])

@ti.kernel
def elasticity_compute_force(obj: ti.template(), nobj: ti.template()):  # f(xt, x0)
    for i in range(obj.part_num[None]):
        # for n in ti.static(range(obj.neighbors_num[i])):
        for n in range(obj.neighbors_num[i]):
            j = obj.neighbors_initial[i, n]
            RLWi = obj.R[i] @ obj.corrected_W[i]
            RLWj = nobj.R[j] @ nobj.corrected_W[j]
            obj.elastic_force[i] += obj.rest_volume[i] * nobj.rest_volume[j] * (
                    ti.Vector(multi_6v_with_3v(obj.stress[i], RLWi)) - ti.Vector(multi_6v_with_3v(nobj.stress[j], RLWj)))
        print("elastic_force", obj.elastic_force[i])


@ti.kernel
def elasticity_compute_rhs(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.b[i] = obj.vel[i] + config.dt[None] * (obj.acce_adv[i] + (1/obj.mass[i]) * obj.elastic_force[i])
        print("obj.b[i]", obj.b[i])
################################################# solve iter############################################

@ti.func
def elasticity_iter_init(i: ti.template(), obj: ti.template()):
    # for i in range(obj.part_num[None]):
    obj.r[i] = obj.b[i]  # r0 = b - Ax0
    obj.p[i] = obj.r[i]  # p0 = r0


@ti.func
def elasticity_iter_stress(i: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):  # grad u -- F(dtp,0) strain(dtp,0) stress(dtp,0)
    # for i in range(obj.part_num[None]):
        # for n in ti.static(range(obj.neighbors_num[i])):
    for n in range(obj.neighbors_num[i]):
        j = obj.neighbors_initial[i, n]
        RLW = obj.R[i] @ obj.corrected_W[i]
        pji = nobj.p[j] - obj.p[i]
        obj.grad_u[i] += nobj.rest_volume[j] * pji @ RLW.transpose()
    obj.grad_u[i] *= config.dt[None]

    # ε = (F+FT)/2 -I , ∇0u = F-I -> ε = (∇0u+∇0uT)/2
    obj.strain[None][0] = obj.grad_u[i][0, 0]
    obj.strain[None][1] = obj.grad_u[i][1, 1]
    obj.strain[None][2] = obj.grad_u[i][2, 2]
    obj.strain[None][3] = 0.5 * (obj.grad_u[i][0, 1] + obj.grad_u[i][1, 0])
    obj.strain[None][4] = 0.5 * (obj.grad_u[i][0, 2] + obj.grad_u[i][2, 0])
    obj.strain[None][5] = 0.5 * (obj.grad_u[i][1, 2] + obj.grad_u[i][2, 1])

    # P = 2μ*strain + (K-2μ/3)*trace(strain)*I
    # K = λ+2μ/3
    obj.stress_adv[i] = obj.strain[None] * 2.0 * config.elasticity_mu[None]
    ltrace = config.elasticity_lambda[None] * (obj.strain[None][0] + obj.strain[None][1] + obj.strain[None][2])
    obj.stress_adv[i][0] += ltrace
    obj.stress_adv[i][1] += ltrace
    obj.stress_adv[i][2] += ltrace


@ti.func
def elasticity_iter_force_adv(i: ti.template(), obj: ti.template(), nobj: ti.template()):  # f(dtp, 0)
    # for i in range(obj.part_num[None]):
        # for n in ti.static(range(obj.neighbors_num[i])):
    for n in range(obj.neighbors_num[i]):
        j = obj.neighbors_initial[i, n]
        RLWi = obj.R[i] @ obj.corrected_W[i]
        RLWj = nobj.R[j] @ nobj.corrected_W[j]
        obj.elastic_force_adv[i] += obj.rest_volume[i] * nobj.rest_volume[j] * (
                ti.Vector(multi_6v_with_3v(obj.stress_adv[i], RLWi)) - ti.Vector(multi_6v_with_3v(nobj.stress_adv[j], RLWj)))


@ti.func
def elasticity_iter_Ap(i: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    elasticity_iter_stress(i, obj, nobj, config)
    elasticity_iter_force_adv(i, obj, nobj)
    # for i in range(obj.part_num[None]):
    obj.Ap[i] = obj.p[i] - (1 / obj.mass[i]) * config.dt[None] * obj.elastic_force_adv[i]


@ti.func
def elasticity_iter_update_p(i: ti.template(), obj: ti.template()):  # v_elastic
    # for i in range(obj.part_num[None]):
        # alpha
    rTr = obj.r[i].transpose() @ obj.r[i]
    pTAp = obj.p[i].transpose() @ obj.Ap[i]
    alpha = rTr[0, 0] / pTAp[0, 0]

    # x
    obj.vel_adv[i] += alpha * obj.p[i]
    # r
    obj.r[i] -= alpha * obj.Ap[i]

    new_rTr = obj.r[i].transpose() @ obj.r[i]
    beta = new_rTr[0, 0] / rTr[0, 0]
    obj.p[i] = obj.r[i] + beta * obj.p[i]

@ti.kernel
def elasticity_iter_vel_adv(obj: ti.template(), nobj: ti.template(), config: ti.template()):
    ''' solve linear system'''
    for i in range(obj.part_num[None]):
        elasticity_iter_init(i, obj)
        iter = 0
        while obj.r[i].norm() >= 1e-3 and iter < 100:
            iter += 1

            elasticity_iter_Ap(i, obj, nobj, config)
            elasticity_iter_update_p(i, obj)
            print("--------------------------", iter, obj.elastic_force_adv[i], obj.Ap[i], obj.p[i])

################################################# acc ############################################

@ti.kernel
def SPH_advection_elasticity_acc(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.acce_adv[i] += (obj.vel_adv[i] - obj.vel[i])/config.dt[None]


def init_elasticity_value(ngrid, fluid, bound, config):
    # initial pos0 and R
    init_values(fluid)

    # initial neighbors
    ngrid.clear_node(config)
    ngrid.encode(fluid, config)
    ngrid.mem_shift(config)
    ngrid.fill_node(fluid, config)
    init_neighbors(ngrid, fluid, fluid, config)

    elasticity_compute_W(fluid, fluid, config)  # todo elastic force only compute in same phase
    print("done init elasticity value")


# todo elasticity
def elasticity_step(fluid, config):
    ''' RHS '''
    elasticity_compute_rotations(fluid, fluid, config)
    elasticity_compute_stress(fluid, fluid, config)
    elasticity_compute_force(fluid, fluid)
    elasticity_compute_rhs(fluid, config)  # compute right hand side vector b
    elasticity_iter_vel_adv(fluid, fluid, config)  # cg

    SPH_advection_elasticity_acc(fluid, config)
    # SPH_advection_elasticity_acc(bound)


def sph_elasticity_step(ngrid, fluid, bound, config):
    cfl_condition(fluid, config)
    """ neighbour search """
    ngrid.clear_node(config)
    ngrid.encode(fluid, config)
    ngrid.encode(bound, config)
    ngrid.mem_shift(config)
    ngrid.fill_node(fluid, config)
    ngrid.fill_node(bound, config)
    """ SPH clean value """
    SPH_clean_value(fluid, config)
    SPH_clean_value(bound, config)
    elasticity_clean_value(fluid, config)  # todo
    """ SPH compute W and W_grad """
    SPH_prepare_attr(ngrid, fluid, fluid, config)
    SPH_prepare_attr(ngrid, fluid, bound, config)
    SPH_prepare_attr(ngrid, bound, bound, config)
    SPH_prepare_attr(ngrid, bound, fluid, config)
    SPH_prepare_alpha_1(ngrid, fluid, fluid, config)
    SPH_prepare_alpha_1(ngrid, fluid, bound, config)
    SPH_prepare_alpha_2(ngrid, fluid, fluid, config)
    SPH_prepare_alpha_2(ngrid, bound, fluid, config)
    SPH_prepare_alpha(fluid)
    SPH_prepare_alpha(bound)
    """ IPPE SPH divergence """
    config.div_iter_count[None] = 0
    SPH_vel_2_vel_adv(fluid)
    config.is_compressible[None] = 1
    while config.div_iter_count[None] < config.iter_threshold_min[None] or config.is_compressible[None] == 1:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid, config)
        IPPE_adv_psi(ngrid, fluid, bound, config)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        IPPE_psi_adv_is_compressible(fluid, config)
        IPPE_update_vel_adv(ngrid, fluid, fluid, config)
        IPPE_update_vel_adv(ngrid, fluid, bound, config)
        config.div_iter_count[None] += 1
        if config.div_iter_count[None] > config.iter_threshold_max[None]:
            break
    SPH_vel_adv_2_vel(fluid)
    """ SPH advection """
    SPH_advection_gravity_acc(fluid, config)
    SPH_advection_viscosity_acc(ngrid, fluid, fluid, config)
    # SPH_advection_viscosity_acc(ngrid, fluid, bound, config)
    # SPH_advection_surface_tension_acc(ngrid, fluid, fluid, config)

    # todo elasticity_step
    elasticity_step(fluid, config)
    elasticity_cfl_condition(fluid, config)

    SPH_advection_update_vel_adv(fluid, config)
    """ IPPE SPH pressure """
    config.incom_iter_count[None] = 0
    while config.incom_iter_count[None] < config.iter_threshold_min[None] or config.is_compressible[None] == 1:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid, config)
        IPPE_adv_psi(ngrid, fluid, bound, config)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        IPPE_psi_adv_is_compressible(fluid, config)
        IPPE_update_vel_adv(ngrid, fluid, fluid, config)
        IPPE_update_vel_adv(ngrid, fluid, bound, config)
        config.incom_iter_count[None] += 1
        if config.incom_iter_count[None] > config.iter_threshold_max[None]:
            break
    """ debug info """
    # print('iter div: ', div_iter_count)
    # print('incom div: ', incom_iter_count)
    """ WC SPH pressure """
    # WC_pressure_val(fluid)
    # WC_pressure_acce(ngrid, fluid, fluid)
    # WC_pressure_acce(ngrid, fluid, bound)
    # SPH_advection_update_vel_adv(fluid)
    """ FBM procedure """
    # while fluid.general_flag[None] > 0:
    #     SPH_FBM_clean_tmp(fluid, config)
    #     SPH_FBM_convect(ngrid, fluid, fluid, config)
    #     SPH_FBM_diffuse(ngrid, fluid, fluid, config)
    #     SPH_FBM_check_tmp(fluid)
    """ SPH update """
    SPH_update_volume_frac(fluid)
    SPH_update_mass(fluid, config)
    SPH_update_pos(fluid, config)
    SPH_update_color(fluid, config)
    # SPH_update_energy(fluid, config)
    # map_velocity(ngrid, grid, fluid)
    return config.div_iter_count[None], config.incom_iter_count[None]
    """ SPH debug """


def run_elasticity_step(ngrid, fluid, bound, config):
    config.time_counter[None] += 1

    while config.time_count[None] < config.time_counter[None] / config.gui_fps[None]:
        """ computation loop """
        config.time_count[None] += config.dt[None]
        if config.solver_type == "DFSPH" or config.solver_type == "VFSPH":
            sph_elasticity_step(ngrid, fluid, bound, config)
        elif config.solver_type == "JL21":
            sph_step_jl21(ngrid, fluid, bound, config)
        else:
            raise Exception('sph ERROR: no solver type', config.solver_type)
        apply_bound_transform(bound, config)
        config.frame_div_iter[None] += config.div_iter_count[None]
        config.frame_incom_iter[None] += config.incom_iter_count[None]









