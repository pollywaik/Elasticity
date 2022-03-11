from sph import *

################################################# obj ############################################
@ti.data_oriented
class Elasticity(Fluid):
    def __init__(self, max_part_num):
        super(Elasticity, self).__init__(max_part_num)
        self.pos0 = ti.Vector.field(dim, int, max_part_num)  # particle initial position

        self.neighbors_initial = ti.field(int, (max_part_num, 80))  # neighbors_initial[i,j] -- initial neighbor particle
        self.neighbors_num = ti.field(int, max_part_num)  # neighbor particle counter

        self.L = ti.Matrix.field(3, 3, float, max_part_num)  # correction matrix
        self.corrected_W = ti.Vector.field(3, float, max_part_num)  # corrected kernel function
        self.R = ti.Matrix.field(3, 3, float, max_part_num)  # rotation matrix
        # self.RL = ti.Matrix.field(3, 3, float, max_part_num)  # rotation matrix * correction matrix

        # solver rhs
        self.F = ti.Matrix.field(3, 3, float, max_part_num)  # deformation gradient
        self.strain = ti.Vector.field(6, float, ())  # temp
        self.stress = ti.Vector.field(6, float, max_part_num)
        self.elastic_force = ti.Vector.field(3, float, max_part_num)

        # matrix free conjugate gradient method
        self.b = ti.Vector.field(3, float, max_part_num)
        self.r = ti.Vector.field(3, float, max_part_num)  # residual r0 = b
        self.p = ti.Vector.field(3, float, max_part_num)  # p0 = r0
        self.grad_u = ti.Matrix.field(3, 3, float, max_part_num)  # gradient of the displacement field
        self.stress_adv = ti.Vector.field(6, float, max_part_num)
        self.elastic_force_adv = ti.Vector.field(3, float, max_part_num)  # next time step, temp force
        self.Ap = ti.Vector.field(3, float, max_part_num)

        # self.beta = ti.field(float, max_part_num)
        # self.alpha = ti.field(float, max_part_num)

        # self.init_pos0()


    def init_pos0(self):
        self.pos0 = self.pos
        print(self.pos0)
        print(self.pos)

    @ti.kernel
    def init_neighbors(self, ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
        for i in range(obj.part_num[None]):
            self.neighbors_num[i] = 0
            for t in range(config.neighb_search_template.shape[0]):
                node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t])
                if 0 < node_code < config.node_num[None]:
                    for j in range(ngrid.node_part_count[node_code]):
                        shift = ngrid.node_part_shift[node_code] + j
                        neighb_uid = ngrid.part_uid_in_node[shift]
                        if neighb_uid == nobj.uid:
                            neighb_pid = ngrid.part_pid_in_node[shift]

                            xij0 = obj.pos0[i] - nobj.pos0[neighb_pid]
                            r = xij0.norm()
                            if r > 0:
                                self.neighbors_initial[i, self.neighbors_num[i]] = neighb_pid
                                self.neighbors_num[i] += 1
                                print("neighb_pid:", neighb_pid)
                                print("neighbors_num:", self.neighbors_num[i])




@ti.data_oriented
class ConfigElasticity(Config):
    def __init__(self):
        super(ConfigElasticity, self).__init__()
        self.youngs_modulus = ti.field(float, ())
        self.poisson_ratio = ti.field(float, ())
        self.elasticity_mu = ti.field(float, ())  # μ =E/2(1+ν)
        self.elasticity_lambda = ti.field(float, ())  # λ=Eν/(1+ν)(1−2ν)

        self.init_elasticity_parameters()

    def init_elasticity_parameters(self):
        self.youngs_modulus[None] = 2e3
        self.poisson_ratio[None] = 0.2
        self.elasticity_mu[None] = self.youngs_modulus[None] / (2.0 * (1.0 + self.poisson_ratio[None]))  # μ =E/2(1+ν)
        self.elasticity_lambda[None] = self.youngs_modulus[None] * self.poisson_ratio[None] / (
                (1.0 + self.poisson_ratio[None]) * (1.0 - 2.0 * self.poisson_ratio[None]))  # λ=Eν/(1+ν)(1−2ν)



############################################# config para #########################################
config_e = ConfigElasticity()




################################################# tool ############################################

@ti.func
def multi_6v_with_3v(M, v):  # use 6vec to represent the matrix
    r1 = M[0] * v[0] + M[3] * v[1] + M[4] * v[2]
    r2 = M[3] * v[0] + M[1] * v[1] + M[5] * v[2]
    r3 = M[4] * v[0] + M[5] * v[1] + M[2] * v[2]
    return [r1, r2, r3]



def init_neighb(ngrid: ti.template(), fluid:ti.template(), bound:ti.template()):
    """ neighbour search """
    ngrid.clear_node()
    ngrid.encode(fluid)
    ngrid.encode(bound)
    ngrid.mem_shift()
    ngrid.fill_node(fluid)
    ngrid.fill_node(bound)


@ti.kernel
def elasticity_cfl_condition(obj: ti.template()):
    config.dt[None] = config.part_size[1] / config.cs[None]
    for i in range(obj.part_num[None]):
        v_norm = obj.vel_adv[i].norm()  # v_elastic
        if v_norm > 1e-4:
            atomic_min(config.dt[None], config.part_size[1] / v_norm * config.cfl_factor[None])


################################################# func ############################################
@ti.kernel
def elasticity_compute_W(obj: ti.template(), nobj: ti.template()):  # L W -- united
    for i in range(obj.part_num[None]):
        for n in ti.static(range(obj.neighbors_num[None])):
            j = obj.neighbors_initial[i, n]
            xij_0 = obj.pos0[i] - nobj.pos0[j]
            xji_0 = -xij_0
            r = xij_0.norm()
            obj.L[i] += nobj.rest_volume[j] * (W_grad(r) * xij_0 / r) @ xji_0.transpose()  # Matrix W,  (xij_0 / r)--A unit vector

        # add 1 to z - component. otherwise we get a singular matrix in 2D
        if config.dim[None] == 2:
            obj.L[i][2, 2] = 1.0

        if obj.L[i].determinant() != 0:
            obj.corrected_W[i] = obj.L[i].inverse() * (W_grad(r) * xij_0 / r)  # corrected_W[i] todo


@ti.kernel
def elasticity_compute_rotations(obj: ti.template(), nobj: ti.template()):  # R RL -- united
    F = ti.Matrix.zero(float, 3, 3)
    for i in range(obj.part_num[None]):
        # print(type(range(obj.neighbors_num[i])))
        for n in range(obj.neighbors_num[i]):
        # for n in ti.static(range(obj.neighbors_num[i])):
            j = obj.neighbors_initial[i, n]
            xji = nobj.pos[j] - obj.pos[i]
            F += nobj.rest_volume[j] * xji @ obj.corrected_W[i].transpose()

        if config.dim[None] == 2:
            F[2, 2] = 1.0

        # obj.R[i] = extract_rotation(F, q, 10)  # todo
        U, sig, V = ti.svd(F)
        obj.R[i] = U @ V.transpose()
        # obj.RL[i] = obj.R[i] * obj.L[i]

        # obj.corrected_W[i] *= obj.R[i]  # rotated kernel gradient


################################################# solve rhs ############################################
@ti.kernel
def elasticity_compute_stress(obj: ti.template(), nobj: ti.template()):  # F(xt, x0)  strain(xt, x0) stress(xt, x0)
    for i in range(obj.part_num[None]):
        # for n in ti.static(range(obj.neighbors_num[i])):
        for n in range(obj.neighbors_num[i]):
            j = obj.neighbors_initial[i, n]
            xji = nobj.pos[j] - obj.pos[i]
            xji_0 = nobj.pos0[j] - obj.pos0[i]  # initial neighbor
            RLW = obj.R[i]@obj.corrected_W[i]
            obj.F[i] += nobj.rest_volume[j] * (xji - obj.R[i] @ xji_0) @ RLW.transpose() # Matrix W_grad(r) todo
            # obj.F[i] += nobj.rest_volume[j] * xji @ RLW.transpose()  # Matrix W_grad(r) todo

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
        obj.stress[i] = obj.strain[None] * 2.0 * config_e.elasticity_mu[None]
        ltrace = config_e.elasticity_lambda[None]*(obj.strain[None][0] + obj.strain[None][1] + obj.strain[None][3])
        obj.stress[i][0] += ltrace
        obj.stress[i][1] += ltrace
        obj.stress[i][2] += ltrace


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


@ti.kernel
def SPH_advection_elasticity_rhs(obj: ti.template()):
    for i in range(obj.part_num[None]):
        # obj.vel_adv[i] = obj.vel[i] + config.dt[None] * (obj.acce_adv[i] + obj.elastic_force[i]/obj.mass[i])
        obj.b[i] = obj.vel[i] + config.dt[None] * (obj.acce_adv[i] + obj.elastic_force[i]/obj.mass[i])
################################################# solve iter############################################

@ti.kernel
def elasticity_iter_init(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.r[i] = obj.b[i]  # r0 = b - Ax0
        # if obj.r[i] <= 10e-4:
        #     break
        obj.p[i] = obj.r[i]  # p0 = r0


@ti.kernel
def elasticity_iter_stress(obj: ti.template(), nobj: ti.template()):  # grad u -- F(dtp,0) strain(dtp,0) stress(dtp,0)
    for i in range(obj.part_num[None]):
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
        obj.stress_adv[i] = obj.strain[None] * 2.0 * config_e.elasticity_mu[None]
        ltrace = config_e.elasticity_lambda[None] * (obj.strain[None][0] + obj.strain[None][1] + obj.strain[None][3])
        obj.stress_adv[i][0] += ltrace
        obj.stress_adv[i][1] += ltrace
        obj.stress_adv[i][2] += ltrace


@ti.kernel
def elasticity_iter_force_adv(obj: ti.template(), nobj: ti.template()):  # f(dtp, 0)
    for i in range(obj.part_num[None]):
        # for n in ti.static(range(obj.neighbors_num[i])):
        for n in range(obj.neighbors_num[i]):
            j = obj.neighbors_initial[i, n]
            RLWi = obj.R[i] @ obj.corrected_W[i]
            RLWj = nobj.R[j] @ nobj.corrected_W[j]
            obj.elastic_force_adv[i] += obj.rest_volume[i] * nobj.rest_volume[j] * (
                    ti.Vector(multi_6v_with_3v(obj.stress_adv[i], RLWi)) - ti.Vector(multi_6v_with_3v(nobj.stress_adv[j], RLWj)))


@ti.kernel
def elasticity_iter_Ap(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.Ap[i] = obj.p[i] - (1 /obj.mass[i]) * config.dt[None] * obj.elastic_force_adv[i]


@ti.kernel
def elasticity_iter_vel_adv(obj: ti.template()):  # v_elastic

    for i in range(obj.part_num[None]):
        while obj.r[i].norm() >= 10e-4:
            # alpha
            rTr = obj.r[i].norm()
            pTAp = obj.p[i].transpose() @ obj.Ap[i]
            alpha = rTr / pTAp[0,0]

            # x
            obj.vel_adv[i] += alpha * obj.p[i]
            # r
            obj.r[i] -= alpha * obj.Ap[i]

            beta = obj.r[i].norm() / rTr
            obj.p[i] = obj.r[i] + beta * obj.p[i]


################################################# acc ############################################

@ti.kernel
def SPH_advection_elasticity_acc(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.acce_adv[i] += (obj.vel_adv[i] - obj.vel[i])/config.dt[None]


def init_elasticity_value(ngrid, fluid, bound):
    init_neighb(ngrid, fluid, bound)
    SPH_clean_value(fluid)
    SPH_clean_value(bound)
    elasticity_compute_W(fluid, fluid)  # todo elastic force only compute in same phase
    elasticity_compute_W(bound, bound)


# todo elasticity
def elasticity_step(fluid, bound):
    ''' RHS '''
    elasticity_compute_rotations(fluid, fluid)
    # elasticity_compute_rotations(bound, bound)
    elasticity_compute_stress(fluid, fluid)
    # elasticity_compute_stress(bound, bound)
    elasticity_compute_force(fluid, fluid)
    # elasticity_compute_force(bound, bound)
    SPH_advection_elasticity_rhs(fluid)  # compute right hand side vector b
    # SPH_advection_elasticity_rhs(bound)

    ''' solve linear system'''
    elasticity_iter_init(fluid)
    # elasticity_iter_init(bound)
    elasticity_iter_stress(fluid, fluid)
    # elasticity_iter_stress(bound, bound)
    elasticity_iter_force_adv(fluid, fluid)
    # elasticity_iter_force_adv(bound, bound)
    elasticity_iter_Ap(fluid)
    # elasticity_iter_Ap(bound)
    elasticity_iter_vel_adv(fluid)
    # elasticity_iter_vel_adv(bound)

    SPH_advection_elasticity_acc(fluid)
    # SPH_advection_elasticity_acc(bound)


def sph_step(ngrid, fluid, bound, globalvar):
    # global div_iter_count, incom_iter_count
    """ neighbour search """
    ngrid.clear_node()
    ngrid.encode(fluid)
    ngrid.encode(bound)
    ngrid.mem_shift()
    ngrid.fill_node(fluid)
    ngrid.fill_node(bound)
    """ SPH clean value """
    SPH_clean_value(fluid)
    SPH_clean_value(bound)
    """ SPH compute W and W_grad """
    SPH_prepare_attr(ngrid, fluid, fluid)
    SPH_prepare_attr(ngrid, fluid, bound)
    SPH_prepare_attr(ngrid, bound, bound)
    SPH_prepare_attr(ngrid, bound, fluid)
    SPH_prepare_alpha_1(ngrid, fluid, fluid)
    SPH_prepare_alpha_1(ngrid, fluid, bound)
    SPH_prepare_alpha_2(ngrid, fluid, fluid)
    SPH_prepare_alpha_2(ngrid, bound, fluid)
    SPH_prepare_alpha(fluid)
    SPH_prepare_alpha(bound)
    """ IPPE SPH divergence """
    globalvar.div_iter_count = 0
    SPH_vel_2_vel_adv(fluid)
    while globalvar.div_iter_count<config.iter_threshold_min[None] or fluid.compression[None]>config.divergence_threshold[None]:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid)
        IPPE_adv_psi(ngrid, fluid, bound)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        IPPE_update_vel_adv(ngrid, fluid, fluid)
        IPPE_update_vel_adv(ngrid, fluid, bound)
        globalvar.div_iter_count+=1
        if globalvar.div_iter_count>config.iter_threshold_max[None]:
            break
    SPH_vel_adv_2_vel(fluid)
    """ SPH advection """
    SPH_advection_gravity_acc(fluid)
    SPH_advection_viscosity_acc(ngrid, fluid, fluid)
    # SPH_advection_surface_tension_acc(ngrid, fluid, fluid)

    # todo elasticity_step
    # elasticity_step(fluid, bound)
    # elasticity_cfl_condition(fluid)


    """ IPPE SPH pressure """
    globalvar.incom_iter_count = 0
    while globalvar.incom_iter_count<config.iter_threshold_min[None] or fluid.compression[None]>config.compression_threshold[None]:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid)
        IPPE_adv_psi(ngrid, fluid, bound)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        IPPE_update_vel_adv(ngrid, fluid, fluid)
        IPPE_update_vel_adv(ngrid, fluid, bound)
        globalvar.incom_iter_count+=1
        if globalvar.incom_iter_count>config.iter_threshold_max[None]:
            break
    """ debug info """
    # print('iter div: ', globalvar.div_iter_count)
    # print('incom div: ', globalvar.incom_iter_count)
    """ WC SPH pressure """
    # WC_pressure_val(fluid)
    # WC_pressure_acce(ngrid, fluid, fluid)
    # WC_pressure_acce(ngrid, fluid, bound)
    # SPH_advection_update_vel_adv(fluid)
    """ FBM procedure """
    # while fluid.general_flag[None] > 0:
    #     SPH_FBM_clean_tmp(fluid)
    #     SPH_FBM_convect(ngrid, fluid, fluid)
    #     SPH_FBM_diffuse(ngrid, fluid, fluid)
    #     SPH_FBM_check_tmp(fluid)
    """ SPH update """
    # SPH_update_volume_frac(fluid)
    SPH_update_mass(fluid)
    SPH_update_pos(fluid)
    SPH_update_energy(fluid)
    # map_velocity(ngrid, grid, fluid)
    # return globalvar.div_iter_count, globalvar.incom_iter_count
    """ SPH debug """











