from sph import *

################################################# obj ############################################
@ti.data_oriented
class Elasticity(Fluid):
    def __init__(self, max_part_num):
        super(Elasticity, self).__init__(max_part_num)
        self.pos0 = ti.Vector.field(dim, int, max_part_num)

        self.neighbors_initial = ti.field(int, (max_part_num, 30))  # neighbors_initial[i,j]
        self.neighbors_num = ti.field(int, ())

        self.L = ti.Matrix.field(3, 3, float, max_part_num)
        self.RL = ti.Matrix.field(3, 3, float, max_part_num)
        self.R = ti.Matrix.field(3, 3, float, max_part_num)
        self.F = ti.Matrix.field(3, 3, float, max_part_num)
        self.strain = ti.Vector.field(6, float, ())
        self.stress = ti.Vector.field(6, float, max_part_num)
        self.elastic_force = ti.Vector.field(3, float, max_part_num)
        self.corrected_W = ti.Vector.field(3, float, max_part_num)

        self.init_pos0()

        # matrix free cg
        self.grad_u = ti.Vector.field(3, 3, float, max_part_num)
        self.iter_p = ti.Vector.field(3, float, max_part_num)
        self.residual = ti.Vector.field(3, float, max_part_num)
        self.v_elast = ti.Vector.field(3, float, max_part_num)
        self.Ap = ti.Vector.field(3, float, max_part_num)
        self.b = ti.Vector.field(3, float, max_part_num)
        self.force_adv = ti.Vector.field(3, float, max_part_num)
        self.beta = ti.field(float, max_part_num)
        self.alpha = ti.field(float, max_part_num)



    def init_pos0(self):
        self.pos0 = self.pos

    @ti.kernel
    def init_neighbors(self, ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
        self.neighbors_num[None] = 0
        for i in range(obj.part_num[None]):
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
                                self.neighbors_initial[i, self.neighbors_num[None]] = neighb_pid
                                self.neighbors_num[None] += 1


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
        self.youngs_modulus = 2e3
        self.poisson_ratio = 0.2
        self.elasticity_mu = self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))  # μ =E/2(1+ν)
        self.elasticity_lambda = self.youngs_modulus * self.poisson_ratio / (
                (1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio))  # λ=Eν/(1+ν)(1−2ν)



############################################# config para #########################################
config_e = ConfigElasticity()




################################################# tool ############################################
@ti.func
def multi_6v_with_3v(M, v):
    r1 = M[0] * v[0] + M[3] * v[1] + M[4] * v[2]
    r2 = M[3] * v[0] + M[1] * v[1] + M[5] * v[2]
    r3 = M[4] * v[0] + M[5] * v[1] + M[2] * v[2]
    return [r1, r2, r3]


@ti.kernel
def init_neighb(ngrid: ti.template(), fluid:ti.template(), bound:ti.template()):
    """ neighbour search """
    ngrid.clear_node()
    ngrid.encode(fluid)
    ngrid.encode(bound)
    ngrid.mem_shift()
    ngrid.fill_node(fluid)
    ngrid.fill_node(bound)


################################################# func ############################################
@ti.kernel
def elasticity_compute_W(obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for j in ti.static(range(obj.neighbors_num[None])):
            xij_0 = obj.pos0[i] - nobj.pos0[j]
            r = xij_0.norm()
            obj.L[i] += nobj.rest_volume[j] * (W_grad(r) * xij_0 / r) * xij_0.transpose()  # Matrix W_grad(r) todo

        # add 1 to z - component. otherwise we get a singular matrix in 2D
        if config.dim[None] == 2:
            obj.L[i][2, 2] = 1.0

        if obj.L[i].determinant() != 0:
            obj.corrected_W[i] = obj.L[i].inverse() * (W_grad(r) * xij_0 / r)  # corrected_W[i] todo


@ti.kernel
def elasticity_compute_rotations(obj: ti.template(), nobj: ti.template()):
    F = ti.Matrix.zero(float, 3, 3)
    for i in range(obj.part_num[None]):
        for j in ti.static(range(obj.neighbors_num[None])):
            xji = nobj.pos[j] - obj.pos[i]
            F += nobj.rest_volume[j] * xji * obj.corrected_W[i].transpose()  # Matrix W_grad(r) todo

        if config.dim[None] == 2:
            F[2, 2] = 1.0

        # obj.R[i] = extract_rotation(F, q, 10)
        U, sig, V = ti.svd(F)
        obj.R[i] = U @ V.transpose()
        obj.RL[i] = obj.R[i] * obj.L[i]

        # obj.corrected_W[i] *= obj.R[i]  # rotated kernel gradient


@ti.kernel
def elasticity_compute_stress(obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for j in ti.static(range(obj.neighbors_num[None])):
            xji = nobj.pos[j] - obj.pos[i]
            xji_0 = nobj.pos0[j] - obj.pos0[i]  # initial neighbor
            RLW = obj.R[i]*obj.corrected_W[i]
            # obj.F[i] += nobj.rest_volume[j] * (xji - obj.R[i] * xji_0) * RLW.transpose() # Matrix W_grad(r) todo
            obj.F[i] += nobj.rest_volume[j] * xji * RLW.transpose()  # Matrix W_grad(r) todo

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
def elasticity_compute_force(obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for j in ti.static(range(obj.neighbors_num[None])):
            RLWi = obj.R[i] * obj.corrected_W[i]
            RLWj = nobj.R[j] * nobj.corrected_W[j]
            obj.elastic_force[i] += obj.rest_volume[i] * nobj.rest_volume[j] * (
                    ti.Vector(multi_6v_with_3v(obj.stress[i], RLWi)) - ti.Vector(multi_6v_with_3v(nobj.stress[j], RLWj)))


@ti.kernel
def elasticity_iter_init(obj: ti.template()):
    obj.r = obj.b  # r0 = b - Ax0
    obj.p = obj.r  # p0 = r0


@ti.kernel
def elasticity_iter_stress(obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for j in ti.static(range(obj.neighbors_num[None])):
            RLW = obj.R[i] * obj.corrected_W[i]
            pji = nobj.iter_p[j] - obj.iter_p[i]
            obj.grad_u[i] += nobj.rest_volume[j] * pji * RLW.transpose()
        obj.grad_u[i] *= config.dt[None]

        # ε = (F+FT)/2 -I symmetric
        obj.strain[None][0] = obj.F[i][0, 0]
        obj.strain[None][1] = obj.F[i][1, 1]
        obj.strain[None][2] = obj.F[i][2, 2]
        obj.strain[None][3] = 0.5 * (obj.F[i][0, 1] + obj.F[i][1, 0])
        obj.strain[None][4] = 0.5 * (obj.F[i][0, 2] + obj.F[i][2, 0])
        obj.strain[None][5] = 0.5 * (obj.F[i][1, 2] + obj.F[i][2, 1])

        # P = 2μ*strain + (K-2μ/3)*trace(strain)*I
        # K = λ+2μ/3
        obj.stress[i] = obj.strain[None] * 2.0 * config_e.elasticity_mu[None]
        ltrace = config_e.elasticity_lambda[None] * (obj.strain[None][0] + obj.strain[None][1] + obj.strain[None][3])
        obj.stress[i][0] += ltrace
        obj.stress[i][1] += ltrace
        obj.stress[i][2] += ltrace


def elasticity_iter_force(obj: ti.template(), nobj: ti.template()):
    elasticity_compute_force(obj, nobj)


################################################# solve ############################################
@ti.kernel
def SPH_advection_elasticity_rhs(obj: ti.template()):
    for i in range(obj.part_num[None]):
        # obj.vel_adv[i] = obj.vel[i] + config.dt[None] * (obj.acce_adv[i] + obj.elastic_force[i]/obj.mass[i])
        obj.b[i] = obj.vel[i] + config.dt[None] * (obj.acce_adv[i] + obj.elastic_force[i]/obj.mass[i])


@ti.kernel
def elasticity_iter_Ap(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.Ap[i] = obj.iter_p[i] - (1 /obj.mass[i]) * config.dt[None] * obj.force_adv[i]

@ti.kernel
def elasticity_iter_alpha(obj: ti.template()):
    for i in range(obj.part_num[None]):
        rTr = obj.r[i].norm()
        pTAp = obj.p[i].T() * obj.Ap[i]
        alpha = rTr / pTAp
        obj.vel_adv[i] += alpha * obj.p[i]


        obj.r[i] -= alpha * obj.Ap[i]

        beta = 


################################################# acc ############################################

@ti.kernel
def SPH_advection_elasticity_acc(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.acce_adv[i] += (obj.vel_adv[i] - obj.vel[i])/config.dt[None]


def init_elastic_value(ngrid, fluid, bound):
    init_neighb(ngrid, fluid, bound)
    SPH_clean_value(fluid)
    SPH_clean_value(bound)
    elasticity_compute_W(fluid, fluid)
    elasticity_compute_W(bound, bound)

# todo elasticity
def elasticity_step(fluid, bound):
    ''' R '''
    elasticity_compute_rotations(fluid, fluid)
    elasticity_compute_rotations(bound, bound)
    elasticity_compute_stress(fluid, fluid)
    elasticity_compute_stress(bound, bound)
    elasticity_compute_force(fluid, fluid)
    elasticity_compute_force(bound, bound)

    ''' solve linear system'''
    SPH_advection_elasticity_rhs(fluid)  # compute right hand side vector b
    SPH_advection_elasticity_rhs(bound)





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
    elasticity_step(fluid, bound)

    SPH_advection_update_vel_adv(fluid)




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











