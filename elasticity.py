import numpy.linalg

from ti_sph.func_spline_kernel import *
from taichi.lang.ops import atomic_min

#################################### init ###################################
@ti.kernel
def init_elasticity_values(obj: ti.template()):
    for i in range(obj.info.stack_top[None]):
        obj.elasticity.pos0[i] = obj.basic.pos[i]
        obj.elasticity.R[i] = ti.Matrix.identity(float, 3)


@ti.kernel
def init_neighbors(obj: ti.template(), nobj: ti.template(), config_discre: ti.template(), config_neighb: ti.template()):
    for i in range(obj.info.stack_top[None]):
        cell_vec = ti.static(obj.located_cell.vec)
        for cell_tpl in range(config_neighb.search_template.shape[0]):
            cell_coded = (cell_vec[i] + config_neighb.search_template[cell_tpl]).dot(config_neighb.cell_coder[None])
            if 0 < cell_coded < config_neighb.cell_num[None]:
                for j in range(nobj.cell.part_count[cell_coded]):
                    shift = nobj.cell.part_shift[cell_coded] + j
                    nid = nobj.located_cell.part_log[shift]
                    xij = obj.basic.pos[i] - nobj.basic.pos[nid]
                    r = xij.norm()
                    if 0 < r < config_discre.kernel_h[None]:
                        obj.elasticity_neighbor.neighbors_initial[i, obj.elasticity.neighbors_num[i]] = nid
                        obj.elasticity.neighbors_num[i] += 1


def test_init_neighbor(obj: ti.template(), nobj: ti.template(), i: ti.i32):
    for n in range(obj.elasticity.neighbors_num[i]):
        j = obj.elasticity_neighbor.neighbors_initial[i, n]
        nobj.color.vec[j] = [0, 0, 1]
    obj.color.vec[i] = [1, 0, 0]


@ti.kernel
def compute_L(obj: ti.template(), nobj: ti.template(), config_discre: ti.template(), config_space: ti.template()):
    for i in range(obj.info.stack_top[None]):
        for n in range(obj.elasticity.neighbors_num[i]):
            j = obj.elasticity_neighbor.neighbors_initial[i, n]
            xij0 = obj.elasticity.pos0[i] - nobj.elasticity.pos0[j]
            r0 = xij0.norm()
            grad_W0 = grad_spline_W(r0, config_discre.kernel_h[None], config_discre.kernel_sig3d[None]) * xij0 / r0
            obj.elasticity.L[i] -= nobj.basic.rest_volume[j] * grad_W0 @ xij0.transpose()

        if config_space.dim[None] == 2:
            obj.elasticity.L[i][2, 2] = 1.0

        # if obj.elasticity.L[i].determinant() != 0:
        #     obj.elasticity.L[i] = obj.elasticity.L[i].inverse()
        # else:
        #     obj.elasticity.L[i] = ti.Matrix.zero(float, 3, 3)
        #     print("Irreversible")


def inverse_L(obj: ti.template()):
    L = obj.elasticity.L.to_numpy()
    for i in range(obj.info.stack_top[None]):
        if numpy.linalg.det(L[i]) != 0:
            L[i] = numpy.linalg.inv(L[i])
        else:
            L[i] = numpy.linalg.pinv(L[i])  # Compute the (Moore-Penrose) pseudo-inverse of a matrix.
            print("Irreversible")
    obj.elasticity.L.from_numpy(L)


def init_L(obj, nobj, config_discre, config_space):
    compute_L(obj, nobj, config_discre, config_space)
    inverse_L(obj)


#################################### elasticity_compute ###################################
@ti.kernel
def elasticity_compute_rotation(obj: ti.template(), nobj: ti.template(), config_discre: ti.template()):
    for i in range(obj.info.stack_top[None]):
        temp_F = ti.Matrix.zero(float, 3, 3)
        for n in range(obj.elasticity.neighbors_num[i]):
            j = obj.elasticity_neighbor.neighbors_initial[i, n]
            xij0 = obj.elasticity.pos0[i] - nobj.elasticity.pos0[j]
            r0 = xij0.norm()
            grad_W0 = grad_spline_W(r0, config_discre.kernel_h[None], config_discre.kernel_sig3d[None]) * xij0 / r0
            LW0 = obj.elasticity.L[i] @ grad_W0

            xji = nobj.basic.pos[j] - obj.basic.pos[i]
            temp_F += nobj.basic.rest_volume[j] * xji @ LW0.transpose()
            # print(xij0, r0, grad_W0, LW0, xji, temp_F)
            
        U, sig, V = ti.svd(temp_F)
        obj.elasticity.R[i] = U @ V.transpose()
        obj.elasticity.RL[i] = obj.elasticity.R[i] @ obj.elasticity.L[i]
        # print(obj.elasticity.R[i], obj.elasticity.RL[i])


@ti.kernel
def elasticity_compute_stress(obj: ti.template(), nobj: ti.template(), config_discre: ti.template(), config_elasticity: ti.template()):
    for i in range(obj.info.stack_top[None]):
        for n in range(obj.elasticity.neighbors_num[i]):
            j = obj.elasticity_neighbor.neighbors_initial[i, n]
            xij0 = obj.elasticity.pos0[i] - nobj.elasticity.pos0[j]
            r0 = xij0.norm()
            grad_W0 = grad_spline_W(r0, config_discre.kernel_h[None], config_discre.kernel_sig3d[None]) * xij0 / r0
            RLW0 = obj.elasticity.RL[i] @ grad_W0

            xji = nobj.basic.pos[j] - obj.basic.pos[i]
            obj.elasticity.F[i] += nobj.basic.rest_volume[j] * xji @ RLW0.transpose()

        obj.elasticity.strain[i] = 0.5 * (obj.elasticity.F[i] + obj.elasticity.F[i].transpose()) - ti.Matrix.identity(float, 3)
        obj.elasticity.stress[i] = 2 * config_elasticity.lame_mu[None] * obj.elasticity.strain[i] + config_elasticity.lame_lambda[None] * obj.elasticity.strain[i].trace() * ti.Matrix.identity(float, 3)
        # todo strain=0, stress!=0  precision problem


@ti.kernel
def elasticity_compute_force(obj: ti.template(), nobj: ti.template(), config_discre: ti.template()):
    for i in range(obj.info.stack_top[None]):
        for n in range(obj.elasticity.neighbors_num[i]):
            j = obj.elasticity_neighbor.neighbors_initial[i, n]
            xij0 = obj.elasticity.pos0[i] - nobj.elasticity.pos0[j]
            r0 = xij0.norm()
            grad_W0_i = grad_spline_W(r0, config_discre.kernel_h[None], config_discre.kernel_sig3d[None]) * xij0 / r0
            grad_W0_j = -grad_spline_W(r0, config_discre.kernel_h[None], config_discre.kernel_sig3d[None]) * xij0 / r0
            RLW0_i = obj.elasticity.RL[i] @ grad_W0_i
            RLW0_j = nobj.elasticity.RL[j] @ grad_W0_j
            obj.elasticity.force[i] += obj.basic.rest_volume[i] * nobj.basic.rest_volume[j] * (obj.elasticity.stress[i] @ RLW0_i - nobj.elasticity.stress[j] @ RLW0_j)


@ti.kernel
def elasticity_compute_rhs(obj: ti.template(), config_discre: ti.template()):
    for i in range(obj.info.stack_top[None]):
        obj.conjugate_gradient.b[i] = obj.basic.vel[i] + config_discre.dt[None] * (obj.implicit_sph.acce_adv[i] + obj.elasticity.force[i] / obj.basic.mass[i])


@ti.func
def elasticity_iter_stress(obj: ti.template(), nobj: ti.template(), config_discre: ti.template(), config_elasticity: ti.template(), i: ti.i32):
    for n in range(obj.elasticity.neighbors_num[i]):
        j = obj.elasticity_neighbor.neighbors_initial[i, n]
        xij0 = obj.elasticity.pos0[i] - nobj.elasticity.pos0[j]
        r0 = xij0.norm()
        grad_W0 = grad_spline_W(r0, config_discre.kernel_h[None], config_discre.kernel_sig3d[None]) * xij0 / r0
        RLW0 = obj.elasticity.RL[i] @ grad_W0

        pji = nobj.conjugate_gradient.p[j] - obj.conjugate_gradient.p[i]
        obj.elasticity.grad_u[i] += nobj.basic.rest_volume[j] * pji @ RLW0.transpose()
    obj.elasticity.grad_u[i] *= config_discre.dt[None]
    obj.elasticity.strain[i] = 0.5 * (obj.elasticity.grad_u[i] + obj.elasticity.grad_u[i].transpose())
    obj.elasticity.stress_iter[i] = 2.0 * config_elasticity.lame_mu[None] * obj.elasticity.strain[i] + config_elasticity.lame_lambda[None] * obj.elasticity.strain[i].trace() * ti.Matrix.identity(float, 3)
    # todo strain=0, stress!=0
    # print(config_elasticity.lame_mu[None],  config_elasticity.lame_lambda[None])
    # print(obj.elasticity.stress[i])


@ti.func
def elasticity_iter_force(obj: ti.template(), nobj: ti.template(), config_discre: ti.template(), i: ti.i32):
    for n in range(obj.elasticity.neighbors_num[i]):
        j = obj.elasticity_neighbor.neighbors_initial[i, n]
        xij0 = obj.elasticity.pos0[i] - nobj.elasticity.pos0[j]
        r0 = xij0.norm()
        grad_W0_i = grad_spline_W(r0, config_discre.kernel_h[None], config_discre.kernel_sig3d[None]) * xij0 / r0
        grad_W0_j = -grad_spline_W(r0, config_discre.kernel_h[None], config_discre.kernel_sig3d[None]) * xij0 / r0
        RLW0_i = obj.elasticity.RL[i] @ grad_W0_i
        RLW0_j = nobj.elasticity.RL[j] @ grad_W0_j
        obj.elasticity.force_iter[i] += obj.basic.rest_volume[i] * nobj.basic.rest_volume[j] * (obj.elasticity.stress_iter[i] @ RLW0_i - nobj.elasticity.stress_iter[j] @ RLW0_j)


@ti.func
def elasticity_iter_Ap(obj: ti.template(), nobj: ti.template(), config_discre: ti.template(), config_elasticity: ti.template(), i: ti.i32):
    elasticity_iter_stress(obj, nobj, config_discre, config_elasticity, i)
    elasticity_iter_force(obj, nobj, config_discre, i)
    obj.conjugate_gradient.Ap[i] = obj.conjugate_gradient.p[i] - (config_discre.dt[None] / obj.basic.mass[i]) * obj.elasticity.force_iter[i]


@ti.kernel
def elasticity_iter(obj: ti.template(), nobj: ti.template(), config_discre: ti.template(), config_elasticity: ti.template()):
    for i in range(obj.info.stack_top[None]):
        obj.conjugate_gradient.r[i] = obj.conjugate_gradient.b[i]
        obj.conjugate_gradient.p[i] = obj.conjugate_gradient.r[i]
        iter = 0
        while iter < 100:
            iter += 1
            elasticity_iter_Ap(obj, nobj, config_discre, config_elasticity, i)
            rTr = obj.conjugate_gradient.r[i].dot(obj.conjugate_gradient.r[i])
            pTAp = obj.conjugate_gradient.p[i].dot(obj.conjugate_gradient.Ap[i])
            alpha = rTr/pTAp
            obj.elasticity.vel_elast[i] += alpha * obj.conjugate_gradient.p[i]
            obj.conjugate_gradient.r[i] -= alpha * obj.conjugate_gradient.Ap[i]
            if obj.conjugate_gradient.r[i].norm() < 1e-3:
                break
            new_rTr = obj.conjugate_gradient.r[i].dot(obj.conjugate_gradient.r[i])
            beta = new_rTr / rTr
            obj.conjugate_gradient.p[i] = obj.conjugate_gradient.r[i] + beta * obj.conjugate_gradient.p[i]

        print("------------",i,"------------", iter)


@ti.kernel
def elasticity_zero_energy(obj: ti.template(), nobj: ti.template(), config_discre: ti.template(), config_elasticity: ti.template()):
    for i in range(obj.info.stack_top[None]):
        fi_hg = ti.Vector([0.0, 0.0, 0.0])
        for n in range(obj.elasticity.neighbors_num[i]):
            j = obj.elasticity_neighbor.neighbors_initial[i, n]
            xji0 = nobj.elasticity.pos0[j] - obj.elasticity.pos0[i]
            r0 = xji0.norm()
            xji = nobj.basic.pos[j] - obj.basic.pos[i]
            r = xji.norm()
            epsilon_i = obj.elasticity.F[i] @ obj.elasticity.R[i] @ xji0 - xji
            epsilon_j = -nobj.elasticity.F[j] @ nobj.elasticity.R[j] @ xji0 + xji
            delta_i = epsilon_i.dot(xji) / r
            delta_j = -epsilon_j.dot(xji) / r
            W = spline_W(r0, config_discre.kernel_h[None], config_discre.kernel_sig3d[None])
            fi_hg -= nobj.basic.rest_volume[j] * (W / (r0 ** 2)) * (delta_i + delta_j) * xji / r

        fi_hg *= config_elasticity.alpha[None] * config_elasticity.youngs_modulus[None] * obj.basic.rest_volume[i]
        obj.implicit_sph.acce_adv[i] += fi_hg / obj.basic.mass[i]


@ti.kernel
def elasticity_clean_value(obj: ti.template()):
    for i in range(obj.info.stack_top[None]):
        obj.elasticity.R[i] = ti.Matrix.identity(float, 3)
        obj.elasticity.RL[i] = ti.Matrix.zero(float, 3, 3)
        obj.elasticity.F[i] = ti.Matrix.zero(float, 3, 3)
        obj.elasticity.strain[i] = ti.Matrix.zero(float, 3, 3)
        obj.elasticity.stress[i] = ti.Matrix.zero(float, 3, 3)

        obj.elasticity.force[i] = [0.0, 0.0, 0.0]
        obj.elasticity.vel_elast[i] = [0.0, 0.0, 0.0]

        obj.elasticity.grad_u[i] = ti.Matrix.zero(float, 3, 3)
        obj.elasticity.stress_iter[i] = ti.Matrix.zero(float, 3, 3)
        obj.elasticity.force_iter[i] = [0.0, 0.0, 0.0]

        obj.conjugate_gradient.b[i] = [0.0, 0.0, 0.0]
        obj.conjugate_gradient.p[i] = [0.0, 0.0, 0.0]
        obj.conjugate_gradient.Ap[i] = [0.0, 0.0, 0.0]


@ti.kernel
def basic_clean_value(obj: ti.template()):
    for i in range(obj.info.stack_top[None]):
        obj.implicit_sph.acce_adv[i] = [0.0, 0.0, 0.0]


@ti.kernel
def elasticity_acc(obj: ti.template(), config_discre: ti.template()):
    for i in range(obj.info.stack_top[None]):
        obj.implicit_sph.acce_adv[i] += (obj.elasticity.vel_elast[i] - obj.basic.vel[i]) / config_discre.dt[None]


@ti.kernel
def SPH_advection_gravity_acc(obj: ti.template(), config_sim: ti.template()):
    for i in range(obj.info.stack_top[None]):
        obj.implicit_sph.acce_adv[i] += config_sim.gravity[None]


@ti.kernel
def SPH_update_pos(obj: ti.template(), config_discre: ti.template()):
    for i in range(obj.info.stack_top[None]):
        obj.basic.vel[i] += obj.implicit_sph.acce_adv[i] * config_discre.dt[None]
        obj.basic.pos[i] += obj.basic.vel[i] * config_discre.dt[None]


@ti.kernel
def cfl_condition(obj: ti.template(), config_discre: ti.template()):
    config_discre.dt[None] = config_discre.part_size[None] / config_discre.cs[None]
    for i in range(obj.info.stack_top[None]):
        v_norm = obj.basic.vel[i].norm()
        if v_norm > 1e-4:
            atomic_min(config_discre.dt[None], config_discre.part_size[None] / v_norm * config_discre.cfl_factor[None])


def elasticity_step(fluid, elastomer, config_discre, config_sim, config_elasticity, config_neighb, config_space):
    cfl_condition(elastomer, config_discre)
    elasticity_clean_value(elastomer)
    basic_clean_value(fluid)
    basic_clean_value(elastomer)

    elastomer.neighb_search(config_neighb, config_space)
    fluid.neighb_search(config_neighb, config_space)

    SPH_advection_gravity_acc(fluid, config_sim)

    elasticity_compute_rotation(elastomer, elastomer, config_discre)
    elasticity_compute_stress(elastomer, elastomer, config_discre, config_elasticity)
    elasticity_compute_force(elastomer, elastomer, config_discre)
    elasticity_zero_energy(elastomer, elastomer, config_discre, config_elasticity)
    elasticity_compute_rhs(elastomer, config_discre)
    elasticity_iter(elastomer, elastomer, config_discre, config_elasticity)

    elasticity_acc(elastomer, config_discre)
    for i in range(elastomer.info.stack_top[None]):
        print("acc", elastomer.implicit_sph.acce_adv[i])

    SPH_update_pos(elastomer, config_discre)

    SPH_update_pos(fluid, config_discre)


def run_elasticity_step(fluid, elastomer, config_discre, config_sim, config_elasticity, config_neighb, config_space):
    # while True:
    elasticity_step(fluid, elastomer, config_discre, config_sim, config_elasticity, config_neighb, config_space)

