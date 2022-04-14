from turtle import shape
import taichi as ti


def struct_node_basic(dim, node_num):
    struct_node_basic = ti.types.struct(
        pos=ti.types.vector(dim, ti.f32),
        vel=ti.types.vector(dim, ti.f32),
        acc=ti.types.vector(dim, ti.f32),
        mass=ti.f32,
        rest_density=ti.f32,
        rest_volume=ti.f32,
        radius=ti.f32
    )
    return struct_node_basic.field(shape=(node_num,))


def struct_node_implicit_sph(dim, node_num):
    struct_node_implicit_sph = ti.types.struct(
        W=ti.f32,
        W_grad=ti.types.vector(dim, ti.f32),

        alpha_1=ti.types.vector(dim, ti.f32),
        alpha_2=ti.f32,

        vel_adv=ti.types.vector(dim, ti.f32),
        acce_adv=ti.types.vector(dim, ti.f32),

        approximated_compression_ratio=ti.f32,
        approximated_density=ti.f32,
        approximated_compression_ratio_adv=ti.f32,
        approximated_density_adv=ti.f32,
    )
    return struct_node_implicit_sph.field(shape=(node_num,))


def struct_node_color(node_num):
    struct_node_color = ti.types.struct(
        hex=ti.i32,
        vec=ti.types.vector(3, ti.f32),
    )
    return struct_node_color.field(shape=(node_num,))


def struct_node_neighb_search(dim, node_num):
    struct_node_neighb_search = ti.types.struct(
        vec=ti.types.vector(dim, ti.i32),
        coded=ti.i32,
        sequence=ti.i32,
        part_log=ti.i32,
    )
    return struct_node_neighb_search.field(shape=(node_num,))


def struct_node_neighb_cell(cell_num):
    struct_node_neighb_cell = ti.types.struct(
        part_count=ti.i32,
        part_shift=ti.i32,
    )
    return struct_node_neighb_cell.field(shape=(cell_num))


def struct_node_elasticity(dim, node_num):
    struct_node_elasticity = ti.types.struct(
        pos0=ti.types.vector(dim, ti.f32),
        neighbors_num=ti.i32,

        L=ti.types.matrix(dim, dim, ti.f32),
        R=ti.types.matrix(dim, dim, ti.f32),
        RL=ti.types.matrix(dim, dim, ti.f32),
        F=ti.types.matrix(dim, dim, ti.f32),
        strain=ti.types.matrix(dim, dim, ti.f32),
        stress=ti.types.matrix(dim, dim, ti.f32),

        force=ti.types.vector(dim, ti.f32),
        vel_elast=ti.types.vector(dim, ti.f32),

        grad_u=ti.types.matrix(dim, dim, ti.f32),
        stress_iter=ti.types.matrix(dim, dim, ti.f32),
        force_iter=ti.types.vector(dim, ti.f32),
    )
    return struct_node_elasticity.field(shape=(node_num,))


def struct_node_conjugate_gradient(dim, node_num):
    struct_node_conjugate_gradient = ti.types.struct(
        b=ti.types.vector(dim, ti.f32),
        r=ti.types.vector(dim, ti.f32),
        p=ti.types.vector(dim, ti.f32),
        Ap=ti.types.vector(dim, ti.f32),

    )
    return struct_node_conjugate_gradient.field(shape=(node_num,))

def struct_node_elasticity_neighbor(node_num):
    struct_node_elasticity_neighbor = ti.types.struct(
        neighbors_initial=ti.i32,
    )
    return struct_node_elasticity_neighbor.field(shape=(node_num, 35))

