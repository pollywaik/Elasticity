import taichi as ti
import ti_sph as tsph
from ti_sph.class_config import Neighb_cell
from ti_sph.class_node import test
import math
from elasticity import *
ti.init()

# CONFIG
config_capacity = ['info_space', 'info_discretization',
                   'info_sim', 'info_gui', 'info_elasticity']
config = tsph.Config(dim=3, capacity_list=config_capacity)
# space
config_space = ti.static(config.space)
config_space.dim[None] = 3
config_space.lb.fill(-8)
config_space.lb[None][1] = -4
config_space.rt.fill(8)
# discretization
config_discre = ti.static(config.discre)
config_discre.part_size[None] = 0.1
config_discre.cs[None] = 100
config_discre.cfl_factor[None] = 0.2
config_discre.dt[None] = tsph.fixed_dt(
    config_discre.cs[None], config_discre.part_size[None], config_discre.cfl_factor[None])
config_discre.kernel_h[None] = config_discre.part_size[None] * 2
config_discre.kernel_sig3d[None] = 8 / math.pi / config_discre.kernel_h[None] ** 3
# gui
config_gui = ti.static(config.gui)
config_gui.res[None] = [1920, 1080]
config_gui.frame_rate[None] = 60
config_gui.cam_fov[None] = 55
config_gui.cam_pos[None] = [6.0, 1.0, 0.0]
config_gui.cam_look[None] = [0.0, 0.0, 0.0]
config_gui.canvas_color[None] = [0.2, 0.2, 0.6]
config_gui.ambient_light_color[None] = [0.7, 0.7, 0.7]
config_gui.point_light_pos[None] = [2, 1.5, -1.5]
config_gui.point_light_color[None] = [0.8, 0.8, 0.8]
# sim
config_sim = ti.static(config.sim)
config_sim.gravity[None] = [0.0, -9.8, 0.0]
# elasticity
config_elasticity = ti.static(config.elasticity)
config_elasticity.youngs_modulus[None] = 25000.0
config_elasticity.poisson_ratio[None] = 0.33
config_elasticity.lame_mu[None] = config_elasticity.youngs_modulus[None] / (2.0 * (1.0 + config_elasticity.poisson_ratio[None]))  # μ =E/2(1+ν)
config_elasticity.lame_lambda[None] = config_elasticity.youngs_modulus[None] * config_elasticity.poisson_ratio[None] / (
                (1.0 + config_elasticity.poisson_ratio[None]) * (1.0 - 2.0 * config_elasticity.poisson_ratio[None]))  # λ=Eν/(1+ν)(1−2ν)
config_elasticity.alpha[None] = 25
# NEIGHB
config_neighb = Neighb_cell(dim=3, struct_space=config_space,
                          cell_size=config_discre.part_size[None] * 4, search_range=1)

# FLUID
fluid_capacity = ["node_basic", 'node_color',
                  'node_implicit_sph', 'node_neighb_search']
fluid = tsph.Node(dim=config_space.dim[None], id=0, part_num=int(1e5),
                  neighb_cell_num=config_neighb.cell_num[None], capacity_list=fluid_capacity)
actual_sum = fluid.push_cube(ti.Vector([-0.2, 0.4, -0.2]), ti.Vector([0.2, 0.8, 0.2]), config_discre.part_size[None], 1)
fluid.color.vec.fill(ti.Vector([1, 1, 0]))

# BOUND
# bound_capacity = ["node_basic", 'node_color',
#                   'node_implicit_sph', 'node_neighb_search']
# bound = tsph.Node(dim=config_space.dim[None], id=0, part_num=int(1e5),
#                   neighb_cell_num=config_neighb.cell_num[None], capacity_list=bound_capacity)
# bound.color.vec.fill(ti.Vector([0.5, 0.5, 0.5]))
# bound.push_box(ti.Vector([-1.5, -1.5, -1.5]),
#                ti.Vector([1.5, 1, 1.5]), config_discre.part_size[None], 1, 2)

# Elastomer
elastomer_capacity = ["node_basic", 'node_color', 'node_implicit_sph', 'node_neighb_search',
                      'node_elasticity', 'node_elasticity_neighbor', 'node_conjugate_gradient']
elastomer = tsph.Node(dim=config_space.dim[None], id=0, part_num=int(1e5),
                  neighb_cell_num=config_neighb.cell_num[None], capacity_list=elastomer_capacity)
elastomer.push_cube(ti.Vector([-1, -0.1, -1]), ti.Vector([1., 0., 1.]), config_discre.part_size[None], 1)
# elastomer.push_cube(ti.Vector([-0.2, -0.2, -0.2]), ti.Vector([0., 0., 0.]), config_discre.part_size[None], 1)
# elastomer.push_box(ti.Vector([-0.5, -0.5, -0.5]), ti.Vector([0., 0., 0.]), config_discre.part_size[None], 1, 2)
elastomer.color.vec.fill(ti.Vector([0.5, 0.5, 0.5]))
elastomer.neighb_search(config_neighb, config_space)



for i in range(elastomer.info.stack_top[None]):
    elastomer.implicit_sph.acce_adv[i] = [0, -9.8, 0]
    elastomer.basic.rest_density[i] = 100.0
    elastomer.basic.mass[i] = elastomer.basic.rest_density[i] * elastomer.basic.rest_volume[i]

init_elasticity_values(elastomer)
elastomer.neighb_search(config_neighb, config_space)
init_neighbors(elastomer, elastomer, config_discre, config_neighb)
init_L(elastomer, elastomer, config_discre, config_space)

# print(elastomer.basic.pos)
# elasticity_step(elastomer, config_discre, config_sim, config_elasticity)

# GUI
gui = tsph.Gui(config.gui)
gui.env_set_up()
while gui.window.running:
    if gui.op_system_run == True:
        # run_elasticity_step(fluid, elastomer, config_discre, config_sim, config_elasticity, config_neighb, config_space)
        pass
    gui.monitor_listen()
    if gui.op_refresh_window:
        gui.scene_setup()
        gui.scene_add_parts(fluid, length=config_discre.part_size[None])
        gui.scene_add_parts(elastomer, length=config_discre.part_size[None])
        # gui.scene_add_parts(bound, length=config_discre.part_size[None])
        gui.scene_render()


# print(config.space)
# print(config.discre)
# print(config.neighb)
# print(config.sim)

