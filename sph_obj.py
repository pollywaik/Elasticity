from taichi.lang.ops import atomic_add, sqrt
from sph_util import *

obj_list = []

@ti.data_oriented
class Fluid:
    def __init__(self, max_part_num, pre_config, config):

        obj_list.append(self)

        self.max_part_num = max_part_num
        self.part_num = ti.field(int, ())
        self.uid = len(obj_list)  # uid of the Fluid object
        # utils
        self.ones = ti.field(int)
        self.flag = ti.field(int)  # ? OBSOLETE
        self.general_flag = ti.field(int, ())  # ? OBSOLETE todo
        self.pushed_part_seq = ti.Vector.field(config.dim[None], int, ())  # ? OBSOLETE todo
        self.pushed_part_seq_coder = ti.field(int, config.dim[None])  # ? OBSOLETE todo

        # Physical properties of particles
        self.color = ti.field(int)
        self.color_vector = ti.Vector.field(3, float)  # for ggui to show
        self.mass = ti.field(float)
        self.rest_density = ti.field(float)
        self.rest_volume = ti.field(float)
        self.pressure = ti.field(float)
        self.pressure_force = ti.Vector.field(config.dim[None], float)
        self.volume_frac = ti.Vector.field(config.phase_num[None], float)
        self.volume_frac_tmp = ti.Vector.field(config.phase_num[None], float)
        self.pos = ti.Vector.field(config.dim[None], float)  # position
        self.gui_2d_pos = ti.Vector.field(config.dim[None], float)  # for ggui to show
        self.vel = ti.Vector.field(config.dim[None], float)  # velocity
        self.vel_adv = ti.Vector.field(config.dim[None], float)
        self.acce = ti.Vector.field(config.dim[None], float)  # acceleration
        self.acce_adv = ti.Vector.field(config.dim[None], float)

        # energy
        self.statistics_kinetic_energy = ti.field(float, ())  # total kinetic energy of particles
        self.statistics_gravity_potential_energy = ti.field(float, ())  # total gravitational potential energy of particles

        # for slover
        self.W = ti.field(float)
        self.W_grad = ti.Vector.field(config.dim[None], float)
        self.compression = ti.field(float, ())  # compression rate gamma for [VFSPH]
        self.sph_compression = ti.field(float)  # diff from compression?
        self.sph_density = ti.field(float)  # density computed from sph approximation
        self.psi_adv = ti.field(float)
        self.alpha = ti.field(float)  # alpha for [DFSPH] and [VFSPH]
        self.alpha_1 = ti.Vector.field(config.dim[None], float)  # 1st term of alpha
        self.alpha_2 = ti.field(float)  # 2nd term of alpha
        self.drift_vel = ti.Vector.field(config.dim[None], float)
        self.phase_vel = ti.Vector.field(config.dim[None], float)
        self.phase_acc = ti.Vector.field(config.dim[None], float)
        # FBM
        self.fbm_zeta = ti.field(float)
        self.fbm_acce = ti.static(self.acce)
        self.normal = ti.Vector.field(config.dim[None], float)  # surface normal in [AKINCI12] for computing curvature force in surface tension

        # neighb
        self.neighb_cell_seq = ti.field(int)  # the seq of the grid which particle is located
        self.neighb_in_cell_seq = ti.field(int)  # the seq of the particle in the grid
        self.neighb_cell_structured_seq = ti.Vector.field(config.dim[None], int)  # the structured seq of the grid

        # [VFSPH] and [DFSPH] use the same framework, aliases for interchangeable variables
        if pre_config.solver_type == 'VFSPH':
            self.X = ti.static(self.rest_volume)
            self.sph_psi = ti.static(self.sph_compression)
            self.rest_psi = ti.static(self.ones)
        elif pre_config.solver_type == 'DFSPH':
            self.X = ti.static(self.mass)
            self.sph_psi = ti.static(self.sph_density)
            self.rest_psi = ti.static(self.rest_density)

        # put for-each-particle attributes in this list to register them!
        self.attr_list = [self.color, self.color_vector, self.mass, self.rest_density, self.rest_volume, self.pressure,self.pressure_force,
                          self.volume_frac, self.volume_frac_tmp, self.pos, self.gui_2d_pos, self.vel, self.vel_adv,self.acce, self.acce_adv,
                          self.W, self.W_grad, self.sph_density, self.sph_compression, self.psi_adv, self.alpha, self.alpha_1, self.alpha_2, self.fbm_zeta, self.normal,
                          self.neighb_cell_seq, self.neighb_in_cell_seq, self.neighb_cell_structured_seq, self.ones,self.flag]

        # allocate memory for attributes (1-D fields)
        for attr in self.attr_list:
            ti.root.dense(ti.i, self.max_part_num).place(attr)  # SOA(see Taichi advanced layout: https://docs.taichi.graphics/docs/lang/articles/advanced/layout#from-shape-to-tirootx)
        # allocate memory for drift velocity (2-D field)
        ti.root.dense(ti.i, self.max_part_num).dense(ti.j, config.phase_num[None]).place(self.drift_vel)
        self.attr_list.append(self.drift_vel)  # add drift velocity to attr_list
        ti.root.dense(ti.i, self.max_part_num).dense(ti.j, config.phase_num[None]).place(self.phase_vel)
        self.attr_list.append(self.phase_vel)  # add drift velocity to attr_list
        ti.root.dense(ti.i, self.max_part_num).dense(ti.j, config.phase_num[None]).place(self.phase_acc)
        self.attr_list.append(self.phase_acc)  # add drift velocity to attr_list

        self.init()

    def init(self):
        for attr in self.attr_list:
            attr.fill(0)
        self.ones.fill(1)

    # update mass for volume fraction multiphase
    @ti.kernel
    def update_mass(self, config: ti.template()):
        for i in range(self.part_num[None]):
            self.mass[i] = config.phase_rest_density[None].dot(self.volume_frac[i])

    # helper function for scene_add functions
    def scene_add_help_centering(self, start_pos, end_pos, spacing):
        end_pos = np.array(end_pos, dtype=np.float32)
        start_pos = np.array(start_pos, dtype=np.float32)
        matrix_shape = ((end_pos - start_pos + 1e-7) / spacing).astype(np.int32)
        padding = (end_pos - start_pos - matrix_shape * spacing) / 2
        return matrix_shape, padding

    # add n dimension cube to scene
    def scene_add_cube(self, start_pos, end_pos, volume_frac, vel, color,
                       relaxing_factor, config):  # add relaxing factor for each cube
        spacing = config.part_size[1] * relaxing_factor
        matrix_shape, padding = self.scene_add_help_centering(start_pos, end_pos, spacing)
        self.push_matrix(np.ones(matrix_shape, dtype=np.bool_), start_pos + padding, spacing, volume_frac, vel, color, config)

    # add 3D or 2D hollow box to scene, with several layers
    def scene_add_box(self, start_pos, end_pos, layers, volume_frac, vel, color, relaxing_factor, config):
        spacing = config.part_size[1] * relaxing_factor
        matrix_shape, padding = self.scene_add_help_centering(start_pos, end_pos, spacing)
        box = np.ones(matrix_shape, dtype=np.bool_)
        if len(matrix_shape) == 2:
            box[layers: matrix_shape[0] - layers, layers: matrix_shape[1] - layers] = False
        elif len(matrix_shape) == 3:
            box[layers: matrix_shape[0] - layers, layers: matrix_shape[1] - layers,
            layers: matrix_shape[2] - layers] = False
        else:
            raise Exception('scenario error: can only add 2D or 3D boxes')
        self.push_matrix(box, start_pos + padding, spacing, volume_frac, vel, color, config)

    def push_part_from_ply(self, p_sum, pos_seq, volume_frac, vel, color, config):
        self.push_part_seq(p_sum, color, pos_seq, ti.Vector(volume_frac), ti.Vector(vel), config)

    # add particles according to true and false in the matrix
    # matrix: np array (dimension: dim, dtype: np.bool)
    def push_matrix(self, matrix, start_position, spacing, volume_frac, vel, color, config):
        if len(matrix.shape) != config.dim[None]:
            raise Exception('push_matrix() [scenario error]: wrong object dimension')
        index = np.where(matrix == True)
        pos_seq = np.stack(index, axis=1) * spacing + start_position
        self.push_part_seq(len(pos_seq), color, pos_seq, ti.Vector(volume_frac), ti.Vector(vel), config)


    @ti.kernel
    def push_pos_seq(self, pos_seq: ti.template(),pushed_part_num: int, current_part_num: int, config: ti.template()):
        dim = ti.static(config.gravity.n)
        for i in range(pushed_part_num):
            i_p = i + current_part_num
            for j in ti.static(range(dim)):
                self.pos[i_p][j] = pos_seq[i][j]


    @ti.kernel
    def push_attrs_seq(self, color: int, volume_frac: ti.template(), vel: ti.template(), pushed_part_num: int, current_part_num: int, config: ti.template()):
        for i in range(pushed_part_num):
            i_p = i + current_part_num
            self.volume_frac[i_p] = volume_frac
            self.vel[i_p] = vel
            self.rest_volume[i_p] = config.part_size[config.dim[None]]  # todo 1
            self.color[i_p] = color
            self.color_vector[i_p] = hex2rgb(color)
            self.rest_density[i_p] = config.phase_rest_density[None].dot(self.volume_frac[i_p])
            self.mass[i_p] = self.rest_density[i_p] * self.rest_volume[i_p]


    def push_part_seq(self, pushed_part_num, color, pos_seq, volume_frac, vel, config):
        print('push ',pushed_part_num, ' particles')
        current_part_num = self.part_num[None]
        new_part_num = current_part_num + pushed_part_num
        pos_seq_ti = ti.Vector.field(config.dim[None], float, pushed_part_num)
        pos_seq_ti.from_numpy(pos_seq)
        self.push_pos_seq(pos_seq_ti, pushed_part_num, current_part_num, config)
        self.push_attrs_seq(color, volume_frac, vel, pushed_part_num, current_part_num, config)
        self.part_num[None] = new_part_num


    @ti.kernel
    def push_cube(self, lb: ti.template(), rt: ti.template(), mask: ti.template(), volume_frac: ti.template(),
                  color: int, relaxing_factor: ti.template(), config: ti.template()):
        current_part_num = self.part_num[None]
        # generate seq (number of particles to push for each dimension)
        self.pushed_part_seq[None] = int(ti.ceil((rt - lb) / config.part_size[1] / relaxing_factor))
        self.pushed_part_seq[None] *= mask
        dim = ti.static(config.gravity.n)
        for i in ti.static(range(dim)):
            if self.pushed_part_seq[None][i] == 0:
                self.pushed_part_seq[None][i] = 1  # at least push one
        # coder for seq
        tmp = 1
        for i in ti.static(range(dim)):
            self.pushed_part_seq_coder[i] = tmp
            tmp *= self.pushed_part_seq[None][i]
        # new part num
        pushed_part_num = 1
        for i in ti.static(range(dim)):
            pushed_part_num *= self.pushed_part_seq[None][i]
        new_part_num = current_part_num + pushed_part_num
        # inject pos [1/2]
        for i in range(pushed_part_num):
            tmp = i
            for j in ti.static(range(dim - 1, -1, -1)):
                self.pos[i + current_part_num][j] = tmp // self.pushed_part_seq_coder[j]
                tmp = tmp % self.pushed_part_seq_coder[j]
        # inject pos [2/2]
        # pos seq times part size minus lb
        for i in range(pushed_part_num):
            self.pos[i + current_part_num] *= config.part_size[1] * relaxing_factor
            self.pos[i + current_part_num] += lb
        # inject volume_frac & rest_volume & color
        for i in range(pushed_part_num):
            self.volume_frac[i + current_part_num] = volume_frac
            self.rest_volume[i + current_part_num] = config.part_size[config.dim[None]]
            self.color[i + current_part_num] = color
        # update part num
        self.part_num[None] = new_part_num
        # update mass and rest_density
        for i in range(self.part_num[None]):
            self.rest_density[i] = config.phase_rest_density[None].dot(self.volume_frac[i])
            self.mass[i] = self.rest_density[i] * self.rest_volume[i]

    def inc_unit(self, seq, length, lim, cur_dim):
        for i in range(length):
            if not seq[cur_dim][i] < lim[cur_dim]:
                seq[cur_dim + 1][i] = seq[cur_dim][i] // lim[cur_dim]
                seq[cur_dim][i] = seq[cur_dim][i] % lim[cur_dim]

    def push_2d_cube(self, center_pos, size, volume_frac, color: int, relaxing_factor, config, layer=0):
        lb = -np.array(size) / 2 + np.array(center_pos)
        rt = np.array(size) / 2 + np.array(center_pos)
        mask = np.ones(config.dim[None], np.int32)
        if layer == 0:
            self.push_cube(ti.Vector(lb), ti.Vector(rt), ti.Vector(mask), ti.Vector(volume_frac), color)
        elif layer > 0:
            cube_part = np.zeros(config.dim[None], np.int32)
            cube_part[:] = np.ceil(np.array(size) / config.part_size[1] / relaxing_factor)[:]
            for i in range(cube_part.shape[0]):
                if cube_part[i] < layer * 2:
                    layer = int(np.floor(cube_part[i] / 2))
            sum = int(1)
            for i in range(cube_part.shape[0]):
                sum *= cube_part[i]
            np_pos_seq = np.zeros(shape=(config.dim[None] + 1, sum), dtype=np.int32)
            counter = int(0)
            for i in range(sum):
                np_pos_seq[0][i] = counter
                counter += 1
            for i in range(0, config.dim[None] - 1):
                self.inc_unit(np_pos_seq, sum, cube_part, i)
            p_sum = int(0)
            for i in range(layer):
                for j in range(config.dim[None]):
                    for k in range(sum):
                        if (np_pos_seq[j][k] == (0 + i) or np_pos_seq[j][k] == (cube_part[j] - i - 1)) and \
                                np_pos_seq[config.dim[None]][k] == 0:
                            np_pos_seq[config.dim[None]][k] = 1
                            p_sum += 1
            pos_seq = np.zeros((p_sum, config.dim[None]), np.float32)
            counter = int(0)
            for i in range(sum):
                if np_pos_seq[config.dim[None]][i] > 0:
                    pos_seq[counter][:] = np_pos_seq[0:config.dim[None], i]
                    counter += 1
            pos_seq *= config.part_size[1] * relaxing_factor
            pos_seq -= (np.array(center_pos) + np.array(size) / 2)
            self.push_part_seq(p_sum, color, pos_seq, ti.Vector(volume_frac), config)

    def push_part_from_ply(self, scenario_buffer, obj_name, config):
        for obj in scenario_buffer:
            if (obj == obj_name):
                for param in scenario_buffer[obj]['objs']:
                    if param['type'] == 'cube':
                        self.scene_add_cube(param['start_pos'], param['end_pos'], param['volume_frac'], param['vel'],
                                            int(param['color'], 16), param['particle_relaxing_factor'], config)
                    elif param['type'] == 'box':
                        self.scene_add_box(param['start_pos'], param['end_pos'], param['layers'], param['volume_frac'],
                                            param['vel'], int(param['color'], 16), param['particle_relaxing_factor'], config)
                    elif param['type'] == 'ply':
                        verts = read_ply(trim_path_dir(param['file_name']))
                        self.push_part_seq(len(verts), int(param['color'], 16), verts, ti.Vector(param['volume_frac']), ti.Vector(param['vel']),
                                                config)
        
        set_unused_par(self, config)

    @ti.kernel
    def update_color_vector_from_color(self):
        for i in range(self.part_num[None]):
            color = hex2rgb(self.color[i])
            self.color_vector[i] = color

class Part_buffer:
    def __init__(self, part_num, config):
        self.rest_volume = np.zeros(shape=part_num, dtype=np.float32)
        self.volume_frac = np.zeros(shape=(config.phase_num[None], part_num), dtype=np.float32)
        self.pos = np.zeros(shape=(config.dim[None], part_num), dtype=np.float32)

@ti.data_oriented
class Ngrid:
    def __init__(self, config):
        self.node_part_count = ti.field(int)
        self.node_part_shift = ti.field(int)
        self.node_part_shift_count = ti.field(int)
        self.part_pid_in_node = ti.field(int)
        self.part_uid_in_node = ti.field(int)

        ti.root.dense(ti.i, config.node_num[None]).place(self.node_part_count)
        ti.root.dense(ti.i, config.node_num[None]).place(self.node_part_shift)
        ti.root.dense(ti.i, config.node_num[None]).place(self.node_part_shift_count)
        ti.root.dense(ti.i, config.max_part_num[None]).place(self.part_pid_in_node)
        ti.root.dense(ti.i, config.max_part_num[None]).place(self.part_uid_in_node)

    @ti.kernel
    def clear_node(self, config: ti.template()):
        for i in range(config.node_num[None]):
            self.node_part_count[i] = 0

    @ti.kernel
    def encode(self, obj: ti.template(), config: ti.template()):
        for i in range(obj.part_num[None]):
            obj.neighb_cell_structured_seq[i] = node_encode(obj.pos[i], config)
            obj.neighb_cell_seq[i] = dim_encode(obj.neighb_cell_structured_seq[i], config)
            if 0 < obj.neighb_cell_seq[i] < config.node_num[None]:
                ti.atomic_add(self.node_part_count[obj.neighb_cell_seq[i]], 1)

    @ti.kernel
    def mem_shift(self, config: ti.template()):
        sum = ti.Vector([0])
        for i in range(config.node_num[None]):
            self.node_part_shift[i] = ti.atomic_add(
                sum[0], self.node_part_count[i])
            self.node_part_shift_count[i] = self.node_part_shift[i]

    @ti.kernel
    def fill_node(self, obj: ti.template(), config: ti.template()):
        for i in range(obj.part_num[None]):
            if 0 < obj.neighb_cell_seq[i] < config.node_num[None]:
                obj.neighb_in_cell_seq[i] = atomic_add(
                    self.node_part_shift_count[obj.neighb_cell_seq[i]], 1)
                self.part_pid_in_node[obj.neighb_in_cell_seq[i]] = i
                self.part_uid_in_node[obj.neighb_in_cell_seq[i]] = obj.uid


