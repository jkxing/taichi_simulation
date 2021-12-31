# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Based on 2D Taichi implementation by Ye Kuang (k-ye)

import taichi as ti
import numpy as np
import math
import time
from pyjet import *
from canvas import Canvas

#import open3d as o3d

ti.init(arch=ti.gpu)

base_size_factor = 400
scaling_size_factor = 1

res_3D = np.array([1, 1, 1]) * base_size_factor * scaling_size_factor
screen_res = res_3D[[0,2]].astype(np.int32) 

screen_to_world_ratio = 10.0 * scaling_size_factor
boundary = res_3D / screen_to_world_ratio
cell_size = 2.51 / scaling_size_factor
cell_recpr = 1.0 / cell_size

def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s

grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1))

dim = 3
bg_color = 0x112f41
particle_color = 0x068587
boundary_color = 0xebaca2
num_particles_x = 40
num_particles_y = 20
num_particles_z = 20
num_particles = num_particles_x * num_particles_y * num_particles_z
max_num_particles_per_cell = 200
max_num_neighbors = 200
time_delta = 1.0 / 20.0
epsilon = 1e-5
particle_radius = 3.0 
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
vorticity_epsilon = 0.01
viscosity_c = 0.01
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001
neighbor_radius = h * 1.05

poly6_factor = 315.0 / 64.0 / np.pi
spiky_grad_factor = -45.0 / np.pi

old_positions = ti.Vector(dim, dt=ti.f32)
positions = ti.Vector(dim, dt=ti.f32)
velocities = ti.Vector(dim, dt=ti.f32)
grid_num_particles = ti.var(ti.i32)
grid2particles = ti.var(ti.i32)
particle_num_neighbors = ti.var(ti.i32)
particle_neighbors = ti.var(ti.i32)
lambdas = ti.var(ti.f32)
position_deltas = ti.Vector(dim, dt=ti.f32)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector(2, dt=ti.f32)

ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)
ti.root.place(board_states)



@ti.func
def get_cell(pos):
    return (pos * cell_recpr).cast(int)

@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    # First coordinate is for the x position of the board, which only moves in x
    bmax = ti.Vector([board_states[None][0], boundary[1], boundary[2]]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p

@ti.kernel
def move_board():
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 8.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b

@ti.func
def in_grid(cell_grid):
    x = True
    for i in ti.static(range(dim)):
        if cell_grid[i]>=grid_size[i]:
            x = False
    return x

@ti.kernel
def prologue():
    for i in positions:
        old_positions[i] = positions[i]
    for i in positions:
        g = ti.Vector([0.0,0.0,-9.8])
        pos,vel = positions[i], velocities[i]
        vel += g*time_delta
        pos += vel*time_delta
        positions[i] = confine_position_to_boundary(pos)

    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in positions:
        cell = get_cell(positions[p_i])
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        if offs< max_num_particles_per_cell:
            grid2particles[cell, offs] = p_i
    
    
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i

@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result

@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result

@ti.func
def compute_scorr(pos_ji):
    x = poly6_value(pos_ji.norm(), h) / poly6_value(corr_deltaQ_coeff * h, h)
    x = x * x
    x = x * x
    return (-corrK) * x

@ti.kernel
def substep():
    for p_i in positions:
        pos_i = positions[p_i]
        grad_i = ti.Vector([0.0,0.0,0.0])
        
        sum_gradient_sqr = 0.0
        density_constraint = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j<0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            density_constraint += poly6_value(pos_ji.norm(), h)
        
        density_constraint = (mass * density_constraint / rho0) - 1.0
        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
                                                lambda_epsilon)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * spiky_gradient(pos_ji, h)

        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i
    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]


@ti.kernel
def epilogue():
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    
    for v_i in velocities:
        pos_i = positions[v_i]
        vel_i = velocities[v_i]
        v_delta_i = ti.Vector([0.0, 0.0, 0.0])

        for j in range(particle_num_neighbors[v_i]):
            p_j = particle_neighbors[v_i, j]
            pos_ji = pos_i - positions[p_j]
            vel_ij = velocities[p_j] - vel_i
            if p_j >= 0:
                pos_ji = pos_i - positions[p_j]
                v_delta_i += vel_ij * poly6_value(pos_ji.norm(), h)

        velocities[v_i] += viscosity_c * v_delta_i

def run_pbf():
    prologue()
    for _ in range(pbf_num_iters):
        substep()
    epilogue()


def init_particles():
    np_positions = np.zeros((num_particles, dim), dtype=np.float32)
    #delta = h * 0.8
    num_x = num_particles_x
    num_y = num_particles_y
    num_z = num_particles_z
    assert num_x * num_y * num_z == num_particles

    @ti.kernel
    def init_board():
        board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])
    init_board()

    for i in range(num_particles):
        #np_positions[i] = np.array([i % num_x, i // num_x]) * delta + offs
        np_positions = np.random.uniform([particle_radius_in_world, boundary[1]/2, particle_radius_in_world], 
                                        np.array([board_states[None][0]/5, boundary[1], boundary[2]/5]) - particle_radius_in_world,
                                        (num_particles,dim))
    np_velocities = (np.random.rand(num_particles, dim).astype(np.float32) -
                     0.5) * 4.0

    @ti.kernel
    def init(p: ti.ext_arr(), v: ti.ext_arr()):
        for i in range(num_particles):
            for c in ti.static(range(dim)):
                positions[i][c] = p[i, c]
                velocities[i][c] = v[i, c]

    init(np_positions, np_velocities)


cam = Canvas(screen_res[0], screen_res[1])
@ti.kernel
def draw_particle():
    for p_i in positions:
        cam.draw_sphere(positions[p_i], ti.Vector([1.0,1.0,1.0]))

def genMesh(cnt):
    pos_np = positions.to_numpy()/(res_3D / screen_to_world_ratio)
    np.save("output/frame_%d"%(cnt),pos_np)
    if cnt>360:
        exit()
    #resX = 256
    #grid_size = 1.0/resX
    #grid = VertexCenteredScalarGrid3((resX, resX, resX), (grid_size,grid_size,grid_size))
    #converter = SphPointsToImplicit3(1.5 * grid_size, 0.5)
    #converter.convert(pos_np.tolist(), grid)
    #surface_mesh = marchingCubes(
    #    grid,
    #    (grid_size,grid_size,grid_size),
    #    (0, 0 ,0),
    #    0.1,
    #    DIRECTION_ALL,
    #    DIRECTION_ALL
    #)
    #surface_mesh.writeObj('output/frame_{:06d}.obj'.format(cnt))
def main():
    init_particles()
    print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')
    gui = ti.GUI('PBF3D', (400,400))
    frames = 0
    while gui.running:
        move_board()
        run_pbf()
        cam.static_cam(5,0,5)
        cam.clear_canvas()
        draw_particle() 
        genMesh(frames)
        frames+=1
        img = cam.img.to_numpy()
        gui.set_image(img)
        gui.show()

if __name__ == '__main__':
    main()
