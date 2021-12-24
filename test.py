# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)

import math
from os import write

import numpy as np
from numpy.core.fromnumeric import mean

import taichi as ti

import meshio
from canvas import Canvas

mx = np.ones((3))*-100
mn = np.ones((3))*100
points = []
face = []
n = 120
for i in range(1,n+1):
    mesh = meshio.read("data/output_mesh_%d_.obj"%(i))
    mesh.points[...,1:3] = mesh.points[...,2:0:-1]
    mean_point = np.average(mesh.points,axis = 0)
    mesh.points = mesh.points*10
    points.append(mesh.points)
    if i==1:
        face = mesh.cells_dict['triangle']
    meshio.write("data/output_mesh_%d_scale.obj"%(i), mesh)

point_num = points[0].shape[0]
face_num = face.shape[0]
points = np.asarray(points)
ti.init(arch=ti.gpu)
screen_res = (1000, 1000)
screen_to_world_ratio = 40
dim = 3
b_min = ti.Vector((-screen_res[0]/screen_to_world_ratio/2,-screen_res[1]/screen_to_world_ratio/2,0))
b_max = ti.Vector((screen_res[0]/screen_to_world_ratio/2,screen_res[1]/screen_to_world_ratio/2,100))

cell_size = 1.25
grid_size = [math.ceil((b_max[x]-b_min[x])//cell_size) for x in range(dim)]

bg_color = 0x112f41
particle_color = 0x068587
boundary_color = 0xebaca2
num_particles_x = 30
num_particles_z = 30
num_particles_y = 30
num_particles = num_particles_x * num_particles_y * num_particles_z
max_num_particles_per_cell=100
max_num_neighbors = 100

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int, shape=(grid_size[0], grid_size[1], grid_size[2], max_num_particles_per_cell))
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)

lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)

frames = ti.field(ti.i32)

ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)
ti.root.dense(ti.i, 1).place(frames)
cam = Canvas(1000,1000)
point_ti = ti.Vector(3, dt=ti.f32, shape=(n, point_num))
face_ti = ti.Vector(3, dt=ti.i32, shape=(face_num))
point_ti.from_numpy(points)
face_ti.from_numpy(face)

h = 1.1
mass = 1.0
rho0 = 4.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001
neighbor_radius = h * 1.05
poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

particle_radius = 3
particle_radius_in_world = 3.0 / screen_to_world_ratio
pbf_num_iters = 5
time_delta = 1.0/24
epsilon = 1e-5

@ti.kernel
def initParticles():
    delta = h * 0.5
    for i in range(num_particles):
        x = i%num_particles_x - num_particles_x/2
        z = i//num_particles_x%num_particles_z - num_particles_z/2
        y = i//(num_particles_x*num_particles_z) - num_particles_y/2+50
        positions[i] = ti.Vector([x,z,y]) * delta


@ti.func
def confineBoundary(pos,old_pos,cnt):
    for i in ti.static(range(dim)):
        if pos[i] < b_min[i] + particle_radius_in_world:
            pos[i] = b_min[i] + particle_radius_in_world + epsilon*ti.random()
        elif pos[i] > b_max[i] - particle_radius_in_world:
            pos[i] = b_max[i] - particle_radius_in_world - epsilon*ti.random()
    frame = 60
    if frames[0]>60:
        frame = frames[0]
    flag = True
    for j in range(face.shape[0]):
        if flag:
            a = point_ti[(frame-60)%120,face_ti[j][0]]
            b = point_ti[(frame-60)%120,face_ti[j][1]]
            c = point_ti[(frame-60)%120,face_ti[j][2]]
            v0 = b-a
            v1 = c-a
            n = v0.cross(v1)
            n = n/n.norm()
            D = n.dot(a-old_pos)
            dir = (pos-old_pos)
            t = D/(n.dot(dir))
            if t<0 or t>1:
                continue
            npos = old_pos+t*(pos-old_pos)
            v2 = npos-a
            dot00 = v0.dot(v0)
            dot01 = v0.dot(v1)
            dot02 = v0.dot(v2)
            dot11 = v1.dot(v1)
            dot12 = v1.dot(v2)
            inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * inverDeno
            if u<0 or u>1:
                continue
            v = (dot00 * dot12 - dot01 * dot02) * inverDeno
            if v<0 or v>1:
                continue
            if u+v>1:
                continue
            pos = old_pos+(t-0.001)*(pos-old_pos)
            flag = False                   
    return pos

@ti.kernel
def test():
    global frames
    frames[0] = 60
    old_pos = ti.Vector([1.05,-0.25,0.9])
    pos = ti.Vector([1.05,-0.25,0.92])
    confineBoundary(pos,old_pos)

@ti.func
def get_cell(pos):
    return (pos-b_min)//cell_size

@ti.func
def in_grid(cell_grid):
    x = True
    for i in ti.static(range(dim)):
        if cell_grid[i]>=grid_size[i]:
            x = False
    return x

@ti.kernel
def prologue():
    frames[0]+=1
    for i in positions:
        old_positions[i] = positions[i]
    for i in positions:
        g = ti.Vector([0.0,0.0,-9.8])
        pos,vel = positions[i], velocities[i]
        vel += g*time_delta
        pos += vel*time_delta
        old_pos = old_positions[i]
        positions[i] = confineBoundary(pos,old_pos,0)
        pos = positions[i]
        positions[i] = confineBoundary(pos,old_pos,1)
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
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h) / poly6_value(corr_deltaQ_coeff * h, h)
    # pow(x, 4)
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
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                spiky_gradient(pos_ji, h)

        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i
    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]

@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        old_pos = old_positions[i]
        positions[i] = confineBoundary(pos,old_pos,0)
        pos = positions[i]
        positions[i] = confineBoundary(pos,old_pos,1)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    # no vorticity/xsph because we cannot do cross product in 2D...

def runPBF():
    print("pro")
    prologue()
    print("substep")
    for _ in range(pbf_num_iters):
        substep()
    print("ep")
    epilogue()

def print_stats():
    print('PBF stats:')
    num = grid_num_particles.to_numpy()
    avg, max = np.mean(num), np.max(num)
    print(f'  #particles per cell: avg={avg:.2f} max={max}')
    num = particle_num_neighbors.to_numpy()
    avg, max = np.mean(num), np.max(num)
    print(f'  #neighbors per particle: avg={avg:.2f} max={max}')

def render(gui, cnt):
    #gui.clear(bg_color)
    pos_np = positions.to_numpy()
    with open("output/%d.ply"%(cnt),"w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n"%(num_particles))
        for i in range(pos_np.shape[0]):
            f.write("%.4lf %.4lf %.4lf\n"%(pos_np[i,0],pos_np[i,1],pos_np[i,2]))
    #for j in range(dim):
    #    pos_np[:, j] = (pos_np[:, j]-b_min[j])/(b_max[j]-b_min[j])
    #gui.circles(pos_np[:,0::2], radius=particle_radius, color=particle_color)
    #gui.rect((0, 0), (1, 1),
    #         radius=1.5,
    #         color=boundary_color)
    #gui.show()
    return


@ti.kernel
def draw_particle():
    for p_i in positions:
        cam.draw_sphere(positions[p_i], ti.Vector([1.0,1.0,1.0]))

def main():
    initParticles()
    gui = ti.GUI('PBF3D', screen_res)
    cnt = 0
    while gui.running:
        cam.static_cam(0,0,0)
        cam.clear_canvas()
        runPBF() 
        draw_particle()
        gui.set_image(cam.img.to_numpy())
        gui.show()
        render(gui, cnt)
        cnt+=1

if __name__ == '__main__':
    main() 
