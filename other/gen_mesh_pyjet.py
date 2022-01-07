from threading import Thread,Lock
import os,time
import numpy as np
from pyjet import *
n=360
start = 0
def work(args):
    global n,start
    while start<n:
        start+=1
        print(start)
        pos_np = np.load("output/frame_%d.npy"%(start))
        print(np.min(pos_np),np.max(pos_np))
        resX = 256
        grid_size = 1.0/resX
        grid = VertexCenteredScalarGrid3((resX, resX, resX), (grid_size,grid_size,grid_size))
        converter = SphPointsToImplicit3(1.5 * grid_size, 0.8)
        converter.convert(pos_np.tolist(), grid)
        surface_mesh = marchingCubes(
            grid,
            (grid_size,grid_size,grid_size),
            (0, 0 ,0),
            0.1,
            DIRECTION_ALL,
            DIRECTION_ALL
        )
        surface_mesh.writeObj('output/frame_{:06d}.obj'.format(start))

if __name__ == '__main__':
    work(1)