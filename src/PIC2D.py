import numpy as np
import scipy as sp
from tqdm.auto import tqdm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from typing import Literal
from scipy import linalg
from src.utils import SORsolver

# 2D two-stream instability simulation by solving 2D vlasov equation solver using PIC method
class PICsolver:
    def __init__(
        self, 
        N : int, 
        N_mesh : int, 
        n0:float,
        Lx : float, 
        Ly : float,
        dt : float, 
        tmin : float, 
        tmax : float, 
        gamma : float,
        vth : float,
        vb : float,
        use_animation:bool=True,
        plot_freq : int = 4,  
        save_dir : str = "./result/simulation-2D.gif"
        ):
        
        # setup
        self.N = N                  # num of particles
        self.N_mesh = N_mesh        # num of mesh cell
        self.n0 = n0                # average electron density
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt                # time difference
        self.tmin = tmin            # minimum time
        self.tmax = tmax            # maximum time
        self.gamma = gamma          # parameters for solving linear equation of variant form of Tri-diagonal matrix
        self.use_animation = use_animation
        self.plot_freq = plot_freq
        self.save_dir = save_dir
        
        # particle information
        self.dx = Lx / N_mesh
        self.dy = Ly / N_mesh
        
        # initialize position, velocity and acceleration of particles
        self.pos = np.zeros((N,2))
        self.vel = np.zeros((N,2))
        self.acc = np.zeros((N,2))
        
        # electric potential, E-field and B-field : field dynamics
        self.phi_mesh = np.zeros((N_mesh, N_mesh))
        self.Ex_mesh = np.zeros((N_mesh, N_mesh))
        self.Ey_mesh = np.zeros((N_mesh, N_mesh))
        self.Bx_mesh = np.zeros((N_mesh, N_mesh))
        self.By_mesh = np.zeros((N_mesh, N_mesh))
        
        # Field representation for particle motion
        self.E = np.zeros((N, 2))
        self.B = np.zeros((N, 2))
        
        self.initialize_condition() # initialize x,v,a that is equivalent to the problem
        
        # index paramters for updating each mesh grid
        self.indx_l = np.zeros((N,2))
        
        self.indx_l[:,0] = np.floor(self.pos[:,0] / self.dx).astype(int)
        self.indx_l[:,1] = np.floor(self.pos[:,1] / self.dy).astype(int)
        self.indx_r = self.indx_l + np.ones_like(self.indx_l)
        
        self.weight_l = np.zeros_like(self.indx_l)
        self.weight_l[:,0] = (self.indx_r[:,0] * self.dx - self.pos[:,0]) / self.dx
        self.weight_l[:,1] = (self.indx_r[:,1] * self.dy - self.pos[:,1]) / self.dy
        
        self.weight_r = np.zeros_like(self.weight_l)
        self.weight_r[:,0] = (self.pos[:,0] - self.indx_l[:,0] * self.dx) / self.dx
        self.weight_r[:,1] = (self.pos[:,1] - self.indx_l[:,1] * self.dy) / self.dy
        
        # periodic BC
        self.indx_r = np.mod(self.indx_r, N_mesh)
        
        # electron density
        nx = np.bincount(self.indx_l[:,0], weights=self.weight_l[:,0], minlength=N_mesh)
        ny = np.bincount(self.indx_l[:,1], weights=self.weight_l[:,1], minlength=N_mesh)
        self.n = nx.reshape(-1,1) * ny.reshape(1,-1)
        
        nx = np.bincount(self.indx_r[:,0], weights=self.weight_r[:,0], minlength=N_mesh)
        ny = np.bincount(self.indx_r[:,1], weights=self.weight_r[:,1], minlength=N_mesh)
        self.n += nx.reshape(-1,1) * ny.reshape(1,-1)
        self.n *= self.n0 * (self.Lx / self.dx) * (self.Ly / self.dy) / self.N 
        
        # gradient field and laplacian of potential
        self.grad_x = np.zeros((N_mesh, N_mesh))
        self.grad_y = np.zeros((N_mesh, N_mesh))
        self.laplacian = np.zeros((N_mesh * N_mesh, N_mesh*N_mesh))
        
        self.generate_grad()
        self.generate_laplacian()
        
    def initialize_condition(self):    
        np.random.seed(42)
        
    def generate_grad(self):
        
        dx = self.dx
        dy = self.dy
        
        for idx_i in range(0,self.N_mesh):
            
            if idx_i > 0:
                self.grad_x[idx_i, idx_i - 1] = -1.0
                self.grad_y[idx_i, idx_i - 1] = -1.0
            
            if idx_i < self.N_mesh - 1:
                self.grad_x[idx_i, idx_i + 1] = 1.0
                self.grad_y[idx_i, idx_i - 1] = -1.0
                    
        # periodic condition
        self.grad_x[0,self.N_mesh - 1] = -1.0
        self.grad_x[self.N_mesh - 1,0] = 1.0
        self.grad_x /= 2*dx
        
        self.grad_y[0,self.N_mesh - 1] = -1.0
        self.grad_y[self.N_mesh - 1,0] = 1.0
        self.grad_y /= 2*dy
        
    def compute_grad(self, A : np.ndarray, axis : Literal['x', 'y'] = 'x'):
        if axis == 'x':
            grad_A = self.grad_x * A.T
        else:
            grad_A = self.grad_y * A
        return grad_A
    
    def generate_laplacian(self):
        dx = self.dx
        dy = self.dy
        
        for idx_i in range(1,self.N_mesh-1):
            for idx_j in range(1,self.N_mesh-1):
                
                idx = self.N_mesh * idx_i + idx_j
                drow = self.N_mesh    
                
                self.laplacian[idx, idx - 1] = 1.0 / dx ** 2
                self.laplacian[idx, idx + 1] = 1.0 / dx ** 2
                
                self.laplacian[idx, idx - drow] = 1.0 / dy ** 2
                self.laplacian[idx, idx + drow] = 1.0 / dy ** 2
                
                self.laplacian[idx, idx] = (-2.0) / dx ** 2 + (-2.0) / dy ** 2
    
    def linear_solve(self, A : np.ndarray, B : np.array):
        B = B.reshape(-1,1)
        x = linalg.solve(A,B)
        return x.reshape(-1,1)
        
    def solve(self):
        
        Nt = int(np.ceil((self.tmax - self.tmin) / self.dt))
        
        # we have to compute initial acceleration
        self.update_acc()
        
        if self.use_animation:
            pos_list = []
            vel_list = []
            
            pos_list.append(self.x)
            vel_list.append(self.v)
            
        else:
            pos_list = None
            vel_list = None
        
        for i in tqdm(range(Nt), 'PIC simulation process'):
            
            # velocity update
            self.update_velocity()
            
            # position update
            self.update_position()
            
            # density update
            self.update_density()
            
            # acceleration update
            self.update_acc()
            
            # velocity update with 1/2 kick
            self.update_velocity()
            
            if pos_list is not None:
                pos_list.append(self.x)
                vel_list.append(self.v)
                
        plt.figure(figsize = (6,4))
        plt.scatter(self.x[0:self.Nh],self.v[0:self.Nh],s=.5,color='blue', alpha=0.5)
        plt.scatter(self.x[self.Nh:], self.v[self.Nh:], s=.5,color='red',  alpha=0.5)
        plt.xlabel("x pos")
        plt.ylabel("vel")
        plt.xlim([0,self.L])
        plt.ylim([-8, 8])
        plt.tight_layout()
        plt.savefig("./result/PIC.png", dpi=160)
        print("# Computation process end")
            
        if self.use_animation:
            print("# Generating animation file")
            fig, ax = plt.subplots(1,1,figsize = (6,4), facecolor = 'white', dpi=160)
            
            def _plot(idx : int, ax:Axes, pos_list, vel_list):
                ax.cla()
                
                pos = pos_list[idx]
                vel = vel_list[idx]
                
                ax.scatter(pos[0:self.Nh],vel[0:self.Nh],s=.5,color='blue', alpha=0.5)
                ax.scatter(pos[self.Nh:], vel[self.Nh:], s=.5,color='red',  alpha=0.5)
                ax.set_xlabel("x pos")
                ax.set_ylabel("vel")
                ax.set_xlim([0,self.L])
                ax.set_ylim([-8, 8])
            
            replay = lambda idx : _plot(idx, ax, pos_list, vel_list)
            idx_max = len(pos_list) - 1
            indices = [i for i in range(idx_max)]
            ani = animation.FuncAnimation(fig, replay, frames = indices)
            writergif = animation.PillowWriter(fps = self.plot_freq, bitrate = False)
            ani.save(self.save_dir, writergif)
        
            print("# Complete")
    
    def update_velocity(self):
        self.vel += self.acc * self.dt / 2.0
  
    def update_position(self):
        self.pos += self.vel * self.dt
        self.pos[:,0] = np.mod(self.pos[:,0], self.Lx)
        self.pos[:,1] = np.mod(self.pos[:,1], self.Ly)
    
    def update_acc(self):
        self.phi_mesh = self.linear_solve(self.laplacian, self.n - self.n0).reshape(self.N_mesh, self.N_mesh)
        self.Ex_mesh = (-1) * self.compute_grad(self.phi_mesh, 'x')
        self.Ey_mesh = (-1) * self.compute_grad(self.phi_mesh, 'y')
        
        for idx in range(self.N):
            self.E[idx,0] = self.weight_l[idx,0] * self.Ex_mesh[self.indx_l[idx]] + self.weight_r[idx,0] * self.Ex_mesh[self.indx_r[idx]]
            self.E[idx,1] = self.weight_l[idx,1] * self.Ey_mesh[self.indx_l[idx]] + self.weight_r[idx,1] * self.Ey_mesh[self.indx_r[idx]]        
     
        # update acceleration
        self.acc[:,0] = (-1) * (self.E[:,0] + self.vel[:,1] * self.B[:,0])
        self.acc[:,1] = (-1) * (self.E[:,1] - self.vel[:,0] * self.B[:,1])
    
    def update_density(self):
        self.indx_l[:,0] = np.floor(self.pos[:,0] / self.dx).astype(int)
        self.indx_l[:,1] = np.floor(self.pos[:,1] / self.dy).astype(int)
        self.indx_r = self.indx_l + np.ones_like(self.indx_l)
        
        self.weight_l = np.zeros_like(self.indx_l)
        self.weight_l[:,0] = (self.indx_r[:,0] * self.dx - self.pos[:,0]) / self.dx
        self.weight_l[:,1] = (self.indx_r[:,1] * self.dy - self.pos[:,1]) / self.dy
        
        self.weight_r = np.zeros_like(self.weight_l)
        self.weight_r[:,0] = (self.pos[:,0] - self.indx_l[:,0] * self.dx) / self.dx
        self.weight_r[:,1] = (self.pos[:,1] - self.indx_l[:,1] * self.dy) / self.dy
        
        # periodic BC
        self.indx_r = np.mod(self.indx_r, self.N_mesh)
        
        # electron density
        nx = np.bincount(self.indx_l[:,0], weights=self.weight_l[:,0], minlength=self.N_mesh)
        ny = np.bincount(self.indx_l[:,1], weights=self.weight_l[:,1], minlength=self.N_mesh)
        self.n = nx.reshape(-1,1) * ny.reshape(1,-1)
        
        nx = np.bincount(self.indx_r[:,0], weights=self.weight_r[:,0], minlength=self.N_mesh)
        ny = np.bincount(self.indx_r[:,1], weights=self.weight_r[:,1], minlength=self.N_mesh)
        self.n += nx.reshape(-1,1) * ny.reshape(1,-1)
        self.n *= self.n0 * (self.Lx / self.dx) * (self.Ly / self.dy) / self.N 