import numpy as np
from typing import Union, Optional
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def compute_hamiltonian(v: np.array, E_mesh: np.array, dx: float, return_all:bool = False):
    KE = 0.5 * np.sum(v * v)
    PE = 0.5 * np.sum(E_mesh * E_mesh) * dx
    H = KE + PE
    
    if return_all:
        return H, KE, PE
    
    else:
        return H

def generate_hamiltonian_analysis(
    tmax:float,
    H:np.ndarray,
    KE:np.ndarray,
    PE:np.ndarray,
    save_dir: Optional[str],
    filename: Optional[str],
):
    # check directory
    filepath = os.path.join(save_dir, filename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    tlins = np.linspace(0, tmax, len(H))
    
    H_init = H[0]

    H /= H_init
    KE /= H_init
    PE /= H_init

    ax.plot(tlins, H, "r-", label="$E_{total}$")
    ax.plot(tlins, KE, "b-", label="$E_{kinetic}$")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Energy")

    ax2 = ax.twinx()
    ax2.plot(tlins, PE, "k-", label="$E_{potential}$")

    ax.set_title("E(t)")
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()
    plt.savefig(filepath, dpi=120)

    return fig, ax

# plot Two-stream instability simulation
def generate_PIC_snapshot(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
    vmin:Optional[float] = -10.0,
    vmax:Optional[float] = 10.0,
):
    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)
    else:
        filepath = None

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nh = N // 2

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), facecolor="white", dpi=120)

    ax.cla()
    ax.scatter(snapshot[0:Nh], snapshot[N:N+Nh], s=0.4, color="blue", alpha=0.5)
    ax.scatter(snapshot[Nh:N], snapshot[N+Nh:], s=0.4, color="red", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.axis([xmin, xmax, vmin, vmax])
    ax.set_title("PIC simulation")

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    return fig, ax

def generate_PIC_figure(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
    vmin:Optional[float] = -10.0,
    vmax:Optional[float] = 10.0,
):
    # check directory
    filepath = os.path.join(save_dir, filename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nh = N // 2
    Nt = snapshot.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white", dpi=120)
    axes = axes.ravel()

    axes[0].cla()
    axes[0].scatter(snapshot[0:Nh,0], snapshot[N:N+Nh,0], s=0.4, color="blue", alpha=0.5)
    axes[0].scatter(snapshot[Nh:N,0], snapshot[N+Nh:,0], s=0.4, color="red", alpha=0.5)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("v")
    axes[0].axis([xmin, xmax, vmin, vmax])
    axes[0].set_title("PIC simulation at $t=0$")

    axes[1].cla()
    axes[1].scatter(snapshot[0:Nh,Nt//2], snapshot[N:N+Nh,Nt//2], s=0.4, color="blue", alpha=0.5)
    axes[1].scatter(snapshot[Nh:N,Nt//2], snapshot[N+Nh:,Nt//2], s=0.4, color="red", alpha=0.5)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v")
    axes[1].axis([xmin, xmax, vmin, vmax])
    axes[1].set_title("PIC simulation at $t=t_{max}/2$")

    axes[2].cla()
    axes[2].scatter(snapshot[0:Nh,-1], snapshot[N:N+Nh,-1], s=0.4, color="blue", alpha=0.5)
    axes[2].scatter(snapshot[Nh:N,-1], snapshot[N+Nh:,-1], s=0.4, color="red", alpha=0.5)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("v")
    axes[2].axis([xmin, xmax, vmin, vmax])
    axes[2].set_title("PIC simulation at $t=t_{max}$")

    fig.tight_layout()
    plt.savefig(filepath, dpi=120)

    return fig, axes

def generate_PIC_gif(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
    vmin:Optional[float] = -10.0,
    vmax:Optional[float] = 10.0,
    plot_freq:int = 32,
    ):

    # check directory
    filepath = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), facecolor="white", dpi=120)

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nh = N // 2
    Nt = snapshot.shape[1]

    ax.cla()
    scatter_b = ax.scatter([], [], s=0.4, color="blue", alpha=0.5)
    scatter_r = ax.scatter([], [], s=0.4, color="red", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(vmin, vmax)
    ax.set_title("PIC simulation")

    fig.tight_layout()

    def _update(idx):
        scatter_b.set_offsets(np.column_stack((snapshot[0:Nh, idx], snapshot[N:N+Nh, idx])))
        scatter_r.set_offsets(np.column_stack((snapshot[Nh:N, idx], snapshot[N+Nh:, idx])))
        fig.tight_layout()
        return scatter_b, scatter_r

    # Create animation
    ani = animation.FuncAnimation(fig, _update, frames=Nt, interval = 1000// plot_freq, blit=True)

    # Save animation
    ani.save(filepath, writer=animation.PillowWriter(fps=plot_freq))
    plt.close(fig)
