import numpy as np
from typing import Union, Optional
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gaussian_kde

def compute_hamiltonian(v: np.array, E_mesh: np.array, dx: float, return_all:bool = False):
    KE = 0.5 * np.sum(v * v)
    PE = 0.5 * np.sum(E_mesh * E_mesh) * dx
    
    # Scale calibration
    N = len(v)
    N_mesh = len(E_mesh)
    L = dx * N_mesh
    f = (N / L)
    PE *= f
    
    H = KE + PE
    
    if return_all:
        return H, KE, PE
    
    else:
        return H

def compute_distribution(v:np.array, vmin:float, vmax:float, interval:int):
    hist, bin_edges = np.histogram(v.reshape(-1), bins = interval, range = [vmin, vmax], density = True)
    return hist, bin_edges

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

    ax.set_title("Energy over time")
    
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

def generate_PIC_dist_gif(
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white", dpi=120)
    axes = axes.ravel()

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nh = N // 2
    Nt = snapshot.shape[1]

    axes[0].cla()
    scatter_b = axes[0].scatter([], [], s=0.4, color="blue", alpha=0.5)
    scatter_r = axes[0].scatter([], [], s=0.4, color="red", alpha=0.5)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("v")
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(vmin, vmax)
    axes[0].set_title("PIC simulation")

    v = np.linspace(vmin, vmax, 64)
    density = gaussian_kde(snapshot[N:, 0])
    density_plot = axes[1].plot(v, density(v), 'b')[0]
    axes[1].set_xlabel("v")
    axes[1].set_ylabel("f(v)")
    axes[1].set_xlim(vmin, vmax)
    axes[1].set_ylim([0, 0.5])
    axes[1].set_title("Distribution f(v)")

    fig.tight_layout()

    def _update(idx):
        scatter_b.set_offsets(np.column_stack((snapshot[0:Nh, idx], snapshot[N:N+Nh, idx])))
        scatter_r.set_offsets(np.column_stack((snapshot[Nh:N, idx], snapshot[N+Nh:, idx])))

        density = gaussian_kde(snapshot[N:, idx])
        density_plot.set_xdata(v)
        density_plot.set_ydata(density(v))
        fig.tight_layout()

        return scatter_b, scatter_r

    # Create animation
    ani = animation.FuncAnimation(fig, _update, frames=Nt, interval = 1000// plot_freq, blit=True)

    # Save animation
    ani.save(filepath, writer=animation.PillowWriter(fps=plot_freq))
    plt.close(fig)

# plot bump-on-tail simulation
def generate_bump_on_tail_snapshot(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
    vmin:Optional[float] = -10.0,
    vmax:Optional[float] = 10.0,
    high_electron_indice:Optional[np.ndarray] = None,
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
    
    if high_electron_indice is not None:
        low_electron_indice = np.array([i for i in range(0,N) if i not in high_electron_indice])
    else:
        low_electron_indice = np.arange(0,N)
        
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), facecolor="white", dpi=120)

    ax.cla()
    ax.scatter(snapshot[low_electron_indice], snapshot[low_electron_indice+N], s=0.4, color="blue", alpha=0.5)
    
    if high_electron_indice is not None:
        ax.scatter(snapshot[high_electron_indice], snapshot[high_electron_indice+N], s=0.4, color="red", alpha=0.5)
        
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.axis([xmin, xmax, vmin, vmax])
    ax.set_title("PIC simulation")

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    return fig, ax

def generate_bump_on_tail_figure(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
    vmin:Optional[float] = -10.0,
    vmax:Optional[float] = 10.0,
    high_electron_indice:Optional[np.ndarray] = None,
):
    # check directory
    filepath = os.path.join(save_dir, filename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]
    
    if high_electron_indice is not None:
        low_electron_indice = np.array([i for i in range(0,N) if i not in high_electron_indice])
    else:
        low_electron_indice = np.arange(0,N)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white", dpi=120)
    axes = axes.ravel()

    axes[0].cla()
    axes[0].scatter(snapshot[low_electron_indice,0], snapshot[low_electron_indice+N,0], s=0.4, color="blue", alpha=0.5)
    
    if high_electron_indice is not None:
        axes[0].scatter(snapshot[high_electron_indice,0], snapshot[high_electron_indice+N,0], s=0.4, color="red", alpha=0.5)
      
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("v")
    axes[0].axis([xmin, xmax, vmin, vmax])
    axes[0].set_title("PIC simulation at $t=0$")

    axes[1].cla()
    axes[1].scatter(snapshot[low_electron_indice,Nt//2], snapshot[low_electron_indice+N,Nt//2], s=0.4, color="blue", alpha=0.5)
    
    if high_electron_indice is not None:
        axes[1].scatter(snapshot[high_electron_indice,Nt//2], snapshot[high_electron_indice+N,Nt//2], s=0.4, color="red", alpha=0.5)
      
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v")
    axes[1].axis([xmin, xmax, vmin, vmax])
    axes[1].set_title("PIC simulation at $t=t_{max}/2$")

    axes[2].cla()
    axes[2].scatter(snapshot[low_electron_indice,-1], snapshot[low_electron_indice+N,-1], s=0.4, color="blue", alpha=0.5)
    
    if high_electron_indice is not None:
        axes[2].scatter(snapshot[high_electron_indice,-1], snapshot[high_electron_indice+N,-1], s=0.4, color="red", alpha=0.5)
      
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("v")
    axes[2].axis([xmin, xmax, vmin, vmax])
    axes[2].set_title("PIC simulation at $t=t_{max}$")

    fig.tight_layout()
    plt.savefig(filepath, dpi=120)

    return fig, axes


def generate_bump_on_tail_gif(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
    vmin:Optional[float] = -10.0,
    vmax:Optional[float] = 10.0,
    plot_freq:int = 32,
    high_electron_indice:Optional[np.ndarray] = None,
    ):

    # check directory
    filepath = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), facecolor="white", dpi=120)

    # Snapshot info
    N = snapshot.shape[0]//2
    Nt = snapshot.shape[1]
    
    if high_electron_indice is not None:
        low_electron_indice = np.array([i for i in range(0,N) if i not in high_electron_indice])
    else:
        low_electron_indice = np.arange(0,N)
    
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
        scatter_b.set_offsets(np.column_stack((snapshot[low_electron_indice, idx], snapshot[low_electron_indice + N, idx])))
        
        if high_electron_indice is not None:
            scatter_r.set_offsets(np.column_stack((snapshot[high_electron_indice, idx], snapshot[high_electron_indice + N, idx])))
        
        fig.tight_layout()
        return scatter_b, scatter_r

    # Create animation
    ani = animation.FuncAnimation(fig, _update, frames=Nt, interval = 1000// plot_freq, blit=True)

    # Save animation
    ani.save(filepath, writer=animation.PillowWriter(fps=plot_freq))
    plt.close(fig)

def generate_bump_on_tail_dist_gif(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
    vmin:Optional[float] = -10.0,
    vmax:Optional[float] = 10.0,
    plot_freq:int = 32,
    high_electron_indice:Optional[np.ndarray] = None,
):

    # check directory
    filepath = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white", dpi=120)
    axes = axes.ravel()

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nh = N // 2
    Nt = snapshot.shape[1]
    
    N = snapshot.shape[0]//2
    Nt = snapshot.shape[1]
    
    if high_electron_indice is not None:
        low_electron_indice = np.array([i for i in range(0,N) if i not in high_electron_indice])
    else:
        low_electron_indice = np.arange(0,N)
    
    axes[0].cla()
    scatter_b = axes[0].scatter([], [], s=0.4, color="blue", alpha=0.5)
    scatter_r = axes[0].scatter([], [], s=0.4, color="red", alpha=0.5)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("v")
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(vmin, vmax)
    axes[0].set_title("PIC simulation")

    v = np.linspace(vmin, vmax, 64)
    density = gaussian_kde(snapshot[N:, 0])
    density_plot = axes[1].plot(v, density(v), 'b')[0]
    axes[1].set_xlabel("v")
    axes[1].set_ylabel("f(v)")
    axes[1].set_xlim(vmin, vmax)
    axes[1].set_ylim([0, 0.5])
    axes[1].set_title("Distribution f(v)")

    fig.tight_layout()

    def _update(idx):
        scatter_b.set_offsets(np.column_stack((snapshot[low_electron_indice, idx], snapshot[low_electron_indice + N, idx])))
        
        if high_electron_indice is not None:
            scatter_r.set_offsets(np.column_stack((snapshot[high_electron_indice, idx], snapshot[high_electron_indice + N, idx])))
        
        density = gaussian_kde(snapshot[N:, idx])
        density_plot.set_xdata(v)
        density_plot.set_ydata(density(v))
        fig.tight_layout()

        return scatter_b, scatter_r

    # Create animation
    ani = animation.FuncAnimation(fig, _update, frames=Nt, interval = 1000// plot_freq, blit=True)

    # Save animation
    ani.save(filepath, writer=animation.PillowWriter(fps=plot_freq))
    plt.close(fig)

def generate_distribution_snapshot(
    snapshot:np.ndarray,
    save_dir: Optional[str],
    filename: Optional[str],
    xmin: Optional[float] = 0.0,
    xmax: Optional[float] = 50.0,
    vmin: Optional[float] = -10.0,
    vmax: Optional[float] = 10.0,
):
    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)
    else:
        filepath = None

    # info
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor="white", dpi=120)
    axes = axes.ravel()

    x = snapshot[:len(snapshot)//2]
    v = snapshot[len(snapshot)//2:]

    fx, fx_bin_edge = compute_distribution(x, xmin, xmax, interval = 64)
    fv, fv_bin_edge = compute_distribution(v, vmin, vmax, interval = 64)

    axes[0].cla()
    axes[0].bar(fx_bin_edge[:-1], fx, width=np.diff(fx_bin_edge), align="edge")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_xlim([xmin, xmax])
    axes[0].set_ylim([0,0.1])

    axes[1].cla()
    axes[1].bar(fv_bin_edge[:-1], fv, width=np.diff(fv_bin_edge), align="edge")
    axes[1].set_xlabel("v")
    axes[1].set_ylabel("f(v)")
    axes[1].set_xlim([vmin, vmax])
    axes[1].set_ylim([0,0.5])

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    return fig, axes

def generate_distribution_figure(
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
    Nt = snapshot.shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), facecolor="white", dpi=120)

    x = snapshot[: len(snapshot) // 2, 0]
    v = snapshot[len(snapshot) // 2 :, 0]

    fx, fx_bin_edge = compute_distribution(x, xmin, xmax, interval=64)
    fv, fv_bin_edge = compute_distribution(v, vmin, vmax, interval=64)

    axes[0,0].cla()
    axes[0,0].bar(fx_bin_edge[:-1], fx, width=np.diff(fx_bin_edge), align="edge")
    axes[0,0].set_xlabel("x")
    axes[0,0].set_ylabel("f(x) at $t=0$")
    axes[0,0].set_xlim([xmin, xmax])
    axes[0,0].set_ylim([0, 0.1])
    
    axes[1,0].cla()
    axes[1,0].bar(fv_bin_edge[:-1], fv, width=np.diff(fv_bin_edge), align="edge")
    axes[1,0].set_xlabel("v")
    axes[1,0].set_ylabel("f(v) at $t=0$")
    axes[1,0].set_xlim([vmin, vmax])
    axes[1,0].set_ylim([0, 0.5])
    
    x = snapshot[: len(snapshot) // 2, Nt//2]
    v = snapshot[len(snapshot) // 2 :, Nt//2]

    fx, fx_bin_edge = compute_distribution(x, xmin, xmax, interval=64)
    fv, fv_bin_edge = compute_distribution(v, vmin, vmax, interval=64)

    axes[0,1].cla()
    axes[0,1].bar(fx_bin_edge[:-1], fx, width=np.diff(fx_bin_edge), align="edge")
    axes[0,1].set_xlabel("x")
    axes[0,1].set_ylabel("f(x) at $t=t_{max}/2$")
    axes[0,1].set_xlim([xmin, xmax])
    axes[0,1].set_ylim([0, 0.1])
    
    axes[1,1].cla()
    axes[1,1].bar(fv_bin_edge[:-1], fv, width=np.diff(fv_bin_edge), align="edge")
    axes[1,1].set_xlabel("v")
    axes[1,1].set_ylabel("f(v) at $t=t_{max}/2$")
    axes[1,1].set_xlim([vmin, vmax])
    axes[1,1].set_ylim([0, 0.5])

    x = snapshot[: len(snapshot) // 2, -1]
    v = snapshot[len(snapshot) // 2 :, -1]

    fx, fx_bin_edge = compute_distribution(x, xmin, xmax, interval=64)
    fv, fv_bin_edge = compute_distribution(v, vmin, vmax, interval=64)

    axes[0,2].cla()
    axes[0,2].bar(fx_bin_edge[:-1], fx, width=np.diff(fx_bin_edge), align="edge")
    axes[0,2].set_xlabel("x")
    axes[0,2].set_ylabel("f(x) at $t=t_{max}$")
    axes[0,2].set_xlim([xmin, xmax])
    axes[0,2].set_ylim([0, 0.1])
    
    axes[1,2].cla()
    axes[1,2].bar(fv_bin_edge[:-1], fv, width=np.diff(fv_bin_edge), align="edge")
    axes[1,2].set_xlabel("v")
    axes[1,2].set_ylabel("f(v) at $t=t_{max}$")
    axes[1,2].set_xlim([vmin, vmax])
    axes[1,2].set_ylim([0, 0.5])

    fig.tight_layout()
    plt.savefig(filepath, dpi=120)

    return fig, axes

def generate_v_distribution_figure(
    snapshot: np.ndarray,
    save_dir: Optional[str],
    filename: Optional[str],
    vmin: Optional[float] = -10.0,
    vmax: Optional[float] = 10.0,
):

    v_axis = np.linspace(vmin, vmax, 64)

    # check directory
    filepath = os.path.join(save_dir, filename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white", dpi=120)

    x = snapshot[: len(snapshot) // 2, 0]
    v = snapshot[len(snapshot) // 2 :, 0]

    density = gaussian_kde(v)

    axes[0].plot(v_axis, density(v_axis), "b")[0]
    axes[0].set_xlabel("v")
    axes[0].set_ylabel("f(v)")
    axes[0].set_xlim(vmin, vmax)
    axes[0].set_ylim([0, 0.5])
    axes[0].set_title("Distribution f(v,$t=0$)")

    x = snapshot[: len(snapshot) // 2, Nt//2]
    v = snapshot[len(snapshot) // 2 :, Nt//2]

    density = gaussian_kde(v)

    axes[1].plot(v_axis, density(v_axis), "b")[0]
    axes[1].set_xlabel("v")
    axes[1].set_ylabel("f(v)")
    axes[1].set_xlim(vmin, vmax)
    axes[1].set_ylim([0, 0.5])
    axes[1].set_title("Distribution f(v,$t=t_{max}/2$)")

    x = snapshot[: len(snapshot) // 2, -1]
    v = snapshot[len(snapshot) // 2 :, -1]

    density = gaussian_kde(v)

    axes[2].plot(v_axis, density(v_axis), "b")[0]
    axes[2].set_xlabel("v")
    axes[2].set_ylabel("f(v)")
    axes[2].set_xlim(vmin, vmax)
    axes[2].set_ylim([0, 0.5])
    axes[2].set_title("Distribution f(v,$t=t_{max}$)")

    fig.tight_layout()
    plt.savefig(filepath, dpi=120)

    return fig, axes
