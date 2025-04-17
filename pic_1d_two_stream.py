import argparse, os
from src.PIC import PIC
from src.util import (
    generate_PIC_figure, 
    generate_PIC_snapshot, 
    generate_hamiltonian_analysis, 
    generate_PIC_gif, 
    generate_distribution_snapshot, 
    generate_distribution_figure, 
    generate_PIC_dist_gif
)

def parsing():
    parser = argparse.ArgumentParser(description="1D Electrostatic Particle-In-Cell code for plasma kinetic simulation")
    parser.add_argument("--num_particle", type = int, default = 40000)
    parser.add_argument("--num_mesh", type = int, default = 1000)
    parser.add_argument("--method", type = str, default = "leapfrog", choices=["midpoint","leapfrog", "verlet", "implicit"])
    parser.add_argument("--solver", type=str, default="Gauss", choices=["SOR", "Gauss"])
    parser.add_argument("--interpol", type = str, default = "CIC", choices=["CIC", "TSC"])
    parser.add_argument("--t_min", type = float, default = 0)
    parser.add_argument("--t_max", type = float, default = 50)
    parser.add_argument("--dt", type = float, default = 0.05)
    parser.add_argument("--L", type = float, default = 50)
    parser.add_argument("--n0", type = float, default = 1.0)
    parser.add_argument("--vb", type = float, default = 3.0)
    parser.add_argument("--vth", type = float, default = 1.0)
    parser.add_argument("--gamma", type = float, default = 5.0)
    parser.add_argument("--A", type = float, default = 0.1)
    parser.add_argument("--n_mode", type=int, default=2)
    parser.add_argument("--use_animation", type = bool, default = True)
    parser.add_argument("--plot_freq", type = int, default = 50)
    parser.add_argument("--save_dir", type = str, default = "./result/")
    parser.add_argument("--simcase", type=str, default="two-stream", choices = ["two-stream", "bump-on-tail"])
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":

    args = parsing()
    sim = PIC(
        N = args['num_particle'],
        N_mesh=args['num_mesh'],
        method = args['method'],
        solver = args["solver"],
        interpol = args["interpol"],
        n0 = args['n0'],
        L = args['L'],
        dt = args['dt'],
        tmin = args['t_min'],
        tmax = args['t_max'],
        gamma = args['gamma'],
        vth = args['vth'],
        vb = args['vb'],
        A = args['A'],
        n_mode = args['n_mode'],
        simcase = args['simcase'],
    )

    snapshot, E, KE, PE = sim.solve()
    
    # file check
    if not os.path.exists(args['save_dir']):
        os.mkdir(args["save_dir"])

    # plot pic simulation figure
    generate_PIC_snapshot(snapshot[:,-1], args['save_dir'], "{}_snapshot_{}_{}.png".format(args['simcase'], args['interpol'], args['method']), xmin = 0, xmax = args['L'], vmin = -10.0, vmax = 10.0)
    generate_PIC_figure(snapshot, args['save_dir'], "{}_evolution_{}_{}.png".format(args['simcase'], args['interpol'], args['method']), xmin = 0, xmax = args['L'], vmin = -10.0, vmax = 10.0)
    generate_hamiltonian_analysis(args['t_max'], E, KE, PE, args['save_dir'], "{}_hamiltonian_{}_{}.png".format(args['simcase'], args['interpol'], args['method']))

    # plot distribution
    generate_distribution_snapshot(snapshot[:,-1], args["save_dir"],"{}_snapshot_dist_{}_{}.png".format(args['simcase'], args['interpol'], args['method']), xmin = 0, xmax = args['L'], vmin = -10.0, vmax = 10.0)
    generate_distribution_figure(snapshot, args["save_dir"],"{}_evolution_dist_{}_{}.png".format(args['simcase'], args['interpol'], args['method']), xmin = 0, xmax = args['L'], vmin = -10.0, vmax = 10.0)

    if args['use_animation']:
        generate_PIC_gif(snapshot, args['save_dir'], "{}_simulation_{}_{}.gif".format(args['simcase'], args['interpol'], args['method']), 0, args['L'], -10.0, 10.0, args['plot_freq'])
        generate_PIC_dist_gif(snapshot, args['save_dir'], "{}_simulation_dist_{}_{}.gif".format(args['simcase'], args['interpol'], args['method']), 0, args['L'], -10.0, 10.0, args['plot_freq'])