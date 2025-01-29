import argparse
from src.PIC1D_dist import PICsolver
from src.dist import BumpOnTail1D

def parsing():
    parser = argparse.ArgumentParser(description="PIC code for plasma simulation: 1D ")
    parser.add_argument("--num_particle", type = int, default = 40000)
    parser.add_argument("--num_mesh", type = int, default = 400)
    parser.add_argument("--t_min", type = float, default = 0)
    parser.add_argument("--t_max", type = float, default = 50)
    parser.add_argument("--dt", type = float, default = 1)
    parser.add_argument("--L", type = float, default = 10)
    parser.add_argument("--n0", type = float, default = 1.0)
    parser.add_argument("--eta", type = float, default = 10.0)
    parser.add_argument("--a", type = float, default = 0.3)
    parser.add_argument("--v0", type = float, default = 4.0)
    parser.add_argument("--sigma", type = float, default = 0.5)
    parser.add_argument("--beta", type = float, default = 5.95)
    parser.add_argument("--gamma", type = float, default = 1.0)
    parser.add_argument("--use_animation", type = bool, default = True)
    parser.add_argument("--plot_freq", type = int, default = 10)
    parser.add_argument("--save_dir", type=str, default="./result/")
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    
    args = parsing()
    
    # Initial distribution: Bump-On-Tail distribution   
    dist = BumpOnTail1D(eta = args['eta'], a = args['a'], v0 = args['v0'], sigma = args['sigma'], beta = args['beta'], n_samples=args['num_particle'], L = args['L'])
    
    # PIC solver
    solver = PICsolver(
        N = args['num_particle'],
        N_mesh=args['num_mesh'],
        n0 = args['n0'],
        L = args['L'],
        dt = args['dt'],
        tmin = args['t_min'],
        tmax = args['t_max'],
        gamma = args['gamma'],
        init_dist= dist,
        E_external=lambda x : dist.compute_E_field(x-args['L']/2),
        use_animation=args['use_animation'],
        plot_freq=args['plot_freq'],  
        save_dir = args['save_dir']
    )
    
    solver.solve()
