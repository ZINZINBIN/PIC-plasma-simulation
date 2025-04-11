# Electrostatic Particle-In-Cell Plasma Simulation
## Introduction

This is a github repository of Particle-In-Cell (PIC) python code for 1-dimensional electrostatic plasma. The baseline code is referred from Philip Mocz. See the <a href = "https://github.com/pmocz/pic-python">github link</a> for original code of Philip Mocz. This code covers not only the two-stream instability case but also the more general case called bump-on-tail instability with different numerical integration methods. 

<div>
    <p float = 'left'>
        <img src="/result/two-stream_simulation_CIC_leapfrog.gif"  width="48.5%">
        <img src="/result/bump-on-tail_simulation_CIC_leapfrog.gif"  width="48.5%">
    </p>
</div>

### Algorithms for field solver
For the faster computation, two different numerical algorithms are used.
- Tridiagonal Gaussian elimination method for two-stream instability simulation
- Successive Over Relaxation (SOR) method for Bump-on-tail instabilty simulation

### Time integration method
For better accuracy during the time integration, we implemented 
- Leapfrog method
- Verlet method
- Explicit midpoint method
- Implicit midpoint method

### Interpolation method
The mesh interpolation process can be proceeded via two methods below.
- Cloud-In-Cell (CIC)
- High-order spline method (TSC)

Each simulation code can take initial sinusoidal perturbation with different modes in electron velocity distribution. The arguments for initial perturbations depends on the simulation and the detailed information can be checked from the code. 

## Simulation
### Two-stream instability
<p>
    Fisrt, we developed two-stream instability simulation for 1D case using PIC method. The below figure shows the result with leapfrog method (left) and explicit midpoint method (right) based on Cloud-In-Cell interpolation method.
</p>
<div>
    <p float = 'left'>
        <img src="/result/two-stream_simulation_dist_CIC_leapfrog.gif"  width="100%">
    </p>
</div>
<p> 
    In this code, we assumed 1-dimensional periodic space for solving electrostatic plasmas applying the velocity perturbation. For short time scale, ions in plasma are assumed to be fixed while only electrons can move along the system. Two cold electron beams injected through different directions results in two-stream instability induced by the plasma-wave interaction as you can see from above figures. 
</p>
<div>
    <p float = 'left'>
        <img src="/result/two-stream_evolution_CIC_leapfrog.png"  width="100%">
    </p>
</div>
<p> 
    The distribution of the particles along the time can be represented as below. Again, the two-stream instability simulation assumed the perturbation of the velocity distribution with high drift velocity induced by two injected electron beams in the periodic system. Thus, the initial velocity distribution of the electron depicts two Gaussian distribution with different mean velocities obtained by two injected beams. However, the particle-wave interaction leads to the change of distribution with the resonant wave observed in phase-velocity space and distribution on position space. 
</p>
<div>
    <p float = 'left'>
        <img src="/result/two-stream_evolution_dist_CIC_leapfrog.png"  width="100%">
    </p>
</div>

### Bump-on-tail instability
<div>
    <p float = 'left'>
        <img src="/result/bump-on-tail_simulation_dist_CIC_leapfrog.gif"  width="100%">
    </p>
</div>
<p>
    The bump-on-tail instability is an example of wave growth and is one of the most fundamental and basic instabilities in plasma physics. Two-stream instability mentioned above is also one of the limiting cases of the bump-on-tail instability. The particle-wave interaction with energy transferred, also called Landau damping, can be observed by PIC simulation.
</p>
<div>
    <p float = 'left'>
        <img src="/result/bump-on-tail_evolution_CIC_leapfrog.png"  width="100%">
    </p>
</div>
<p> 
    This code can simulate any given initial distributions. Rejection sampling is utilized to estimate the follwoing distribution when samples the particles' position and velocity. For stable computation, SOR algorithm is applied with JIT compiler. In this simulation, the initial distirbution is given as the sumation of Maxwellian distribution of thermal electrons and high energy electrons. The code can be exectued through pic_1d_bump_on_tail.py. It is available to simulate other initial distribution function by modifying dist.py. 
</p>
<p> 
    The distribution of the particles along the time in this case shows that the Landau damping process, which indicates the quasi-linear diffusion of the velocity distribution due to wave-particle interaction. 
</p>
<div>
    <p float = 'left'>
        <img src="/result/bump-on-tail_evolution_dist_CIC_leapfrog.png"  width="100%">
    </p>
</div>

## How to execute
### Two-stream instability simulation

```
    python3 pic_1d_two_stream.py    --num_particles {# of particles} 
                                    --num_mesh {# of mesh grid}
                                    --method {"midpoint","implicit","leapfrog","verlet"}
                                    --solver {"Gauss", "SOR"}
                                    --interpol {"CIC","TSC"}
                                    --vb {beam velocity}
                                    --vth {thermal velocity}
                                    --A {Amplitude of perturbation}
                                    --n_mode {mode number of perturbation}
```

### Bump-on-tail instability simluation

```
    python3 pic_1d_bump_on_tail.py  --num_particles {# of particles} 
                                    --num_mesh {# of mesh grid}
                                    --method {"midpoint","implicit","leapfrog","verlet"}
                                    --solver {"Gauss", "SOR"}
                                    --interpol {"CIC","TSC"}
                                    --v0 {high energy electron velocity}
                                    --a {ratio of perturbation}
                                    --sigma {Deviation of distribution}
                                    --A {Amplitude of perturbation}
                                    --n_mode {mode number of perturbation}
```

## Reference
- <a href = "https://medium.com/swlh/create-your-own-plasma-pic-simulation-with-python-39145c66578b">Create Your Own Plasma PIC Simulation, Philip Mocz</a>
- Jianyuan XIAO et al 2018 Plasma Sci. Technol. 20 110501
- Hong Qin et al 2016 Nucl. Fusion. 56 014001