# Electrostatic Particle-In-Cell Plasma Simulation
## Introduction
<p> 
    This is a github repository of python code for Two-stream instability based on PIC method. The baseline code referred is from Philip Mocz. See the <a href = "https://github.com/pmocz/pic-python">github link</a> for original code of Philip Mocz. For the faster computation in field solver, Tri-diagonal Gaussian elimiation method is utilized for 1-dimensional two-stream instability simulation. For initial condition, we applied adding beam velocity with reversion of the sign in half of electrons, then applying the sinusoidal perturabtion in overall electron velocity distribution.
</p>

## How to execute
### Two-stream instability simulation
```
    python3 pic_1d_two_stream.py --num_particles {# of particles}
```

### Bump-on-tail instability simluation
```
    python3 pic_1d_bump_on_tail.py --num_particles {# of particles}
```

## Simulation
<p>
    We developed two-stream instability simulation for 1D case using PIC method. The below shows the result of 1D case with leapfrog method (left) and implicit midpoint method (right).
</p>
<div>
    <p float = 'left'>
        <img src="/result/two-stream_simulation_TSC_leapfrog.gif"  width="47.5%">
        <img src="/result/two-stream_simulation_TSC_midpoint.gif"  width="47.5%">
    </p>
</div>

<p> 
    In this code, we assumed 1D space with periodic condition for solving electrostatic electron plasmas with applying the velocity perturbation. Two different beam streams of plasmas shows interesting motion of electrons as you can see above. I will develop more complex situations including 2D case and develop the code based on parallel computation. Since the explicit leapfrog method is not a symplectic integrator, we also developed implicit midpoint method for preserving hamiltonian. The below describes two different results obtained from leapfrog (left) and implicit midpoint method (right).
</p>

<div>
    <p float = 'left'>
        <img src="/result/two-stream_evolution_CIC_leapfrog.png"  width="100%">
    </p>
</div>

<div>
    <p float = 'left'>
        <img src="/result/two-stream_evolution_dist_CIC_leapfrog.png"  width="100%">
    </p>
</div>

<p> 
    The hamiltonian computed for each step shows that implicit midpoint method has better performance in terms of energy conservation compared with explicit method. 
</p>

<div>
    <p float = 'left'>
        <img src="/result/hamiltonian_comparsion.png"  width="50%">
    </p>
</div>

<p> 
    This code also simulates any initial condition which is given by any probability distribution. Rejection sampling is used to select random particles following the distribution, while the SOR algorithm is applied for stable computation during the simulation. The example below is a "Bump-on-tail" distribution with externel electric field. The analytic solution is already known, thus we compare two results from analytic solution (left) and PIC simluation (right).
</p>

## Reference
- <a href = "https://medium.com/swlh/create-your-own-plasma-pic-simulation-with-python-39145c66578b">Create Your Own Plasma PIC Simulation, Philip Mocz</a>
- Jianyuan XIAO et al 2018 Plasma Sci. Technol. 20 110501
- Hong Qin et al 2016 Nucl. Fusion. 56 014001