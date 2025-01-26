# PIC code for plasma simulation
## Introduction
<p> 
    This is a github repository of python code for Two-stream instability based on PIC method. The baseline code referred is from Philip Mocz. See the <a href = "https://github.com/pmocz/pic-python">github link</a> for original code of Philip Mocz. For the faster computation in field solver, Tri-diagonal Gaussian elimiation method is utilized for 1-dimensional two-stream instability simulation. For initial condition, we applied adding beam velocity with reversion of the sign in half of electrons, then applying the sinusoidal perturabtion in overall electron velocity distribution.
</p>

## Simulation Result
<p>
    We developed two-stream instability simulation for 1D case using PIC method. The below shows the result of 1D case with leapfrog method (left) and implicit midpoint method (right).
</p>
<div>
    <p float = 'left'>
        <img src="/result/simulation_leapfrog.gif"  width="360" height="240">
        <img src="/result/simulation_midpoint.gif"  width="360" height="240">
    </p>
</div>

<p> 
    In this code, we assumed 1D space with periodic condition for solving electrostatic electron plasmas with applying the velocity perturbation. Two different beam streams of plasmas shows interesting motion of electrons as you can see above. I will develop more complex situations including 2D case and develop the code based on parallel computation. Since the explicit leapfrog method is not a symplectic integrator, we also developed implicit midpoint method for preserving hamiltonian. The below describes two different results obtained from leapfrog (left) and implicit midpoint method (right).
</p>

<div>
    <p float = 'left'>
        <img src="/result/PIC_leapfrog.png"  width="360" height="240">
        <img src="/result/PIC_midpoint.png"  width="360" height="240">
    </p>
</div>

<p> 
    The hamiltonian computed for each step shows that implicit midpoint method has better performance in terms of energy conservation compared with explicit method. 
</p>

<div>
    <p float = 'left'>
        <img src="/result/hamiltonian_comparsion.png"  width="360" height="240">
    </p>
</div>

## Reference
- <a href = "https://medium.com/swlh/create-your-own-plasma-pic-simulation-with-python-39145c66578b">Create Your Own Plasma PIC Simulation, Philip Mocz</a>
- Jianyuan XIAO et al 2018 Plasma Sci. Technol. 20 110501
- Hong Qin et al 2016 Nucl. Fusion. 56 014001