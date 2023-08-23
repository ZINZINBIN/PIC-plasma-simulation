# Particle-In-Cell code for plasma simulation
## Introduction
<p> 
This is a github repository of python code for 1D Two-stream instability based on PIC method. The baseline code I referred is from Philip Mocz. See the <a href = "https://github.com/pmocz/pic-python">github link</a> for original code of Philip Mocz. 
</p>

<div>
    <p float = 'left'>
        <img src="/result/PIC.png"  width="360" height="240">
        <img src="/result/simulation.gif"  width="360" height="240">
    </p>
</div>

<p> 
In this code, we assumed 1D space with periodic condition for solving electrostatic electron plasmas with applying the velocity perturbation. Two different beam streams of plasmas shows interesting motion of electrons as you can see above. I will develop more complex situations including 2D case and develop the code based on parallel computation.
</p>