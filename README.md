## **Cuda Flocking Boids Simulation**


## Showcase
(Simulation with 10K particles)

![](images/CudaBoidsSimulation.gif)

(Simulation with 25K particles)

![](images/CudaBoidsSimulation25K.gif)

## Description
A simple boids simulation done using Cuda and C++. \
Techniques implemented :
  * **Brute Force** approach (Each boid iterates over every other boid).
  * **Uniform grid** approach (Each boid iterates over boids only in neighbouring spatial grids).
  * **Coherent grid** approach (Similar to Uniform grid, but memory access is more linear (fewer indrections required)).
  

> Starting code / template obtained from : https://github.com/CIS565-Fall-2022/Project1-CUDA-Flocking.

**Performance Analysis**

![Alt text](images/perf_no_visualization.png)

![Alt text](images/perf_with_visualization.png) 

## References
![CIS 5650 GPU Programming And Architecture Course](https://cis565-fall-2023.github.io/)