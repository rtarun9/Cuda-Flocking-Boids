#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <thrust/gather.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

#ifndef imax
#define imax(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef imin
#define imin(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

// Check for CUDA errors; print and exit if there was a problem.
void checkCUDAError(const char *msg, int line = -1)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    if (line >= 0)
    {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    
    // Adding this here so the terminal does not immediatly close in case of error.
    
    int keyboardValue = 0;
    scanf("%d", &keyboardValue);

    exit(EXIT_FAILURE);
  }
}

/*****************
 * Configuration *
 *****************/

// Block size used for CUDA kernel launch.
#define blockSize 128

// Parameters for the boids algorithm.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 2.0f

// Size of the starting area in simulation space.
#define scene_scale 100.0f

/***********************************************
 * Kernel state (pointers are device pointers) *
 ***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// For efficient sorting and the uniform grid. These should always be parallel.
// dev_particleArrayIndieces stores for a boid 'b', what index in dev_pos and dev_velX represents its pos / vel.
int *dev_particleArrayIndices; 

// dev_particleGridIndices stores which grid cell this particle is in.
int *dev_particleGridIndices;  

// Needed for use with thrust when usign the sort_by_key function.
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

// These two arrays store the part of dev_particleArrayIndices that belong to this cell.
int *dev_gridCellStartIndices; 
int *dev_gridCellEndIndices;   

// In the coherent uniform grid implementation, we try to avoid the indirection caused 
// due the dev_particleArrayIndices. Instead, we reshuffle the position and velocity
// buffers (which can potentially lead to performance gains to due spatial locality).
// The thrust::gather function can be used for this resizing activity.
glm::vec3 *dev_reshuffledPositions;
glm::vec3 *dev_reshuffledVelocities;

thrust::device_ptr<glm::vec3> dev_thrust_reshuffledPositions;
thrust::device_ptr<glm::vec3> dev_thrust_reshuffledVelocities; 

thrust::device_ptr<glm::vec3> dev_thrust_positions;
thrust::device_ptr<glm::vec3> dev_thrust_velocities;

// Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
 * initSimulation *
 ******************/

__host__ __device__ unsigned int hash(unsigned int a)
{
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

// Function for generating a random vec3.
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index)
{
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

// CUDA kernel for generating boids with a specified mass randomly around the star.
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 *arr, float scale)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N)
  {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

// Initialize memory, update some globals
void Boids::initSimulation(int N)
{
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  cudaMalloc((void **)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void **)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void **)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
                                                               dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // We want the gridCellWidth to ALWAYS be atleast 2x of the rule distances.
  // This will guarentee that in a 2D scenario, to check for neighbours / boids which lie in a distance of 
  // rule X, we need to check a set number of neighbouring cells (4 for 2D, and 8 for 3D).
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  
  // The scene goes from -scene_scale to +scene_scale, hence why the gridSizeCount is 2X halfSizeCount.
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // For uniform grid.
  cudaMalloc((void **)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("Failed to allocate memory for dev_particleArrayIndices");

  cudaMalloc((void **)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("Failed to allocate memory for dev_particleGridIndices");

  cudaMalloc((void **)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("Failed to allocate memory for dev_gridCellStartIndices");

  cudaMalloc((void **)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("Failed to allocate memory for dev_gridCellEndIndices");

  cudaMalloc((void**)&dev_reshuffledPositions, N * sizeof(glm::vec3));
  cudaMalloc((void**)&dev_reshuffledVelocities, N * sizeof(glm::vec3));

  cudaDeviceSynchronize();
}

/******************
 * copyBoidsToVBO *
 ******************/

// Copy the boid positions into the VBO so that they can be drawn by OpenGL.
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale)
{
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N)
  {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale)
{
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N)
  {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

// Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities)
{
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}

/******************
 * stepSimulation *
 ******************/

// Compute the new velocity on the body with index `iSelf` due to the `N` boids
// in the `pos` and `vel` arrays.
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel)
{
  const glm::vec3 currentPosition = pos[iSelf];

  // Rule 1 : Cohesion.
  // Boids move towards the percieved center of mass of neighbours.
  glm::vec3 percievedCenter = glm::vec3(0.0f, 0.0f, 0.0f);
  int neighboursInRangeForCohesion = 0;

  // Rule 2 : Separation.
  // Boids must avoid colliding with its close neighbours.
  glm::vec3 separationVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

  // Rule 3 : Alignment.
  // Boids try to move with same direction and velocity as thier neighbours.
  glm::vec3 alignmentVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
  int neighboursInRangeForAlignment = 0;

  for (int b = 0; b < N; b++)
  {
    glm::vec3 bPosition = pos[b];

    const float distance = glm::distance(currentPosition, bPosition);
    if (b != iSelf && distance < rule1Distance)
    {
      ++neighboursInRangeForCohesion;
      percievedCenter += (bPosition);
    }

    if (b != iSelf && distance < rule2Distance)
    {
      separationVelocity += (currentPosition - bPosition);
    }

    if (b != iSelf && distance < rule3Distance)
    {
      ++neighboursInRangeForAlignment;
      alignmentVelocity += vel[b];
    }
  }

  if (neighboursInRangeForCohesion > 0)
  {
    percievedCenter /= neighboursInRangeForCohesion;
  }

  if (neighboursInRangeForAlignment > 0)
  {
    alignmentVelocity /= neighboursInRangeForAlignment;
  }

  const glm::vec3 v1 = (percievedCenter - currentPosition) * rule1Scale;
  const glm::vec3 v2 = separationVelocity * rule2Scale;
  const glm::vec3 v3 = alignmentVelocity * rule3Scale;

  return v1 + v2 + v3;
}

// For each of the `N` bodies, update its position based on its current velocity.
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
                                             glm::vec3 *vel1, glm::vec3 *vel2)
{
  // Each boid is basically represented by a 1D index.
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index >= N)
  {
    return;
  }

  // Compute a new velocity based on pos and vel1
  glm::vec3 newVelocity = vel1[index] + computeVelocityChange(N, index, pos, vel1);

  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
  // Reason for using vel1 : vel1 is being overwritten in the computeVelocityChange function.
  // After the dispatch call, we swap vel1 and vel2 (i.e ping pong-ing the buffers).
  if (glm::length(newVelocity) > maxSpeed)
  {
    newVelocity = glm::normalize(newVelocity) * maxSpeed;
  }

  vel2[index] = newVelocity;

  return;
}

// For each of the `N` bodies, update its position based on its current velocity.
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel)
{
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N)
  {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution)
{
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
                                   glm::vec3 gridMin, float inverseCellWidth,
                                   glm::vec3 *pos, int *indices, int *gridIndices)
{
  // Label each boid with the index of its grid cell.
  // Set up a parallel array of integer indices as pointers to the actual
  // boid data in pos and vel1/vel2

  const int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N)
  {
    return;
  }

  // Find the 3d index of the current boid based on its position in the grid.
  glm::ivec3 boidCellIndex3D = (pos[index] - gridMin) * inverseCellWidth;

  // store the 1d index of the cell the current boid is in.
  gridIndices[index] = gridIndex3Dto1D(boidCellIndex3D.x, boidCellIndex3D.y, boidCellIndex3D.z, gridResolution);

  // This array indices will store the index of each boid (this is not done automatically).
  // This looks very simple, but is required because we do a sort by key's in the driver code
  // and during that time, indices[i] != i.
  indices[index] = index;
}

__global__ void kernResetIntBuffer(int N, int *intBuffer, int value)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N)
  {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
                                         int *gridCellStartIndices, int *gridCellEndIndices)
{
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

  const int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index >= N)
  {
    return;
  }

  if (index == 0)
  {
    gridCellStartIndices[particleGridIndices[index]] = 0;
  }
  else if (index == N - 1)
  {
    gridCellEndIndices[particleGridIndices[index]] = N - 1;
  }
  else if (particleGridIndices[index] != particleGridIndices[index + 1])
  {
    gridCellEndIndices[particleGridIndices[index]] = index;
    gridCellStartIndices[particleGridIndices[index + 1]] = index + 1;
  }
}

__global__ void kernUpdateVelNeighborSearchScattered(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int *gridCellStartIndices, int *gridCellEndIndices,
    int *particleArrayIndices,
    glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2)
{
  // Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= N)
  {
    return;
  }

  const glm::vec3 currentPosition = pos[index];

  // Rule 1 : Cohesion.
  // Boids move towards the percieved center of mass of neighbours.
  glm::vec3 percievedCenter = glm::vec3(0.0f, 0.0f, 0.0f);
  int neighboursInRangeForCohesion = 0;

  // Rule 2 : Separation.
  // Boids must avoid colliding with its close neighbours.
  glm::vec3 separationVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

  // Rule 3 : Alignment.
  // Boids try to move with same direction and velocity as thier neighbours.
  glm::vec3 alignmentVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
  int neighboursInRangeForAlignment = 0;

  // Find the 3d index of the boid in the grid.
  const glm::ivec3 gridIndex3D = (pos[index] - gridMin) * inverseCellWidth;

  for (int x = -1; x <= 1; x++)
  {
    for (int y = -1; y <= 1; y++)
    {
      for (int z = -1; z <= 1; z++)
      {
        int gridIndex = gridIndex3Dto1D(imin(imax(0, gridIndex3D.x + x), gridResolution - 1), imin(imax(gridIndex3D.y + y, 0), gridResolution - 1), imin(imax(gridIndex3D.z + z, 0), gridResolution - 1), gridResolution);

        // Perform rule 1, 2, 3 for all boids within this grid index (if gridCellStartIndices == -1, means there is no cell in this spacial grid).
        if (gridCellStartIndices[gridIndex] != -1)
        {
          for (int k = gridCellStartIndices[gridIndex]; k <= gridCellEndIndices[gridIndex]; k++)
          {
            int boidIndex = particleArrayIndices[k];
            glm::vec3 bPosition = pos[boidIndex];

            const float distance = glm::distance(currentPosition, bPosition);

            if (boidIndex != index && distance < rule1Distance)
            {
              ++neighboursInRangeForCohesion;
              percievedCenter += (bPosition);
            }

            if (boidIndex != index && distance < rule2Distance)
            {
              separationVelocity += (currentPosition - bPosition);
            }

            if (boidIndex != index && distance < rule3Distance)
            {
              ++neighboursInRangeForAlignment;
              alignmentVelocity += vel1[boidIndex];
            }
          }
        }
      }
    }
  }

  if (neighboursInRangeForCohesion > 0)
  {
    percievedCenter /= neighboursInRangeForCohesion;
  }

  if (neighboursInRangeForAlignment > 0)
  {
    alignmentVelocity /= neighboursInRangeForAlignment;
  }

  const glm::vec3 v1 = (percievedCenter - currentPosition) * rule1Scale;
  const glm::vec3 v2 = separationVelocity * rule2Scale;
  const glm::vec3 v3 = alignmentVelocity * rule3Scale;

    // Compute a new velocity based on pos and vel1
  glm::vec3 newVelocity = vel1[index] + v1 + v2 + v3;

  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
  // Reason for using vel1 : vel1 is being overwritten in the computeVelocityChange function.
  // After the dispatch call, we swap vel1 and vel2 (i.e ping pong-ing the buffers).
  if (glm::length(newVelocity) > maxSpeed)
  {
    newVelocity = glm::normalize(newVelocity) * maxSpeed;
  }

  vel2[index] = newVelocity;

  return;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int *gridCellStartIndices, int *gridCellEndIndices,
    glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2)
{
  // This is very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= N)
  {
    return;
  }

  const glm::vec3 currentPosition = pos[index];

  // Rule 1 : Cohesion.
  // Boids move towards the percieved center of mass of neighbours.
  glm::vec3 percievedCenter = glm::vec3(0.0f, 0.0f, 0.0f);
  int neighboursInRangeForCohesion = 0;

  // Rule 2 : Separation.
  // Boids must avoid colliding with its close neighbours.
  glm::vec3 separationVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

  // Rule 3 : Alignment.
  // Boids try to move with same direction and velocity as thier neighbours.
  glm::vec3 alignmentVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
  int neighboursInRangeForAlignment = 0;

  // Find the 3d index of the boid in the grid.
  const glm::ivec3 gridIndex3D = (currentPosition - gridMin) * inverseCellWidth;

  for (int x = -1; x <= 1; x++)
  {
    for (int y = -1; y <= 1; y++)
    {
      for (int z = -1; z <= 1; z++)
      {
        int gridIndex = gridIndex3Dto1D(imin(imax(0, gridIndex3D.x + x), gridResolution - 1), imin(imax(gridIndex3D.y + y, 0), gridResolution - 1), imin(imax(gridIndex3D.z + z, 0), gridResolution - 1), gridResolution);

        // Perform rule 1, 2, 3 for all boids within this grid index (if gridCellStartIndices == -1, means there is no cell in this grid block).
        if (gridCellStartIndices[gridIndex] != -1)
        {
          for (int k = gridCellStartIndices[gridIndex]; k <= gridCellEndIndices[gridIndex]; k++)
          {
            glm::vec3 bPosition = pos[k];

            const float distance = glm::distance(currentPosition, bPosition);

            if (k != index && distance < rule1Distance)
            {
              ++neighboursInRangeForCohesion;
              percievedCenter += (bPosition);
            }

            if (k != index && distance < rule2Distance)
            {
              separationVelocity += (currentPosition - bPosition);
            }

            if (k != index && distance < rule3Distance)
            {
              ++neighboursInRangeForAlignment;
              alignmentVelocity += vel1[k];
            }
          }
        }
      }
    }
  }

  if (neighboursInRangeForCohesion > 0)
  {
    percievedCenter /= neighboursInRangeForCohesion;
  }

  if (neighboursInRangeForAlignment > 0)
  {
    alignmentVelocity /= neighboursInRangeForAlignment;
  }

  const glm::vec3 v1 = (percievedCenter - currentPosition) * rule1Scale;
  const glm::vec3 v2 = separationVelocity * rule2Scale;
  const glm::vec3 v3 = alignmentVelocity * rule3Scale;

    // Compute a new velocity based on pos and vel1
  glm::vec3 newVelocity = vel1[index] + v1 + v2 + v3;

  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
  // Reason for using vel1 : vel1 is being overwritten in the computeVelocityChange function.
  // After the dispatch call, we swap vel1 and vel2 (i.e ping pong-ing the buffers).
  if (glm::length(newVelocity) > maxSpeed)
  {
    newVelocity = glm::normalize(newVelocity) * maxSpeed;
  }

  vel2[index] = newVelocity;

  return;
}

// Step the entire N-body simulation by `dt` seconds.
void Boids::stepSimulationNaive(float dt)
{
  // Idea : All objects are given a 1D id.
  dim3 gridDim((numObjects + blockSize - 1) / blockSize);

  kernUpdateVelocityBruteForce<<<gridDim, threadsPerBlock>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  checkCUDAError("Failed to run kernel updateVelocityBruteForce");

  kernUpdatePos<<<gridDim, threadsPerBlock>>>(numObjects, dt, dev_pos, dev_vel1);
  checkCUDAError("Failed to run kernel kernUpdatePos");

  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt)
{
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

  // General idea : Have a key + value pair array, where the key is the grid index, and value is the block index.
  // If you sort by key, you can easily see which boids are in close proximity in the uniform grid.
  // For example, say you are boid X in grid 4. In a 2D configuration, if you sort by grid index,
  // you can easily see the boid indices who are in grid 1, 2, 3, 4.
  // thrust has a sort by key function for this purpose.

  // dev_particleArrayIndices : Buffer containing pointer for each boid to its data in
  // dev_vel1 and dev_vel2.
  // dev particleGridIndices : Buffer containing grid index of each boid.
  // dev_gridCellStartIndices : Buffer containing a pointer for each cell to the beginning of its data in dev_particleArrayIndices.
  // Basically, Pointer to the first cell with corresponding index that belongs in that respective grid.
  // dev_gridCellEndIndices : The same as mentioned above, but pointer to the end of data.

  dim3 gridDimIterateOverBoids((numObjects + blockSize - 1) / blockSize);
  dim3 gridDimIterateOverUniformGrid((gridCellCount + blockSize - 1) / blockSize);

  kernResetIntBuffer<<<gridDimIterateOverBoids, threadsPerBlock>>>(numObjects, dev_particleArrayIndices, -1);
  checkCUDAErrorWithLine("Failed to run kernel kernResetIntBuffer.");

  kernResetIntBuffer<<<gridDimIterateOverBoids, threadsPerBlock>>>(numObjects, dev_particleGridIndices, -1);
  checkCUDAErrorWithLine("Failed to run kernel kernResetIntBuffer.");

  kernResetIntBuffer<<<gridDimIterateOverUniformGrid, threadsPerBlock>>>(gridCellCount, dev_gridCellStartIndices, -1);
  checkCUDAErrorWithLine("Failed to run kernel kernResetIntBuffer.");

  kernResetIntBuffer<<<gridDimIterateOverUniformGrid, threadsPerBlock>>>(gridCellCount, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("Failed to run kernel kernResetIntBuffer.");

  kernComputeIndices<<<gridDimIterateOverBoids, threadsPerBlock>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("failed to run kernel kernComputeIndices");

  // Sort by keys (key here is dev_particleGridIndices, value is dev_particleArrayIndices).
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // Now, setup the start and end indices.
  kernIdentifyCellStartEnd<<<gridDimIterateOverBoids, threadsPerBlock>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("failed to run kernel kernIdentifyCellStartEnd");
  
  // Run the core algorithm.
  kernUpdateVelNeighborSearchScattered<<<gridDimIterateOverBoids, threadsPerBlock>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("Failed to run kernel kernUpdateValNeighborSearchScattered");

  kernUpdatePos<<<gridDimIterateOverBoids, threadsPerBlock>>>(numObjects, dt, dev_pos, dev_vel1);
  checkCUDAErrorWithLine("Failed to run kernel kernUpdatePos");

  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationCoherentGrid(float dt)
{

  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed.

  // General idea : Have a key + value pair array, where the key is the grid index, and value is the block index.
  // If you sort by key, you can easily see which boids are in close proximity in the uniform grid.
  // For example, say you are boid X in grid 4. In a 2D configuration, if you sort by grid index,
  // you can easily see the boid indices who are in grid 1, 2, 3, 4.
  // thrust has a sort by key function for this purpose.

  // dev_particleArrayIndices : Buffer containing pointer for each boid to its data in
  // dev_vel1 and dev_vel2.
  // dev particleGridIndices : Buffer containing grid index of each boid.
  // dev_gridCellStartIndices : Buffer containing a pointer for each cell to the beginning of its data in dev_particleArrayIndices.
  // Basically, Pointer to the first cell with corresponding index that belongs in that respective grid.
  // dev_gridCellEndIndices : The same as mentioned above, but pointer to the end of data.

  const dim3 gridDimIterateOverBoids((numObjects + blockSize - 1) / blockSize);
  const dim3 gridDimIterateOverUniformGrid((gridCellCount + blockSize - 1) / blockSize);

  kernResetIntBuffer<<<gridDimIterateOverBoids, threadsPerBlock>>>(numObjects, dev_particleArrayIndices, -1);
  checkCUDAErrorWithLine("Failed to run kernel kernResetIntBuffer.");

  kernResetIntBuffer<<<gridDimIterateOverBoids, threadsPerBlock>>>(numObjects, dev_particleGridIndices, -1);
  checkCUDAErrorWithLine("Failed to run kernel kernResetIntBuffer.");

  kernResetIntBuffer<<<gridDimIterateOverUniformGrid, threadsPerBlock>>>(gridCellCount, dev_gridCellStartIndices, -1);
  checkCUDAErrorWithLine("Failed to run kernel kernResetIntBuffer.");

  kernResetIntBuffer<<<gridDimIterateOverUniformGrid, threadsPerBlock>>>(gridCellCount, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("Failed to run kernel kernResetIntBuffer.");

  kernComputeIndices<<<gridDimIterateOverBoids, threadsPerBlock>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("failed to run kernel kernComputeIndices");

  // Sort by keys (key here is dev_particleGridIndices, value is dev_particleArrayIndices).
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // Now, setup the start and end indices.
  kernIdentifyCellStartEnd<<<gridDimIterateOverBoids, threadsPerBlock>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("failed to run kernel kernIdentifyCellStartEnd");
  
  // Reshuffle the position & velocities.
  dev_thrust_reshuffledPositions = thrust::device_ptr<glm::vec3>(dev_reshuffledPositions);
  dev_thrust_reshuffledVelocities = thrust::device_ptr<glm::vec3>(dev_reshuffledVelocities);

  dev_thrust_positions = thrust::device_ptr<glm::vec3>(dev_pos);
  dev_thrust_velocities = thrust::device_ptr<glm::vec3>(dev_vel1);

  thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects, dev_thrust_positions, dev_thrust_reshuffledPositions);
  thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects, dev_thrust_velocities, dev_thrust_reshuffledVelocities);

  // Run the core algorithm.
  kernUpdateVelNeighborSearchCoherent<<<gridDimIterateOverBoids, threadsPerBlock>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_reshuffledPositions, dev_reshuffledVelocities, dev_vel2);
  checkCUDAErrorWithLine("Failed to run kernel kernUpdateValNeighborSearchScattered");

  kernUpdatePos<<<gridDimIterateOverBoids, threadsPerBlock>>>(numObjects, dt, dev_reshuffledPositions, dev_vel2);
  checkCUDAErrorWithLine("Failed to run kernel kernUpdatePos");

  std::swap(dev_vel1, dev_vel2);
  std::swap(dev_pos, dev_reshuffledPositions);
}

void Boids::endSimulation()
{
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);

  cudaFree(dev_reshuffledPositions);
  cudaFree(dev_reshuffledVelocities);
}

void Boids::unitTest()
{
  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]> intKeys{new int[N]};
  std::unique_ptr<int[]> intValues{new int[N]};

  intKeys[0] = 0;
  intValues[0] = 0;
  intKeys[1] = 1;
  intValues[1] = 1;
  intKeys[2] = 0;
  intValues[2] = 2;
  intKeys[3] = 3;
  intValues[3] = 3;
  intKeys[4] = 0;
  intValues[4] = 4;
  intKeys[5] = 2;
  intValues[5] = 5;
  intKeys[6] = 2;
  intValues[6] = 6;
  intKeys[7] = 0;
  intValues[7] = 7;
  intKeys[8] = 5;
  intValues[8] = 8;
  intKeys[9] = 6;
  intValues[9] = 9;

  cudaMalloc((void **)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void **)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++)
  {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);

  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++)
  {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
  