#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief Compute pin position from node position 
template <typename T, typename K>
__global__ void computePinPos(
	const T* x, const T* y, const T* z, 
	const T* pin_offset_x,
	const T* pin_offset_y,
	const T* pin_offset_z,
	const K* pin2node_map,
	const int num_pins,
	T* pin_x, T* pin_y, T* pin_z
	)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
	{
		int node_id = pin2node_map[i];
		pin_x[i] = pin_offset_x[i] + x[node_id];
		pin_y[i] = pin_offset_y[i] + y[node_id];
		pin_z[i] = pin_offset_z[i] + z[node_id]; //3D modify
	}
}

template <typename T>
int computePinPosCudaLauncher(  //3D modify
	const T* x, const T* y, const T* z,
	const T* pin_offset_x,
	const T* pin_offset_y,
	const T* pin_offset_z,
	const long* pin2node_map,
	const int* flat_node2pin_map,
	const int* flat_node2pin_start_map,
	int num_pins,
	T* pin_x, T* pin_y, T* pin_z
    )
{
	int thread_count = 512;

	computePinPos<<<(num_pins+thread_count-1) / thread_count, thread_count>>>(
		x, y, z, pin_offset_x, pin_offset_y, pin_offset_z, 
		pin2node_map, num_pins, pin_x, pin_y, pin_z);

    return 0;
}

/// @brief Compute pin position from node position 
template <typename T>
__global__ void computeNodeGrad(
	const T* grad_out_x,
	const T* grad_out_y,
	const T* grad_out_z,
	const int* flat_node2pin_map,
    const int* flat_node2pin_start_map, 
    const int num_nodes, 
	T* grad_x,
	T* grad_y,
	T* grad_z
	)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes)
	{
        T& gx = grad_x[i];
        T& gy = grad_y[i];
		T& gz = grad_z[i];
        gx = 0; 
        gy = 0; 
		gz = 0;
        for (int j = flat_node2pin_start_map[i]; j < flat_node2pin_start_map[i+1]; ++j)
        {
            int pin_id = flat_node2pin_map[j]; 
            gx += grad_out_x[pin_id]; 
            gy += grad_out_y[pin_id]; 
			gz += grad_out_z[pin_id];
        }
	}
}

template <typename T>
int computePinPosGradCudaLauncher(
	const T* grad_out_x, const T* grad_out_y, const T* grad_out_z,
	const T* x, const T* y, const T* z,
	const T* pin_offset_x,
	const T* pin_offset_y,
	const T* pin_offset_z,
	const long* pin2node_map,
	const int* flat_node2pin_map,
	const int* flat_node2pin_start_map,
	int num_nodes,
	int num_pins,
	T* grad_x, T* grad_y, T* grad_z
    )
{
    int thread_count = 512;

    computeNodeGrad<<<(num_nodes + thread_count - 1) / thread_count, thread_count>>>(
            grad_out_x, 
            grad_out_y, 
			grad_out_z,
            flat_node2pin_map, 
            flat_node2pin_start_map, 
            num_nodes, 
            grad_x, 
            grad_y,
			grad_z
            );

    return 0;	
}


#define REGISTER_KERNEL_LAUNCHER(T) \
    template int computePinPosCudaLauncher<T>(\
    	    const T* x, const T* y, const T* z,\
    	    const T* pin_offset_x, \
	        const T* pin_offset_y, \
			const T* pin_offset_z, \
	        const long* pin2node_map, \
	        const int* flat_node2pin_map, \
	        const int* flat_node2pin_start_map, \
	        int num_pins, \
	        T* pin_x, T* pin_y, T* pin_z\
            ); \
    \
    template int computePinPosGradCudaLauncher<T>(\
        	const T* grad_out_x, const T* grad_out_y, const T* grad_out_z,\
	        const T* x, const T* y, const T* z,\
	        const T* pin_offset_x, \
	        const T* pin_offset_y, \
			const T* pin_offset_z, \
	        const long* pin2node_map, \
	        const int* flat_node2pin_map, \
	        const int* flat_node2pin_start_map, \
	        int num_nodes, \
	        int num_pins, \
	        T* grad_x, T* grad_y, T* grad_z \
		); 

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

// template int computePinPosCudaLauncher<float>(
// 		const float* x, const float* y, const float* z
// 		const float* pin_offset_x, 
// 		const float* pin_offset_y, 
// 		const float* pin_offset_z, 
// 		const long* pin2node_map, 
// 		const int* flat_node2pin_map, 
// 		const int* flat_node2pin_start_map, 
// 		int num_pins, 
// 		float* pin_x, float* pin_y, float* pin_z
// 		); 

// template int computePinPosGradCudaLauncher<float>(\
// 		const float* grad_out_x, const float* grad_out_y, const float* grad_out_z,\
// 		const float* x, const float* y, const float* z,\
// 		const float* pin_offset_x, \
// 		const float* pin_offset_y, \
// 		const float* pin_offset_z, \
// 		const long* pin2node_map, \
// 		const int* flat_node2pin_map, \
// 		const int* flat_node2pin_start_map, \
// 		int num_nodes, \
// 		int num_pins, \
// 		float* grad_x, float* grad_y, float* grad_z \
// 	); 


// template int computePinPosCudaLauncher<double>(
// 		const double* x, const double* y, const double* z
// 		const double* pin_offset_x, 
// 		const double* pin_offset_y, 
// 		const double* pin_offset_z, 
// 		const long* pin2node_map, 
// 		const int* flat_node2pin_map, 
// 		const int* flat_node2pin_start_map, 
// 		int num_pins, 
// 		double* pin_x, double* pin_y, double* pin_z
// 		); 

// template int computePinPosGradCudaLauncher<double>(\
// 		const double* grad_out_x, const double* grad_out_y, const double* grad_out_z,\
// 		const double* x, const double* y, const double* z,\
// 		const double* pin_offset_x, \
// 		const double* pin_offset_y, \
// 		const double* pin_offset_z, \
// 		const long* pin2node_map, \
// 		const int* flat_node2pin_map, \
// 		const int* flat_node2pin_start_map, \
// 		int num_nodes, \
// 		int num_pins, \
// 		double* grad_x, double* grad_y, double* grad_z \
// 	); 

DREAMPLACE_END_NAMESPACE
