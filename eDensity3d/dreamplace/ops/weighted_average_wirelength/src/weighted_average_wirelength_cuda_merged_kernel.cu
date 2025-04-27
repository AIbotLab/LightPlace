/**
 * @file   weighted_average_wirelength_cuda_merged_kernel.cu
 * @author Yibo Lin
 * @date   Sep 2019
 */

#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computeWeightedAverageWirelength(
    const T *x, const T *y, const T *z,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    const T *inv_gamma,
    T *partial_wl,
    T *grad_intermediate_x, T *grad_intermediate_y, T* grad_intermediate_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = i / 3;  // 3D modify
    if (ii < num_nets && net_mask[ii])
    {
        const T *values;
        T *grads;
        if (i % 3 == 0)  // 3D modify
        {
            values = y;
            grads = grad_intermediate_y;
        }
        else if(i % 3 == 1)
        {
            values = x;
            grads = grad_intermediate_x;
        }
        else{  // 3D modify
            values = z;
            grads = grad_intermediate_z;
        }

        // int degree = netpin_start[ii+1]-netpin_start[ii];
        T x_max = -FLT_MAX;
        T x_min = FLT_MAX;
        
        // to find the max & min position of module pin
        for (int j = netpin_start[ii]; j < netpin_start[ii + 1]; ++j)
        {
            T xx = values[flat_netpin[j]]; //get position of pin
            x_max = max(xx, x_max);
            x_min = min(xx, x_min);
        }

        T xexp_x_sum = 0;
        T xexp_nx_sum = 0;
        T exp_x_sum = 0;
        T exp_nx_sum = 0;
        for (int j = netpin_start[ii]; j < netpin_start[ii + 1]; ++j)
        {
            T xx = values[flat_netpin[j]];
            T exp_x = exp((xx - x_max) * (*inv_gamma));  //对应文章的a_i_+
            T exp_nx = exp((x_min - xx) * (*inv_gamma)); //对应文章的a_i_-

            xexp_x_sum += xx * exp_x;    //对应文章的c_e_+
            xexp_nx_sum += xx * exp_nx;  //对应文章的c_e_-
            exp_x_sum += exp_x;          //对应文章的b_e_+
            exp_nx_sum += exp_nx;        //对应文章的b_e_-
        }

        partial_wl[i] = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum; //算出一个net的近似线长WA

        //3D modify  作用应该是记录每个pin的中间梯度  暂时没看懂下面怎么算的 10.24 
        // 已看懂    对应的就是论文中WA的求导公式
        T b_x = (*inv_gamma) / (exp_x_sum);  
        T a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
        T b_nx = -(*inv_gamma) / (exp_nx_sum);
        T a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

        for (int j = netpin_start[ii]; j < netpin_start[ii + 1]; ++j)
        {
            T xx = values[flat_netpin[j]];
            T exp_x = exp((xx - x_max) * (*inv_gamma));
            T exp_nx = exp((x_min - xx) * (*inv_gamma));

            grads[flat_netpin[j]] = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;
        }
    }
}

template <typename T>
int computeWeightedAverageWirelengthCudaMergedLauncher(
    const T *x, const T *y, const T* z,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    const T *inv_gamma,
    T *partial_wl,
    T *grad_intermediate_x, T *grad_intermediate_y, T* grad_intermediate_z)
{
    int thread_count = 64;
    int block_count = (num_nets * 3 + thread_count - 1) / thread_count; // separate x and y
    // 3D modify  separate x y and z
    computeWeightedAverageWirelength<<<block_count, thread_count>>>(
        x, y, z,
        flat_netpin,
        netpin_start,
        net_mask,
        num_nets,
        inv_gamma,
        partial_wl,
        grad_intermediate_x, grad_intermediate_y, grad_intermediate_z);

    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                    \
    template int computeWeightedAverageWirelengthCudaMergedLauncher<T>( \
        const T *x, const T *y, const T* z,                             \
        const int *flat_netpin,                                        \
        const int *netpin_start,                                       \
        const unsigned char *net_mask,                                 \
        int num_nets,                                                  \
        const T *inv_gamma,                                            \
        T *partial_wl,                                                 \
        T *grad_intermediate_x, T *grad_intermediate_y, T *grad_intermediate_z);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
