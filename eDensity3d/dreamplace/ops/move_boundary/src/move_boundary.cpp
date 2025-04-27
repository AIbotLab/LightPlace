/**
 * @file   move_boundary.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Move out-of-bound cells back to inside placement region
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeMoveBoundaryMapLauncher(
    T* x_tensor, T* y_tensor, T* z_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T* node_size_z_tensor, 
    const T xl,  const T yl, const T zl, const T xh, const T yh, const T zh,
    const int num_nodes, const int num_movable_nodes,
    const int num_filler_nodes, const int num_bins_z, const int num_threads);

at::Tensor move_boundary_forward(at::Tensor pos, at::Tensor node_size_x,
                                 at::Tensor node_size_y, at::Tensor node_size_z, 
                                 double xl, double yl, double zl,
                                 double xh, double yh, double zh, int num_movable_nodes,
                                 int num_filler_nodes, int num_bins_z) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  int num_nodes = pos.numel() / 3;  //3D modify
  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeMoveBoundaryMapLauncher", [&] {
        computeMoveBoundaryMapLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes*2,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t), 
            DREAMPLACE_TENSOR_DATA_PTR(node_size_z, scalar_t),
            xl, yl, zl, xh, yh, zh,
            num_nodes, num_movable_nodes, num_filler_nodes,
            num_bins_z,
            at::get_num_threads());
      });

  return pos;
}

template <typename T>
int computeMoveBoundaryMapLauncher(
    T* x_tensor, T* y_tensor, T* z_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T* node_size_z_tensor, 
    const T xl,  const T yl, const T zl, const T xh, const T yh, const T zh,
    const int num_nodes, const int num_movable_nodes,
    const int num_filler_nodes, const int num_bins_z, const int num_threads) {
    //num_bins_z 只考虑为2的情况
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nodes; ++i) {
    if (i < num_movable_nodes || i >= num_nodes - num_filler_nodes) {
      x_tensor[i] = std::max(x_tensor[i], xl);
      x_tensor[i] = std::min(x_tensor[i], xh - node_size_x_tensor[i]);

      y_tensor[i] = std::max(y_tensor[i], yl);
      y_tensor[i] = std::min(y_tensor[i], yh - node_size_y_tensor[i]);

      z_tensor[i] = std::max(z_tensor[i], zl);
      z_tensor[i] = std::min(z_tensor[i], zh - node_size_z_tensor[i]);
      
      // int mid = zh / num_bins_z;
      // if (num_bins_z == 2) {
      //   if (std::abs(z_tensor[i] - 0) > std::abs(z_tensor[i] - mid)){
      //     z_tensor[i] = mid;
      //   }
      //   else{
      //     z_tensor[i] = 0;
      //   }
      // }

    }
  }

  return 0;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::move_boundary_forward,
        "MoveBoundary forward");
  // m.def("backward", &DREAMPLACE_NAMESPACE::move_boundary_backward,
  // "MoveBoundary backward");
}
