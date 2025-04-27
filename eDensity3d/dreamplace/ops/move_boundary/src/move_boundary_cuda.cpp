/**
 * @file   hpwl_cuda.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Move out-of-bound cells back to inside placement region
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeMoveBoundaryMapCudaLauncher(T* x_tensor, T* y_tensor, T* z_tensor,
                                       const T* node_size_x_tensor,
                                       const T* node_size_y_tensor, 
                                       const T* node_size_z_tensor,
                                       const T xl,const T yl, const T zl,
                                       const T xh, const T yh, const T zh,
                                       const int num_nodes,
                                       const int num_movable_nodes,
                                       const int num_filler_nodes);

at::Tensor move_boundary_forward(at::Tensor pos, at::Tensor node_size_x,
                                 at::Tensor node_size_y,at::Tensor node_size_z,
                                 double xl, double yl, double zl,
                                 double xh, double yh, double zh,
                                 int num_movable_nodes,
                                 int num_filler_nodes) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  int num_nodes = pos.numel() / 3;  //3D modify

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeMoveBoundaryMapCudaLauncher", [&] {
        computeMoveBoundaryMapCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes*2,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t), 
            DREAMPLACE_TENSOR_DATA_PTR(node_size_z, scalar_t), 
            xl, yl, zl, xh, yh, zh,
            num_nodes, num_movable_nodes, num_filler_nodes);
      });

  return pos;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::move_boundary_forward,
        "MoveBoundary forward (CUDA)");
  // m.def("backward", &DREAMPLACE_NAMESPACE::move_boundary_backward,
  // "MoveBoundary backward (CUDA)");
}
