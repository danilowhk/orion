use array::SpanTrait;

use orion::numbers::fixed_point::core::FixedType;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_i8::{Tensor_i8, i8TensorAdd};
use orion::performance::core::PerfomanceTrait;
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::implementations::impl_tensor_fp::FixedTypeTensorAdd;



/// Cf: NNTrait::linear docstring
fn linear_ft(z: Tensor<FixedType>, weights: Tensor<FixedType>, bias: Tensor<FixedType>) -> Tensor<FixedType> {
    assert(z.shape.len() == 1, 'input tensor must be 1D');
    assert(weights.shape.len() == 2, 'weights tensor must be 2D');
    assert(bias.shape.len() == 1, 'bias tensor must be 1D');

    let dot = weights.matmul(@z);
    let sum = dot + bias;

    return sum;
}
