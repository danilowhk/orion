use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;
use orion::numbers::signed_integer::i8::i8;
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::core::{Tensor, TensorTrait};

use orion::operators::tensor::helpers::check_compatibility;

fn and(y: @Tensor<i8>, z: @Tensor<i8>) -> Tensor<usize> {
    check_compatibility(*y.shape, *z.shape);

    let mut data_result = ArrayTrait::<usize>::new();
}