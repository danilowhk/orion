use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::{Tensor};
use orion::operators::tensor::math::and::and_fp::fp8x23;
use orion::operators::tensor::math::and::and_fp::f16x16;

/// Cf: TensorTrait::and docstring
fn and(y: @Tensor<FixedType>, z: @Tensor<FixedType>) -> Option<Tensor<usize>> {
    match *y.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::and(y,z)),
                FixedImpl::FP16x16(()) => Option::Some(f16x16::and(y,z)),
            },
            Option::None(_) => Option::Some(fp16x16::and(y,z)),
        },
        Option::None(_) => Option::Some(fp16x16::and(y,z)),
    }
}