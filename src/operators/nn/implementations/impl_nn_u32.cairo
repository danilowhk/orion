use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::{NNTrait};
use orion::operators::nn::functional::relu::relu_u32::relu_u32;
use orion::operators::nn::functional::softmax::softmax_u32::softmax_u32_fp8x23;
use orion::operators::nn::functional::softsign::softsign_u32::softsign_u32_fp8x23;
use orion::operators::nn::functional::softplus::softplus_u32::softplus_u32_fp8x23;
use orion::operators::nn::functional::linear::linear_u32::linear_u32;
use orion::operators::nn::functional::leaky_relu::leaky_relu_u32::leaky_relu_u32_fp8x23;
use orion::numbers::fixed_point::core::{FixedType};
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;

impl NN_u32_fp8x23 of NNTrait<u32, fp8x23> {

    fn relu(tensor: @Tensor<u32>, threshold: u32) -> Tensor<u32> {
        relu_u32(tensor, threshold)
    }

    fn softmax(tensor: @Tensor<u32>, axis: usize) -> Tensor<FixedType<fp8x23>> {
        softmax_u32_fp8x23(tensor, axis)
    }

    fn softsign(tensor: @Tensor<u32>) -> Tensor<FixedType<fp8x23>> {
        softsign_u32_fp8x23(tensor)
    }

    fn softplus(tensor: @Tensor<u32>) -> Tensor<FixedType<fp8x23>> {
        softplus_u32_fp8x23(tensor)
    }

    fn linear(
        inputs: Tensor<u32>, weights: Tensor<u32>, bias: Tensor<u32>, quantized: bool
    ) -> Tensor<u32> {
        linear_u32(inputs, weights, bias, quantized)
    }

    fn leaky_relu(
        inputs: @Tensor<u32>, alpha: @FixedType<fp8x23>, threshold: u32
    ) -> Tensor<FixedType<fp8x23>> {
        leaky_relu_u32_fp8x23(inputs, alpha, threshold)
    }
}
