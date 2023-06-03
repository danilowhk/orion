use orion::operators::tensor::core::Tensor;
use orion::numbers::signed_integer::i32::i32;
use orion::operators::nn::core::{NNTrait};
use orion::operators::nn::functional::relu::relu_i32::relu_i32;
use orion::operators::nn::functional::softmax::softmax_i32::softmax_i32_fp8x23;
use orion::operators::nn::functional::softsign::softsign_i32::softsign_i32_fp8x23;
use orion::operators::nn::functional::softplus::softplus_i32::softplus_i32_fp8x23;
use orion::operators::nn::functional::linear::linear_i32::linear_i32;
use orion::operators::nn::functional::leaky_relu::leaky_relu_i32::leaky_relu_i32_fp8x23;
use orion::numbers::fixed_point::core::{FixedType};
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;

impl NN_i32_fp8x23 of NNTrait<i32, fp8x23> {

    fn relu(tensor: @Tensor<i32>, threshold: i32) -> Tensor<i32> {
        relu_i32(tensor, threshold)
    }

    fn softmax(tensor: @Tensor<i32>, axis: usize) -> Tensor<FixedType<fp8x23>> {
        softmax_i32_fp8x23(tensor, axis)
    }

    fn softsign(tensor: @Tensor<i32>) -> Tensor<FixedType<fp8x23>> {
        softsign_i32_fp8x23(tensor)
    }

    fn softplus(tensor: @Tensor<i32>) -> Tensor<FixedType<fp8x23>> {
        softplus_i32_fp8x23(tensor)
    }

    fn linear(
        inputs: Tensor<i32>, weights: Tensor<i32>, bias: Tensor<i32>, quantized: bool
    ) -> Tensor<i32> {
        linear_i32(inputs, weights, bias, quantized)
    }

    fn leaky_relu(
        inputs: @Tensor<i32>, alpha: @FixedType<fp8x23>, threshold: i32
    ) -> Tensor<FixedType<fp8x23>> {
        leaky_relu_i32_fp8x23(inputs, alpha, threshold)
    }
}
