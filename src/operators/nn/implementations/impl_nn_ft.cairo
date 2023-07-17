use core::option::OptionTrait;
use orion::operators::tensor::core::Tensor;
use orion::numbers::signed_integer::i8::i8;
use orion::operators::nn::core::{NNTrait};
use orion::operators::nn::functional::relu::relu_i8::relu_i8;
use orion::operators::nn::functional::sigmoid::sigmoid_i8::core::sigmoid_i8;
use orion::operators::nn::functional::softmax::softmax_i8::softmax_i8;
use orion::operators::nn::functional::logsoftmax::logsoftmax_i8::logsoftmax_i8;
use orion::operators::nn::functional::softsign::softsign_i8::core::softsign_i8;
use orion::operators::nn::functional::softplus::softplus_i8::core::softplus_i8;
use orion::operators::nn::functional::linear::linear_ft::linear_ft;
use orion::operators::nn::functional::leaky_relu::leaky_relu_i8::core::leaky_relu_i8;
use orion::numbers::fixed_point::core::{FixedType};


impl NN_i8 of NNTrait<FixedType> {

    fn linear(inputs: Tensor<FixedType>, weights: Tensor<FixedType>, bias: Tensor<FixedType>) -> Tensor<FixedType> {
        linear_i8(inputs, weights, bias)
    }

}
