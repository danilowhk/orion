use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;
use orion::numbers::signed_integer::i8::{i8,i8_logical_and};

use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::core::{Tensor, TensorTrait};

use orion::operators::tensor::helpers::check_compatibility;


fn generate_output(tensor_a_shape: Array<i8>, tensor_b_shape: Array<i8>, output_shape: Array<i8>, y: @Tensor<i8>, z: @Tensor<i8>) -> Tensor<i8> {
    //1. Create dict_of_indexes inside tensar_a_shape, tensar_b_shape, output_shape
    let mut indexes_dict_u8 = felt252_dict_new::<u8>();
    let mut indexes_dict_counter = 0;

    let mut shape_a_counter = Array.len();
    let mut shape_b_counter = Array.len();
    let mut shape_output_counter = Array.len();
    // Loop for shape_a and add all indexes in indexes_dict_u8
    loop {
        //TODO: Maybe dont need counter, just check tensor_a_shape.len()
        if shape_a_counter == 0 {
            break ();
        };
        //TODO: Check order or array
        let shape_value = tensor_a_shape.pop_front();
        let value = dict_u8[shape_value];
        match value {
            Some(value) => {
                shape_a_counter = shape_a_counter - 1;
            },
            None => {
                dict_u8[indexes_dict_counter] = 1;
                indexes_dict_counter = indexes_dict_counter + 1;
                shape_a_counter = shape_a_counter - 1;
            }
        }
    }
    // Loop for shape_b and add all indexes in indexes_dict_u8
    loop {
        //TODO: Maybe dont need counter, just check tensor_b_shape.len()
        if shape_b_counter == 0 {
            break ();
        };
        //TODO: Check order or array
        let shape_value = tensor_b_shape.pop_front();
        let value = dict_u8[shape_value];
        match value {
            Some(value) => {
                shape_b_counter = shape_b_counter - 1;
            },
            None => {
                dict_u8[shape_value] = 1;
                shape_b_counter = shape_b_counter - 1;
            }
        }
    }
    //3.  Loop for shape_b and create not_output_dict (List from all indexes that is not in output_shape)
    let mut not_output_dict = felt252_dict_new::<u8>();
    let mut not_output_counter = 0;
    // Loop for shape_output
    loop {
        let counter = 0;
        if shape_output_counter == 0 {
            break ();
        };
        let shape_value = shape_output[counter];
        let value = dict_u8[shape_value];
        match value {
            Some(value) => {
                counter = counter + 1;
            },
            None => {
                not_output_dict[not_output_counter] = shape_value;
                not_output_counter = not_output_counter + 1;
                counter = counter + 1;
            }
        }
    }
    let mut result_tensor = Tensor::new::<i8>();
    //TODO: 4. Loop through out shape
    // einsum_nested_loop();
    return result_tensor;
}

// Nested loop function

fn einsum_nested_loop(n : usize, array_n: Array<usize>, m: usize, array_m: Array<usize> , tensor_a: Tensor<i8> , tensor_b: Tensor<i8>, a_indices: Array<i8> , b_indices: Array<i8>, results_indices: Array<i8> , n_index: usize, m_index: usize, indices: Array<usize>) {
    if n_index < n {
        let i = 0;
        loop {
            if i >= array_n[n_index] {
                break ();
            };
            // TODO: fix indices , or consider using dict
            einsum_nested_loop(n, array_n, m, array_m, tensor_a, tensor_b, a_indices, b_indices, results_indices, n_index + 1, m_index, indices + [i]);
            i = i + 1;
        }
    } else if (m_index < m) {
        let i = 0;
        loop {
            if i >= array_m[m_index] {
                break ();
            };
            // TODO: fix indices , or consider using dict
            einsum_nested_loop(n, array_n, m, array_m, tensor_a, tensor_b, a_indices, b_indices, results_indices, n_index, m_index + 1, indices + [i]);
            i = i + 1;
        }
    } else {
        let results_tensor_indices = ArrayTrait::new::<i8>();
        let idx = 0;
        loop {
            if idx >= indices.len() {
                break ();
            };
            let index = indices[idx];
            results_tensor_indices.append(index);
            idx = idx + 1;
        }

        let a_tensor_indices = ArrayTrait::new::<i8>();
        let idx = 0;
        loop {
            if idx >= a_indices.len() {
                break ();
            };
            let index = a_indices[idx];
            a_tensor_indices.append(index);
            idx = idx + 1;
        }

        let b_tensor_indices = ArrayTrait::new::<i8>();
        let idx = 0;
        loop {
            if idx >= b_indices.len() {
                break ();
            };
            let index = b_indices[idx];
            b_tensor_indices.append(index);
            idx = idx + 1;
        }

        set_element(result_tensor, results_tensor_indices, get_element(tensor_a, a_tensor_indices) * get_element(tensor_b, b_tensor_indices));
    }

}
// TODO: Test get_element
fn get_element(tensor: Tensor<i8>, indices: Array<i8>) -> i8 {
    let mut tensor = tensor;
    let mut indices = indices;
    let index = indices.pop_front();
    match index {
        Some(index) => {
            let tensor = tensor[index];
            get_element(tensor, indices)
        },
        None => {
            tensor
        }
    }
}
// TODO: Test set_element
fn set_element(tensor: Tensor<i8>, indices: Array<i8>, value: i8) {
    let mut tensor = tensor;
    let mut indices = indices;
    let mut value = value;
    let index = indices.pop_front();
    match index {
        Some(index) => {
            let tensor = tensor[index];
            set_element(tensor, indices, value);
        },
        None => {
            tensor = value;
        }
    }
}





