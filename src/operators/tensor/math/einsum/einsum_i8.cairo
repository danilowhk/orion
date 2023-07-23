use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;
use orion::numbers::signed_integer::i8::{i8,i8_logical_and};

use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::core::{Tensor, TensorTrait};

use orion::operators::tensor::helpers::check_compatibility;

fn and(y: @Tensor<i8>, z: @Tensor<i8>) -> Tensor<usize> {
    check_compatibility(*y.shape, *z.shape);

    let mut data_result = ArrayTrait::<usize>::new();

    let (mut smaller, mut bigger, retains_input_order) = if (*y.data).len() < (*z.data).len() {
        (y, z, true)
    } else {
        (z, y , false)
    };

    let mut bigger_data = *bigger.data;
    let mut smaller_data = *smaller.data;
    let mut smaller_index = 0;

    loop {
        if bigger_data.len() == 0 {
            break ();
        };

        let bigger_current_index = *bigger_data.pop_front().unwrap();
        let smaller_current_index = *smaller_data[smaller_index];

        let (y_value, z_value) = if retains_input_order {
            (smaller_current_index, bigger_current_index)
        } else {
            (bigger_current_index, smaller_current_index)
        };

        if i8_logical_and(y_value, z_value) {
            data_result.append(1);
        } else {
            data_result.append(0);
        }

        smaller_index = (1 + smaller_index) % smaller_data.len();
    };

    return TensorTrait::<usize>::new(*bigger.shape, data_result.span(), *y.extra);
}

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

    //4. Loop through out shape
    let n = 3;
    let value_n1 = 3;
    let value_n2 = 4;
    let value_n3 = 4;


    loop {
        // n = 3
        if value_n1 == 0 {
            break ();
        };
        loop {
            // n = 2
            if value_n2 == 0 {
                break ();
            } 
            // n = 1
            loop {
                if value_n3 == 0 {
                    break ();
                }
            }

        }
    }

    }



