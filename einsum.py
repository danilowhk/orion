import torch

# n: The number of dimensions in output tensor
# array_n: List containing the size of each dimension in output tensor
# m: The number of dimensions in non repeated non output tensor
# array_m: List containing the size of each dimension in non repeated non output tensor
# tensor_a: The first tensor for the operation
# tensor_b: The second tensor for the operation
# result_tensor: The tensor where the result of the operation will be stored
# a_indices: List containing the indices in "indices" list that are used to index tensor_a
# b_indices: List containing the indices in "indices" list that are used to index tensor_b
# result_indices: List containing the indices in "indices" list that are used to index result_tensor
# n_index: Current index for the dimensions of tensor_a being looped over (for recursion)
# m_index: Current index for the dimensions of tensor_b being looped over (for recursion)
# indices: List of current indices for each dimension (for recursion)
def nested_loop(n, array_n, m, array_m, tensor_a, tensor_b, result_tensor, a_indices, b_indices, result_indices, n_index=0, m_index=0, indices=[]):
    # Run first through the dimensions of output_tensor then on values that are not in output_tensor
    if n_index < n:

        i = 0
        while i < array_n[n_index]:
            nested_loop(n, array_n, m, array_m, tensor_a, tensor_b, result_tensor, a_indices, b_indices, result_indices, n_index + 1, m_index, indices + [i])
            i += 1
    elif m_index < m:

        i = 0
        while i < array_m[m_index]:
            nested_loop(n, array_n, m, array_m, tensor_a, tensor_b, result_tensor, a_indices, b_indices, result_indices, n_index, m_index + 1, indices + [i])
            i += 1
    else:
    # Calculate the indices for result_tensor based on result_indices


        result_tensor_indices = []
        idx = 0
        while idx < len(result_indices):
            result_tensor_indices.append(indices[result_indices[idx]])
            idx += 1

        # Calculate the indices for tensor_a based on a_indices
        tensor_a_indices = []
        idx = 0
        while idx < len(a_indices):
            tensor_a_indices.append(indices[a_indices[idx]])
            idx += 1            

        # Calculate the indices for tensor_b based on b_indices
        tensor_b_indices = []
        idx = 0
        while idx < len(b_indices):
            tensor_b_indices.append(indices[b_indices[idx]])
            idx += 1
        
        print("Before set_element indices" , indices)
        print("Before set_element result_tensor_indices" , result_tensor_indices)
        print("Before set_element tensor_a_indices" , a_indices)
        print("Before set_element tensor_b_indices" , b_indices)
        set_element(result_tensor, result_tensor_indices, get_element(tensor_a, tensor_a_indices) * get_element(tensor_b, tensor_b_indices))


# Perform the multiplication and assign the result to the corresponding position in result_tensor
# We use a recursive function to handle the variable number of dimensions
def get_element(tensor, indices):
    if len(indices) == 1:
        return tensor[indices[0]]
    else:
        return get_element(tensor[indices[0]], indices[1:])    

def set_element(tensor, indices, value):
    if len(indices) == 1:
        tensor[indices[0]] = value
    else:
        set_element(tensor[indices[0]], indices[1:], value)


      


def test_nested_loop():
    # Test 1
    tensor_a = torch.randn(3, 4, 4)
    tensor_b = torch.randn(4, 2)
    result_tensor = torch.zeros(3, 4, 4, 2)
    a_indices = [0, 1, 2]
    b_indices = [2, 3]
    result_indices = [0, 1, 2, 3]
    nested_loop(3, [3, 4, 4], 1, [2], tensor_a, tensor_b, result_tensor, a_indices, b_indices, result_indices)
    result_einsum = torch.einsum('ijk,kl->ijkl', tensor_a, tensor_b)
    print("TEST 1")
    # Loop over all indices
    result_for_loop = torch.zeros(3, 4, 4, 2)
    assert torch.allclose(result_tensor, result_einsum), "Test 1 Failed"

    # Test 2
    tensor_a = torch.randn(5, 7)
    tensor_b = torch.randn(7, 3)
    result_tensor = torch.zeros(5, 7, 3)
    a_indices = [0, 1]
    b_indices = [1, 2]
    result_indices = [0, 1, 2]
    # n is 3 because the output tensor has 3 dimensions
    # array_n is [5, 7, 3] because the output tensor's dimensions are 5, 7, and 3
    # m is 0 because there are no unique dimensions in tensor_a and tensor_b that are not in the output tensor
    # array_m is [] because there are no unique dimensions in tensor_a and tensor_b that are not in the output tensor
    nested_loop(3, [5, 7, 3], 0, [], tensor_a, tensor_b, result_tensor, a_indices, b_indices, result_indices)
    result_einsum = torch.einsum('ij,jk->ijk', tensor_a, tensor_b)
    print("TEST 1")



    assert torch.allclose(result_tensor, result_einsum), "Test 2 Failed"


    # # Test 3
    tensor_a = torch.randn(2, 3, 4)
    tensor_b = torch.randn(4, 5)
    result_tensor = torch.zeros(2, 3, 4, 5)
    a_indices = [0, 1, 2]
    b_indices = [2, 3]
    result_indices = [0, 1, 2, 3]
    nested_loop(4, [2, 3, 4, 5], 0, [], tensor_a, tensor_b, result_tensor, a_indices, b_indices, result_indices)
    result_einsum = torch.einsum('ijk,kl->ijkl', tensor_a, tensor_b)
    print("TEST 3")
    assert torch.allclose(result_tensor, result_einsum), "Test 3 Failed"

    # Test 4
    tensor_a = torch.randn(2, 3)
    tensor_b = torch.randn(3, 2)
    result_tensor = torch.zeros(2, 3, 2)
    a_indices = [0, 1]
    b_indices = [1, 2]
    result_indices = [0, 1, 2]
    nested_loop(3, [2, 3, 2], 0, [], tensor_a, tensor_b, result_tensor, a_indices, b_indices, result_indices)
    result_einsum = torch.einsum('ij,jk->ijk', tensor_a, tensor_b)
    print("TEST 4")
    assert torch.allclose(result_tensor, result_einsum), "Test 4 Failed"

    # Test with 1D tensors
    tensor_a = torch.randn(5)
    tensor_b = torch.randn(5)
    result_tensor = torch.zeros(5)
    a_indices = [0]
    b_indices = [0]
    result_indices = [0]
    nested_loop(1, [5], 0, [], tensor_a, tensor_b, result_tensor, a_indices, b_indices, result_indices)
    result_einsum = torch.einsum('i,i->i', tensor_a, tensor_b)
    print("TEST 5")
    assert torch.allclose(result_tensor, result_einsum), "1D Test Failed"

    #Test with 2D tensors
    tensor_a = torch.randn(3, 4)
    tensor_b = torch.randn(4, 2)

    result_einsum = torch.einsum('ij,jk->ik', tensor_a, tensor_b)
    print("result_einsum: ", result_einsum)

        # assuming tensor_a and tensor_b are already defined
    n, m = tensor_a.shape  # tensor_a is of shape (n, m)
    m, p = tensor_b.shape  # tensor_b is of shape (m, p)

    result_for_loop = torch.zeros((3, 2))  # initialize result tensor with shape (n, p)

    for i in range(3):
        for k in range(2):
            for j in range(4):
                result_for_loop[i, k] += tensor_a[i, j] * tensor_b[j, k]
                print(f"i: {i}, k: {k}, -  i: {i}, j: {j},- j: {j} k: {k}")
                # print(f"i: {i}, j: {j}, k: {k}, result_for_loop: {result_for_loop}")
    # assert torch.allclose(result_tensor, result_einsum), "2D Test Failed"

    print("---------------------------")
    result_tensor = torch.zeros(3, 2)
    a_indices = [0, 2]
    b_indices = [2, 1]
    result_indices = [0, 1]
    nested_loop(2, [3, 2], 1, [4], tensor_a, tensor_b, result_tensor, a_indices, b_indices, result_indices)

    print("TEST 6")


    print("All tests passed!")

test_nested_loop()


