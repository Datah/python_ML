def replace_in_tensor(r, c, l, new_val):
    layer_matrix = tensor[l];
    row_matrix = layer_matrix[r]
    print("Entry: {}".format(row_matrix[c]))
    new_row_matrix = row_matrix[:c] + ((new_val),) + row_matrix[c+1:]
    new_layer_matrix = layer_matrix[:r] + (new_row_matrix,) + layer_matrix[r+1:]
    new_tensor = tensor[:l] + [new_layer_matrix] + tensor[l+1:]
    return new_tensor


def incr_in_tensor(tensor, r, c, l, incr):
    layer_matrix = tensor[l];
    row_matrix = layer_matrix[r]
    print("Entry: {}".format(row_matrix[c]))
    new_row_matrix = row_matrix[:c] + ((row_matrix[c] + incr),) + row_matrix[c+1:]
    new_layer_matrix = layer_matrix[:r] + (new_row_matrix,) + layer_matrix[r+1:]
    new_tensor = tensor[:l] + [new_layer_matrix] + tensor[l+1:]
    return new_tensor