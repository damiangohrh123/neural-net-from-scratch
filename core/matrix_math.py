def dot_product(A, B):
    """
    Computes matrix multiplication: C = A @ B
    Required for Forward Pass and Weight Gradients
    """
    # Get dimensions of both matrices
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    # Make sure A's columns match B's rows for valid multiplication 
    if cols_A != rows_B:
        raise ValueError(f"Incompatible dimensions: {cols_A} != {rows_B}")

    # Transpose B to make columns accessible as lists
    B_T = [[B[i][j] for i in range(rows_B)] for j in range(cols_B)]

    result = []
    for row_A in A:
            new_row = []
            for col_B in B_T:
                cell_value = 0
                # Pair up elements from row_A and col_B using zip, and perform dot product
                for a, b in zip(row_A, col_B):
                    cell_value += a * b
                
                new_row.append(cell_value)
                
            result.append(new_row)
        
    return result

def transpose(A):
    """
    Required for mapping error signals backward
    """
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def add_matrices(A, B):
    """
    Required for adding Biases
    """
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def subtract_matrices(A, B):
    """
    Required for calculating the Error Term: (y_hat - y)
    """
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def hadamard_product(A, B):
    """
    Element-wise multiplication
    Used for: delta = (W_next^T * delta_next) * f'(z)
    """
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def scalar_multiply(A, scalar):
    """
    Used for the learning rate update: W = W - (eta * gradient) (Sec 6.1)
    """
    return [[val * scalar for val in row] for row in A]