__kernel void matrix_mul(__global const int *A,__global const int *B, __global int *C, __local int *blk_a, __local int *blk_b)
{
    int dim = get_global_size(0);
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    // Get the index of the current element to be processed
    int lrow = get_local_id(0);
    int lcol = get_local_id(1);
    int ldim = get_local_size(0);
    
    // Do the operation
    int i, j;
    int sum = 0;
    for (i=0; i< (dim/ldim); i++)
    {
        blk_a [lrow*ldim+lcol] = A[row*dim+(i*ldim+lcol)];
        blk_b [lrow*ldim+lcol] = B[(i*ldim+lrow)*dim + col];
        
        // Synchronise
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (j = 0; j<ldim ; j++)
        {
            sum += blk_a[lrow*ldim+j]*blk_b[j*ldim+lcol];
        }
        
        // Synchronise
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[row*dim+col] = sum;
}
