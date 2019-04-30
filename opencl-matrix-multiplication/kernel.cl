
__kernel void addVectors(__global float* tab1, __global float* tab2, __global float* tab3)
{
    int idx=get_global_id(0);//*get_local_size(0)+get_local_id(0);
    tab3[idx]=tab1[idx]+tab2[idx];
}

