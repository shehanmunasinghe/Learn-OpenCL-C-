
void kernel add(global const int* v1, global const int* v2, global int* v3) {
    int ID;
    ID = get_global_id(0);
    v3[ID] = v1[ID] + v2[ID];
}

void kernel add_looped_1(global const int* v1, global const int* v2, global int* v3, const int n, const int k) {
    int ID, NUM_GLOBAL_WITEMS, ratio, start, stop;
    ID = get_global_id(0);
    NUM_GLOBAL_WITEMS = get_global_size(0);

    ratio = (n / NUM_GLOBAL_WITEMS); // elements per thread
    start = ratio * ID;
    stop  = ratio * (ID+1);

    int i, j; // will the compiler optimize this anyway? probably.
    for (i=0; i<k; i++) {
        for (j=start; j<stop; j++)
            v3[j] = v1[j] + v2[j];
    }
}

void kernel add_looped_2(global const int* v1, global const int* v2, global int* v3, const int n, const int k) {
    int ID, NUM_GLOBAL_WITEMS, step;
    ID = get_global_id(0);
    NUM_GLOBAL_WITEMS = get_global_size(0);
    step = (n / NUM_GLOBAL_WITEMS);

    int i,j;
    for (i=0; i<k; i++) {
        for (j=ID; j<n; j+=step)
            v3[j] = v1[j] + v2[j];
    }
}

void kernel add_single(global const int* v1, global const int* v2, global int* v3, 
                        const int k) { 
    int ID = get_global_id(0);
    for (int i=0; i<k; i++)
        v3[ID] = v1[ID] + v2[ID];
}

