void add3d_double(double* input1, double* input2, double* output, int xgrid, int ygrid, int zgrid)
{
	int i,j,k;
	int idx;

	for(i=0; i < xgrid; i++){
		for(j=0; j < ygrid; j++){
			for(k=0; k < zgrid; k++){
	
				idx = k + j * zgrid + i * zgrid * ygrid;
				output[idx] = input1[idx] + input2[idx];
				
			}
		}
	}
	return;
}

void mul3d_double(double* input1, double* input2, double* output, int xgrid, int ygrid, int zgrid)
{
	int i,j,k;
	int idx;

	for(i=0; i < xgrid; i++){
		for(j=0; j < ygrid; j++){
			for(k=0; k < zgrid; k++){
	
				idx = k + j * zgrid + i * zgrid * ygrid;
				output[idx] = input1[idx] * input2[idx];
				
			}
		}
	}
	return;
}

void sub3d_double(double* input1, double* input2, double* output, int xgrid, int ygrid, int zgrid)
{
	int i,j,k;
	int idx;

	for(i=0; i < xgrid; i++){
		for(j=0; j < ygrid; j++){
			for(k=0; k < zgrid; k++){
	
				idx = k + j * zgrid + i * zgrid * ygrid;
				output[idx] = input1[idx] - input2[idx];
				
			}
		}
	}
	return;
}

void div3d_double(double* input1, double* input2, double* output, int xgrid, int ygrid, int zgrid)
{
	int i,j,k;
	int idx;

	for(i=0; i < xgrid; i++){
		for(j=0; j < ygrid; j++){
			for(k=0; k < zgrid; k++){
	
				idx = k + j * zgrid + i * zgrid * ygrid;
				output[idx] = input1[idx] / input2[idx];
				
			}
		}
	}
	return;
}
