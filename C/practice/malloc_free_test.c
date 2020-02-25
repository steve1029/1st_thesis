#include <stdlib.h>
#include <stdio.h>

void alloc3Darray_double(double**** pointer, int Nx, int Ny, int Nz);
void alloc3Darray_int(int*** pointer, int Nx, int Ny, int Nz);
void free3Darray_double(double*** pointer, int Nx, int Ny);
void free3Darray_int(void*** pointer, int Nx, int Ny);

int main(void)
{
	int i = 0, j = 0, k = 0;
	int Nx = 3, Ny = 4, Nz = 5;

	int*   		array1D;
	int**  		array2D;
	int*** 		array3D_int;
	double*** 	array3D_double;

	printf("%s%lu\n", "size of int : ",sizeof(int ));
	printf("%s%lu\n", "size of int*: ",sizeof(int*));
	printf("%s%lu\n", "size of long: ",sizeof(long));

	/* calloc: Allocate 1D array and initialize each element as zero. */
	array1D = (int*) calloc(Nz, sizeof(int));

	printf("\n\n");
	printf("print 1D array\n");
	for(k=0; k<Nz; k++){
		array1D[k] = k;
		printf("%3d", array1D[k]);
	}
	printf("\n\n");
	free(array1D);

	/* Allocate 2D array with malloc function and initialize. */
	array2D = (int**) malloc(Ny*sizeof(int*));
	
	printf("print 2D array\n");
	for(j=0; j<Ny; j++){
		array2D[j] = (int*) calloc(Nz, sizeof(int));
		
		for(k=0; k<Nz; k++){
			array2D[j][k] = Nz*j + k;
			printf("%4d", array2D[j][k]);
		}
		printf("\n");
	}

	for(j=0; j<Ny; j++){
		free(array2D[j]);
	}
	
	free(array2D);

	printf("\n\n");

	printf("print 3D array int\n");
	alloc3Darray_int(array3D_int, Nx, Ny, Nz);
	printf("print 3D array double\n");
	alloc3Darray_double(&array3D_double, Nx, Ny, Nz);

	printf("%-10s%p\n", "double: " , array3D_double);
	printf("%-10s%p\n", "int   : " , array3D_int);

	free3Darray_double(array3D_double, Nx, Ny);

	printf("%-10s%p\n", "double: " , &array3D_double);
	printf("%-10s%p\n", "int   : " , array3D_int);

	return 0;
}

void alloc3Darray_double(double**** pointer, int Nx, int Ny, int Nz)
{
	int i,j,k;

	*pointer = (double***) malloc(Nx*sizeof(long));
	
	for(i=0; i<Nx; i++)
	{
		(*pointer)[i] = (double**) malloc(Ny*sizeof(long));
		
		for(j=0; j<Ny; j++)
		{
			(*pointer)[i][j] = (double*) malloc(Nz*sizeof(double));
			
			for(k=0; k<Nz; k++)
			{
				(*pointer)[i][j][k] = Ny*Nz*i + Nz*j + k;
				printf("%-8.2lf", (*pointer)[i][j][k]);
			}
			printf("\n");
		}
		printf("\n\n");
	}
	
	return;
}

void alloc3Darray_int(int*** pointer, int Nx, int Ny, int Nz)
{
	int i,j,k;	

	pointer = (int***) malloc(Nx*sizeof(int**));
	
	for(i=0; i<Nx; i++){
		pointer[i] = (int**) malloc(Ny*sizeof(int*));
		for(j=0; j<Ny; j++){
			pointer[i][j] = (int*) malloc(Nz*sizeof(int));
			for(k=0; k<Nz; k++){
				pointer[i][j][k] = Ny*Nz*i + Nz*j + k;
				printf("%4d", pointer[i][j][k]);
			}
			printf("\n");
		}
		printf("\n\n");
	}
	
	return;
}

void free3Darray_double(double*** pointer, int Nx, int Ny)
{
	int i,j;

	for(i=Nx-1; i>=0; i--)
	{
		for(j=Ny-1; j>=0; j--)
		{
			free(pointer[i][j]);
			printf("%3d%3d\n", i, j);
		}
	}
	
	for(i=Nx-1; i>=0; i--)
	{
		free(pointer[i]);
		printf("%3d\n", i);
	}

	free(pointer);
	printf("%s\n", "MEM_FREE_SUCCES");

	return;
}
