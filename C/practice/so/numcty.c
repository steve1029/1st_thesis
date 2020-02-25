#include <stdio.h>
#include <math.h>
#include <assert.h>

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

double* add(double *a, double *b, double *c,\
				int alen)
{
	int i;

	for(i=0;i<alen;i++){
		c[i] = a[i] + b[i];
	}

	return c;
}

double* product(double *a, double *b, double *c, int alen)
{
	int i;
	
	for(i=0;i<alen;i++){
		c[i] = a[i] * b[i];
	}

	return c;
}

double* subtract(double *a, double *b, double *c, int alen)
{
	int i;
	
	for(i=0;i<alen;i++){
		c[i] = a[i] - b[i];
	}

	return c;
}

//double* updateHx(double ***CB1, double ***CB2,\
//				 double ***CH1, double ***CH2, double ***CH3,\
//				 double ***Bstorage, double ***Hstorage,\
//				 int nx, int ny, int nz)
//{
//	// All parameters are given as a numpy.ctypes.float array	
//
//	
//	for(i=0;i<nx;i++){
//		for(j=0;j<ny;,j++){
//			for(k=0;k<nz;,k++){
//				
//			}
//		}
//	}
//
//	return storage;
//}

int main(void)
{

	double a[5] = {0.,1.,2.,3.,4.};
	double b[5] = {0.,1.,2.,3.,4.};
	int i,l=ARRAY_LEN(a);	

	double sum[l];
	add(a,b,sum,l);
	
	for(i=0;i<l;i++){
		printf("a: %.2f, b: %.2f, c: %.2f\n",a[i],b[i],sum[i]);
	}

	subtract(a,b,sum,l);
	
	for(i=0;i<l;i++){
		printf("a: %.2f, b: %.2f, c: %.2f\n",a[i],b[i],sum[i]);
	}
	
	product(a,b,sum,l);
	
	for(i=0;i<l;i++){
		printf("a: %.2f, b: %.2f, c: %.2f\n",a[i],b[i],sum[i]);
	}
	return 0;
}
