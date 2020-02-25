#include "stdio.h"
#include "stdlib.h"

int  someFunction (unsigned char**** buff, int *nR, int *nC)
{
  int i,j;
  *buff = (unsigned char ***)malloc(*nR * sizeof(char**));
  for(i = 0; i < *nR; ++i)
  {
    (*buff)[i] = (unsigned char**)malloc(*nC * sizeof(char**));
    for(j = 0; j < *nC; ++j)
    {
      (*buff)[i][j] = (unsigned char*)malloc(256);

      (*buff)[i][j][0] ='1';
    }
  }
}


int main()
{
unsigned char ***buff1;
int r = 3, c= 2,i,j;
someFunction(&buff1, &r, &c);
for( i = 0; i < r; ++i)
{
  for(j = 0; j < c; ++j)
  {
        printf("        %c\n",buff1[i][j][0]);
    free(buff1[i][j]);
  }
  free(buff1[i]);
}
free(buff1);
}
