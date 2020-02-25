#include <stdio.h>

int sum(int i, int j)
{
	return i + j;
}

int mystrlen(char *s)
{
	char *p = s;
	for (;*p;++p);
	return p-s;
}

int hexdump(void *v, int len)
{
	unsigned char *p = v;
	int i =0;
	for (;i<len;++i){
		printf("%02x",p[i]);
		if (i>0 && i % 8 ==0) printf("\n");
	}
	printf("\n");
	return len*2;
}
