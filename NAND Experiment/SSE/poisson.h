/*
*  This file contains the routines for generating random number based on the poisson distribution
*
*
*/

#ifndef _POISSON_H
#define _POISSON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <time.h>

#define GOODRANDMAX INT_MAX
#define ERR_NO_NUM -1
#define ERR_NO_MEM -2

struct drand48_data buffer; // used as a seed for generating random number

/* rndExp generates pseudorandom numbers following the exponential
   distribution of parameter lambda*/
double rndExp(double lambda){
  double random_value;
  drand48_r(&buffer, &random_value);
  return -log(1.0-random_value)/lambda;
}

/**
*  Function to generate Poisson distributed random variables            
*   - Input:  Mean value of distribution                                
*   - Output: Returns with Poisson distributed random variable          
*/
int poissonRandom(double lambda)
{
  struct timeval tv;   
  int    poi_value;             // Computed Poisson value to be returned
  double t_sum;                 // Time sum value

  // Loop to generate Poisson values using exponential distribution
  poi_value = 0;
  t_sum = 0.0;
  while(1)
  {
	gettimeofday(&tv, NULL);
	srand48_r(tv.tv_sec + tv.tv_usec, &buffer);
	
    t_sum = t_sum + rndExp(lambda);
    if (t_sum >= 1.0) break;
    poi_value++;
  }

  return(poi_value);
}

int Random(){
	
	struct drand48_data buffer; // used as a seed for generating random number
	long int random_value; 
	struct timeval tv; 
	
	gettimeofday(&tv, NULL);
	srand48_r(tv.tv_sec + tv.tv_usec, &buffer);
	lrand48_r(&buffer, &random_value);
	
	return random_value;
		
}

void BubbleSort(int a[], int array_size)
 {
 int i, j, temp;
 for (i = 0; i < (array_size - 1); ++i)
 {
      for (j = 0; j < array_size - 1 - i; ++j )
      {
           if (a[j] > a[j+1])
           {
                temp = a[j+1];
                a[j+1] = a[j];
                a[j] = temp;
           }
      }
 }
 } 

int distinct_random (int size, int lambda) {
    int i, n;
    static int numNums = 0;
    static int *numArr = NULL;

    if(size < 0) return -1;

    if (numArr == NULL){
        numArr = malloc (sizeof(int) * size);
        for (i = 0; i  < size; i++)
                numArr[i] = i;
        numNums = size;
    }

    if (numNums == 0)
       return ERR_NO_NUM;

    //n = rand() % numNums;
    n = poissonRandom(lambda) % numNums;
    i = numArr[n];
    numArr[n] = numArr[numNums-1];
    numNums--;
    if (numNums == 0) {
        free (numArr);
        numArr = NULL;
    }

    return i;
}

#endif
