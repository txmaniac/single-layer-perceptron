#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

typedef struct my_record {
    float A;
    float B;
    float C;
    float D;
    float Y;
    float err;
} rec;


int flag = 0;
float act_oup[44];
float weights[4];

void shuffle(rec *shuffled, int length, int times)
{
    int i,j,k;
    rec aux;

    j = k = 0;
    for(i=0;i<times;i++)
    {
        do
        {
            j=rand() % length;
            k=rand() % length;
        } while(j==k);

        aux = shuffled[j];
        shuffled[j] = shuffled[k];
        shuffled[k] = aux;
    }
}

float sigmoid(float x)
{
     float exp_value;
     float return_value;

     exp_value = (float) exp((double)-x);
     return_value = (1.0) / (1.0 + exp_value);

     return return_value;
}

float act_fun(float x){
	if(sigmoid(x)>0.5)
		return(1);
	
	if(sigmoid(x)<=0.5)
		return(0);
}

float Sum(float x1,float x2,float x3,float x4,float w1,float w2,float w3,float w4){
	return((x1*w1)+(x2*w2)+(x3*w3)+(x4*w4));
}

void slp(rec* records, int size, float bias){
	int i,j;
	float sum = 0.0;
	for(i=0;i<size;i++){
		act_oup[i] = act_fun(Sum(records[i].A,records[i].B,records[i].C,records[i].D,weights[0],weights[1],weights[2],weights[3])+bias);
		
		// error calculation
		
		records[i].err = records[i].Y - act_oup[i];
		weights[0]+=(records[i].err*records[i].A);
		weights[1]+=(records[i].err*records[i].B);
		weights[2]+=(records[i].err*records[i].C);
		weights[3]+=(records[i].err*records[i].D);
		sum+=records[i].err;	
	}
	
	sum/=size;
	for(j=0;j<4;j++) 
			printf("weights : %f\n",weights[j]);
	if((sum <= 0.005) && (sum >= -0.005))
		flag = 1;
}

float test_slp(rec* records,int size,int times){
	float expected;
	int acc=0;
	int i=0;
	
	shuffle(records,size,times);
		
	//printf("Desired \t Actual\n");
	for(i=0;i<size;i++){
		expected = act_fun(Sum(records[i].A,records[i].B,records[i].C,records[i].D,weights[0],weights[1],weights[2],weights[3]));
		
		if((records[i].Y-expected <= 0.005) && (records[i].Y-expected >= -0.005))
			acc+=1;
			
		//printf("%f\t %f \n",records[i].Y,expected); use this for checking the output. If this doesn't work properly then logical error!!!
	}
	
	return(acc*100/size);
}

int main(int argc, char** argv){
	
	int i;
	char inp[30];
	float bias = (rand()%10 + 1)/10;
	strcpy(inp,"/home/txm/Desktop/SLP/");
	strcat(inp,argv[1]);
	FILE* my_file = fopen(inp,"r");
	time_t t;
	rec records[1000];
	//rec records2[22];
	size_t count = 0;
	float accuracy;
	//float weights[4];
	int epochs;
	int times,j;
	
	srand((unsigned) time(&t));
	times = (rand()%(10000)) + 1;
	// Initializing weights
	for(i=0;i<4;i++)
		weights[i]=0.0;
		
	for (;count < sizeof(records)/sizeof(records[0]);++count)
	{
		int got = fscanf(my_file, "%f,%f,%f,%f,%f",&records[count].A,&records[count].B,&records[count].C,&records[count].D, &records[count].Y);
		records[count].Y-=1;
		records[count].err = 0;
		if (got != 5) break; // wrong number of tokens - maybe end of file
	}
	
	fclose(my_file);
	printf("Number of elements in dataset : %ld\n",count);
	int train_size = (count*2)/3;
	int test_size = count - train_size;
	
	
	shuffle(records,count,times);
	/*printf("\n-------Training Data-------\n\n"); use this for debugging and for checking if data is corrupted.
	for(i=0;i<train_size;i++){
		printf("%f %f %f %f %f \n",records[i].A,records[i].B,records[i].C,records[i].D,records[i].Y);
	}*/
	
	epochs = 0;
		
	while(!flag && epochs<1000){
	
		times = (rand()%(10000)) + 1;
		printf("\n%d epochs executed \n\n",++epochs);
		shuffle(records,train_size,times);
		
		slp(records,train_size,bias);
	}
	
	if(flag)
		printf("\n\nNo more learning is possible\n\n");
	
	printf("Number of epochs required for training the model : %d\n\n",epochs);		
	accuracy = test_slp(&records[train_size],test_size,times);
	printf("\n");
	for(i=0;i<4;i++)
		printf("Final weights : %f\n",weights[i]);
		
	printf("\nAccuracy of the model : %f\n",accuracy);	
		
	return(0);
}
