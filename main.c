/*
 *Alicia Guerra ID: 0586561
 *MATH 230
 *Project II
 *Artificial Neural Network Node
 *Last Revision: 10/03/2014
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 

int main(void) {       

/*We start out by declaring all of our variables.*/
float weight[1][16] = {0., 0., 0., 0.1, 0., 0., 0.2, 0., 0.1, 0.3, 0.1, 0.1, 0.3, 0., 0., 0.4};
float desired_output, network, maxiter, error=0.05;//you said to start with an error of 5%
double learning_rate = 0.1;
const double threshold = 0.5;
int i;
int count = 0, errorcount=0, correctcount=0;
int x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14,x15;
float weightsum, deltaw0, deltaw1, deltaw2, deltaw3, deltaw4, deltaw5, deltaw6, deltaw7, deltaw8, deltaw9, deltaw10, deltaw11, deltaw12, deltaw13, deltaw14, deltaw15;

/*Since I was having so much trouble reading in the values of the csv file, I decided to copy and paste the values of the csv
  file into the IDE and create the following two-dimensional array to be able to perform the calculations:*/
int training_set[128][17]= {
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,1,0},
    {0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0},
    {0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0},
    {0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1},
    {0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,0,1},
    {0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,0},
    {0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,0},
    {0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0},
    {0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0}};
/*What I was originally trying to do was read in each line of the csv file as a string. I was going to use 
fgets(str,1,18) in order to do this. And after that, I was going to use string copy (strncpy(les,str,16) to
take all these values and populate them into an array. To covert characters into integers, I was going to perform 
a very simple subtraction. Since all the integers in the file were bounded by 0 and 9, to find integer ia, all I had 
to do was int ia = a - '0'. However, since I couldn't get the code I was writing to do this to work in time, I saw myself 
forced to  copy and paste the file values and store them in an array. Since I was playing around with fprintf statements 
earlier, it's certainly possibly that I accidentally overwrote the in.csv file.*/

/* By the way, I tried this and it didn't work AT ALL (it printed out "error opening file"):
    #include <stdio.h>
    int main()
    {
    FILE *fp;
    char str[60];
    fp = fopen("in.csv" , "r");
    if(fp == NULL) {
    perror("Error opening file");
    return(-1);
   }
   if( fgets (str, 60, fp)!=NULL ) {
   puts(str);}
   fclose(fp);
   return(0);} 
 */
 
printf("Enter the learning rate, error, and maximum number of iterations:"); 
scanf("%lf %f %f", &learning_rate, &error, &maxiter);

/*First, we print out our initial weights.*/
   printf("\nINITAL WEIGHTS\n");
   printf("w0 = %lf\n", weight[0][0]);
   printf("w1 = %lf\n", weight[0][1]);
   printf("w2 = %lf\n", weight[0][2]);
   printf("w3 = %lf\n", weight[0][3]);
   printf("w4 = %lf\n", weight[0][4]);
   printf("w5 = %lf\n", weight[0][5]);
   printf("w6 = %lf\n", weight[0][6]);
   printf("w7 = %lf\n", weight[0][7]);
   printf("w8 = %lf\n", weight[0][8]);
   printf("w9 = %lf\n", weight[0][9]);
   printf("w10 = %lf\n", weight[0][10]);
   printf("w11 = %lf\n", weight[0][11]);
   printf("w12 = %lf\n", weight[0][12]);
   printf("w13 = %lf\n", weight[0][13]);
   printf("w14 = %lf\n", weight[0][14]);
   printf("w15 = %lf\n\n", weight[0][15]);
  
 /*This while loop continues iterating until the error value equals zero.*/
   while(error != 0 && count <= maxiter)
   {
 /*The for loop below goes through each column of the training set so we can acquire 
   the corresponding sensor/desired output values to run through all the 
   calculations.*/       
       for(i=0; i<128; i++){
            x0=training_set[i][0];
            x1=training_set[i][1];
            x2=training_set[i][2];
            x3=training_set[i][3];
            x4=training_set[i][4];
            x5=training_set[i][5];
            x6=training_set[i][6];
            x7=training_set[i][7];
            x8=training_set[i][8];
            x9=training_set[i][9];
            x10=training_set[i][10];
            x11=training_set[i][11];
            x12=training_set[i][12];
            x13=training_set[i][13];
            x14=training_set[i][14];
            x15=training_set[i][15];
            desired_output=training_set[i][16];

/*We find the weightsum, which is the dot product of our sensor value and initial weight vectors.*/
weightsum = (x0 * weight[0][0]) + (x1 * weight[0][1]) + (x2 * weight[0][2]) + (x3 * weight[0][3]) + (x4 * weight[0][4]) + (x5 * weight[0][5]) + (x6 * weight[0][6])+ (x7 * weight[0][7]) + (x8 * weight[0][8]) + (x9 * weight[0][9]) + (x10 * weight[0][10]) + (x11 * weight[0][11]) + (x12 * weight[0][12]) + (x13 * weight[0][13]) + (x14 * weight[0][14]);

/*We use the dot product of the sensor value and initial weight vectors to find the network values.*/
            if (weightsum > threshold) network = 1;
            else if (weightsum <= threshold) network = 0;

/*We use the network values to find the error values.*/            
            error=desired_output-network;
 
/*We increment the count value by one each time we go through the loop in order to record the
   number of iterations.*/           
            count++;

/*If the error is not zero and we are within the first 128 iterations, then the errorcount value is 
  incremented by 1. We use the errorcount value to give us the correctvalue, which indicates how
  often during the first 128 iterations we were able to correctly calculate the network value.*/
            if((abs(network - desired_output)) != 0 && count<= 128)errorcount++;
            correctcount = 128 - errorcount;

/*We calculate the corrections of each weight by multiplying the learning_rate and error values.*/
            deltaw0 = learning_rate * error;
            deltaw1 = learning_rate * error;
            deltaw2 = learning_rate * error;
            deltaw3 = learning_rate * error;
            deltaw4 = learning_rate * error;
            deltaw5 = learning_rate * error;
            deltaw6 = learning_rate * error;
            deltaw7 = learning_rate * error;
            deltaw8 = learning_rate * error;
            deltaw9 = learning_rate * error;
            deltaw10 = learning_rate * error;
            deltaw11 = learning_rate * error;
            deltaw12 = learning_rate * error;
            deltaw13 = learning_rate * error;
            deltaw14 = learning_rate * error;
            deltaw15 = learning_rate * error;
   
/*We calculate our new weights by taking our initial weights and adding them to 
  the product of the corresponding correction and sensor values.*/        
            weight[0][0] = weight[0][0]+(deltaw0*x0);
            weight[0][1] = weight[0][1]+(deltaw1*x1);
            weight[0][2] = weight[0][2]+(deltaw2*x2);
            weight[0][3] = weight[0][3]+(deltaw3*x3);
            weight[0][4] = weight[0][4]+(deltaw4*x4);
            weight[0][5] = weight[0][5]+(deltaw5*x5);
            weight[0][6] = weight[0][6]+(deltaw6*x6);
            weight[0][7] = weight[0][7]+(deltaw7*x7);
            weight[0][8] = weight[0][8]+(deltaw8*x8);
            weight[0][9] = weight[0][9]+(deltaw9*x9);
            weight[0][10] = weight[0][10]+(deltaw10*x10);
            weight[0][11] = weight[0][11]+(deltaw11*x11);
            weight[0][12] = weight[0][12]+(deltaw12*x12);
            weight[0][13] = weight[0][13]+(deltaw13*x13);
            weight[0][14] = weight[0][14]+(deltaw14*x14);
            weight[0][15] = weight[0][15]+(deltaw15*x15);
        }}
 
 /*After we get out of the while loop, we print out our final weights, the number of
   iterations we did, and our ratio of correctly calculated outputs (x/128).*/                            
            printf("FINAL WEIGHTS\n");
            printf("w0 = %lf\n", weight[0][0]);
            printf("w1 = %lf\n", weight[0][1]);
            printf("w2 = %lf\n", weight[0][2]);
            printf("w3 = %lf\n", weight[0][3]);
            printf("w4 = %lf\n", weight[0][4]);
            printf("w5 = %lf\n", weight[0][5]);
            printf("w6 = %lf\n", weight[0][6]);
            printf("w7 = %lf\n", weight[0][7]);
            printf("w8 = %lf\n", weight[0][8]);
            printf("w9 = %lf\n", weight[0][9]);
            printf("w10 = %lf\n", weight[0][10]);
            printf("w11 = %lf\n", weight[0][11]);
            printf("w12 = %lf\n", weight[0][12]);
            printf("w13 = %lf\n", weight[0][13]);
            printf("w14 = %lf\n", weight[0][14]);
            printf("w15 = %lf\n", weight[0][15]);
            printf("The perceptron algorithm was invoked %d times so that the error would equal zero.\n", count);
            printf("The desired output was correctly attained %d out of 128 times.\n\n", correctcount);
    return (0);
}