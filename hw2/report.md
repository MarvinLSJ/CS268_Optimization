## Assignment 2: Conjugate gradients Report

### 1. Explanation

I've implemented conjugate gradient and steepest descent using golden section from assignment 1 as **line search method**, and a GS helper function to calculate mean optimum value among 20 attempts, I thought that might be helpful, but it remains to be verified. 

**Initial point** is selected by a normal distribution with $\mu=0, \sigma=5$ over randomly chosen points: (-1,1) for Rosenbrock and (1,3,3,2,2,1,1,1,2,1) for 10 dimension function. 

**Stopping criterion** is $|F_{new}-F_{old}|\leq F_{tolerance}$  where $F_{tolerance}=10^{-5}$ by default. For conjugate gradients optimization in Rosenbrock, it easily goes wrong and expand quickly, thus I set an early stopping criterion $F_{new} \ge 5F_{old} $ to constrain the result within a rational range.

### 2. Evaluation

Two **test functions** are listed as follows:  

RosenBrock:

 $f_{rb}=(1-x_0)^2 + 100(x_1-x_0^2)^2$

$f^\prime_{rb}$: $[x_0+400x_0^3-400x_0x_1-2,  200(x_1-x_0^2)]$

$x_{opt} = (1,1), f_{opt} = 0$

10D function:

$f_{10d} =\sum_{i=0}^9x_i^2+x_0x_1 + x_2x_3 + x_4x_5 + x_6x_7+ x_8x_9$

$f_{10d}^{\prime} = [2x_0+x_1, 2x_1+x_0, 2x_2+x_3, 2x_3+x_2, 2x_4+x_5, 2x_5+x_4, 2x_6+x_7, 2x_7+x_6, 2x_8+x_9, 2x_9+x8]$

$x_{opt} = (0,0,0,0,0,0,0,0,0,0), f_{opt} = 0$

#### Problem 1. conjugate_gradients

**Average line search & CG restart iteration**

![image-20181017161118549](/Users/liushijing/Library/Application Support/typora-user-images/image-20181017161118549.png)

The average line search method is not helpful as we can concluded from above, and as CG inner iteration grows, the errors become bigger.

#### Problem 2. Comparing with steepest descent

![image-20181017161425953](/Users/liushijing/Library/Application Support/typora-user-images/image-20181017161425953.png)

As we can see from above, if we evaluate these two optimizer 50 times, the CG performs slightly better at X and F error distances. CG is much more efficient since the it iterates far less than SD, we can also tell by running time(ms). Although SD is less accurate and efficient, but it's more stable than CG as its standard deviation in errors are smaller.

### 3. Implementation

**hw2.ipynb:** Detailed implementation and experiments 

**testfile.py:** Required basic functions for testing

conjugate_gradients: 

Different format of fprime input: my code take in fprime as a array, storing every partial derivatives for each dimension (fprime = [fp1, fp2, fp3] , it's easier to implement this way by lambda function).

The example function in assignment is hard to optimize with this CG, after adding try-except, there may be one out ten success rate.