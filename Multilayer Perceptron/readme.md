
# Name Generator Using Multi Layer Perceptron

This project is an implementation of a multilayer perceptron (MLP) for generating new names using a dataset containing 30000 names.
The MLP is used to learn a mapping from input some letters (of given block size) to output classes (27 classes consisting of 26 words and a end of word character ".").

## Part 1

In part 1 , I started with embedding size of 2 for each letter , the neural network consists of only 2 layers ,
initially I used a model of 13 thousand parameters. After that i increased the embedding size from 2 to 10 .
To find out the best learning rate , plot of loss vs the exponent of learning rate .

As you can see i found out that best learning rate would be somewhere near -1 exponent(0.1) 

![loss vs exponent of learning rate](https://user-images.githubusercontent.com/102567732/227607681-966f4459-0538-4fa4-a9cc-d950a959827c.png)

### Here is the visualisation of the "Letter Embeddings "

![mlp part1](https://user-images.githubusercontent.com/102567732/227604486-6bc2ecd9-36a6-42d6-8782-59c64eab2033.png)

#### You can see that  q is a outlier because it does not appear often in names and "." character is also away

#### Train Loss -> 2.1846
#### Dev Loss -> 2.2103

### Here are some names that our model generated
1. carmah.
2. amelle.
3. khiim.
4. shreety.
   Not bad but can be better.

##Part 2

In part 2 , I experimented more with the word embeddings and increased the number of parameters significantly to try to minimize the loss
Model was of 30000 parameters approximately .
After several attempts loss decreased
#### Train Loss -> 1.8495
#### Dev Loss -> 2.1930

### Here are some names the model generated
1. carl
2. quinn
3. kaleigh
4. dellynn
Definitely unique and better names than previous... 

