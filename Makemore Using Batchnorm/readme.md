# Makemore Using Batchnorm From Scratch And Analysing Different Initialisations

Batch normalization (BatchNorm) is a technique used in deep learning models to normalize the inputs of each layer.
It is commonly used in convolutional neural networks (CNNs) and has been shown to improve training speed, generalization, and model accuracy.

#### Here we will use batch norm to improve out name generation model (12.3K parameters)

### First off we fix the initialisation because the initial loss was too high than expected .
* To fix initialisation we removed the bias from last layer and reduced the weights magnitude (*0.01)
* Here all characters(we have 27 in this model) should have same probability at initialisation so expected loss will be -torch.tensor(1/27.0).log()  ->  (3.2958)

#### Loss when initialisation was bad ->

![bad initialisation](https://user-images.githubusercontent.com/102567732/229861108-39d5537e-1ce3-47f0-89f5-a879f968a5a9.png)

#### Loss after fixing initialisation ->

![after fixing initialisation](https://user-images.githubusercontent.com/102567732/229861226-ce384b95-d77c-4a0b-a197-3fbc5b1ad151.png)

* Now we see that the hockey stick appearance of loss is not there because we initialised well 
* This helps because the more training time will be given to improve the model rather than shrinking it

### Fixing the saturation of tanh layer 
* We see that tanh is too saturated and most values are -1 and 1 which is bad for training because the slope is inf there and the model does not learn .
* As derivative of tanh is (1- t^2)x(out.grad) at 1 and -1 this becomes 0 ,so  no matter how much you change the gradient model will not learn
![tanh saturation](https://user-images.githubusercontent.com/102567732/229862234-5e94a01b-d766-4865-9fa9-fcd4b8dddb76.png)
* We fix this by reducing weights by a factor off 100 , results in a very stable tanh with almost no saturation
![tanh after fixing](https://user-images.githubusercontent.com/102567732/229862530-5ffdda15-1ce4-4eb3-b89b-2975210602db.png)

### Adding batchnorm 
### Loss Log ->
#### LOSS WHEN INITIALISATION WAS BAD
* TRAIN ->  2.2308623790740967
* VAL -> 2.249389886856079
#### LOSS WHEN INITIALISATION WAS GOOD
* TRAIN -> 2.117876768112182
* VAL -> 2.157973527908325
#### LOSS WHEN WE FIXED TANH LAYER BEING TOO SATURATED AT INIT
* TRAIN ->2.0703351497650146
* VAL -> 2.11810302734375
#### LOSS WHEN WE FIXED TANH ACCORDING TO THE PAPER
* TRAIN ->2.026627540588379
* VAL -> 2.106279134750366
#### LOSS USING BATCH NORM
* TRAIN -> 2.0443265438079834 ,not expecting improvement
* VAL -> 2.095064401626587
#### LOSS USING BNMEAN RUNNING AND BNSTD RUNNING
* TRAIN -> 2.0435707569122314 ,expecting similar results as above
* VAL -> 2.0435707569122314

## Analsying activations distribution , gradient distributions and update to data ratio distributions
### Activation distributions
#### When gain is 0.5 ( activations are shrinking to zero ) -> 

![activation distribution (1)](https://user-images.githubusercontent.com/102567732/229870248-98e8d1ef-0dd6-4487-9199-f141040fe255.png)

We need activation distribution for all layers shd be similar

#### When gain is 5/3 ->

![activation distribution ](https://user-images.githubusercontent.com/102567732/229870304-767598f6-7358-4f3f-9b94-40e815b745d6.png)

#### When we use batch norm ->

![activation distribution with batch norm](https://user-images.githubusercontent.com/102567732/229870388-7847a460-52c0-48a9-89f3-8c63be11bccb.png)

### Gradient Distributions 

#### When gain is 5/3 ->
![gradient distribution 1](https://user-images.githubusercontent.com/102567732/229872140-73732775-a1be-4cd7-b680-fe0b114cf7cc.png)

#### When using batch norm ->
![gradient distribution 2](https://user-images.githubusercontent.com/102567732/229872156-45e7c1ef-85ff-404d-925c-ff35cd36e03a.png)

### Update to data ratio 

### Below is exactly how we dont want it to be ->
![update to data ratio 0](https://user-images.githubusercontent.com/102567732/229872174-895e8859-ecac-4877-bc5e-ea53a0aaa87b.png)

We want all layers to have uniform and above is example of when we dint using kaiming initialisation

### When using kaiming initialisation ->
The black line shows us where it should be.. if they are below black line it means that our modell is training slowly and learning rate can be increased

![update to data ratio 1](https://user-images.githubusercontent.com/102567732/229872192-9eb94a89-d087-4a15-92b2-d6cbcda1e031.png)

### When using batchnorm ->

![update to data ratio 2](https://user-images.githubusercontent.com/102567732/229872220-b0002425-3bc7-49ff-a75d-ad5a35452f4d.png)
