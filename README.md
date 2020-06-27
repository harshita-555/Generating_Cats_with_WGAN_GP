# Generating Cats using WGAN_GP
This is a Pytorch implementation of Wasserstein GANs with gradient penalty.<br>
Link to the paper is : https://arxiv.org/pdf/1704.00028.pdf

We are using a Dataset consisting of around 15,700 images of cats, and then generating pictures of cats .The hyperparameters such as learning rate, n_critic, beta1, beta2 are assigned the same values as mentioned in the paper . The noise dimension is set to 100 as suggested in the paper.

# Results
iteration_1 :<br><br>
![img](https://github.com/harshita-555/WGAN_GP_cats/blob/master/images/iter_1.png)
<br><br>iteration_10000 :<br><br>
![img](https://github.com/harshita-555/WGAN_GP_cats/blob/master/images/iter_10000.png)
<br><br>iteration_17000 :<br><br>
![img](https://github.com/harshita-555/WGAN_GP_cats/blob/master/images/iter_17000.png)
<br><br>iteration_25000 :<br><br>
![img](https://github.com/harshita-555/WGAN_GP_cats/blob/master/images/iter_25000.png)
