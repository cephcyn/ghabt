# ghabt

"Bullying Models with Perturbation" (secret name "GANna Have A Bad Time"): Final project for CSE455 2021WI (Computer Vision)

View our demo page here! https://cephcyn.github.io/ghabt/

## Project Proposal

We want to find out what models / model architectures are more resistant to adversarial attack. To do this, we will try to perform an adversarial attack on a range of MNIST/CIFAR models with different architectures based on the model we implemented for Homework 5. We’ll perform adversarial attacks by using a GAN where the discriminative networks are the various models we want to test. We’ll measure the performance of the models on the generated images, and measure a given model’s resistance to adversarial attacks by how large the dropoff in performance is. 

We will use the CIFAR-10 dataset. To train the models to be attacked, we will use PyTorch. 
