r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0.0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 150
    hypers['seq_len'] = 100
    hypers['h_dim'] = 1200
    hypers['n_layers'] = 6
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 0.0008
    hypers['lr_sched_factor'] = 0.2
    hypers['lr_sched_patience'] = 2
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ARNON. Oh my, I forgot to pay the water bill. "
    start_seq = "ACT I. Scene 1. \n" \
                "Haifa. A students apartment. \n\n" \
                "ARNON. Oh my, our refrigerator seems to be dysfunctional. What shall we do? \n" \
                "IDO. To refrige, or not to refrige? That is the question. \n" \
                "    Ye must ask the neighbours for storage room in theirs. \n"
    temperature = 0.0001
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

To create a training set of many samples, rather than just one long sample. 
Learning with just one train sample is not good, 
because the training set should have various samples different from one another. 

Also, making the samples short allows the hidden state to cost less space.  

"""


part1_q2 = r"""
**Your answer:**

The generator is not trained by memorizing the train set, but by learning a function to predict the next character.  
The hidden state is learnt on multiple epochs, so it knows how to respond to different inputs.   

"""


part1_q3 = r"""
**Your answer:**

So that we'll have the same order in every iteration.
It matters, for the hidden state to learn from all batches.

"""

part1_q4 = r"""
**Your answer:**

1. To make the generated text more conservative and make real words, not too random with non-existent words.

2. For high temperature, the generated distribution is more uniform. 
So the generator chooses a random letter at every iteration.    

3. For low temperature, the generated distribution is more like the learnt distribution. 
So the generator chooses letters to make sequences that are likely to be drawn from the target distribution.  

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0.0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 10
    hypers['h_dim'] = 124
    hypers['z_dim'] = 64
    hypers['x_sigma2'] = 0.0999
    hypers['learn_rate'] = 0.0001
    hypers['betas'] = (0.6, 0.8)
    # ========================
    return hypers


part2_q1 = r"""
It determines the variance of the generated images, 
how creative the model should be in generating different images.  

Low values lead to a model that generates images very similar to the average input image. 

Too high values would lead to a model that generated images that are different from the input. 

"""

part2_q2 = r"""
**Your answer:**

1. 
The KL-divergence loss penalizes the differences between the train and the generated distributions 
in latent space,  
in order to learn the train distribution. 
The reconstruction loss penalizes the difference between the input image and the encoded-decoded image, 
in order to learn a latent space that represents the input well. 

2. 
It becomes a better estimation of the input distribution. 

3. It allows the model to generate new images from the input distributions, 
that are not identical to the train set. 


"""

part2_q3 = r"""
**Your answer:**

In order to start by learning a good latent space, 
before learning the generated distribution. 

"""

part2_q4 = r"""
**Your answer:**

To make it more robust against small changes. 

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 4
    hypers['z_dim'] = 128
    hypers['data_label'] = 0
    hypers['label_noise'] = 0.3
    hypers['discriminator_optimizer'] = dict(
        type="Adam",  # Any name in nn.optim like SGD, Adam
        lr=0.0002,
        betas=(0.6, 0.998)
        # You an add extra args for the optimizer here
    )
    hypers['generator_optimizer'] = dict(
        type="Adam",  # Any name in nn.optim like SGD, Adam
        lr=0.0002,
        betas=(0.6, 0.998)
        # You an add extra args for the optimizer here
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

The gradients are used when training the generator, and not when training the discriminator.  

The generator is trained to generate better samples, by gradient descent optimizers 
with the gradient of the generated images. 

The discriminator is trained to make better predictions against a given generator, by gradient descent optimizers 
with the gradient of the predictions. 
The gradient of the generated samples does not affect it. 
It treats the generator as given, frozen. 
The gradient of the predictions does not depend in the above mentioned gradient of the generated samples. 

"""

part3_q2 = r"""
**Your answer:**

1. No. The discriminator might improve, and consequentially the generator loss will be estimated better, 
allowing the generator to continue improving. 

2. It is not a good sign. 
It means that the discriminator stopped improving, and consequentially the generator loss is not well estimated. 
The generator will learn to trick the current discriminator, and won't generalize.  
The advantage of GAN being adversarial works only if both continue to improve.  
"""

part3_q3 = r"""
**Your answer:**

The GAN results are much more varied, colorful and detailed, 
whereas the VAE results are all similar, all blurry and with few interesting details. 

This is because VAE is trained to learn the train set's distribution,
and its best result would be an average of the training images with some little variance. 

GAN is trained to provide variance that might risk differing from the train set, but 
safely enough to make a well trained discriminator believe it is genuine.  


"""

# ==============
