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

An iteration over a very large text requires 

To create a training set of many samples, rather than just one sample. 
Afterwards, in every epoch will have many iterations, each on a batch of few samples.

Also, making the samples smaller allows the hidden state to cost less space.  

"""

part1_q2 = r"""
**Your answer:**

The hidden state. ...

"""

part1_q3 = r"""
**Your answer:**

So that we'll have the same order in every iteration. 

"""

part1_q4 = r"""
**Your answer:**

1. To make the distribution less uniform
2. For high temperature, the distribution is less uniform
3. For low temperature, the distribution is more uniform 

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
