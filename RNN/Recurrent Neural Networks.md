# Recurrent Neural Networks

![RNN type](C:/Users/kang1/Desktop/LLM/Paper/Img/RNN0.png)

- one to one : Image classification
- one to many : Image captioning
- many to one : Sentiment classification
- many to many : Machine Translation
- many to many : Video classification

RNNs are a class of neural networks that allow previous outputs to be used as inputs while having hidden states.

![Untitled](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%201.png)

RNNs have an internal state that is updated as a sequence is processed.

before this rnn, we didnt have any kind of feedback loop.

we always have just a unidirectional relationship as possible from the input to the output.

now, we have some feedback loop like above.

rnn module gets some input and uses its previous states.

then, it updates the current states  by itself using **the most recent input and based on its previous states.**

so, the new value is determined by its old value as well as the input.

and then it gets the next input and repeats the same thing again and again.

![Untitled](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%202.png)

We can express with expanded view so instead of that feedback loop.

- Each RNN cell takes an input $x_i$, updates its hidden state $h_i$ from $h_{t-1}$, then (optionally) returns an output $y_i$
- The RNN cell needs to be initialized somehow ($h_0$)

The input is a sequence of something like $x_1$ to $x_t$

it is similar to what we’ve done with the fully connected layers.

RNNs are a type of neural network that can be used to model sequence data. 

Simply said, recurrent neural networks can anticipate sequential data in a way that other algorithms can’t.

![*Source: Quora.com*](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%203.png)

*Source: Quora.com*

The hidden states should be initialized to some value because we will need its previous value and the input as its input in the very first input.

and then it updates to the next hidden state which is called h1.

and then based on $h_1$ we output $y_1$, based on $h_1$ we take $x_2$ as input.

and based on these two we updates the hidden states and that is called $h_2$

in that way, we can self update rnn cells based on the input and the previous hiddens.

let’s do this in mathematical formula.

- A sequence of vectors $\{ x_1, x_2, …, x_T\}$ is processed by applying a recurrence formula at every time step
$h_t = f_W(h_{t-1}, x_t)$
- it is important to note that **the same function** and **the same set of parameters** are used at every time step.
- That is the key of RNN so let’s see how it works.

this model takes input token and the previous hidden states.

and then it updates to the new hidden states.

and we generate some outpus from there

![source: [https://medium.com/@navarai/the-architecture-of-a-basic-rnn-eb5ffe7f571e](https://medium.com/@navarai/the-architecture-of-a-basic-rnn-eb5ffe7f571e)](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%204.png)

source: [https://medium.com/@navarai/the-architecture-of-a-basic-rnn-eb5ffe7f571e](https://medium.com/@navarai/the-architecture-of-a-basic-rnn-eb5ffe7f571e)

- let’s follow what wd do with a fully-connected
    - one simple way is taking linear transformations (with W_{hh} and $W_{xh}$ ) on the two inputs (previous hidden state $h_{t-1}$ and input $x_t$)
    - then take a nonlinearity before updating it as $h_t$
    - For the output, we may put another **linear transformation** form $h_t$
    
    $h_t = tanh(W_{hh}h_{t-1} + W_{xh}x_t)$
    
    $y_t = W_{hy}h_t$
    

![Untitled](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%205.png)

- Again, all weights are shared across time!
    - During the forward pass, the same weights are used rpetitively.

we always use the same W values for each steps regardless of the index of the inputs.

and the model should optimize to best reflect the relationship between x and y.

we can best approximate this relationship between the sequence x and sequence y using these same values that is a restriction.

that’s how we actually **can take arbitrary lengths** of these input sentences or input sequence.

- Then, where do we compare with the ground truth?
    - At the output $\{y_1. y_2, …\}$
    - Each output is combined to compute the overall loss.

they always share weights to encode inputs and apply to the sequence and the general it’s learned.

- What if our problem is many-to-one?
    - We output only once at the end of the sequence.
    - Often, intermediate hidden states are used as well when the output is determined.
        - The sequence is too long. so then feeding the input in the given order tends to remember the recent inputs are better thean the previous input(the earlier inputs)  so these ealier hidden states are better representing the earlier input.

![Untitled](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%206.png)

it extracted its meaning and then the semantics of the entire video is compactly represented within this hidden states after consuming all of these input frames and thend based on this we output.

- What about one-to-many?
    - We still must input something at each step, given the formula:
    $\mathbf h_t = tanh(\mathbf W_{hh} \mathbf h_{t-1} + \mathbf W_{xh}\mathbf x_{t-1})$
    - **Autoregressive** input: For time series data, the lagged(autoregressive) values of the time series are used as inputs to a neural network.
    
    ![Untitled](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%207.png)
    

**Autoregressive : we input the output from the previous step.**

- Lastly, how to implement many-to-many
    - Many-to-one as an encoder, then one-to-many as a decoder
    - The input sequence is encoded as a single vector at the end of the encoder.
    - From this single vector, the decoder generates output sequence.
    - Called Sequence-to-sequence, or seq2seq.

![Untitled](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%208.png)

It just stores the meaning or semantics of this input sequence in the compact representation of the hidden states.

And in the end h3 will contain the entire semantics of this input video or sentence.

This part is called encoder so, we don’t output anything

Hidden state is initialized from this encoded hidden states from the encoder.

We expect $s_0$ contains the semantics of this video.

And It starts with an initial token, a very special token that marks the beginning of a sentence.

> Pytorch API: Vanilla RNN
> 

```python
import torch
import torch.nn as nn

input = torch.randn(5, 3, 10)  # (batch_size, sequence_length, input_dim)
h0 = torch.randn(2, 3, 20)  # (D∗num_layers, batch_size, hidden_size)
rnn = nn.RNN(10, 20, 2)  # (input_size, hidden_size, num_layers)
output, hn = rnn(input, h0) 
print(output.shape) # [5, 3, 20]
print(hn.shape) # [2, 3, 20]
```

## RNN Trade-offs

- RNN Advantages:
    - Can process input sequence of any length
    - Model size doesnt increase for longer input.
    - Computation at step t can (in theory) use information from many steps back.
    - Same weights are applied at every timestep.

- RNN Disadvantages:
    - Recurrent computation is slow.
    - A sequence output inference is hard to be parallelized.
        - because it need previous hidden states to produce the next output.
        - it has to be sequentially produced the output.
    - Vanilla RNN suffers from vanishing gradient in training
    - Vanilla RNN often fails to model long-range dependence in a sequence.
        - if the sequence is too long, then it forgets the information at the first
    - In practice, difficult to access information from many steps back.

## Multi-layer RNN

- we may put more than one hidden layers.

![Untitled](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%209.png)

RNN doesn’t necessarily be just a single layer.

so, you can actually stack multiple layers of hidden states from the input

## backpropagation of RNN

![source: [https://mmuratarat.github.io/2019-02-07/bptt-of-rnn](https://mmuratarat.github.io/2019-02-07/bptt-of-rnn)](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%2010.png)

source: [https://mmuratarat.github.io/2019-02-07/bptt-of-rnn](https://mmuratarat.github.io/2019-02-07/bptt-of-rnn)

## Useage of RNN

- Image Captioning
- Visual Question and Answering
- Visual dialogue (a kind of chatbot)

## Towards Modeling Longer Dependence

Gradient Flow Problem with Vanilla RNN

![Untitled](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%2011.png)

Backporp from h_t to h_{t-1} multiplies by W_hh

$$
\begin{equation} {\partial h_t \over \partial h_{t-1}}= tanh' (W_{hh}h_{t-1}+W_{xh}x_t)W_{hh}\end{equation}
$$

we do this not just one step but multiple steps

We need the partial derivative of the entire loss (with chain-rule)

What does this formula mean?

![Untitled](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%2012.png)

That actually picks at the input is zero which is one.

And all other parts the value is between zero and one

so, at this part we always multiply some value which is less than one

this will be getting smaller and smaller all the time when we have long sequences.

- Exploding gradients can be addressed by gradient clipping (scale gradient down if its norm is above some threshold).
- Vanishing gradients again? Harder to treat this without changing architecture.

Notation

For simplicity, let’s use “FC” (fully-connected) box instead of each weight maatrix.

![Untitled](Recurrent%20Neural%20Networks%20f062d59a492e40aaa96ffe5593d01683/Untitled%2013.png)

⇒ Long Short Term Memory (LSTM)

## Reference

- **[Joonseok Lee](https://www.youtube.com/@LeeJoonseok)**
- [CS 230- RNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks#architecture)
- **[The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)**
- **[NLP FROM SCRATCH: CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)**
