# Long Short Term Memory

## Motivation

![Untitled](https://github.com/kang952175/Paper/blob/main/Img/LSTM1.png?raw=true)

Recall that vanilla RNN had vanishing gradient problem, as we the backward passses through an FC.

how should we convey those gradients or the information from long away to the beginning.

it is the main motivation of the LSTM

to avoid this, we add a “highway” detouring the FC layer, and a new set of hidden states called cell state($c_t$).

An additional non-linearity added after adding with $c_{t-1}$.

because we are multiplying those  $W_{hh}$ many times.

we’d like to avoid that but this FC is needed.

because without FC we can not model the relationship between inputs and previous hidden state.

## Forget gate

![Untitled](https://github.com/kang952175/Paper/blob/main/Img/LSTM2.png?raw=true)

Even if the cell state is for long-term memory, we still need some mechanism to control it. The forget gate is added this purpose

We read some multiple sentences when the sentence is done and then when you start the next sentence we have to forget what we had in the previous states.

Given the input and hidden states it decides whether this current cell states should be preserved or not be by this forget gate

## Input gate

![Untitled](https://github.com/kang952175/Paper/blob/main/Img/LSTM3.png?raw=true)

Similarly to the input side, we add the input gate to control the flow from the input.

cell states is basically the sum of the previous state and input

## Output gate

![Untitled](https://github.com/kang952175/Paper/blob/main/Img/LSTM4.png?raw=true)

Lastly, we add the output gate to control the value updated to the hidden state $h_t$

These three gates are in the same form.

They are just taking the FC form from the $h_{t-1}$ and $x_t$ and then followed by the sigmoid function which outputs between zero and one.

And then they are multiplied to control these cell states and the hidden states.

they are just gating to controls its ratio when we should forget or updates the input 

## overall

![Untitled](https://github.com/kang952175/Paper/blob/main/Img/LSTM5.png?raw=true)

Overall, the input $x_t$ and previous hidden state $h_{t-1}$ determine the next hidden and cell states ($c_t, h_t$) as well as how much they keep old value or update to new value.

![Untitled](https://github.com/kang952175/Paper/blob/main/Img/LSTM6.png?raw=true)

- With the cell states and the uninterrupted gradient highway, LSTM can **preserve long-term information** better than vanilla RNN can.
    - If forget gate = 1 and input gate = 0, cell state is preserved indefinitely.
    - if this is needed, the model will learn this from the data
    - vice versa, if the input gate  = 1 and forget gate = 0, it forgets completely and a new sentence.
- LSTM does NOT guarantee that there is no vanishing/exploding gradient, it just helps to reduce it. but it does provide an easier way for the model to learn long-distance dependencies.

LSTM tries to reduce that problem to make it a little bit more practical in the range that we are interested in

## Reference

- **[Joonseok Lee](https://www.youtube.com/@LeeJoonseok)**