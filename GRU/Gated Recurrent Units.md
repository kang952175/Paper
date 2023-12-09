# Gated Recurrent Units

## GRU

![Untitled]((https://github.com/kang952175/Paper/blob/main/Img/GRU1.png?raw=true)

- Another idea similar to LSTM, providing long-range dependency on RNNS
    - No additional cell states as in LSTM
    - Fewer parameters compared to LSTM
    - Provide a gradient highway similar to LSTM, using **a convex combination** of previous hidden state and new one computed from the input.

## Reference

- **[Joonseok Lee](https://www.youtube.com/@LeeJoonseok)**