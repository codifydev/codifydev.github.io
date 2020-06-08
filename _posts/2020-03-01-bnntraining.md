---
title: "Training Binarized Neural Networks with binarized weights"
date: 2020-03-01
---

Currently training BNNs still require weights in full/half precision floating point.
Training a BNN with its weights in binarized form would accelerate training time and boost ML applications at the edge.
This approach contrasts with the lottery ticket hypothesis, in which one would start with a larger network, find a winning ticket (model weights), followed by pruning.

Taking inspiration from binary search trees, I experimented with a training algorithm in which we operate on a "chunk" of weights, a subset of the weights matrices.
There are three options on each iteration: clear (0), set (1), or don't touch. 
A greedy approach determines which option to take, based on whether the loss decreases.
Emulating a learning schedule, the chunk size decreases as training progresses. In other words, in the later epochs we operate at a much finer chunk granularity. 

I implemented this training algorithm in PyTorch using a recursive approach. Changing a subset of the weights matrix can be achieved in-place using 'torch.narrow'.  
The algorithm converges slower in terms of epochs compared to backpropagation, since the entire network is exercised only to change a subset of weights (as opposed to obtaining a gradient for every weight).
So far The best accuracy I was able to achieve using this BNN training method for MNIST was 27%. The lack of stochasticity made it stuck on a local minima.

One possible improvement would be to leverage the error value to introduce stochasticity into the equation.
There are other ideas I've yet to try (after brainstorming with friends).
This was done as a side project for me to familiarize with PyTorch and BNNs, and will continue developing when I could spare some time.