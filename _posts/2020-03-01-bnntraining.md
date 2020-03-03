---
title: "Training Binarized Neural Networks with binarized weights"
date: 2020-03-01
---

So far the techniques I have seen for training BNNs still require keeping weights in full/half precision floating point.
Training a BNN with its weights in binarized form would boost ML applications at the edge.
This approach contrasts with the lottery ticket hypothesis, in which one would start with a larger network, find a winning ticket (model weights), and then start pruning.

Taking inspiration from binary search trees, I experimented with a training algorithm in which we operate on "chunk" of weights (i.e. decide the 0/1 value).
To determine which route (0/1/don't touch), we'd refer to whether the loss decreased, hence the greedy approach. 
The chunk size it operates on decreases as training progresses. In other words, in the later epochs we operate at a much finer chunk granularity. 

I implemented this training algorithm in PyTorch using a recursive approach. Changing a subset of the weights matrix can be achieved in-place using 'torch.narrow'.  
The algorithm runs slow compared to backpropagation, since the entire network is exercised only to change a subset of weights (as opposed to obtaining a gradient for every weight).
The best accuracy I was able to achieve for MNIST was 27%. The lack of stochasticity likely made it stuck on a local minima.

One possible improvement would be to leverage the error value to introduce stochasticity into the equation.
There are other ideas I've yet to try (after brainstorming with friends), which I'd like to publish if I get satisfactory measurements.
This was done as a side project for me to familiarize with PyTorch and BNNs, which I'd revisit on free time.
I'm looking forward to seeing advances in this topic. 