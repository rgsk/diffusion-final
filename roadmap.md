# Roadmap

A clean rewrite of ../diffusion in the style of ../llm-final: .pyn files with
byname records, one module plus its tests per step. Flow matching only, no
DDPM or DDIM, generating MNIST digits.

The idea: draw a straight line from each image x0 (t = 0) to a noise sample
(t = 1), x_t = (1 - t) x0 + t noise. Its velocity, v = noise - x0, is the same
at every t. The model learns to predict v from (x_t, t). Sampling starts from
noise and follows the predicted velocity back to t = 0.

## Done
1. Data. All of MNIST as two in-memory tensors, pixels scaled to [-1, 1], random
   batches by index with no DataLoader.
2. Flow. The straight path from data to noise and its constant velocity, plus
   flow_batch, which draws the noise and t one training step needs.
3. Model. A small UNet (3.15M params, 28 -> 14 -> 7) with a sinusoidal time
   embedding and zero-init output, so it starts out predicting v = 0.
4. Train. llm-final's loop, logger and checkpoint, with val loss measured on
   fixed noise and t so evals are comparable.
5. Sample. An Euler sampler from t = 1 to 0, checked against the exact velocity
   for Gaussian data, plus grid saving.
6. Main. 10k steps in about 9 minutes, val 1.93 -> 0.167, clean digits at 50
   steps and blurry ones at 2.

## Next
7. EMA of the weights. Usually the biggest jump in sample quality for the least code.
8. Class conditioning plus classifier-free guidance (CFG). Lets you ask for a
   particular digit, using the labels get_batch already returns.
9. Heun sampler. Gives better samples at low step counts, like 10.
