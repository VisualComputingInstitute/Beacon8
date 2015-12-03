# Beacon8
A Torch-inspired library for high-level deep learning with Theano.

Thorough documentation will follow very soon.

### Building a model
```python
  model = bb8.Sequential()
  model.add(bb8.Reshape(-1, 1, 28, 28))
  model.add(bb8.SpatialConvolutionCUDNN(1, 32, 5, 5, 1, 1, 2, 2, with_bias=False))
  model.add(bb8.BatchNormalization(32))
  model.add(bb8.ReLU())
  model.add(bb8.SpatialMaxPoolingCUDNN(2, 2))

  model.add(bb8.SpatialConvolutionCUDNN(32, 64, 5, 5, 1, 1, 2, 2, with_bias=False))
  model.add(bb8.BatchNormalization(64))
  model.add(bb8.ReLU())
  model.add(bb8.SpatialMaxPoolingCUDNN(2, 2))
  model.add(bb8.Reshape(-1, 7*7*64))

  model.add(bb8.Linear(7*7*64, 100, with_bias=False))
  model.add(bb8.BatchNormalization(100))
  model.add(bb8.ReLU())
  model.add(bb8.Dropout(0.5))

  model.add(bb8.Linear(100, 10))
  model.add(bb8.SoftMax())
```

### Training
```python
  criterion = bb8.ClassNLLCriterion()
  optimiser = optim.Momentum(lr=0.01, momentum=0.9)
  model.zero_grad_parameters()
  model.accumulate_gradients(mini_batch_input, mini_batch_targets, criterion)
  optimiser.update_parameters(model)
```

### Inference
```python
  mini_batch_prediction = model.forward(mini_batch_input)
```
