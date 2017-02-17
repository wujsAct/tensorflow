# CRF

The CRF module implements a linear-chain CRF layer for learning to predict tag sequences. This variant of the CRF is factored into unary potentials for every element in the sequence and binary potentials for every transition between output tags.

### Usage

Below is an example of the API, which learns a CRF for some random data. The linear layer in the example can be replaced by any neural network.


```python
import numpy as np
import tensorflow as tf

# Data settings.
num_examples = 10
num_words = 20
num_features = 100
num_tags = 5

# Random features.
x = np.random.rand(num_examples, num_words, num_features).astype(np.float32)

# Random tag indices representing the gold sequence.
y = np.random.randint(num_tags, size=[num_examples, num_words]).astype(np.int32)

# All sequences in this example have the same length, but they can be variable in a real model.
sequence_lengths = np.full(num_examples, num_words - 1, dtype=np.int32)

# Train and evaluate the model.
with tf.Graph().as_default():
  with tf.Session() as session:
    # Add the data to the TensorFlow graph.
    x_t = tf.constant(x)
    y_t = tf.constant(y)
    sequence_lengths_t = tf.constant(sequence_lengths)
    '''
    @read time: 2017/2/17  learning the whole process of linear chain crf
    strategy1: context window: feed the window to a single neural network!
    1. unary log-factors: only denpendy on single output(y_k)   a_u(y_k) = a^(L+1)(x_k-1,x_k,x_k+1)y_k
    2. pairwise log-factors:  a_p(y_k,y_k+1) = 1_(a<=k<K)V_(y_k,y_k+1)
    3. Then we have: Z(X) is the partition function, very diffcult to compute!
    p(y|X) = exp(\sum_k=1^K a_u(y_k) + \sum_k=1^K-1 a_p(y_k,y_k+1))/Z(X)
    ?how to compute
    4. how to compute the partition function
    Z(X) = \sum_(y'_1) \sum_(y'_2)...\sum_(y'_K) exp(\sum_k=1^K a_u(y'_k) + \sum_k=1^K-1 a_p(y'_k,y'_k+1))
    dynamic programming methods... 
    5. forward/backward or belief propagation
    * computing both tables is often referred to as the forward/backward algorithm for CRFs
      alpha is computed with a forward pass  ===> give the summation from the left
      belta is computed with a backward pass ===> give the summation from the right
     *It has other names
       *belief propagation  / sum product
        有时间可以重新推到一下子，不是特别难的，
        教程：https://www.youtube.com/watch?v=ZYUnyyVgtyA&t=4s
      *stable implementation of belief propagation:we should work in log space
      
      log \sum_i exp(z_i) = max_i(z_i) + log \sum_i exp(z_i - max_i(z_i))  ===>numerucakkt stable 
    '''
    # Compute unary scores from a linear layer.
    weights = tf.get_variable("weights", [num_features, num_tags])
    matricized_x_t = tf.reshape(x_t, [-1, num_features])
    matricized_unary_scores = tf.matmul(matricized_x_t, weights)
    unary_scores = tf.reshape(matricized_unary_scores,
                              [num_examples, num_words, num_tags])
    
    # Compute the log-likelihood of the gold sequences and keep the transition
    # params for inference at test time.
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
        unary_scores, y_t, sequence_lengths_t)

    # Add a training op to tune the parameters.
    loss = tf.reduce_mean(-log_likelihood)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # Train for a fixed number of iterations.
    session.run(tf.global_variables_initializer())
    for i in range(1000):
      tf_unary_scores, tf_transition_params, _ = session.run(
          [unary_scores, transition_params, train_op])
      if i % 100 == 0:
        correct_labels = 0
        total_labels = 0
        for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores, y,
                                                          sequence_lengths):
          # Remove padding from the scores and tag sequence.
          tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
          y_ = y_[:sequence_length_]

          # Compute the highest scoring sequence.
          viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
              tf_unary_scores_, tf_transition_params)

          # Evaluate word-level accuracy.
          correct_labels += np.sum(np.equal(viterbi_sequence, y_))
          total_labels += sequence_length_
        accuracy = 100.0 * correct_labels / float(total_labels)
        print("Accuracy: %.2f%%" % accuracy)
```
