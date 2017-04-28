---
title: Language Detection using LSTM Networks in Keras
author: "**Timothee Monceaux** and **Robin Ricard**"
---

LSTM Networks are a kind of Recurrent Neural Network that can be used for language processing. In this post, we're going to use them for language detection. But you can generalize this technique to do more advanced things such as completing texts automatically.

So LSTMs are recurrent models. That means that previous data that did go through the model is used as input of the model. Basically, the input in a recurrent layer for an instant $t$ is:

$$x'_t=x_t+y_{t-1}$$

We introduce the notion of memory into our network and we can see how it would work with texts: when we read a character, we have the information of previous characters that our recurrent model read.

However, basic RNNs are too trivial to process efficiently text input, they have a great difficulty to distinguish efficiently between short and long term antecedents. Long Short Term Memory (LSTM) Networks with a more complex inner structure will be able to add or drop data from the past under certain conditions. This solves the issues we had with basic RNNs and makes it great for processing text.

In the following example, we'll show how to create two LSTMs supposed to generate text in English or French and how to exploit them to perform some language detection. This will be done with the Python scientific stack and Keras.

## Building the dataset from the Universal Declaration of the Human Rights

... or any other corpus for that matter!

First, we simplify the character set by removing all special chars (unicode, uppercase letters, etc...) and reduce the possible chars to 47 different chars including digits and punctuation.

Then we split the cleaned texts (English and French) in two parts: the 80% first chars for training and the 20% last chars for testing.

We can now encode our chars into boolean vectors of 47 values.

This is our raw dataset. From that, we'll treat our train and test set differently:

- The train set can be split into any n-gram substrings that will be used to predict the next chars. However, we'll need to augment that set by perforing some padding in order to support smaller n-gram values when testing. For instance from this 6-gram: `articl`, we'll derive those:
```
      X       y
      artic -> l
      -arti -> c
      --art -> i
      ---ar -> t
      ----a -> r
      ----- -> a
```

```python
self.X_train = numpy.array(X_train)
self.y_train = numpy.array(y_train)
for i in range(0, self.substr_size):
    y_train = numpy.array(X_train[:, -1]) # In order always to predict the very next char
    X_train[:, 1:] = X_train[:, :-1] # We swap every column to the left
    X_train[:, 0] = False # A padded space is a char where all bools are false
    self.X_train = numpy.concatenate([self.X_train, X_train], axis=0)
    self.y_train = numpy.concatenate([self.y_train, y_train], axis=0)
```

- The test set will be only composed of 5-grams. We'll pick randomly from this. However, once picked, we'll do the same padding manipulation to get the probabilities of getting the next char knowing the partial part of the 5-gram:

```
      Pr(a|-----)
      Pr(r|----a)
      Pr(t|---ar)
      Pr(i|--art)
      Pr(c|-arti)
```

## Building and training the LSTM models

We're going to create two LSTM Nets, one for each language, with the same hyperparameters. We'll train them on our previous augmented training datasets. One dataset per model!

```python
self.model = keras.models.Sequential()
self.model.add(keras.layers.LSTM(128, input_shape=(self.dataset.substr_size, self.dataset.n_chars)))
self.model.add(keras.layers.Dense(self.dataset.n_chars))
self.model.add(keras.layers.Activation('softmax'))
optimizer = keras.optimizers.RMSprop(lr=0.01)
self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

We're then going to train our model by carefully keeping the dataset in order (no shuffle) since there is information that we want to get from this order:

```python
self.model.fit(
  self.dataset.X_train, self.dataset.y_train,
  batch_size=128, epochs=epochs, shuffle=False
)
```

## Evaluating the probability that a model is able to generate a 5-gram from an empty seed

Let's get back to our main goal: language detection. Predicting the next char is useful but does not help us classifying a language right out of the box. However we have an interesting information already. We have the probability of generating a given char after feeding a substring. Let's go back the example from earlier:

      Pr(a|-----), Pr(r|----a), Pr(t|---ar), Pr(i|--art), Pr(c|-arti)

By multiplying all of those probabilities with each other, we get the probability that the model will come up with the sequence `artic` from an empty sequence. Since the probabilities are very small, we prefer using the log-likelyhood that gives easier to compare numbers:

```python
# Order of the columns: ["xxxxx", "xxxxT", "xxxTR", "xxTRU", "xTRUM"]  
res = numpy.zeros((X_test.shape[0], self.dataset.substr_size))
for i in range(self.dataset.substr_size):
    y_test = numpy.array(X_test[:, -1]) # In order always to predict the very next char
    X_test[:, 1:] = X_test[:, :-1] # We swap every column to the left
    X_test[:, 0] = False # A padded space is a char where all bools are false
    # We only want the prob on the next char
    res[:, self.dataset.substr_size-i-1] = self.model.predict(X_test)[y_test]
return numpy.sum(numpy.log(res), axis=1) # we return the overall log-likelyhood
```

## Predicting languages

The model with the highest probability to generarate the sequence is the model corresponding to the language in which the sequence has been taken from.

```python
return 1-numpy.abs(numpy.argmax(preds, axis=1)-y).sum()/X.shape[0]
```

By applying this simple test we get the raw accuracy (average on 10 runs) of `0.8215`.

Finally, we can evaluate this model using a ROC curve. This ROC curve is based on the ratio between the log-likelyhoods.

```python
# We build y_hat as the ratio between both class loglikelyhood
y_hat = preds[:,1] - preds[:, 0]

# We compute the ROC & AUC for our predictions
fpr, tpr, _ = sklearn.metrics.roc_curve(y, y_hat)
roc_auc = sklearn.metrics.auc(fpr, tpr)
```

![ROC curve on semilogx](./roc.png)

## Evaluating the model

It's a fair model overall, especially if we consider the small size of the corpus, but it should not be used for super critical language detection since we still see a significant false positive rate. However, let's take the example of a translation service like Google Translate: getting a false match sometimes is ok since the user can correct by himself the detected language.

Let's also keep in mind that the model is trained and tested on a very specific document. Usually, we would try to train LSTMs on a larger corpus of more varied documents to avoid the bias we have now.

However, on the same data, it seems to compare fairly to other models. We tested a simple **n-gram** matching that is known to be simple and we find a better accuracy with LSTMs (by sacrifying simplicity): we had an accuracy around `0.75` on n-gram.

We could have considered other statistical approaches as well based on **term frequency** or **mutual information** but those approaches tend to ignore the structure of the language. That is a clear disavantage compared to LSTMs, but once again, they are very simple to construct.

Finally, how can we do better? First of all we did not try to tune yet the LSTM, we can easily get better results by changing the size of the LSTM layer, the batch size and even the amount of epochs/learning rate, all we need is time to compute it.

Then, as said earlier, we should try to augment the corpus for each language. Note that the train/test split must be generated by doing the split in each text we put in the corpus.

Changing the charset we're keeping could also influence the LSTM, we could reduce the dimensionality of X by deleting things like digits or punctuation but we may lose some meaningful structural information.

We could also try a form of bagging where a same language would get multiple LSTM nets each trained on different corpuses in the same language. The average or maximum probability of the LSTMs for a given language would be used for comparison against other languages.

Finally, we did not try yet to mix LSTM layers with anything else than a dense layer. Maybe adding something like a 1D-Convolution (interesting for "temporal" data) could give interesting results upstream of the LSTM.

## Going further

We considered a lot of things to try previously, here are some additional experiments with LSTMs and language detection.

### Trying a simpler model: n-gram frequencies

This model is simple, all we need to do is record apparition frequencies of bigrams and trigrams in a corpus:

```python
def gen_ngram_freqs(ngrams):
    apparitions = {}
    tot_app = len(ngrams)
    for ngram in ngrams:
        if ngram in apparitions:
            apparitions[ngram] += 1
        else:
            apparitions[ngram] = 1
    ng_freq = {}
    for ng, app in apparitions.items():
        ng_freq[ng] = float(app)/float(tot_app)
    return ng_freq
```

From that, we can score the bigrams and trigrams from a test string with those frequencies:

```python
def score_ng(ngram_freqs, ngrams_test):
    score = 0
    for gram in ngrams_test:
        if gram in ngram_freqs:
            score += ngram_freqs[gram]
    return score
```

Then, we compare the scores between all of the languages we want to distinguish: with the same data as previously, we get around `0.75` of accuracy.

## References

- [Getting started with the Keras Sequential model](https://keras.io/getting-started/sequential-model-guide/)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
