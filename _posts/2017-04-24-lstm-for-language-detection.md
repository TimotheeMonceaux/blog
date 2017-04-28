---
layout: post
title: Language Detection using LSTM Networks in Keras
date:   2017-04-24
author: Timothée Monceaux, Robin Ricard
---
LSTM Networks are a kind of Recurrent Neural Network that can be used for language processing. In this post, we're going to use them for language detection. But you can generalize this technique to do more advanced things such as completing texts automatically.

So LSTMs are recurrent models. That means that previous data that did go through the model is used as input of the model. Basically, the input in a recurrent layer for an instant *t* is: *x'(t) = x(t) + y(t-1)*

We introduce the notion of memory into our network and we can see how it would work with texts: when we read a character, we have the information of previous characters that our recurrent model read.

However, basic RNNs are too trivial to process efficiently text input, they have a great difficulty to distinguish efficiently between short and long term antecedents. Long Short Term Memory (LSTM) Networks with a more complex inner structure will be able to add or drop data from the past under certain conditions. This solves the issues we had with basic RNNs and makes it great for processing text.

In the following example, we'll show how to create two LSTMs supposed to generate text in English or French and how to exploit them to perform some language detection. This will be done with the Python scientific stack and Keras.

You can download the jupyter notebook used for this experemiment [here](/blog/assets/2014-04-24/2017-04-24-lstm-for-language-detection/lstm-for-language-detection.ipynb)

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

The code to generate the padding is the following:

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

### Training for multiple languages

Instead of only considering two languages to compare with each other, we also tried to see if we were efficient at comparing any language with any other language. We can expect languages like French and Italian to be harder to distinguish from each other since they have similar latin roots. Here is the resulting matrix:

![Languages Matrix](/blog/assets/2014-04-24/2017-04-24-lstm-for-language-detection/langmat.png)

As we can tell from these results, our model is pretty consistent with the all tested languages, with the exception or ltn vs. ltn1, for these languages are extremely close together. Our best results are yps vs frn and yps vs por with an accuracy of 0.97, which is an excellent score. On the other side, as expected frn and itn have a poorer accuracy: 0.81. 

### Optimizing hyperparameters

As suggested earlier, we can make things better by varying the hyperparameters of the model.

We started by changing the length of the feeded substrings:

![Accuracy and computational time in function of the length of the input sequences](/blog/assets/2014-04-24/2017-04-24-lstm-for-language-detection/strlen.png)

As we can tell from the output above, the size of the substring has a great influence over the the accuracy and computational time. It grows continuously until 40, and at that point it starts to overfit and decrease rapidly to a point at which it's only slightly better than chance. 

At substring size of 30 and 40, the computed accuracy is 1. That doesn't mean that the actual accuracy of our model is 100%. Indeed, as we only tested it on 200 samples, and it got every single one of them right, that means that the accuracy of our model is >0.995

The size of the substring isn't the only influential parameter in this experience, as it also has a side effect. Indeed, as we are also training on every padded version of each substring, increasing their size will also increase the size of the training set. This partly explains the better results, but also the longer computational time.

We also tried to see the influence of the number of training epochs:

![Accuracy and computational time in function of the number of epochs](/blog/assets/2014-04-24/2017-04-24-lstm-for-language-detection/epochs.png)

As we can tell from the output above, the number of training epochs doesn't have much influence over the accuracy of our model. The accuracy seems to gradually, slowly increase intil about 20 epochs, where it starts to overfit and fall back. The computational time increases linearly to the number of training epochs, which is kind of intuitive.

Finally, we computed four different versions of both eng.txt and frn.txt to the following different sets of characters:

- *everything*: default version
- *no_punctuation*: keep only letters, numbers and spaces
- *no_punctuation_digits*: keep only letters and spaces
- *no_punctuation_digits_spaces*: keep only letters

![Accuracy and computational time in function of the charset](/blog/assets/2014-04-24/2017-04-24-lstm-for-language-detection/charset.png)

As we can tell from the output above, the set of characters used has very little influence over the accuracy of our model. The best score is still when the charset used is the default one, the one with everything. We can guess that punctuation and spaces help captures the internal structure of English and French languages.

### Training with more texts

For this part, we tried training our LSTM models on different datasets and different types of texts to see what quality of content it could generate.

#### From wikipedia

Here is a text generated from some wikipedia articles:

```
Generated from the seed " by the revenge":
 by the revenge ason was mostook to producers of the brong, knownine nerds
 aired corralio and he was hor the say affece comby, thetrated held the 
 contucky of the season wassond securly frouge or new york ciedt as ans" he was 
 arred only badully undwa ibborst. tan is sid layonksord it aumod only culs 
 assoce relth so audy to produce only spy: currdard th stealicathedrapic eali 
 is added the trogen, sometor instruments well-squestrume hed with the roon, 
 boham also abson was hosted by actord ...
```

We can see from the output above that the results are not as good as what we could expect. This may be due to the fact that Wikipedia articles tells about a wide variety of subjects with a lot a really specific/technical words and names, thus quickly become over-complicated for our model. We should try on a text with a much smaller vocabulary.

#### From novel extracts

Here is a text generated from some random paragraphs from Jack London - Martin Eden:

```
Generated from the seed "n she had taken":
 n she had taken the was no so far nearer, talking nearybeahing come than the 
 world, and to the fast looking of soul out lefior to feie caced ly of eyes ff 
 orld so lave gyinssinuthe mean worked the poare, anot him, like fee scove a 
 scarions and was was a nurry. pointent of anothe so faist oper dient and form 
 him and the two fight-fly he the old of broak, ruth. he so faint o blact morie 
 wee no so fal coass in the time and she would no make sudeasing so making 
 worlh-ly in thand to the fieriepsoon on toly, lone of soul ,umm noworked with
```

As we can tell form the output above, the results are already much better than wikipedia articles. It doesn't quite tell a story, nor even always output real words, but the overall structure is respected. In order to improve the results, maybe we can try working on set of words instead of a set of characters.

#### From rap lyrics

Here is a text generated from some cult Eminem lyrics

```
Generated from the seed "that is gaping ":
 that is gaping too it with your cannin' yo be the be how why wigh like yo, in 
 want to seour just lost it i'm job for with the with my han to examain than to 
 stain to i cannto tan' to start it, i don't warn to be the beside, who 
 lose-mout lick to, shit we acaust so i ain' to kneed to sk it that with the 
 soldiers up in hance to know it the s in it i ain't way to words
```

The results are quite encouraging, and even really funny as long you've got some imagination: *"so i went the shit we acat i were lost it to cross"*, *"so i ain lour to a start a rhyme i'm back"*, *"try to forge an exatow to forget yo you so i ail jost show in it", "your jawsf never will, fuc wint!"*. This model obviously isn't ready to write some real rap lyrics, but it definitely shows that there are a lot of things we can do in this domain.

## References

- [Getting started with the Keras Sequential model](https://keras.io/getting-started/sequential-model-guide/)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
