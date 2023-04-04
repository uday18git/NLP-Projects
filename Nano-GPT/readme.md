# Nano GPT
## Overview
This project involves building a language model called Nano GPT that uses the Shakespeare dataset, which has a length of 1115394, and has a total of 10.7 million parameters.
The Nano GPT model is based on the transformer architecture, which has been widely used in natural language processing tasks such as machine translation, language modeling, and text classification.
The main goal of this project is to train the Nano GPT model to generate text that mimics the writing style of Shakespeare. 
To achieve this, we have used the Shakespeare dataset,which contains a large number of lines of text written by Shakespeare, including plays, sonnets, and other works.

## Changes Log
We start from a bigram model then we make the following changes ->

### After introducing a intermediate layer -> 
* train loss -> 2.5006 
* val loss -> 2.5117
 so far we have taken indices and we have encoded them based on the identities of the tokens.
 next we will encode them also using thier position in the sequence
### When applied a single head ->
* train loss -> 2.4 
* val loss -> 2.44
### When applied multi headed attention 
* train loss ->2.2424 
* val loss -> 2.2696
### When we added a feed forward ->
* train loss ->2.2179 
* val loss ->2.2359 (better)
 when we add many blocks of feed forward and multi head attention , it does not do that well because now our network is becoming deep and we need to add resnets and dropout and norm
### When we apply resnets with the blocks 
* train loss -> 1.9727 
* val loss ->2.0686 (much better , and we also see that it is overfitting a little bit so we add layer norm)
 layer norm is similar to batch norm
 spoiler it is very complicated lol
 in call function of batch norm we change the 0 to 1 to make layer norm
 we use xmean= x.mean(dim=1, keepdim=True)
 we use xvar= x.var(dim=1, keepdim=True) we normalize the rows
 in paper we see that the layer norm is applied before the feed forward and after the multi head attention but we are applying 
 it after the feed forward and before the multi head attention it is nowadays common to do so.
### After adding layer norm 
* train loss ->1.9883 
* val loss -> 2.0828
### After scaling the model ->
* train loss -> 1.4780
* val loss -> 1.4876

## About transformer architecture
The transformer architecture is a type of neural network that has revolutionized the field of natural language processing (NLP) by improving the performance of language models on a wide range of tasks.
The architecture was introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017.
The transformer architecture is based on the concept of self-attention, which allows the model to focus on different parts of the input sequence at each step of the computation.
The model is composed of an encoder and a decoder, each consisting of a stack of multiple layers.
Each layer of the encoder and decoder contains two sub-layers: a multi-head self-attention mechanism and a feed-forward network.
The multi-head self-attention mechanism computes a weighted sum of the input sequence, where the weights are determined by the similarities between each pair of input tokens. 
This allows the model to capture long-range dependencies between different parts of the sequence.
![transformer](https://user-images.githubusercontent.com/102567732/229882013-df299348-02c3-4e07-b0b1-109289f3b87d.png)
