---
layout: post
title:  "Birdbrain batching and tips on never using it"
date:   2025-06-08 00:00:00
categories:
---

$$\require{amsmath}$$

Batching is the process of passing multiple datapoints through a machine learning model at once.
It is nearly always used in the context of training these models, where the batches themselves are often referred to as *mini-batches*.
In this context, it is crucial for both computational efficiency and convergence of the training process.
However, the former motivation also applies to inference, meaning that batching is also at the core of using any machine learning model after training.

This post is about batching in scenarios where the feature vector lengths differ, and how it can be done in a way that is either **clever**, **industrious** or **birdbrained**.
For the uninitiated, there are also short introductions to the prerequisites.
Each subchapter has a title, whose cursive segment signifies its subject.
If the subject matter seems obvious, you can go ahead and skip forward. 

# Laying the groundwork *(perceptrons)*

The individual units of "intelligence" used in "artificially intelligent" systems today did not come out of nowhere, but instead had an evolutionary process.
First came the McCulloch-Pitts neurons in the 40s, then the perceptrons in the 50s, and finally the neurons that we know today from MultiLayer Perceptrons (MLP) in the 80s.
Let's skip the McCulloch-Pitts neurons as they are not terribly relevant, and begin with the perceptrons.

Let's define $i$ as the dimensionality of the input vector ie. the number of features.
Mentally you can replace this with anything you'd like.
For the sake of an example, I'm going to use the **price**, **gas mileage** and **production year** of some car.
In all simplicity, a perceptron consists of a weight vector $\boldsymbol{w} \in \mathbb{R}^{i}$, and a bias $b$, which is just a scalar value.
Then, given a feature vector $\boldsymbol{x}$, we take the dot product $\boldsymbol{x} \cdot \boldsymbol{w}$ and sum it with the bias $b$.
So, with the (arbitrary) weights $\boldsymbol{w} = [-2, 1522, 8]$ and bias term of $10$, our feature vector $\boldsymbol{x} = [10000 \text{ (€)}, 10 \text{ (l/100km)}, 2000]$ produces the following result:

$$\boldsymbol{x} \cdot \boldsymbol{w} + b = [10000, 10, 2000] \cdot [-2, 1522, 8] + 10 = 11230$$

The values used could just as well be floating point numbers. I've used whole numbers just for the sake of simplicity.

These two operations are then wrapped in a function that outputs 1 if its input is greater than zero, and 0 otherwise.
The fancy name for this is the *Heaviside step function*.
Here, the perceptron would therefore return 1.
Again, this could mean that this car is worth purchasing, or whatever you'd like.
This is all a perceptron is; take in a vector of features, and map it into a scalar that signifies some interpretation of the features.
This simplicity is both a blessing and a curse - it means that we can [guarantee convergence](https://en.wikipedia.org/wiki/Perceptron#Convergence_of_one_perceptron_on_a_linearly_separable_dataset), but also that we cannot solve any system where the data is not linearly separable.
Even modeling a simple XOR function is not possible for this perceptron.

# Increasing the units of intelligence *(multilayer perceptrons)*
Let's increase the number of these perceptrons, say, by two. So now we have three perceptrons, all of which take in the same vector and map it to a scalar value using their respective weights and biases. Each perceptron is trying to interpret some different thing. Let's say they're interpreting whether the
car is worth buying for three different people, all with different desires and requirements. Passing a feature vector through the three perceptrons could be done separately for all of them, simply by repeating
the aforementioned process of taking the dot-products and summing the results with the biases.

However, we can also stack the (columnar) weight vectors on top of each other, creating a matrix, and multiply its transpose with the feature vector.
After also combining the biases into a column vector and summing it with the output of the vector-matrix product, we have calculated the results of all three perceptrons, prior to the Heaviside step function.

More concretely, with the (arbitrarily chosen) weights and biases

$$
\boldsymbol{W} =
  \begin{bmatrix}
    -2 & 1522 & 8 \\
    -3 & 111 & 23 \\
    3 & -3263 & 0
  \end{bmatrix}

\boldsymbol{b} = [10, 0, -44]
$$

the same feature vector passes through as follows:

$$
\boldsymbol{x} \boldsymbol{W}^T + b = [10000, 10, 2000]
  \begin{bmatrix}
    -2 & 1522 & 8 \\
    -3 & 111 & 23 \\
    3 & -3263 & 0
  \end{bmatrix}^T
+ [10, 0, -44] = [11230, 17110, -2674]
$$

If we were to take the Heaviside step function, we would see that the first person should buy the car (as we already knew) along with the second one, but the third person should not. But let's forget about the Heaviside step function for now, and just look at the values produced.

We have mapped our feature vector into an output vector containing three unique interpretations of the features, calculated using three perceptrons.
As we have omitted the Heaviside step function, let's change the name of our units of intelligence from perceptrons into *neurons*. 
What we have produced is perhaps the simplest, and without question the most foundational component in modern artificial neural networks: the linear layer.
These are otherwise known as Fully Connected (FC) layers, dense layers, affine layers, Feed-Forward (FF) layers or seemingly any other word salad you could to come up with.
Naming conventions in machine learning follow the long, traditional engineering approach of *None*. 

These linear layers can be chained together with a non-linear function of your choice between them to already produce some fairly powerful decision makers.
Combining at least two of these layers is what is referred to as a MLP or Feed Forward Network (FFN) - the original meaning of the term *neural network*.
Although the latter has evolved to mean many different types of architectures, the MLP is still present within effectively all of them, including the transformer architecture used universally by large language models:
{:refdef: style="text-align: center;"}
![](/blog/assets/transformer_ff_layers.png){:width="400px"}
{: refdef}

# Bumping up the dimensionality *(batch processing of constant sized feature vectors)*

Now, let's say we have four feature vectors we want to pass through the layer.
Keeping up with our previous analogy, let's say we have four different cars whose suitability we want to evaluate for the three people from before.
If we process each vector independently, this means roughly four times the work from before: A cycle of loading the feature vector into VRAM, performing the two computations, and then writing the resulting vector back into RAM.
This is not the best way to do things, as we can do something very similar to how we processed multiple perceptrons at once.

Let's create a feature matrix, which is simply the concatenation of the feature vectors.
The bias vector acts a bit odd here, as it is added row-wise to the matrix.
Linear algebra libraries typically express this through a process known as *broadcasting*.
However, in terms of mathematical notation, this is the same as if we were to stack copies of the row-vector on top of each other, forming a matrix where the number of rows equals the number of datapoints, and then performed normal element-wise addition.
Therefore, let's define it as such here.

With the feature vectors (cars) of $\boldsymbol{x}_1 = [10000, 10, 2000]$, $\boldsymbol{x}_2 = [5500, 15, 1988]$, $\boldsymbol{x}_3 = [78000, 32, 2021]$ and $\boldsymbol{x}_4 = [100000, 25, 1999]$ we have this feature matrix:

$$
  \begin{bmatrix}
    10000 & 10 & 2000 \\
    5500 & 15 & 1988 \\
    78000 & 32 & 2021 \\
    100000 & 25 & 1999 \\
  \end{bmatrix}
$$

Passing it through the linear layer from before, made up of three neurons...

$$
\boldsymbol{X} \boldsymbol{W}^T + \boldsymbol{B} = \\

  \begin{bmatrix}
    10000 & 10 & 2000 \\
    5500 & 15 & 1988 \\
    78000 & 32 & 2021 \\
    100000 & 25 & 1999 \\
  \end{bmatrix}
  \begin{bmatrix}
    -2 & 1522 & 8 \\
    -3 & 111 & 23 \\
    3 & -3263 & 0
  \end{bmatrix}^T
+ 
  \begin{bmatrix}
    10 & 0 & -44 \\
    10 & 0 & -44 \\
    10 & 0 & -44 \\
    10 & 0 & -44
  \end{bmatrix} = \\

  \begin{bmatrix}
    11230 & 17110 & -2674 \\
    27744 & 30889 & -32489 \\
    -91118 & -183965 & 129540 \\
    -145948 & -251248 & 218381 \\
  \end{bmatrix}
$$

You'll notice that the first row is in fact the same vector you'd get for passing the first feature vector through the linear layer alone.
This notion is true for the other rows as well.
We have not altered the results, we have just calculated them all at once.
Continuing with our logic, the first and second people should buy one of the first two cars, and the third person should buy one of the later two cars.

In total, we have passed **four** different feature vectors through a network of **three** neurons in **one** matrix-matrix multiplication and **one** matrix-matrix addition.
In the naive approach we would be performing 12 operations using seven different vectors.
What does this stark decrease in the number of individual operations actually do for us in practice, if anything?

# A quick peek under the hood *(parallelized matrix-matrix multiplication)*

I wrote a [post](./mulling-over-matrix-multiplications-in-cuda) about matrix multiplications within CUDA.
I would recommend reading it, as it is quite relevant, but wouldn't necessarily consider it to be a prerequisite as it is also needlessly in-depth. 

The bare minimum to understand for the purposes of this post is this: modern GPUs are able to calculate large matrix-matrix products extremely efficiently, achieving exceptional levels of parallelism.
If we are able to turn our model from many small dot products into a (large) matrix-matrix multiplication, we are able to tap into this efficiency.

In an ideal world, we are able to process many datapoints in a way where processing all of them is roughly as fast as processing just one of them, given that you have the required cores and memory bandwidth available.
In our example, evaluating **four feature vectors for three perceptrons** is just as fast as evaluating **one feature vector for one perceptron**. 
This is the magic of parallelized matrix multiplications.

## The clever, the industrious and the birdbrained
This is where we arrive at the real subject of this post.

When the feature vectors are statically sized (details of car listings, sensor readings, color intensities of pixels), parallelism is straightforward and obvious.
However, things get interesting when the feature vector sizes differ.
One pertinent scenario where this typically occurs is in natural language.
*From this point onwards, we're going to focus on movie reviews as an example, but the ideas apply generally to all language models and document types.*

# Getting our feature vectors
A countless number of randomly occurring things in this world produce the exponential distribution. More specifically, any homogeneous Poisson process creates inter-arrival times that form the exponential distribution. Breaking that statement down,
a homogeneous Poisson process is simply a random process where in a given interval of time, the probability of an event occurring is proportional to the length of the interval and independent of other intervals.
This is a continuous time version of the (discrete) Bernoulli process, where each individual experiment has a (constant) probability of success that is independent of all previous and later experiments; think of a coin flip.

In the Bernoulli, we have discrete *events* that have a *success rate*. 
In the Poisson, we have continuous *time intervals* that have an *arrival rate*.
A Bernoulli process generates a geometric distribution, whereas a Poisson process generates an exponential distribution.


As an everyday example, let's say you sit by a highway and spot red cars as they pass by.
The probability of seeing at least one red car in a time interval of 5 minutes is, say, 30%.
This probability, for any given interval, does not vary based previous or subsequent intervals.
Similarly, let's say that the probability of any given car being red is 5%.
This probability, for a given car, does not vary based on the cars around it.
The (discrete) number of non-red cars between each spotting of a red car would produce a geometric distribution; The (continuous) lengths of time intervals between red cars would produce an exponential distribution.

You can probably come up with a dozen ways this could be used off the top of your head: customer service request durations, equipment usage times between malfunctions, radioactive decay...
If we parameterize the distribution to match our historical data, we can evaluate likelihoods for future events with some degree of confidence.
Here, let's use an exponential distribution to model the lengths of our movie reviews.
You could just as well (and perhaps more sensibly) use a geometric distribution.
In any case, we sample 1000 values from an exponential distribution and discretize them to signify movie review lengths: 

![](/blog/assets/review_lengths.png){:width="850px"}

For practical purposes, I've turned all the zeroes we sampled into ones. The longest review we got was just about 400 words long, while most of them landed within 1-10 words.
The rest of the lengths are scattered around 10-200 in decreasing fashion, with a few popping up between 200 and 400 as well.
Makes sense, most people write short reviews, some put in more effort. A handful decide to write a novel.

# Optimal brain damage

Let's say we now have a sentiment analysis model and want to analyze these reviews.
Let's further define that our model's weights are in half-precision (16 bits).
The model consists of the following things:
* Embedding matrix for mapping tokens into embedding vectors
* Multiple *blocks* made up of a non-masked single-head self-attention layer and a linear layer
* A decision head (just another linear layer), which then maps its input into a vector of two values.
    * The first of these is interpreted as being a measure of positivity, the second as being a measure of negativity. The larger of the two is chosen as our label.

Here is an implementation without any libraries or frameworks (other than for basic linear algebra operations and data types):

{% highlight python %}
hidden_dim = 4096
vocab_size = 65535
num_layers = 70
dtype = float16

class WhateverModel:
    def __init__(self):
        self.padding_token_id = -100

        # Embeddings
        self.embeddings = torch.rand(
            [vocab_size, 1, hidden_dim], dtype=dtype, device="cuda"
        )

        # Attention weights
        self.q_proj_weights = [
            torch.rand([hidden_dim, hidden_dim], dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]
        self.k_proj_weights = [
            torch.rand([hidden_dim, hidden_dim], dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]
        self.v_proj_weights = [
            torch.rand([hidden_dim, hidden_dim], dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]
        self.o_proj_weights = [
            torch.rand([hidden_dim, hidden_dim], dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]

        # Linear layers
        self.linear_weights = [
            torch.rand([hidden_dim, hidden_dim], dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]
        self.linear_biases = [
            torch.rand([1, hidden_dim], dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]

        # Decision head
        self.decision_head = torch.rand([hidden_dim, 2], dtype=dtype, device="cuda")

{% endhighlight %}

The model is missing all of the bells and whistles of modern large language models. No grouped query attention, no GeLU (or any other nonlinearity), no residual connections.

This is by design, as we are not trying to build an intelligent model, but a model that embodies the computational burdens of an intelligent model without all the distractions.
We have added one component where the tokens are interconnected (the self-attention) and one where they are not (the linear layer), and that's it.
Consequently, our forward pass is effectively just 7 matrix multiplications, performed for each of our *blocks*.
Goes to show how simple this stuff gets when you remove the cruft.

Here's the forward pass:

{% highlight python %}
    @staticmethod
    def normalize(X: Tensor):
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)
        return (X - mean) / (std + 1e-5)

    def forward(self, documents: list[list[int]]) -> list[bool]:
        X, attention_mask = self.embed_documents(documents)

        attention_mask = attention_mask.unsqueeze(2)
        attention_mask = torch.cat(
            [attention_mask] * attention_mask.shape[1], dim=2
        ).transpose(1, 2)

        for i in range(num_layers):
            # Attention layer
            q_states = X @ self.q_proj_weights[i]
            k_states = X @ self.k_proj_weights[i]
            v_states = X @ self.v_proj_weights[i]

            attn_weights = q_states @ k_states.transpose(1, 2) / math.sqrt(hidden_dim)
            attn_weights += attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights @ v_states

            X = attn_weights @ self.o_proj_weights[i]

            # Linear layer
            X = X @ self.linear_weights[i] + self.linear_biases[i]

            # Normalization so the values don't explode
            X = self.normalize(X)

        # Just take the first token's output as a pseudo "class token"
        cls_tokens = (X @ self.decision_head)[:, 0, :]
        decisions: list[bool] = [(tok[1] >= tok[0]).item() for tok in cls_tokens]  # type: ignore

        return decisions

{% endhighlight %}

In terms of vocabulary, let's define that our tokenizer does not map subwords, but only knows whole words.
This is just so we don't have to get bogged down by the token/word distinction.

# Passing a feature vector through the model

When we have a review that is made up of $100$ words.
These words are first mapped into $100$ tokens based on our vocabulary.
Then, each of these tokens is given an embedding vector simply by looking it up from the embedding matrix.
Notice that our model uses an embedding dimension (or hidden dimension) of $4096$.
As such, the input to the rest of the model is now of size $\mathbb{R}^{100 \times 4096}$.

We can pass this through the linear layers just fine, as they just perform a matrix multiplication with the transpose of a matrix whose shape is $\mathbb{R}^{4096 \times 4096}$.
We will receive a new $\mathbb{R}^{100 \times 4096}$ matrix as a result, keeping the dimensionality of our matrix unchanged.
The self-attention also starts off with three linear layers, which we know to not alter the dimensionality.
This produces three different matrices of identical dimensions.
The first is multiplied with the second's transpose, the result of which is then multiplied with the third.
As these three matrices are of the same dimensions and we don't care about their contents right now, let's call them all $\boldsymbol{X}$.
For any matrix $\boldsymbol{X}$ of arbitrary dimensions, $\boldsymbol{X X^T X}$ will always produce a matrix of identical dimensions to $\boldsymbol{X}$.
So our single review passes through a full block no problem, producing a matrix of identical dimensions.
The final linear layer has two columns, so it will output a 2-length vector, from which we then pick out our label.

Therefore, the naive way to process our 1000 reviews is to just pass them through the model individually.
In this case, we pass through a matrix of shape $\mathbb{R}^{\text{number of words in review} \times 4096}$ for each review, and receive a binary output.
However, this does not utilize the aforementioned GPU parallelism all that well.
A better idea would be to process the reviews "all at once". Is this possible?

# Testing the waters
Let's try to plug in a second review. Let's say that it is 90 words long. Well, through an identical starting process it turns into a $\mathbb{R}^{90 \times 4096}$ matrix.
However, if we were to pass these two reviews through the first linear layer of the first self-attention layer, what would we be using as the input?
As is emphasized in my [post on CUDA programming](./mulling-over-matrix-multiplications-in-CUDA), the speed of our calculations originates from collections of threads performing the same instructions in parallel.
We need to be able to express our data as a tensor to be able to reasonably create such instructions and reuse them throughout the differently sized batches.
Sadly, the dimensions of our matrices do not match, so we cannot concatenate them directly to form a tensor.

This is where padding comes in. We define a special token which acts as a "padding token", and add 10 of those at the end of our shorter review.
Now, we have two $\mathbb{R}^{100 \times 4096}$ matrices that can be concatenated along the third axis, producing a $\mathbb{R}^{2 \times 100 \times 4096}$ tensor which is then passed through the model.
In components where the tokens are intra-connected, namely the self-attention, we can set the attention values between all real tokens and these padding tokens to zero.
In components where the tokens are not intra-connected, the padding tokens do not interact with our real tokens.
Therefore, we avoid any modifications to our result, no matter how much padding we add. Fantastic.

So what we can do with our 1000 movie reviews is to find the longest review, pad all other reviews to its length with these padding tokens, and then pass all the reviews through the model in one go?
In theory, yes, but this is where reality kicks in.
With 1000 reviews and a maximum review length of 400 tokens, how big is our input matrix?
Well, it's $\mathbb{R}^{1000 \times 400 \times 4096}$, which means our **input matrix alone** would take $1000\cdot400\cdot4096\cdot16\cdot0.5\cdot10^{\ -9} = 13.1072$ GiB of VRAM, assuming 16-bit floating point values and zero overhead.
This is before any of the matrix multiplications have been performed and have had their results stored in memory, only to be recalled in the future in order to produce a new matrix that uses up yet another 13GiB. What happens if our hidden dimension increases? Or we get more data?
Or we get a really long review? No matter the depth of your pockets, there is going to be a limit. This is a non-starter.

So we have to sort of *reduce* the batching. In essence, we see how many reviews our hardware can process at a time, then take that many reviews from our $\mathbb{R}^{1000 \times 400 \times 4096}$ matrix, load them into VRAM and pass them through the model.
Rinse and repeat until we're done. Here's the implementation:

{% highlight python %}
def pad_documents(model: WhateverModel, documents: list[list[int]]) -> list[list[int]]:
    max_tokens = max([len(doc) for doc in documents])

    output = documents.copy()

    for i in range(len(documents)):
        num_padding = max_tokens - len(documents[i])
        output[i].extend([model.padding_token_id] * num_padding)

    return output

def batch(model: WhateverModel, documents: list[list[int]]):
    sentiments = []

    documents = pad_documents(model, documents)
    for batch_start in range(0, num_docs, batch_size):
        batch = documents[batch_start : batch_start + batch_size]

        # Forward pass of the model
        sentiments.extend(model.forward(batch))

    return sentiments

{% endhighlight %}
For my 16 GiB of VRAM this limit turns out to be eight reviews.
So, we take eight reviews at a time from our large tensor, forming a new tensor of shape $\mathbb{R}^{8 \times 400 \times 4096}$, and pass it through the model.

This works.
It stays within the memory limits and oh boy does it thrash around the GPU, getting it to a constant 100% utilization.
This is exactly what we want, so all signs point to us having done a great job utilizing our resources.

We have succesfully built the **birdbrained** approach.

# Bird's eye view of the problem

Don't get me wrong, I don't mean to knock on anyone for thinking that the previous approach is a serviceable idea. I've built this kind of a system myself, and seen a few written by others since.
However, in any task that is performance critical or is going to be run often (consuming electricity), you really don't want to do this.
The reason becomes obvious when we look at the very first batch:

{:refdef: style="text-align: center;"}
![](/blog/assets/batch_heatmap.png){:width="850px"}
{: refdef}

This "heatmap" shows the first batch of 8 reviews. The colors go up to 60000 or so, as that is the largest vocabulary index present in the batch.
The entire purple section is just padding.
When we pass this batch though the model, the first step is a linear layer with a matrix of size $\mathbb{R}^{4096 \times 4096}$.
Each of the eight rows in the above image will be a $\mathbb{R}^{409 \times 4096}$ matrix that gets multiplied with the linear layer's matrix.
All the values in the resulting $\mathbb{R}^{409 \times 4096}$ matrix will be calculated, but for the first review, only **24** of the rows are **ever** used.
For the 7th review, all but **9** of the resulting 409 rows will **never** be used.

This process gets repeated over and over, with us either discarding or masking out all of the values that get calculated as a result of the padding tokens.
Thinking back to the processing within the GPU, there is a limited amount raw computations we can do during any given timeframe.
When we process a batch like this, we are saturating that capacity with garbage calculations whose results will get thrown out as soon as they finish.
Here is the heatmap for the entire dataset:

{:refdef: style="text-align: center;"}
![](/blog/assets/all_heatmap.png){:width="850px"}
{: refdef}

We can see that this is not an isolated indicent.
It almost doesn't even matter where you slice your batch of 8 reviews, you're mostly just processing garbage.
There's exactly one batch where there is any sort of justification for this level of padding, and that's the one with the longest review.

# Working hard

Here's an idea that probably comes to mind: Let's first choose our 8 reviews and place them in a batch, and then pad according to the maximum length in that batch.
Our implementation only changes a tiny bit as we shift the padding two lines downwards, applying padding only after forming the batch:

{% highlight python %}
def batch(model: WhateverModel, documents: list[list[int]]):
    sentiments = []

    for batch_start in range(0, num_docs, batch_size):
        batch = documents[batch_start : batch_start + batch_size]
        batch = pad_documents(model, batch)

        # Forward pass of the model
        sentiments.extend(model.forward(batch))

    return sentiments

{% endhighlight %}

The padding is now done inside a loop instead of all at once, but that process is effectively instant. More importantly, look at how much the first batch's heatmap improves:

{:refdef: style="text-align: center;"}
![](/blog/assets/hard_heatmap.png){:width="850px"}
{: refdef}

Now we're cooking. The massive wave of purple at the end of each review is gone, and we are only processing what we need to process.
Yeah, we're still calculating *some* garbage, but that's just the price you pay for batch processing.
After all, we can't do any less padding, as we have padded to the length of the shortest review in the batch.
The only way we could do better would be to reduce the length of the shortest review in the batch.

# Hardly working
Here's what you should actually do: simply sort your reviews based on length, and then pad each batch separately. In terms of code, we add one line before the loop:

{% highlight python %}
def batch(model: WhateverModel, documents: list[list[int]]):
    sentiments = []

    documents.sort(key=lambda x: len(x))
    for batch_start in range(0, num_docs, batch_size):
        batch = documents[batch_start : batch_start + batch_size]
        batch = pad_documents(model, batch)

        # Forward pass of the model
        sentiments.extend(model.forward(batch))

    return sentiments
{% endhighlight %}

It doesn't get any simpler than this.

We "group" reviews based on their length, and then place reviews from the same "group" to the same batches.
After all, the downside in the previous approach was that the batch contained reviews of differing lengths due to chance.
Don't count on your luck to segment the reviews into perfect batches, it's not going to happen.

I won't show you the first batch, as it's just a matrix of 8 scalars (ie. 1-word reviews). Instead, here's the 101st batch, containing the reviews indexed 800 to 808:

{:refdef: style="text-align: center;"}
![](/blog/assets/smart_heatmap.png){:width="850px"}
{: refdef}

There's barely any padding, even as we approach the end of our dataset, where there are less reviews and length differences between reviews become more pronounced.

# So what?

So does any of this matter? Is this really all that useful?
Are the additional computations introduced by the sorting even worth it?
Here are the results on a RTX 4080 SUPER:

| Batching      | Time taken (seconds) |
| ----------- | ----------- |
| Birdbrained   | 58.681        |
| Industrious   | 17.579        |
| Clever      | 8.406       |

Quite the jump, and remember: we are performing *exact inference*, producing the same output.
There are no stochastic approximations or unfounded assumptions here, the results are **unchanged**. 
And yet, the runtime drops sevenfold.

This can also be motivated in a more visual sense through an Area Under Curve (AUC):

![](/blog/assets/batch_sizes_subplot.png){:width="850px"}

Here, we have plotted the matrix dimensions (ie. padded length of reviews, or number of rows) for each review in our batches.
As the padded length of the reviews defines the dimensionality of the matrix multiplications occurring during the forward pass, it is a good litmus test for computational complexity.
The larger the colored area, the more computation is required. Let's place these graphs on top of each other:

![](/blog/assets/batch_sizes_oneplot.png){:width="850px"}

It becomes clear just how much better we end up doing by changing two lines of code.

# Honing the blade

There are also further optimizations that could be performed.
As mentioned, when using the birdbrained batching approach, the maximum batch size possible with my GPU is 8.
This is also true for the later two, as we still need to process the batch that contains the longest review *at some point*.
Then we will need to allocate space for the full $\mathbb{R}^{\text{batch size} \times 409 \times 4096}$ matrix anyways.
And yes, with a constant batch size there's no getting away from this fact.
However, if we allow the batch size to change dynamically, we can actually squeeze out just a bit more perforamance yet.

Instead of setting a batch size limit, let's set both a token limit and a length delta limit between the reviews within a single batch.
Now, we add reviews to our batch until one of those limits is reached, and then process the batch. The first limit makes sure our GPU doesn't run out of global memory,
and the second makes sure we don't start processing too much garbage again.
This gets a bit more involved than the previous approaches, but it drops the runtime down to 7.306 seconds.

{% highlight python %}
def batch(model: WhateverModel, documents: list[list[int]]):
    sentiments = []
    documents.sort(key=lambda x: len(x))

    max_tokens_per_batch = 21000
    max_length_diff_within_batch = 8

    batch_start = 0
    curr_index = 0
    while curr_index < num_docs:
        curr_batch_tokens = 0
        curr_batch_min_length = len(documents[batch_start])

        batch = []
        while curr_index < num_docs:
            next_doc_len = len(documents[curr_index])
            if (
                next_doc_len - curr_batch_min_length > max_length_diff_within_batch
                or curr_batch_tokens + next_doc_len >= max_tokens_per_batch
            ):
                break

            curr_batch_tokens += len(documents[curr_index])
            curr_index += 1

        batch = documents[batch_start:curr_index]
        batch = pad_documents(model, batch)

        # Forward pass of the model
        sentiments.extend(model.forward(batch))

        batch_start = curr_index

    return sentiments
{% endhighlight %}

# Wrapping up
I'm hesitant to call batching a necessary evil, as there really is nothing nefarious about it, as long as you do it right.
And this is not to say that I think it's trivial, far from it.
There's always going to be trade-offs and it's never going to be quite perfect as long as your data stream is not 100% consistent and predictable.
It's part of every ML model training cycle and effectively every deployment of said models, so it's important to figure out.
But the most important thing is that if you are batching feature vectors that cannot be concatenated directly into a tensor, don't just pad them naively to make the problem go away.
Think about it for just a bit, or you'll end up doing birdbrain batching.

The code for these experiments can be found [in this Github repository](https://github.com/roopekj/smart-batching).

