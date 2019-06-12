---
layout: post
title:  "Alternate Realities: Partisan N-Gram Use"
date:   2017-01-02 11:48:00 -0500
categories: [partisan media analysis]
---
## Partisan Word Use
### [In progress]

In the prior post, we created a means of reading tokenized data from our database of articles and a list of n-grams (like *supreme_court*) that we'll impose on our data to count certain n-grams as a single token.

In this post, we're going to engage in a preliminary and fairly rudimentary analysis of the text data as it relates to partisanship. In order to do so, we'll look at which n-grams are most characteristic of the left and the right (later looking at sources individually, perhaps at a future date). I'm expecting results sort of in line with this [538 piece](http://fivethirtyeight.com/features/these-are-the-phrases-each-gop-candidate-repeats-most/) on the text of the GOP debates.

This analysis will also set us up for topic modeling (out of fashion as it may be) since we'll just be building a large document-term matrix. I also hope to use it on weekly cuts of the data to see how emphasis in coverage changed over the election cycle.

This ended up being a little more technical than I anticipated, so if you're just looking for results you can skip down <a href='#visualizing-the-data'>to the end</a>.

### Reading the Data
First, we want to collect the data at the level of each source, which will require us to subclass the `SentenceStream` generator we built in the last post. `CountVectorizer` from `scikit-learn` treats each element in an iterator as a document, so we'll restructure the generator such that all words from each source are combined into one list of strings.


```python
from itertools import groupby, imap
from nltk.corpus import stopwords

import numpy as np

from stream import SentenceStream #from the prior post

sql_url = 'postgres://postgres:**PASSWORD**@localhost/articles'

# ordering is a necessity, else groupby won't work
# we also have a few strange base_url's with '{xyz.com}',
# as well as reprinted articles from The Daily Caller we'll remove
query = """
        SELECT base_url, article_text
        FROM articles
        WHERE num_words > 100 and not
              (lower(article_text) like '%%daily caller news foundation%%' and
               base_url != 'dailycaller.com') and not
               lower(article_text) like '%%copyright 20__ the associated press%%'
        ORDER BY base_url
        """

class SourceNGramStream(SentenceStream):
    """
    Get a stream of pre-identified n-grams from each source
    """
    def __iter__(self):
        rows = super(SourceNGramStream, self).__iter__()
        source_sentences = groupby(rows, lambda x: x[0])
        
        for source, sentences in source_sentences:
            source_ngrams = [word for id, sentence in sentences for word in sentence if '_' in word]
            if source_ngrams:
                yield source_ngrams
```

We'll load in the identified n-grams from last time.


```python
import cPickle as pickle
with open('../intermediate/phrasegrams_all.pkl', 'rb') as infile:
    ngrams = pickle.load(infile)

# the n-grams we've located will now be identified in the stream of text
# using the MWETokenizer from nltk
src_stream = SourceNGramStream(sqldb=sql_url, query=query,
                               ngrams=ngrams, idcol='base_url')
```

### Building a document-term matrix
Now we can create a document-term matrix, where each source represents a document and our columns will be the n-grams from earlier.

Importantly, we're going to limit ourselves to tokens and n-grams that appear in two or more sources. I believe this choice enables us to consider these sources as a network, where there might be patterns of mutual influence on rhetoric and thinking. We don't really care about one-off uses of a particular word or phrase. (It also has the nice side-effect of getting rid of a lot of junk phrases).


```python
# we use this dummy tokenizer since nltk is doing the tokenizing in the stream
# (we can't do lambda x: x because it is unpicklable)
def no_tokenizer(x):
    return x

# in fact, all processing is done, and we just need to place it
# in the appropriate data structure 
vectorizer = CountVectorizer(analyzer='word', preprocessor=None,
                             lowercase=False, tokenizer=no_tokenizer,
                             min_df=2)
dtm_source = vectorizer.fit_transform(src_stream)
```


```python
from scipy import io

with open('../intermediate/vec_source_phrasegram.pkl', 'wb') as vecf:
    pickle.dump(vectorizer, vecf)
io.mmwrite('../intermediate/dtm_source_phrasegram.mtx',  dtm_source)
```

Despite having limited ourselves to only n-grams that appear in more than one source, there are still some phrases that muddy up the waters necessarily (like 'washington examiner news desk'). As a result, we'll remove those that appear in only one source over 95% of the time.


```python
import numpy as np
from scipy.sparse import csr_matrix

idx_norm_terms = np.all(np.true_divide(dtm_source.toarray(), dtm_source.sum(axis=0)) <= 0.95, axis=0).A[0]
features = vectorizer.get_feature_names()
features = [f for i, f in enumerate(features) if idx_norm_terms[i]]
dtm_source = csr_matrix(dtm_source[:,idx_norm_terms])
```

### An Initial Examination of Sources on the Left and Right
We'll return to this source-based matrix later, and will for now collapse this matrix to two rows, representing left- and right- aligned sources. This enables a more straightforward analysis of which ideas are of greatest concern to each side of the aisle.

We need to pull the old alignment data from ["Blue Feed, Red Feed"](https://github.com/jonkeegan/blue-feed-red-feed-sources) to correctly identify who is on the left and the right, then sum up the rows of our source matrix, grouping by alignment.


```python
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

# same order as earlier query
source_query = """
               SELECT base_url,
                      split_part(post_id, '_', 1) as fb_id
               FROM articles 
               WHERE num_words > 100
               GROUP BY base_url,
                        split_part(post_id, '_', 1) 
               ORDER BY base_url
               """

# collect source alignment data
sources = pd.read_sql(source_query, create_engine(sql_url))  
alignment_data = pd.read_csv('./input/included_sources.csv', dtype={'fb_id':object})\
                   .drop_duplicates('fb_id')
sources = sources.merge(alignment_data, how='left')

# we supplemented the data with infowars
sources.loc[sources.base_url == 'infowars.com', 'side'] = 'right'

# get indexes of left and right sources
sources_left = np.where(sources.side == 'left')[0]
sources_right = np.where(sources.side == 'right')[0]

# create a new document-term matrix of 2 rows
dtm_side = csr_matrix(np.append(dtm_source[sources_left,:].sum(axis=0),
                                dtm_source[sources_right,:].sum(axis=0),
                                axis=0))
```


```python
dtm_side
```




    <2x46945 sparse matrix of type '<type 'numpy.int64'>'
    	with 92574 stored elements in Compressed Sparse Row format>



And behold, our matrix! We'll now transform this count matrix (where A<sub>ij</sub> is the number of times term j appears on side i) into a [normalized term-frequency matrix](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html). This means that phrases that are highly common to each side are discounted, therefore promoting those that are more distinctive to each political side.


```python
dtm_side_tfidf = TfidfTransformer().fit_transform(dtm_side)

#get the column indices from largest to smallest
idx_sorted_tfidf_left = np.argsort(dtm_side_tfidf[0, ].toarray()[0])[::-1]
idx_sorted_tfidf_right = np.argsort(dtm_side_tfidf[1, ].toarray()[0])[::-1]

#nonzero terms
terms_sorted_tfidf_left = [features[i] for i in idx_sorted_tfidf_left if dtm_side[0,i]] 
terms_sorted_tfidf_right = [features[i] for i in idx_sorted_tfidf_right if dtm_side[1,i]] 
```


```python
# let's put together this information for plotting
pd.DataFrame({'term': terms_sorted_tfidf_left[:10000],
              'tfidf': dtm_side_tfidf[0, idx_sorted_tfidf_left[:10000]].A[0],
              'side': 'left'})\
  .to_csv('./output/ngrams_top10000_left.csv', index_label='rank')
    
pd.DataFrame({'term': terms_sorted_tfidf_right[:10000],
              'tfidf': dtm_side_tfidf[1, idx_sorted_tfidf_right[:10000]].A[0],
              'side': 'right'})\
  .to_csv('./output/ngrams_top10000_right.csv', index_label='rank')

pd.DataFrame(dtm_source.T.toarray(),
             columns=sources.base_url)\
  .assign(term=features)\
  .to_csv('./output/dtm_source_counts.csv', index=False)

sources[['base_url', 'fb_id', 'side', 'avg_align']].to_csv('./output/source_info.csv', index=False)
```

### Visualizing the Data
In a separate R script, I've summarized the above data to determine which n-grams are most associated with each side. In order to do accomplish this task, I sum up the number of times each n-gram appears in sources on the left and the right (normalizing for the total word count on each side), then find the relative share that each term appears in right-aligned sources. 

$$ \sum_{t=1}^n \frac{tf_{t, r}}{tf_{t, r} + tf_{t, l}} $$

This constitutes a measure of term partisanship, the distribution of which is displayed below.

<img src="{{ site.url }}/assets/img/term_alignment_dist.png" width="800">

With that established, we can take a look at the top terms on each side, as ranked by tf-idf. The size and color of these n-grams corresponds to the above measure. Since many terms appear on both the left and right, I've also created a filtered version of the data where we only look at terms with a partisanship measure greater than 1 standard deviation beyond the mean (per the chart above).

While this is only an exploratory look at the data, I think we're starting to see some interesting results (despite the obvious junk). *islamic_state* features more heavily on the right than on the left, for example.

If I had to make an initial and wholly speculative interpretation, I'd say that these data support the idea that hyperpartisan news sources focus on threats to their respective ideologies and values, rather than establishing them from first principles (or, if they are constructive, it is in a negative sense: "we're against" rather than "we're for"). I guess this is unsurprising, but I think we now have some hard evidence corroborating what we already knew (such is the bane of [positivist disciplines](https://tni-back-soon.github.io/essays/podcast-out/), apparently).

But just look: the left talks about *gun_violence* and the right *gun_control*. Conservative media focuses on *illegal_immigrants*, *syrian_refuges*, *sharia_law*, and *islamic_terrorism*. The left is concerned with the *religious_right*, *white_supremacist*, and *wall_street*. To be sure, the left media does have terms aligned with values and reform, like *mass_incarceration*, *climate_change*, and *human_rights* (as well as *lgbt_rights*). 

But it's also my bias that I view *climate_change* as non-reactionary. What *is* patently clear is the substantial difference in the topics covered and promoted on each side.

{% include term_viz.html %}
