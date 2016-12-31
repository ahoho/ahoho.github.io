---
layout: post
title:  "Alternate Realities: Article Processing"
date:   2016-12-04 20:01:00 -0500
categories: partisan media analysis
---

## Text Processing
### [In progress]
In a previous post, we collected roughly 500,000 articles from 80 left- and right-aligned online news sources, going back to July 2015. Here, we'll start to clean and process the text data to enable future analyses.

This is sort of an iterative problem -- I'll likely be revising this process to account for noise that makes it through. This is another post heavy on mechanics and light on findings, but the next one will have some interesting results. 

First, we'll develop a means of accessing text from the database by building an extensible generator. Then, we'll stream sentences to find commonly occuring n-grams in the data, and finally we'll pick those n-grams that we want to treat as single words in the future. 

We'll rely on a combination of [nltk](http://www.nltk.org/), [gensim](https://radimrehurek.com/gensim/index.html), and [scikit-learn's](http://scikit-learn.org/stable/index.html) to tokenize our documents, generate n-grams, do some POS tagging, and place the data in a term-document-matrix.

### Streaming tokenized text

To start, we'll create a generator to return all articles from a query. We'll be using similar functionality throughout this process, so I've opted to make it a class so that we may inherit from it in the future and stream, for e.g., sentences for use in Word2Vec models.


```python
from sqlalchemy import create_engine

class QueryStream(object):
    """ 
    Stream documents from the articles database
    Can be subclassed to stream words or sentences from each document
    """
    def __init__(self, sqldb, query=None, idcol='post_id', 
                 textcol='article_text', chunksize=1000):

        self.sql_engine = create_engine(sqldb)
        self.query = query
        self.chunksize = chunksize
        self.textcol = textcol
        self.idcol = idcol

    def __iter__(self):
        """ Iterate through each row in the query """
        query_results = self.sql_engine.execute(self.query)
        result_set = query_results.fetchmany(self.chunksize)
        while result_set:
            for row in result_set:
                yield row
            result_set = query_results.fetchmany(self.chunksize)
```

We want to create a pipeline for tokenizing documents that will split a document into sentences, then those sentences into words. Here, we'll use nltk for tokenization. 


```python
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import MWETokenizer

import regex as re

class SentenceStream(QueryStream):
    """
    Stream tokenized sentences from a query
    """
    def __init__(self, ngrams=None, *args, **kwargs):
        super(SentenceStream, self).__init__(*args, **kwargs)
        # this will allow us to treat pre-defined n-grams as
        # a single word, e.g., 'hillary_clinton', once
        # we've identified them
        self.mwe = MWETokenizer(ngrams)
        
    def __iter__(self):
        rows = super(SentenceStream, self).__iter__()
        # remove all punctuation, except hyphens
        punct = re.compile("[^A-Za-z0-9\-]")
        
        for doc in rows:
            id = getattr(doc, self.idcol)
            text = getattr(doc, self.textcol)
            
            for sentence in sent_tokenize(text):
                split_sentence = [punct.sub('', word).lower()
                                  for word in word_tokenize(sentence)]
                yield id, self.mwe.tokenize([word for word in split_sentence 
                                             if word.replace('-', '')])
```

### Identifying Collocations

We'll use the article text we've scraped to identify n-grams like 'donald trump'. Should we deem it necessary, we can later identify synonyms like 'senator sanders' and 'bernie sanders' using Word2Vec.


```python
from gensim.models.phrases import Phrases, Phraser
from itertools import imap 

sql_url = 'postgres://postgres:**PASSWORD**@localhost/articles'

full_query = """
             SELECT post_id, article_text
             FROM articles
             WHERE num_words > 100
             """

# Since I'm running this on a Google Compute instance, I can afford
# to load everything in memory as a list. While this isn't strictly necessary,
# I can now avoid pulling from the database multiple times
stream = list(SentenceStream(sqldb=sql_url, query=full_query))
```

To illustrate, here's the 9th sentence in the data.


```python
stream[9]
```




    [(u'144317282271701_1045074145529339',
      [u'a', u'mixture', u'of', u'motives', u'is', u'on', u'display'])]



We must iteratively generate collocations from the sentence stream. The bigram object contains things like 'marco rubio', the trigram might now include 'senator marco rubio'. I imagine this can be improved down the line but for now it's sufficient.


```python
phrase_kwargs = {'threshold': 10,
                  'min_count': 50}

bigram = Phrases(imap(lambda x: x[1], stream), **phrase_kwargs)
trigram = Phrases(bigram[imap(lambda x: x[1], stream)], **phrase_kwargs)
quadgram = Phrases(trigram[imap(lambda x: x[1], stream)], **phrase_kwargs)

phraser = Phraser(quadgram)
phraser.save('../intermediate/phraser_all.pkl')
```

### Trimming n-grams
Many of the n-grams we found have stopwords at either the end or beginning, for instance, 'the supreme court'. We'd like to trim these so that they are individually meaningful components.


```python
from nltk.corpus import stopwords
from collections import defaultdict
from pickle import cPickle

def trim_phrases(phraser):
    """
    Remove stopwords at the start and end of an ngram,
    generate list of unique ngrams in corpus
    """
    stop = stopwords.words('english')
    ngrams = defaultdict(tuple)
    for bigram, score in phraser.phrasegrams.items():
        ngram = bigram[0].split('_') + bigram[1].split('_')
        
        idx = [i for i, v in enumerate(ngram) if v not in stop]  
        ngram = ngram[idx[0]:idx[-1] + 1] if idx else []
        
        
        if len(ngram) > 1:
            ngrams[tuple(ngram)] = score
                        
    return ngrams
  
ngrams = trim_phrases(phraser)

with open('../intermediate/phrasegrams_all.pkl', 'wb') as o:
    pickle.save(ngrams, o)
```

We can now use these phrases, which include n-grams like "united_states" to "bragging_about_sexual_assault", to join together tokens in the data and see which ones are particularly prevalent. Before we get to that, we'll filter down the set to include only noun phrases using POS tagging, which will be used with Word2Vec and topic modeling down the road.

### POS Tagging
Of the n-grams collected above, we're largely interested in noun-phrases like 'aborted_baby_parts' (appearing 1,062 times: uh, ok) rather than adjective or verb phrases like 'radical_islamic'. To subset to only nouns, we'll use [Part-of-Speech tagging](http://www.nltk.org/book/ch05.html) to see how these n-grams are employed in the data.


```python
import cPickle as pickle
sql_url = 'postgres://postgres:**PASSWORD**@localhost/articles'
full_query = """
             SELECT post_id, article_text
             FROM articles
             WHERE num_words > 100
             """
with open('../intermediate/phrasegrams_all.pkl', 'rb') as i:
    ngrams = pickle.load(i)
```


```python
from nltk import pos_tag

class POSSentenceStream(SentenceStream):
    """
    Assign parts of speech to n-grams
    """
    def __iter__(self):
        sentences = super(POSSentenceStream, self).__iter__()
        for id, sentence in sentences:
            for word, pos in pos_tag(sentence):
                if '_' in word:
                    yield word, pos
            
pos_words = POSSentenceStream(sqldb=sql_url, query=full_query, ngrams=ngrams)
```

To take a look at what I'm referring to, see some examples below. Just look: already we have the hugely loaded term *conversion_therapy*!


```python
pos_words[:10]
```




    [(u'hasnt_stopped', 'VBD'),
     (u'republican_politicians', 'NNS'),
     (u'radio_show', 'NN'),
     (u'rafael_cruz', 'VBD'),
     (u'ted_cruz', 'NN'),
     (u'barack_obama', 'NN'),
     (u'conversion_therapy', 'NN'),
     (u'reminds_us', 'NN'),
     (u'conversion_therapy', 'NN'),
     (u'sharp_contrast', 'NN')]



We might expect that certain n-grams, like 'presidential_candidate', could be considered both a noun or adjective phrase. As a result, we'll determine the POS distribution for each phrase, selecting only those that are nouns over 75% of the time.


```python
from collections import defaultdict
from collections import Counter
import numpy as np

def count_ngram_occurances(pos_stream):
    """
    Tally up the different POS associated with each n-gram
    """
    pos_counts = {word: Counter() for word, pos in pos_stream}
    
    for word, pos in pos_stream:
        pos_counts[word].update([pos])
        
    return pos_counts

def identify_np_ngrams(pos_counts):
    """
    Determine the n-grams that are most often employed as noun phrases
    """     
    np_ngrams = defaultdict(str)
    
    for word, counts in pos_counts.items():
        split_word = tuple(word.split('_'))
                
        word_count = sum(counts.values())        
        noun_count = sum([v for k, v in counts.items() if 'NN' in k])
        
        # is it usually used as a noun?
        if np.true_divide(noun_count, word_count) > 0.75:
            np_ngrams[split_word] = word_count
            
    return np_ngrams
            
pos_counts = count_ngram_occurances(pos_words)  
noun_ngrams = identify_np_ngrams(pos_counts)
```

Already, we're seeing some moderately interesting findings in the data. For instance, *rigged election by hillary clinton* appears 113 times in 57 sources, *congress must investigate planned parenthood* 58 times over 16 sources. For posterity, I've saved these n-grams and their counts to github.


```python
# save n-grams appearing over 25 times in the data
ngram_data = pd.DataFrame.from_records(pos_counts)\
                         .transpose()
    
ngram_data['total'] = ngram_data.sum(axis=1)
ngram_data.loc[ngram_data.total >= 25]\
          .to_csv('../output/ngram_pos_counts.csv', index_label='n-gram')
```


```python
with open('../intermediate/noun_ngrams_all.pkl', 'wb') as o:
    pickle.dump(noun_ngrams, o)
```

With that, we're done with this initial stage of processing. I imagine that I'll be expanding this post as time goes on to accomodate other issues that arise (already, I'm aware that there's a lot of noise in the data, to the tune of "follow breitbart on Twitter" or "photo credit AP", but I'd rather get some results first).
