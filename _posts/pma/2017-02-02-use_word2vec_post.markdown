---
layout: post
title:  "Alternate Realities: Generating the word2vec model"
date:   2017-02-14 10:49:00 -0500
categories: [partisan media analysis]
---
## Creating the Model
### [In progress]

This will be a relatively short post where we're going to use `gensim` to create two `word2vec` models; one for right-aligned sources, another for left-aligned sources.

We'll align the left-and right- models using an orthogonal procrustes matrix, following a [method](http://nlp.stanford.edu/projects/histwords/) developed to detect changes in language using word embeddings. This suggestion comes from [Ben Schmidt](http://benschmidt.org/), a professor at Northeastern, who developed the R package `wordVectors` and used it in a phenomenal [omnibus analysis](http://bookworm.benschmidt.org/posts/2015-10-30-rejecting-the-gender-binary.html) of gendered word use in RateMyProfessor reviews.


```python
import cPickle as pickle

from stream import SentenceStream #from the prior post

sql_url = 'postgres://postgres:**PASSWORD**@localhost/articles'

# query yields same text as last time (now includes alignment data)
query = """
        SELECT side, article_text
        FROM articles a
        LEFT JOIN alignment s
        ON split_part(a.post_id, '_', 1) = s.fb_id
        WHERE num_words > 100 and not
              (lower(article_text) like '%%daily caller news foundation%%' and
               base_url != 'dailycaller.com') and not
               lower(article_text) like '%%copyright 20__ the associated press%%'
        ORDER BY side
        """

# pull in the noun n-grams
with open('../intermediate/noun_ngrams_all.pkl', 'rb') as infile:
    noun_ngrams = pickle.load(infile)

# limit to those that appear with some frequency
noun_ngrams = [n for n in noun_ngrams if noun_ngrams[n] > 100]
    
# the n-grams we've located will now be identified in the stream of text
# using the MWETokenizer from nltk
# since I'm using a Google Compute Engine I have RAM to spare
# and will gluttonously store it all in memory
sentences = list(SentenceStream(sqldb=sql_url, query=query,
                                ngrams=noun_ngrams, idcol='side'))
```


```python
from gensim.models import Word2Vec

model_left = Word2Vec([x[1] for x in sentences if x[0] == 'left'],
                      min_count=50, iter=5, sg=1, hs=1, workers=10, size=500)
model_left.save('../intermediate/word2vec_left.pkl')

model_right = Word2Vec([x[1] for x in sentences if x[0] == 'right'],
                       min_count=50, iter=5, sg=1, hs=1, workers=10, size=500)
model_right.save('../intermediate/word2vec_right.pkl')
```

## Some initial explorations

Before we get into anything serious, we can begin by doing a little playing around with our models. One of the most basic things we can do with word embeddings is to find word vectors that are "close" to others, as defined by their cosine similarity. 

I thought it would be interesting to see what terms are considered the most similar to another in one model, but not the other (among some arbitrary number of most similar words). This method is really unsophisticated, but it's fairly intuitive way of looking at what the partisan components of a given term are.


```python
import pandas as pd

def partition_similar_terms(term, model_a, model_b, pool_n=50, n=10,
                            a_lab='left', b_lab='right'):
    """
    For the `pool_n` terms most similar to `term`
    in `model_a` and `model_b`, return
    A - B
    B - A
    A & B
    """
    labs = {'a':a_lab, 'b':b_lab}
    
    terms_a = [t for t, s in model_a.most_similar(term, topn=pool_n)]
    terms_b = [t for t, s in model_b.most_similar(term, topn=pool_n)]
    
    a_not_b = [t for t in terms_a if t not in terms_b][:n]
    b_not_a = [t for t in terms_b if t not in terms_a][:n]
    a_and_b = [t for t in terms_a if t in terms_b][:n]
    
    return pd.DataFrame({'{a} not {b}'.format(**labs): a_not_b,
                         '{b} not {a}'.format(**labs): b_not_a,
                         '{a} and {b}'.format(**labs): a_and_b })
```


```python
# what does it mean to be "shady" on the right, that is different from what it means on the left, and vice versa?
partition_similar_terms('shady', model_left, model_right)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left and right</th>
      <th>left not right</th>
      <th>right not left</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>questionable</td>
      <td>fraudulent</td>
      <td>influence-peddling</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dealings</td>
      <td>business_practices</td>
      <td>scandals</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sleazy</td>
      <td>deceptive</td>
      <td>backroom</td>
    </tr>
    <tr>
      <th>3</th>
      <td>unsavory</td>
      <td>dirty_tricks</td>
      <td>clinton_foundation</td>
    </tr>
    <tr>
      <th>4</th>
      <td>financial_dealings</td>
      <td>clever</td>
      <td>seedy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sketchy</td>
      <td>scam</td>
      <td>connections</td>
    </tr>
    <tr>
      <th>6</th>
      <td>unscrupulous</td>
      <td>frauds</td>
      <td>sordid</td>
    </tr>
    <tr>
      <th>7</th>
      <td>business_dealings</td>
      <td>astroturf</td>
      <td>cozy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>unethical</td>
      <td>disreputable</td>
      <td>foreign_entities</td>
    </tr>
    <tr>
      <th>9</th>
      <td>dodgy</td>
      <td>slimy</td>
      <td>clinton_foundations</td>
    </tr>
  </tbody>
</table>
</div>



Fairly interesting stuff! The right focuses on the Clinton Foundation and its influence, as expected, and the left appears to make more on the unseemliness of (presumably) Trump's business practices. Let's try something else:


```python
partition_similar_terms('radical', model_left, model_right)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left and right</th>
      <th>left not right</th>
      <th>right not left</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>extremist</td>
      <td>far-right</td>
      <td>jihadism</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fringe</td>
      <td>right-wing</td>
      <td>islamic_ideology</td>
    </tr>
    <tr>
      <th>2</th>
      <td>progressive</td>
      <td>anti-imperialist</td>
      <td>hardline</td>
    </tr>
    <tr>
      <th>3</th>
      <td>militant</td>
      <td>liberal</td>
      <td>anti-american</td>
    </tr>
    <tr>
      <th>4</th>
      <td>revolutionary</td>
      <td>leftwing</td>
      <td>islamic</td>
    </tr>
    <tr>
      <th>5</th>
      <td>fundamentalist</td>
      <td>populist</td>
      <td>muslim_brotherhood</td>
    </tr>
    <tr>
      <th>6</th>
      <td>extremists</td>
      <td>socialist</td>
      <td>salafist</td>
    </tr>
    <tr>
      <th>7</th>
      <td>leftist</td>
      <td>conservatism</td>
      <td>extremist_groups</td>
    </tr>
    <tr>
      <th>8</th>
      <td>radicals</td>
      <td>hard-line</td>
      <td>deobandi</td>
    </tr>
    <tr>
      <th>9</th>
      <td>reactionary</td>
      <td>doctrinaire</td>
      <td>islamists</td>
    </tr>
  </tbody>
</table>
</div>



The right uses "radical" to refer to muslim groups almost exclusively. I've collected a few more interesting ones below.


```python
partition_similar_terms('alt-right', model_left, model_right)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left and right</th>
      <th>left not right</th>
      <th>right not left</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>white_nationalists</td>
      <td>breitbart</td>
      <td>nevertrump</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nationalist</td>
      <td>white_nationalism</td>
      <td>bigot</td>
    </tr>
    <tr>
      <th>2</th>
      <td>supremacist</td>
      <td>daily_stormer</td>
      <td>nevertrump_movement</td>
    </tr>
    <tr>
      <th>3</th>
      <td>neo-nazis</td>
      <td>stormfront</td>
      <td>pepe</td>
    </tr>
    <tr>
      <th>4</th>
      <td>neo-nazi</td>
      <td>vdarecom</td>
      <td>movement</td>
    </tr>
    <tr>
      <th>5</th>
      <td>white_supremacists</td>
      <td>breitbart_news</td>
      <td>donald_trumps_supporters</td>
    </tr>
    <tr>
      <th>6</th>
      <td>anti-semitic</td>
      <td>white-supremacist</td>
      <td>feminism</td>
    </tr>
    <tr>
      <th>7</th>
      <td>alt</td>
      <td>bannon</td>
      <td>conservatives</td>
    </tr>
    <tr>
      <th>8</th>
      <td>anti-semites</td>
      <td>steve_bannon</td>
      <td>leftist</td>
    </tr>
    <tr>
      <th>9</th>
      <td>far-right</td>
      <td>jared_taylor</td>
      <td>black_lives_matter_movement</td>
    </tr>
  </tbody>
</table>
</div>




```python
partition_similar_terms(['feminism', 'feminist', 'feminists'], model_left, model_right)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left and right</th>
      <th>left not right</th>
      <th>right not left</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>feminist_movement</td>
      <td>womanhood</td>
      <td>social_justice_warriors</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gloria_steinem</td>
      <td>queer</td>
      <td>radical_feminist</td>
    </tr>
    <tr>
      <th>2</th>
      <td>intersectional</td>
      <td>sex-positive</td>
      <td>leftist</td>
    </tr>
    <tr>
      <th>3</th>
      <td>womens_rights</td>
      <td>motherhood</td>
      <td>radical_feminists</td>
    </tr>
    <tr>
      <th>4</th>
      <td>liberals</td>
      <td>women</td>
      <td>progressivism</td>
    </tr>
    <tr>
      <th>5</th>
      <td>liberal</td>
      <td>intersectionality</td>
      <td>left-wing</td>
    </tr>
    <tr>
      <th>6</th>
      <td>patriarchy</td>
      <td>reproductive_justice</td>
      <td>lefty</td>
    </tr>
    <tr>
      <th>7</th>
      <td>progressives</td>
      <td>traister</td>
      <td>dunham</td>
    </tr>
    <tr>
      <th>8</th>
      <td>progressive</td>
      <td>womens_issues</td>
      <td>lena</td>
    </tr>
    <tr>
      <th>9</th>
      <td>leftists</td>
      <td>freethenipple</td>
      <td>far-left</td>
    </tr>
  </tbody>
</table>
</div>




```python
partition_similar_terms('voter_fraud', model_left, model_right)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left and right</th>
      <th>left not right</th>
      <th>right not left</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>voter_id_laws</td>
      <td>voter_impersonation</td>
      <td>schulkin</td>
    </tr>
    <tr>
      <th>1</th>
      <td>election_fraud</td>
      <td>in-person</td>
      <td>corruption</td>
    </tr>
    <tr>
      <th>2</th>
      <td>voter_suppression</td>
      <td>non-existent</td>
      <td>welfare_fraud</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fraud</td>
      <td>rigged_election</td>
      <td>criminal_activity</td>
    </tr>
    <tr>
      <th>4</th>
      <td>voter_id</td>
      <td>voter-id_laws</td>
      <td>dirty_tricks</td>
    </tr>
    <tr>
      <th>5</th>
      <td>voter_intimidation</td>
      <td>noncitizens</td>
      <td>illegal_activity</td>
    </tr>
    <tr>
      <th>6</th>
      <td>rigging</td>
      <td>voter_id_law</td>
      <td>bribery</td>
    </tr>
    <tr>
      <th>7</th>
      <td>disenfranchisement</td>
      <td>turnout</td>
      <td>elections</td>
    </tr>
    <tr>
      <th>8</th>
      <td>polling_places</td>
      <td>gerrymandering</td>
      <td>absentee_ballot</td>
    </tr>
    <tr>
      <th>9</th>
      <td>non-citizens</td>
      <td>voter</td>
      <td>upcoming_election</td>
    </tr>
  </tbody>
</table>
</div>


