# Standard library imports
import codecs
import json
import os
import re
import time
import warnings

# this is to avoid warning messages from the transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.chdir(os.path.dirname(os.path.realpath(__file__)))
warnings.filterwarnings("ignore")

# Third-party imports
import base64
import markdown
import numpy as np
import pandas as pd
from tqdm import tqdm
import translators as ts

# Vizualizations
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt 
import seaborn as sns
from wordcloud import WordCloud

# Topic modelling and NLP
import gensim
import spacy
import nltk

from unidecode import unidecode
from pyLDAvis.gensim_models import *
from dms2dec.dms_convert import dms2dec

from gensim import corpora, models
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaModel, LdaMulticore, LsiModel, HdpModel

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, FreqDist, Tree
from transformers import AutoTokenizer, BertForSequenceClassification, pipeline

# Local application imports
from mgp import MovieGroupProcess


def remove_unecessary_characters(text):
    """
    Remove unnecessary characters from the text such as quotation marks, brackets, parentheses, and extra whitespace.
    
    Args:
        text (str): The input text to clean.
        
    Returns:
        str: The cleaned text with unnecessary characters removed.
    """
    # Remove quotation marks, brackets, and parentheses
    text = re.sub(r"[“”@\(\)\[\]\{\}]+", "", text)
    
    # Remove string of multiple whitespace characters and replace with a single space
    text = re.sub(r"  ", " ", text)
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces into one
    
    # Remove remaining quotation marks
    text = re.sub(r"\"", "", text)

    return text


def get_doc_type(text):
    """
    Extracts the document type from the input text. Excludes common irrelevant terms.

    Args:
        text (str): The input text to extract the document type from.
        
    Returns:
        str: The extracted document type, or 'OUTRO' if not found.
    """
    remove_list = ['AHU', 'ANT', 'XIX', 'XVII', 'XVIII', 'ACL', 'ALAGOA', 'CEARA', 'DOC', 'CEARA', 'CEARÁ',
                   'INF', 'POS', 'CA', 'CU', 'CX', 'PP']
    
    try:
        # Extract words in uppercase and not in the remove list
        doc_type = [x for x in re.findall(r"[A-ZÃÁÀÂÇÉÊÍÕÓÔÚÜ]{2,}", text) if x not in remove_list][0]

        # Adjust the document type if it ends with certain characters
        if doc_type[-1] == 'S':
            if doc_type[-2] not in ['E', 'I', 'N']:
                doc_type = doc_type[:-1]
            elif doc_type[-3:] == "ÕES":
                doc_type = doc_type[:-3] + "ÃO"
            elif doc_type[-2:] == "NS":
                doc_type = doc_type[:-2] + "M"
            elif doc_type[-2:] == "IS":
                doc_type = doc_type[:-2] + "L"
            else:
                doc_type = doc_type[:-1]
    except:
        # Return "OUTRO" if no document type is found
        doc_type = "OUTRO"

    return unidecode(doc_type)


def get_doc_year(text):
    """
    Extracts the year from the text, removing false year information.

    Args:
        text (str): The input text to extract the year from.
        
    Returns:
        int or None: The extracted year if found, otherwise None.
    """
    # Remove false year information (e.g., '1xxx-' or '1xxx.')
    text = re.sub(r'1\d{3}\-', '', text, 1)
    text = re.sub(r'1\d{3}\.', '', text, 1)
    
    # Fix the year if multiple years are found in sequence
    text = re.sub(r'(\d{4})\s*(\d{4})', r'\2', text, 1)
    text = re.sub(r'(\d{4})\s*(\[)', r'\2', text, 1)  # Handle case like "1897 [ant. 1662"
    
    # Find all year-like patterns and filter out any that are not valid
    re_year = [x.group(1) for x in re.finditer(r"(1\d{3})[^a-zA-Z\d]", text)]
    re_year = [int(y) for y in re_year if 1515 < int(y) < 1900]  # Consider valid years between 1515 and 1900

    return re_year[0] if re_year else None


def remove_irrelenvant_parts(text, max_len=None):
    """
    Removes irrelevant parts of the document, such as document type and AHU references, and trims the text to a specified length.
    
    Args:
        text (str): The input text to clean.
        max_len (int, optional): The maximum length of the output text. If None, the entire text is returned.
        
    Returns:
        str: The cleaned and possibly truncated text.
    """
    # Remove unnecessary characters from the text
    text = remove_unecessary_characters(text)

    try:
        # Extract document type and AHU reference and remove irrelevant parts
        doc_type = re.search(r"[A-ZÃÁÀÂÇÉÊÍÕÓÔÚÜ ]{4,}", text)
        doc_ahu = re.search(r"AHU-", text)
        
        # Trim the text between document type and AHU reference
        text = text[doc_type.end():doc_ahu.start()]
    except:
        pass

    # Return the cleaned text, possibly truncated to max_len
    return text[:max_len] if max_len else text


def sent_to_words(sentences):
    """
    Converts a list of sentences into a list of words by using simple_preprocess.
    
    Args:
        sentences (list): A list of sentences to convert into words.
        
    Yields:
        list: A list of words for each sentence after preprocessing.
    """
    for sentence in sentences:
        yield simple_preprocess(str(sentence), deacc=False)


def pre_processing_regex(texts):
    """
    Preprocesses the input texts by removing document codes and specific patterns using regular expressions.
    
    Args:
        texts (list): A list of strings (sentences or documents) to preprocess.
        
    Returns:
        list: A list of preprocessed texts with document codes removed.
    """
    # Remove patterns with digits surrounded by non-whitespace characters
    texts = [re.sub(r"(\S*)(\d+)(\S*)", "", sent) for sent in texts]
    
    # Remove underscores between alphanumeric characters
    texts = [re.sub(r"([a-zA-Z0-9]*)_([a-zA-Z0-9]*)", "", sent) for sent in texts]
    
    return texts


def make_n_grams(texts, min_count=5):
    """
    Generates bigrams (or n-grams) from a list of tokenized texts.
    
    Args:
        texts (list): A list of tokenized sentences or documents.
        min_count (int): The minimum frequency count for a phrase to be considered (default is 5).
        
    Returns:
        list: A list of bigrams or n-grams for each document.
    """
    # Create a bigram model using gensim
    bigram = gensim.models.Phrases(texts, min_count=min_count, threshold=100)  # higher threshold = fewer phrases.
    
    # Apply the bigram model to the text to create bigrams
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    bigrams_text = [bigram_mod[doc] for doc in tqdm(texts, desc="Making n-grams", leave=False)]

    return bigrams_text


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"], model="pt_core_news_lg"):
    """
    Lemmatizes the input texts by using spaCy's lemmatizer. Filters tokens based on specified POS tags.
    
    Args:
        texts (list): A list of tokenized sentences to lemmatize.
        allowed_postags (list): A list of allowed part-of-speech tags for lemmatization (default includes NOUN, ADJ, VERB, ADV).
        model (str): The spaCy language model to use (default is "pt_core_news_lg").
        
    Returns:
        list: A list of lemmatized words from each sentence.
    """
    texts_out = []
    nlp = spacy.load(model, disable=["parser", "ner"])  # Load spaCy model without parser and NER

    for sent in tqdm(texts, desc="Lemmatizing", leave=False):
        # Process each sentence through the spaCy pipeline
        doc = nlp(" ".join(sent)) 
        
        # Filter tokens based on allowed POS tags
        if allowed_postags:
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        else:
            texts_out.append([token.lemma_ for token in doc])

    return texts_out


def remove_entities(texts, forbidden_entities=["LOC", "MALE", "FEM"], model="other_files/model-best-mape"):
    """
    Removes specified named entities (such as locations or gender labels) from the texts using spaCy.
    
    Args:
        texts (list): A list of tokenized sentences to process.
        forbidden_entities (list): A list of entity labels to remove from the text (e.g., "LOC", "MALE", "FEM").
        model (str): The spaCy model to use for entity recognition (default is "model-best-mape").
        
    Returns:
        list: A list of sentences with forbidden entities removed.
    """
    results = []
    nlp = spacy.load(model)  # Load the custom spaCy model for named entity recognition

    for sent in tqdm(texts, desc="Filtering entities", leave=False):
        # Process each sentence to extract named entities
        doc = nlp(" ".join(sent)) 
        
        # Identify and remove forbidden entities
        to_remove = [token.text for token in doc.ents if token.label_ in forbidden_entities]
        
        # Append sentence with forbidden entities removed
        results.append([token for token in sent if token not in " ".join(to_remove)])
        
    return results


def remove_stopwords(texts, additional_words=[]):
    """
    Removes stopwords from the input texts. Stopwords are removed from both a pre-defined list 
    (loaded from a file) and any additional words provided as input.
    
    Args:
        texts (list): A list of tokenized sentences or documents.
        additional_words (list, optional): A list of additional stopwords to remove (default is an empty list).
        
    Returns:
        list: A list of tokenized sentences with stopwords removed.
    """
    # Load the pre-defined stopwords from a file
    with open("data/stopwords.txt", encoding="utf-8") as f:
        pt_stopwords = f.readlines()

    # Clean up the stopwords list (removing newline characters)
    pt_stopwords = [re.sub("[ ]*\n", "", sent) for sent in tqdm(pt_stopwords, desc="Removing stopwords", leave=False)]

    # Add any additional stopwords provided by the user
    [pt_stopwords.append(w) for w in additional_words]

    # Combine pre-defined and additional stopwords into one set
    remove_list = gensim.parsing.preprocessing.STOPWORDS.union(pt_stopwords)

    # Remove stopwords from the texts
    return [[word for word in simple_preprocess(str(doc)) if word not in remove_list] for doc in tqdm(texts, leave=False)]


def process_topic_modelling(root_data, col='full_text'):
    """
    Processes the given DataFrame for topic modeling by applying various text preprocessing steps:
    regex cleaning, tokenization, n-gram generation, lemmatization, entity removal, and stopword removal.
    
    Args:
        root_data (pd.DataFrame): The input DataFrame containing the text data.
        col (str): The column name containing the text to process (default is 'full_text').
    """
    # Pre-process the text using regex cleaning
    root_data[col + "_processed"] = pre_processing_regex(root_data[col])

    # Tokenize the processed text and generate n-grams
    tokens_ = list(sent_to_words(root_data[col + "_processed"]))
    tokens_ = make_n_grams(tokens_)

    # Lemmatize the tokens and remove entities like locations, male/female terms, etc.
    lemma_text = lemmatization(tokens_, allowed_postags=["NOUN", "ADJ", "VERB", "X"])
    lemma_text = remove_entities(lemma_text, forbidden_entities=["LOC", "MALE", "FEM", "OCC"])
    lemma_text = remove_stopwords(lemma_text)

    # Join the lemmatized tokens back into a single string
    root_data[col + "_processed"] = [" ".join(w) for w in lemma_text]


def filter_df_composite_keywords(df, keywords, group_name=None):
    """
    Filters the DataFrame based on the presence of any keyword from a list. Creates a new column for each keyword
    indicating whether it appears in the 'full_text' column. Optionally groups keywords into a single column.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the text data.
        keywords (list): A list of keywords to search for in the text.
        group_name (str, optional): The name of the group to create if keywords are grouped together (default is None).
        
    Returns:
        pd.DataFrame: The filtered DataFrame with new columns indicating keyword presence.
    """
    # Clean up the 'full_text' column by removing unnecessary characters
    df['full_text'] = df.full_text.apply(remove_unecessary_characters)
    
    # Normalize the keywords to lowercase and remove accents
    keywords = [unidecode(x.lower()) for x in keywords]
    
    # If no group name is provided, create individual columns for each keyword
    if not group_name:
        for term in tqdm(keywords, leave=False):
            df[term] = df.full_text.apply(lambda x: term in unidecode(x.lower()))
        
        # Create a column indicating if any keyword is present
        df["any_term"] = df.full_text.apply(lambda x: any(item for item in keywords if item in unidecode(x.lower())))
    else:
        # Create a single column for the group of keywords
        df[group_name] = df.full_text.apply(lambda x: any(item for item in keywords if item in unidecode(x.lower())))
    
    return df


def filter_df_keywords(df, keywords, group_name=None):
    """
    Filters the DataFrame based on the presence of any keyword from a list. This function is similar to `filter_df_composite_keywords`,
    but it processes keywords in a slightly different manner (splitting text before checking for matches).
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the text data.
        keywords (list): A list of keywords to search for in the text.
        group_name (str, optional): The name of the group to create if keywords are grouped together (default is None).
        
    Returns:
        pd.DataFrame: The filtered DataFrame with new columns indicating keyword presence.
    """
    # Clean up the 'full_text' column by removing unnecessary characters
    df['full_text'] = df.full_text.apply(remove_unecessary_characters)
    
    # Normalize the keywords to lowercase and remove accents
    keywords = [unidecode(x.lower()) for x in keywords]
    
    # If no group name is provided, create individual columns for each keyword
    if not group_name:
        for term in tqdm(keywords, leave=False):
            df[term] = df.full_text.apply(lambda x: term in unidecode(x.lower()).split())
        
        # Create a column indicating if any keyword is present
        df["any_term"] = df.full_text.apply(lambda x: any(item for item in keywords if item in unidecode(x.lower()).split()))
    else:
        # Create a single column for the group of keywords
        df[group_name] = df.full_text.apply(lambda x: any(item for item in keywords if item in unidecode(x.lower()).split())) 
    
    return df


def filter_doc_type(row, group):
    """
    Filters the DataFrame rows based on the document type and the presence of specified inclusion and exclusion keywords.
    
    Args:
        row (pd.Series): A single row of the DataFrame containing the text and doc_type.
        group (dict): A dictionary containing inclusion and exclusion criteria for each document type.
        
    Returns:
        bool: True if the row meets the criteria, False otherwise.
    """
    for t, opts in group.items():
        text = row["full_text"].lower()
        
        # Check if the document type matches and if the inclusion/exclusion criteria are met
        if row["doc_type"] == t and [v for v in opts["incl"] if v in text] and not [v for v in opts["excl"] if v in text]:
            return True
    else:
        return False


def filter_df_keywords_lemm(df, keywords, group_name=None, col='full_text'):
    """
    Filters a DataFrame to check if any of the keywords are present in the lemmatized text.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing textual data.
        keywords (list): A list of keywords to search for in the text.
        group_name (str, optional): The name of the group for the resulting column (if grouping).
        col (str): The column name in the DataFrame containing the text to search through.

    Returns:
        pd.DataFrame: The modified DataFrame with columns indicating whether each keyword is present.
    """
    # Preprocess the text (remove irrelevant parts, tokenize, and lemmatize)
    temp = df[col].apply(lambda t: remove_irrelenvant_parts(t, 500)) 
    temp = list(sent_to_words(temp)) 
    tokens_ = make_n_grams(temp)
    lemma_text = lemmatization(tokens_, allowed_postags=None) 
    temp = [" ".join(w) for w in lemma_text]

    # Check if the keywords are present in the text
    if not group_name:
        for term in tqdm(keywords, leave=False):
            df[term] = [term in unidecode(x.lower()) for x in temp]
        
        # Create a column to check if any keyword exists in the text
        df["any_term"] = [any([item for item in keywords if item in unidecode(x.lower())]) for x in temp]
    else:
        df[group_name] = [any([item for item in keywords if item in unidecode(x.lower())]) for x in temp]
    
    return df


def train_gsdmm(n_topics, lemma_text):
    """
    Trains a GSDMM (Generative Stochastic Dirichlet Allocation for short texts) model.
    
    Args:
        n_topics (int): The number of topics to be modeled.
        lemma_text (list): The preprocessed and lemmatized text data.
    
    Returns:
        tuple: A tuple containing the trained GSDMM model and the fitted model.
    """
    np.random.seed(0)
    mgp = MovieGroupProcess(K=n_topics, alpha=0.01, beta=0.01, n_iters=50)

    # Prepare the vocabulary
    vocab = set(x for lemma in lemma_text for x in lemma)
    n_terms = len(vocab)

    # Fit the model
    model = mgp.fit(lemma_text, n_terms)

    return mgp, model


def get_topics_lists(model, top_clusters, n_words=100):
    """
    Extracts the top words for each topic from the trained GSDMM model.
    
    Args:
        model (GSDMM): The trained GSDMM model.
        top_clusters (list): The list of top clusters (topics).
        n_words (int): The number of top words to extract for each topic.
    
    Returns:
        list: A list of lists, where each inner list contains the top words of a topic.
    """
    topics = []
    
    # Iterate over the top clusters (topics)
    for cluster in top_clusters:
        # Sort the words in the cluster by frequency
        sorted_dict = sorted(model.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:n_words]
        
        # Extract the words
        topic = [k for k, v in sorted_dict]
        
        # Add the topic to the list
        topics.append(topic)
    
    return topics


def plot_coherence(start, limit, step, coherence_values):
    """
    Plots the coherence values to visualize the optimal number of topics.
    
    Args:
        start (int): The starting number of topics.
        limit (int): The upper limit for the number of topics.
        step (int): The step size for the number of topics.
        coherence_values (list): The list of coherence values for each number of topics.
    
    Returns:
        int: The index of the best number of topics.
    """
    sns.set(font_scale=2)
    sns.set_style('white')

    # Find the best coherence value and its corresponding number of topics
    best_index = coherence_values.index(max(coherence_values))
    x_axix, best_val = list(range(start, limit, step)), coherence_values[best_index]
    coherence_df = pd.DataFrame(zip(x_axix, coherence_values), columns=['num_topics', 'coherence'])

    # Plot the coherence values
    ax = sns.lineplot(data=coherence_df, x='num_topics', y='coherence', lw=3, color="#d3d3d3", zorder=0)
    line = plt.axvline(x=x_axix[best_index], linestyle='dashed', lw=2, color='#5c677d', label='Mean')
    
    # Annotate the best value
    el = ax.scatter(x_axix[best_index], best_val, c='#89c2d9', s=100, zorder=1)
    ax.annotate(f'Best Value: {best_val:0.2f}\nBest N: {x_axix[best_index]}', size=14, 
                xy=(x_axix[best_index] * 1.01, best_val * 0.984), xycoords='data', 
                xytext=(10, 30), textcoords='offset points', va="center", 
                bbox=dict(boxstyle="round", fc="#89c2d9", ec="none"))

    plt.xlabel("Number of topics")
    plt.ylabel("Coherence")
    plt.show()


def calc_coherence_values(dictionary, corpus, texts, gensim_model, limit=12, start=1, step=1):
    """
    Calculates coherence values for a range of topic numbers using a Gensim model.
    
    Args:
        dictionary (gensim.corpora.Dictionary): The dictionary for the corpus.
        corpus (list): The corpus of documents.
        texts (list): The list of documents in text form.
        gensim_model (Gensim model class): The Gensim model (e.g., LDA, LSI) to train.
        limit (int): The upper limit of topics to test.
        start (int): The starting number of topics.
        step (int): The step to increment the number of topics.
    
    Returns:
        tuple: A tuple containing a list of trained models and their corresponding coherence values.
    """
    coherence_values = []
    model_list = []

    # Train models for a range of topic numbers
    for num_topics in tqdm(range(start, limit, step), leave=False):
        if gensim_model is LsiModel:
            model = gensim_model(corpus=corpus, id2word=dictionary, num_topics=num_topics, chunksize=200)
        else:
            model = gensim_model(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, 
                                 chunksize=200, passes=100, per_word_topics=True)
        
        # Append the model to the list
        model_list.append(model)
        
        # Calculate coherence score
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence="c_v")
        coherence_values.append(coherencemodel.get_coherence())
    
    return model_list, coherence_values


def calc_coherence_values_gsdmm(dictionary, corpus, texts, lemma_text, limit=12, start=1, step=1):
    """
    Calculates coherence values for a range of topic numbers using the GSDMM model.

    Args:
        dictionary (gensim.corpora.Dictionary): The dictionary for the corpus.
        corpus (list): The corpus of documents.
        texts (list): The list of documents in text form.
        lemma_text (list): The lemmatized text list.
        limit (int): The upper limit of topics to test.
        start (int): The starting number of topics.
        step (int): The step to increment the number of topics.

    Returns:
        model_list (list): A list of GSDMM models.
        coherence_values (list): A list of coherence values for each topic number.
    """
    coherence_values = []
    model_list = []
    
    for num_topics in tqdm(range(start, limit, step), leave=False):
        mgp, model = train_gsdmm(num_topics, texts) 
        model_list.append(mgp)
        
        # Extract topics from the model
        topics = get_topics_lists(mgp, list(range(num_topics)), 100) 
        
        # Calculate coherence score
        coherencemodel = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, corpus=corpus, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    
    return model_list, coherence_values


def create_topics_dataframe_gsdmm(data, mgp, threshold, topic_dict):
    """
    Creates a DataFrame with topics, text, and metadata based on GSDMM model results.

    Args:
        data (pd.DataFrame): The data frame containing the documents.
        mgp (GSDMM model): The trained GSDMM model.
        threshold (float): The probability threshold to assign a topic to a document.
        topic_dict (dict): A dictionary mapping topic numbers to topic labels.

    Returns:
        pd.DataFrame: A DataFrame containing the assigned topics and additional document metadata.
    """
    result = pd.DataFrame(columns=["id_document", "Text", "Topic", "Len", "Lemma_text", "Source_file"])
    data.lemma_text = data.lemma_text.fillna(' ')
    
    for i, row in tqdm(data.iterrows(), leave=False, total=len(data)):
        result.at[i, "Text"] = row["full_text"]
        
        # Get the best topic label for the document
        prob = mgp.choose_best_label(row["lemma_text"].split())
        
        # Assign topic if the probability is above the threshold
        if prob[1] >= threshold:
            result.at[i, "Topic"] = topic_dict[prob[0]]
        else:
            result.at[i, "Topic"] = "Other"
        
        # Fill additional metadata
        result.at[i, "Len"] = len(row["full_text"])
        result.at[i, "Lemma_text"] = row["lemma_text"]
        result.at[i, "Source_file"] = row["source_file"]
        result.at[i, "id_document"] = row["id_document"]
        
    return result


def atof(text):
    """
    Converts a string to a float if possible, otherwise returns the original string.

    Args:
        text (str): The text to convert.

    Returns:
        float or str: The converted float or the original text.
    """
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    """
    Natural sorting for alphanumeric text. It sorts numbers in a human-readable order.

    Args:
        text (str): The text to be sorted.

    Returns:
        list: A list of numeric and non-numeric components of the text.
    """
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def df_to_dict(topics_df):
    """
    Converts a DataFrame of topics into a dictionary of word frequency distributions.

    Args:
        topics_df (pd.DataFrame): A DataFrame containing topic and lemmatized text data.

    Returns:
        dict: A dictionary where keys are topics and values are word frequency distributions.
    """
    res_dict = {}
    
    for t in tqdm(topics_df.Topic.unique(), leave=False):
        f_df = topics_df[topics_df.Topic == t]
        fdist1 = FreqDist(" ".join(f_df.Lemma_text.values).split())
        res_dict[t] = dict(fdist1.most_common(100))
    
    return res_dict


def random_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    """
    Generates a random color for word cloud visualization.

    Args:
        word (str): The word for which the color is being generated (not used).
        font_size (int): The font size of the word (not used).
        position (tuple): The position of the word (not used).
        orientation (str): The orientation of the word (not used).
        font_path (str): The font path (not used).
        random_state (np.random.RandomState): The random state for generating the color.

    Returns:
        str: The generated color in HSL format.
    """
    h, s, l = 0, 0, random_state.randint(20, 60)
    return "hsl({}, {}%, {}%)".format(h, s, l)


def show_wordcloud_dict_grid(dict_, mask=None, topics_order=None, topics_titles=None):
    """
    Displays a grid of word clouds for each topic in the dictionary.

    Args:
        dict_ (dict): A dictionary where keys are topics and values are word frequency distributions.
        mask (str, optional): The image mask to shape the word clouds (default is None).
        topics_order (list, optional): The order in which to display topics (default is None).
        topics_titles (dict, optional): A dictionary of titles for the topics (default is None).
    """
    count, num_topics = 0, len(dict_)
    cols = int(np.ceil(num_topics / 2))
    
    dict_ = {k: v for k, v in sorted(dict_.items(), key=lambda item: natural_keys(item[0]))}
    
    # Create the mask image for the word clouds
    mask_img = np.array(Image.open(mask)) if mask else None
    
    # Set up the subplots for displaying the word clouds
    fig, axs = plt.subplots(cols, 2, figsize=(12, 3 * num_topics))
    
    for n, freq in tqdm(dict_.items(), leave=False):
        ax_ = axs[count] if num_topics < 4 else axs[count // 2][count % 2]
        
        # Set topic title
        if topics_titles:
            n = topics_titles[n]
        
        title = f"Most used words in {n}"
        
        # Generate the word cloud
        wordcloud = WordCloud(mask=mask_img, width=500, height=500, 
                              background_color="white", font_path='arial', 
                              min_font_size=10, collocations=False, 
                              color_func=random_color_func).generate_from_frequencies(freq)
        
        # Display the word cloud
        ax_.imshow(wordcloud, interpolation='bilinear')
        ax_.set_title(title, fontsize=20)
        
        count += 1
    
    # Turn off axes for all subplots
    [ax_.axis("off") for ax_ in axs.ravel()]
    
    plt.show()
    plt.close()


def format_topics_sentences(model_, corpus, text, data, topic_dict):
    """
    Formats the topics and sentences into a DataFrame. The dominant topic is assigned to each sentence 
    based on the highest probability topic in the model's output.

    Args:
        model_ (LsiModel or other model): The topic model (e.g., LSI model) used to assign topics to sentences.
        corpus (list): The corpus of documents used in the topic modeling.
        text (list): The original text documents corresponding to the corpus.
        data (list): Original content data for reference.
        topic_dict (dict): A dictionary mapping topic numbers to topic names.

    Returns:
        pd.DataFrame: A DataFrame with the following columns: "Text", "Topic", "Len", and "Lemma_text".
    """
    sent_topics_df = []
    
    # If the model is LSI (Latent Semantic Indexing), process accordingly
    if isinstance(model_, LsiModel):
        for n, row in tqdm(enumerate(model_[corpus]), leave=False):
            row = sorted(row, key=lambda x: x[1], reverse=True)
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # Dominant topic
                    sent_topics_df.append([text[n], topic_dict[topic_num], len(text[n].split())])
                else:
                    break
    else:
        # If it's any other model type (e.g., LDA)
        for n, row in tqdm(enumerate(model_[corpus]), leave=False):
            row = row[0]
            row = sorted(row, key=lambda x: x[1], reverse=True)
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # Dominant topic
                    sent_topics_df.append([text[n], topic_dict[topic_num], len(text[n].split())])
                else:
                    break

    # Convert list to DataFrame
    sent_topics_df = pd.DataFrame(sent_topics_df)
    
    # Add original text to the output
    contents = pd.Series(data)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df.reset_index(inplace=True)
    
    # Drop unnecessary index column and rename columns
    sent_topics_df.drop(columns=sent_topics_df.columns[0], axis=1, inplace=True)
    sent_topics_df.columns = ["Text", "Topic", "Len", "Lemma_text"]

    return sent_topics_df


def show_wordcloud_dict(dict_, mask=None, topics_order=None, prefix=""):
    """
    Generates and displays a word cloud for each topic in the dictionary and saves the images in SVG format.

    Args:
        dict_ (dict): A dictionary where keys are topics and values are frequency distributions of words.
        mask (optional): Mask image for word cloud shape (default is None).
        topics_order (optional): Order in which to display the topics (default is None).
        prefix (str, optional): Prefix for the saved file names.
    """
    for n, freq in dict_.items():
        if len(freq):
            title = f"Most used words in {n}"
            try:
                # Generate word cloud from the frequency dictionary
                wordcloud = WordCloud(
                    width=500, height=500, background_color="white", 
                    min_font_size=10, collocations=False, color_func=random_color_func
                ).generate_from_frequencies(freq)
                
                # Display word cloud
                plt.imshow(wordcloud, interpolation='bilinear')
            except:
                pass
            
            # Customize plot appearance
            plt.axis("off")
            plt.title(title, fontsize=20)
    
            # Save the plot to SVG format
            plt.savefig(f'figures/{prefix} {title}.svg', dpi=700, format="svg")
            plt.show()
            plt.close()


def translate_topics(top_words):
    """
    Translates the top words for each topic to English using an online translation service.

    Args:
        top_words (dict or pd.DataFrame): A dictionary or DataFrame containing words for each topic.

    Returns:
        dict or pd.DataFrame: Translated words for each topic.
    """
    if isinstance(top_words, dict):
        trans_words = {}
        for t, words in tqdm(top_words.items(), leave=False):
            try:
                temp = {}
                for k, v in words.items():
                    if k in ["baía", "pará"]:  # Skip specific words from translation
                        temp[k] = v
                    else:
                        temp[ts.translate_text(k, to_language='en', from_language='pt').lower()] = v
                trans_words[t] = temp
            except:
                pass
    else:
        trans_words = pd.DataFrame(columns=list(top_words.columns))

        for index, row in tqdm(top_words.iterrows(), leave=False):
            trans_words.at[index, "Text"] = row["Text"]
            trans_words.at[index, "Topic"] = row["Topic"]
            trans_words.at[index, "Len"] = row["Len"]
            trans_ = [ts.translate_text(k).lower() for k in row["Lemma_text"].split()]
            trans_words.at[index, "Lemma_text"] = " ".join(trans_)

    return trans_words


def plot_word_frequency(word_freq_topics_trans, prefix, top_n=10):
    """
    Plots and saves a bar chart of the top N most frequent words for each topic.
    
    Args:
        word_freq_topics_trans (dict): A dictionary of word frequencies for each topic.
        prefix (str): Prefix for the saved file names.
        top_n (int, optional): The number of top words to display (default is 10).
    """
    for t, word_frequencies in word_freq_topics_trans.items():
        if len(word_frequencies):
            # Sort words by frequency in descending order and select top N
            sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
            sorted_words = sorted_words[:top_n]
            
            # Extract words and frequencies
            words, frequencies = zip(*sorted_words)
            
            # Create a bar plot
            plt.figure(figsize=(8, 5))
            plt.bar(words, frequencies, color='#219ebc')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Frequency')
            plt.title(f'Word Frequency Distribution for Topic {t}')
            
            # Save the plot to SVG format
            plt.savefig(f'figures/word_freq_{prefix}_{t}.svg', dpi=700, format="svg")
            plt.show()
            plt.close()
        