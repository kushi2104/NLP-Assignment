#Importing important libraries
import openpyxl
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import nltk
import string
import numpy as np
import re
import pyphen
#Reading the excel file
df = pd.read_excel('Input.xlsx')
#Extracting URL text from URL
def extract_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('title').get_text()
    article_text = ''
    for paragraph in soup.find_all('p'):
        article_text += paragraph.get_text() + '\n'
    return title, article_text.strip()
for index, row in df.iterrows():
    url = row['URL']
    url_id = row['URL_ID']
    title, article_text = extract_article_text(url)
    filename = f"{url_id}.txt"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(title + '\n\n')
        file.write(article_text)
#Creating new DataFrame
text_df=pd.DataFrame(columns=['URL_ID','Text'])
for index, row in df.iterrows(): 
    url_id= row['URL_ID']
    filename = f"{url_id}.txt"
    with open(filename,'r', encoding='utf-8') as file:
        word=file.read()
    text_df.loc[len(text_df)] = [url_id,word]
nltk.download('punkt')
#Tokenizing the words
text_df['Sentences'] = text_df['Text'].apply(nltk.sent_tokenize)
text_df['Tokenized_text'] = text_df['Text'].apply(nltk.word_tokenize)
#Removing stopwords from the Tokenized text
stopwords_files = [
    'Stopwords_Names.txt',
    'StopWords_Geographic.txt',
    'StopWords_GenericLong.txt',
    'StopWords_Generic.txt',
    'StopWords_DatesandNumbers.txt',
    'StopWords_Currencies.txt',
    'StopWords_Auditor.txt'
]
stopwords = set()
for file_name in stopwords_files:
    with open(file_name, 'r', encoding='latin-1') as file:
        stopwords.update(file.read().splitlines())
def remove_stopwords(tokens):
    return [token for token in tokens if token.lower() not in stopwords]
text_df['Cleaned_text'] = text_df['Tokenized_text'].apply(remove_stopwords)
text_df['Cleaned_text'] = text_df['Cleaned_text'].apply(lambda x: [re.sub(r'[^\w]','',s)for s in x])
text_df['Cleaned_text'] =text_df['Cleaned_text'].apply(lambda x:[element for element in x if element])
#Finding the Positive Score
positive_words=set()
with open('positive-words.txt', 'r', encoding='latin-1') as file:
    for line in file:
         positive_words.add(line.strip())
text_df['Positive_Sentiment_score'] = text_df['Cleaned_text'].apply(lambda tokens: sum(1 for token in tokens if token.lower() in positive_words))
#Finding Negative Score
negative_words=set()
with open('negative-words.txt', 'r', encoding='latin-1') as file:
    for line in file:
        negative_words.add(line.strip())
text_df['Negative_Sentiment_score'] = text_df['Cleaned_text'].apply(lambda tokens: sum(1 for token in tokens if token.lower() in negative_words))
#Finding Polarity Score
text_df['Polarity_score']=(text_df['Positive_Sentiment_score'] -text_df['Negative_Sentiment_score'])/((text_df['Positive_Sentiment_score'] +text_df['Negative_Sentiment_score'])+0.000001)
#Finding Subjective Score
text_df['Subjective_score']=(text_df['Positive_Sentiment_score']+text_df['Negative_Sentiment_score'])/(len(text_df['Cleaned_text'])+0.000001 )
#Analysis of Readability
text_df['Average']=text_df.apply(lambda row:len(row['Tokenized_text'])/len(row['Sentences']),axis=1)
dic = pyphen.Pyphen(lang='en')
def count_syllables(word):
    hyphenated = dic.inserted(word)
    syllables = hyphenated.split('-')
    return len(syllables)
def complex_word(words):
    return [word for word in words if count_syllables(word) > 2]
text_df['Complex_Words'] = text_df['Cleaned_text'].apply(complex_word)
text_df['Percentage']=text_df.apply(lambda row:len(row['Complex_Words'])/len(row['Tokenized_text']),axis=1)
text_df['Fog_Index']=0.4*(text_df['Percentage']+text_df['Average'])
#Average word per sentences
text_df['Average']=text_df.apply(lambda row:len(row['Tokenized_text'])/len(row['Sentences']),axis=1)
#Finding Complex word count
text_df['Complex_word_count']=text_df.apply(lambda row:len(row['Complex_Words']),axis=1)
#Finding word cound
text_df['word_count']=text_df.apply(lambda row:len(row['Tokenized_text']),axis=1)
#Finding Syllable count per word
def count_syllables(word):
    word = word.lower()
    vowels = 'aeiou'
    count = 0
    if word.endswith('es') or word.endswith('ed'):
        word = word[:-2]
    for char in word:
        if char in vowels:
            count += 1
    return count
def count_syllables_in_text(text):
    if isinstance(text,list):
        return[count_syllables(word) for word in text]
    return[]
text_df['Syllables_Count']=text_df['Cleaned_text'].apply(count_syllables_in_text)
#find personal pronoun
personal_pronouns = ['I', 'we', 'my', 'ours', 'us']
def count_personal_pronouns(tokens):
    if not isinstance(tokens, list):
        return 0
    count = 0
    for token in tokens:
        if token.lower() in [pronoun.lower() for pronoun in personal_pronouns]:
            # Ensure 'us' is not mistaken for the country 'US'
            if token.lower() == 'us' and not re.match(r'\bUS\b', token):
                continue
            count += 1
    return count
text_df['Personal_Pronouns_Count'] = text_df['Tokenized_text'].apply(count_personal_pronouns)
#Finding average word length
def sum_characters(words):
    if isinstance(words, list):
        return sum(len(word) for word in words)
    return 0
text_df['Total_Characters'] = text_df['Cleaned_text'].apply(sum_characters)
text_df['Total_Characters'] 
text_df['Average_word_length']=text_df['Total_Characters']/text_df.apply(lambda row:len(row['Cleaned_text']),axis=1)
#Creating Final DataFrame
final_df=pd.DataFrame()
final_df['URL_ID']=df['URL_ID']
final_df['URL']=df['URL']
final_df['POSITIVE SCORE']=text_df['Positive_Sentiment_score'] 
final_df['NEGATIVE SCORE']=text_df['Negative_Sentiment_score']
final_df['POLARITY SCORE']=text_df['Polarity_score']
final_df['SUBJECTIVITY SCORE']=text_df['Subjective_score']
final_df['AVG SENTENCE LENGTH']=text_df['Average']
final_df['PERCENTAGE OF COMPLEX WORDS']=text_df['Percentage']
final_df['FOG INDEX']=text_df['Fog_Index']
final_df['AVG NUMBER OF WORDS PER SENTENCE']=text_df['Average']
final_df['COMPLEX WORD COUNT']=text_df['Complex_word_count']
final_df['WORD COUNT']=text_df['word_count']
final_df['SYLLABLE PER WORD']=text_df['Syllables_Count']
final_df['PERSONAL PRONOUNS']=text_df['Personal_Pronouns_Count'] 
final_df['AVG WORD LENGTH']=text_df['Average_word_length']
final_df.to_excel('Output Data Structure.xlsx',index=False)