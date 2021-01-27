# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:19:27 2021

@author: amit saini
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""

# Stemming
sentence_stem = nltk.sent_tokenize(paragraph)
stemmer=PorterStemmer()

for i in range(len(sentence_stem)):
    words_stem=nltk.word_tokenize(sentence_stem[i])
    words_stem=[stemmer.stem(word) for word in words_stem if word not in set(stopwords.words('english'))]
    sentence_stem[i]=' '.join(words_stem)
    
    
#Lemmatization
sentence_lemma=nltk.sent_tokenize(paragraph)
lemmatizer=WordNetLemmatizer()

for i in range(len(sentence_lemma)):
    word_lemma=nltk.word_tokenize(sentence_lemma[i])
    word_lemma=[lemmatizer.lemmatize(word) for word in word_lemma if word not in set(stopwords.words('english'))]
    sentence_lemma[i]=' '.join(word_lemma)
	


    