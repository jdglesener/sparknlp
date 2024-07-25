# Introduction
Fyodor Dostoevksy is a very popular russian author from the mid 19th century. 
# Methodology
### Data
For our data we used the online book texts from project Gutenberg. \
https://www.gutenberg.org/cache/epub/2554/pg2554.txt \
https://www.gutenberg.org/cache/epub/28054/pg28054.txt \
https://www.gutenberg.org/cache/epub/600/pg600.txt
### Libraries
We used the Pandas library for Data Management. For the NLP, we used pyspark as well as the punkt tokenizer models from the nltk library.
### Code
With this chunk we start up the spark nlp object as well as import all the necessary libraries. AFter that we create a books directory and curl the text files into the directory
```{python}
!wget http://setup.johnsnowlabs.com/colab.sh -O - | bash
import sparknlp
spark = sparknlp.start()
import os
import pandas as pd
import nltk
nltk.download('punkt')  # Download the Punkt tokenizer models
from nltk.tokenize import sent_tokenize

from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline("dependency_parse")

!mkdir books
!curl "https://www.gutenberg.org/cache/epub/600/pg600.txt" -o books/underground.txt
!curl "https://www.gutenberg.org/cache/epub/28054/pg28054.txt" -o books/brothers.txt
!curl "https://www.gutenberg.org/cache/epub/2554/pg2554.txt" -o books/candp.txt
```
Now we get into the program. First we created two functions that we will use later. First we will need a function to extract the subjects and objects from the sentence. Second we will need a function that keeps track of all the counts in the text.
```{python}
def extract_subjects_objects(data):
    subjects = []
    objects = []

    for dep, word in zip(data[2], data[3]):
        if dep in ['nsubj', 'nsubjpass']:  # Nominal subjects (active and passive)
            subjects.append(word.lower())
        elif dep in ['dobj', 'iobj', 'pobj']:  # Direct objects, indirect objects, and objects of prepositions
            objects.append(word.lower())

    return subjects, objects

def count_filtered_pronouns(subjects, objects, feminine_pronouns, masculine_pronouns):
    counts_dict = {
        'feminine_subjects': 0,
        'feminine_objects': 0,
        'masculine_subjects': 0,
        'masculine_objects': 0
    }

    for subject in subjects:
        if subject in feminine_pronouns:
            counts_dict['feminine_subjects'] += 1
        elif subject in masculine_pronouns:
            counts_dict['masculine_subjects'] += 1

    for obj in objects:
        if obj in feminine_pronouns:
            counts_dict['feminine_objects'] += 1
        elif obj in masculine_pronouns:
            counts_dict['masculine_objects'] += 1

    return counts_dict
```
Now we loop through each of the files and use the pipeline to analyze and annotate the texts. We then use the above functions to break down the dictionary output that we get from the file. After that we print out the results.
```{python}
#Assuming your functions extract_subjects_objects and count_filtered_pronouns are defined as before
books_directory = 'books'

#Initialize an empty list to store counts dicts
rows = []

#Lists of masculine and feminine pronouns (both lowercase)
feminine_pronouns = ["she", "her", "hers"]
masculine_pronouns = ["he", "him", "his"]

#Iterate through each file in the directory
for filename in os.listdir(books_directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(books_directory, filename)
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            text = ''.join(lines[50:])
            sentences = sent_tokenize(text)
            sentences_list = []
            for sentence in sentences:
                sentences_list.append(sentence)

            # Annotate sentences with your pipeline
            dfs = pipeline.annotate(sentences_list)
            all_subjects = []
            all_objects = []

            # Extract subjects and objects from each annotated dataframe
            for df in dfs:
                subjects, objects = extract_subjects_objects(list(df.values()))
                all_subjects.extend(subjects)
                all_objects.extend(objects)

            # Get counts of filtered pronouns
            counts_dict = count_filtered_pronouns(all_subjects, all_objects, feminine_pronouns, masculine_pronouns)

            # Add the filename to the counts_dict
            counts_dict['filename'] = filename

            # Append counts to the list of rows
            rows.append(counts_dict)

#Create DataFrame from the list of rows
df_counts = pd.DataFrame(rows)
print(df_counts)
```
# Hypothesis
# Results
# Conclusion