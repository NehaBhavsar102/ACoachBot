from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
import pytorch_lightning as pl
from nltk.corpus import wordnet
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load('en_core_web_md')
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)

pl.seed_everything(17)

dataset = pd.read_csv('fairytale_data.csv')
dataset = dataset.dropna()
dataset.columns = ['ids', 'context', 'question', 'answer_text', 'title']
dataset.drop_duplicates(subset=['ids'], inplace = True)
dataset.reset_index(inplace=True, drop=True)

Model_Name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(Model_Name)


train_data = dataset[['question', 'answer_text', 'context']]
    
class MyModel(pl.LightningModule):
  
  def __init__(self):
    super().__init__()
    self.model = T5ForConditionalGeneration.from_pretrained(Model_Name, return_dict = True)

  def forward(self, input_ids, attention_mask, labels=None):
    output = self.model(
        input_ids = input_ids,
        attention_mask = attention_mask,
        labels = labels
    )
    return output.loss, output.logits


trained_t5_model = MyModel.load_from_checkpoint('best-checkpoint.ckpt')
trained_t5_model.freeze()

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'], methods=['GET', 'POST'], allow_headers=['Content-Type'])

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if(request.method == 'GET'):
        
        random = np.random.randint(0, len(train_data))
        context = train_data.loc[random]['context']
        context_encoding = tokenizer(
            context,
            max_length = 350,
            padding='max_length',
            truncation = 'only_second',
            return_attention_mask = True,
            add_special_tokens = True,
            return_tensors = 'pt'
        )
        
        generated_ids = trained_t5_model.model.generate(
            input_ids = context_encoding['input_ids'],
            attention_mask = context_encoding['attention_mask'],
            num_beams = 10,
            max_length = 40,
            repetition_penalty = 5.0,
            early_stopping = True,
            use_cache = True
        )

        preds = [
            tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces = True)
            for generated_id in generated_ids
        ]

        predictions = ' '.join(preds)
        
        question = predictions.split(':')[1].replace('answer', '')[1:-2]
        answer = predictions.split(':')[2]
        return jsonify({'gen_question': question, 'gen_answer': answer, 'context': context})


@app.route('/score', methods=['GET', 'POST'])
def score():
    if(request.method == 'POST'):
        data = request.get_json(force=True)
        sentence1 = data['gen_answer']
        sentence2 = data['provided_answer']
        # vectorizer = TfidfVectorizer()
        # vectors = vectorizer.fit_transform([sentence1, sentence2])
        # cosine_similarities = cosine_similarity(vectors)
        # score = round(cosine_similarities[0][1] * 10, 2)
        
        # sentence1_vec = nlp(sentence1).vector
        # sentence2_vec = nlp(sentence2).vector
        # similarity = cosine_similarity([sentence1_vec], [sentence2_vec])[0][0]
        # score = (similarity + 1) * 5
        # print(score)
        
        tokens1 = nltk.word_tokenize(sentence1)
        tokens2 = nltk.word_tokenize(sentence2)
        synonyms1 = set()
        synonyms2 = set()

        for token in tokens1:
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    synonyms1.add(lemma.name())

        for token in tokens2:
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    synonyms2.add(lemma.name())

        # Convert the sentences to vectors using TF-IDF
        vectorizer = TfidfVectorizer()
        corpus = [sentence1, sentence2]
        X = vectorizer.fit_transform(corpus)

        # Calculate the Soft Cosine Measure between the two vectors
        score = cosine_similarity(X)[0,1]*10
        
        return jsonify({'score':score})
    
if __name__== "__main__":
    app.run(debug=True)
