import spacy

#create an nlp  object
#create tokens by passing it through nlp object
#remove stop words and then lemmatize the rest

nlp = spacy.load("en_core_web_sm")

#get stop words
#use .is_stop instead to avoid loading time
#stopwords  = nlp.Defaults.stop_words

#print(stopwords)
#create the tokens
input_sentence  = input("Enter: ")

corpus = ["Transformers did change the way Large language models are trained, taking a look at BERT,T5 and GPT-5",
"Transformers is a replacement for reccurent neural networks(superior replacement)",
"The realization of trnsformers and it's implemetation can help in solving the protein folding problem",
"Tensorflow is a mature framework but OpenAi choose Pytorch for their project.Amazing!",
"I will become a deep learning expert, no excuses."
]
#preprocessed_sentence = ""
def preprocess(sentence):

    sentence = nlp(sentence)

    token = [token.lemma_  for token in sentence if token.is_stop == False]

    return " ".join(token)
    
#inferior concept to check similarity between objects
#check similarity using spacy
def check_Similarity():
    input_sent = preprocess(input_sentence)
    for sentence in corpus:
        new_sentence = preprocess(sentence)
        
        similarity = nlp(new_sentence).similarity(nlp(input_sent))

        print(new_sentence, similarity)
#check_Similarity()
#output = preprocess(input_sentence)

#print(output)