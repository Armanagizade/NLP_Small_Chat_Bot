import numpy as np 
import json
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout , Input
from keras.optimizers import SGD
import random
from nltk import ngrams
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import tkinter
from tkinter import *

data_file = open('intents.json').read()
intents = json.loads(data_file)

classes = []
sentences = []
docs = []
words = [' ']
for intent in range(len(intents['intents'])):
    classes.append(intents['intents'][intent]['tag'])
    for pattern in intents['intents'][intent]['patterns']:
        tmp_word = nltk.tokenize.word_tokenize(pattern)
        tmp_word = [lemmatizer.lemmatize(w.lower()) for w in tmp_word if w not in ['?','!',',']]
        docs.append((tmp_word ,intents['intents'][intent]['tag']))
        sentences.append(tmp_word)
        words.extend(tmp_word)
classes.sort()        
set_words = set(words)
list_words = []
for set_word in set_words:
    list_words.append(set_word) 
list_words.sort()
#padding
for i in range(len(sentences)):
    sentences[i] = list(ngrams(sentences[i], 8, pad_left=False, pad_right=True, right_pad_symbol=' '))[0]
    
    
y_train = np.zeros((len(docs),len(classes)))
x_train = np.zeros((len(docs),len(list_words)))
for d in range(len(docs)):
    index_y = classes.index(f"{docs[d][1]}")
    y_train[d,index_y] = 1
    for s in range(len(sentences[d])):
        index_x = list_words.index(sentences[d][s])
        x_train[d,index_x] = 1
        
model = Sequential([
                    Input(np.shape(x_train[0])),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(len(y_train[0]), activation='softmax')])



sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#fitting and saving the model 
hist = model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")


def chatbot_response(msg):
    chatbot_tmp_word = nltk.tokenize.word_tokenize(msg)
    chatbot_tmp_word = [lemmatizer.lemmatize(w.lower()) for w in chatbot_tmp_word if w not in ['?','!',',']]
    train_data = np.zeros((len(list_words)))
    for s in range(len(chatbot_tmp_word)):
        if chatbot_tmp_word[s] in list_words:
            index_x = list_words.index(chatbot_tmp_word[s])
            train_data[index_x] = 1
    res = model.predict(np.array([train_data]))
    class_res = str(classes[np.argmax(res)])
    for i in range(len(intents["intents"])):
        if intents["intents"][i]['tag'] == class_res:
            condidate_res = intents["intents"][i]['responses']
    final_res=  random.choice(condidate_res)
    return str(final_res)

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
