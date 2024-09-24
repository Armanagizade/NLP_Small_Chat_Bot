ChatBot Application
This repository contains a simple chatbot application built using Python, Keras, and NLTK. The chatbot is trained using intents data and can classify user input into predefined categories. The project also includes a basic GUI created using Tkinter for user interaction.

Features
Trainable model using custom intents.json file
Simple neural network-based classification using Keras
Tokenization, lemmatization, and padding using NLTK
GUI-based interaction using Tkinter
Responses are generated based on user input using a trained neural network
Requirements
To run the project, ensure you have the following dependencies installed:

Python 3.x
TensorFlow / Keras
Numpy
NLTK
Tkinter
You can install the required Python packages using:

bash
Copy code
pip install numpy tensorflow nltk
How to Run the ChatBot
Install Dependencies: Make sure you have all the required libraries by running:

bash
Copy code
pip install -r requirements.txt
Prepare Intents File: Edit or provide your own intents.json file, which contains the categories, patterns, and responses that the chatbot will use.

Train the Model: The chatbot uses a neural network to classify the input patterns. You can train the model by running the script:

bash
Copy code
python chatbot.py
The model will be saved as chatbot_model.h5.

Run the Chatbot Application: After training the model, run the following command to start the chatbot with a graphical interface:

bash
Copy code
python chatbot_gui.py
Folder Structure
intents.json: Contains the data for training the chatbot model, including patterns and corresponding responses.
chatbot_model.h5: The saved neural network model after training.
chatbot.py: Script to train the model and save it.
chatbot_gui.py: The main script to run the chatbot GUI.
How It Works
Data Processing: The chatbot uses NLTK for tokenizing and lemmatizing the user input, and then processes it into a bag of words.
Model Training: The chatbot model is a simple feedforward neural network with two hidden layers. It uses categorical crossentropy loss and Stochastic Gradient Descent (SGD) as the optimizer.
User Interaction: The chatbot uses a Tkinter GUI for user input and displays the bot's responses in real-time.
Future Improvements
Add more complex sentence processing (e.g., handling negation).
Improve the intent classification model.
Implement additional features such as context-based responses.
License
This project is open-source under the MIT License. Feel free to use and modify it.
