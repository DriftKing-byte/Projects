Image Tagger & Description Generator (Multimodal RAG, LLaVA, CLIP)

A multimodal rag application for tagging images or providing image descriptions leveraging locally run LLaVA:7b using Ollama. The user can feed any image and choose whether to get tags or descriptions, and the model will generate such respectively. Image tags consist of 5 unique words and image descriptions contain up to 3 sentences. OpenAI’s CLIP model for unified embeddings of text and images was used for data preprocessing.

Visa Information Chatbot (LangGraph, LangChain, LLMs) (File: Visa_Chatbot_Notebook.ipynb)

A QA rag application for visa questions built leveraging locally run llama 3.1 using Ollama based on the ReAct agent architecture. The user can ask visa related questions, and the chatbot will search credible, government sources for curating a high quality response. The chatbot also has memory within the session and human feedback in the loop. So, the chatbot remembers previous human messages and asks clarifying questions when needed. The production grade app was built using Streamlit.

QA Chatbot for Any Topic (RAG-based LLM Application) (File: FAQ_Chatbot_For_Any_Topic.ipynb)

A question-answering rag application built leveraging locally run llama 3.1 using Ollama. The user can choose whether to upload a url of a static webpage or a pdf and ask questions related to the content within. The model will output an appropriate answer that is accurate. Streamlit was used for building the production grade app. 

Clinical Note Summarizer (LLM Fine-Tuning, NLP) (File: NLP_Transformers_Clinical_Note_Summarizer_Via_Gen_AI_(tiny_llama).ipynb)

TinyLlama was leveraged to summarize lengthy clinical notes. The dataset included nearly 20,000 clinical notes with AI styled prompts. A causal LM pipeline from Hugging Face with LoRA fine-tuning was plugged in SFTTrainer for training TinyLlama to curate highly accurate 200 token summaries of the clinical notes.

Consumer Complaint Classifier (NLP, Machine Learning) (File: Consumer_Complaint_Classifier.ipynb)

Implemented a NLP classifier, where 30,000 plus consumer complaints sent to the Consumer Financial Protection Bureau were classified as either credit reporting, general reporting, and among other categories. The wordnet lemmatizer in the natural language toolkit (nltk) was used to preprocess the text which was transferred to the Word2Vec model for vectorization. The fully preprocessed data was loaded to a Random Forest Classifier for the classification. An overall accuracy of 89.3% accuracy was achieved.  

NVIDIA Stock Price Forecasting (Time Series, LSTM) (File: NVIDIA_Stock_Market_Forecast.ipynb)

An LSTM was used to visualize the next 10 days’ stock market opening price for NVIDIA. 3 years of NVIDIA stocks data was compiled. Pandas, NumPy, scikit-learn were used for data preprocessing and Matplotlib for data visualization. A validation loss of near 0 was achieved for historical data. The model has capacity for predicting as many days as needed but a sample of 10 was used. The final output was a line graph with historical data and next 10 days for visualization.

Pneumonia Classification from Chest X-Rays (Computer Vision, Transfer Learning) (File: Computer_Vision_Image_Classification_(Non_Transformer).ipynb)

ConvNeXt Tiny model was instanced on GPU to classify whether pneumonia was present or not on a lung scan. The dataset, obtained from PubMed, consisted of over 4000 scans. Image preprocessing, in Keras, was used to first correct the image height, width, set to grayscale, and split to train and test. Through transfer learning, imagenet pretrained was trained on the data. A validation accuracy of 99.84% was achieved.

Airline Ticket Price Prediction (Tabular Regression, GPU Acceleration) (File: Airline_Ticket_Price_Prediction.ipynb)

A categorical regression analysis to predict the price of an airline ticket with the objective of the customer to make cost-effective decisions. Categorical columns such as source city, departure time, arrival time, destination city, and stops were used with numerical columns such as days left for a total of over 20 features. Pandas, NumPy, and scikit-learn were used for data preprocessing. Random Forest Regression algorithm was implemented and evaluation metric MAE was used for final prediction. 
