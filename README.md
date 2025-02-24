# Sentiment_Analysis_LSTM_BiLSTM
 Deep Learning and Edge Computing Lab Assignment-3
Exp: Sentiment Analysis Using LSTM and BiLSTM
Name: Nairit Das
Reg No: 24MAI0097

Submitted To: Dr. Anil Kumar K.



Literature Review
Sentiment Analysis of Marathi News Using LSTM
Sentiment analysis (SA) in low-resource languages like Marathi faces challenges due to limited lexical tools (e.g., parsers, SentiWordNet) and annotated datasets. Recent studies highlight LSTM’s effectiveness in capturing long-term dependencies in text, making it suitable for SA in morphologically rich languages. Prior works on Indian languages, such as Hindi and Bengali, employed SVM and hybrid models but relied on manual feature engineering or bilingual dictionaries, which are resource-intensive. For Marathi, the absence of robust NLP tools necessitates approaches like LSTM, which bypass dependency on external lexicons by learning contextual patterns directly from data. Existing research on Arabic and Vietnamese corpora demonstrates LSTM’s superior accuracy (up to 94%) compared to traditional methods, attributed to its ability to model sequential and semantic relationships. In this study, a bidirectional LSTM model is proposed for Marathi e-news, trained on a small annotated dataset (121 sentences). Results show moderate precision (0.55) but high recall (1.00), suggesting effective identification of sentiment-bearing features despite dataset constraints. Comparative analysis with Telugu SA (73% accuracy) and English IMDB reviews underscores the impact of training data size on performance. While hybrid models (e.g., CNN-LSTM) and knowledge-based techniques show promise in other languages, this work emphasizes LSTM’s adaptability to Marathi’s linguistic complexities, advocating for scaled datasets and enhanced preprocessing to bridge accuracy gaps [1].



Sentiment Analysis for Stock Price Prediction Using LSTM and TLBO
Stock price prediction, influenced by volatile social and economic factors, increasingly leverages sentiment analysis of social media data, where Twitter’s real-time opinions serve as critical indicators. Traditional models often overlook unstructured, noisy Twitter data, characterized by brevity and irregular syntax, necessitating robust preprocessing to extract actionable insights. Recent advancements integrate deep learning, with Long Short-Term Memory (LSTM) networks excelling in capturing sequential dependencies and contextual sentiment in textual data. This study introduces a hybrid Teaching and Learning Based Optimization (TLBO)-LSTM model, optimizing hyperparameters via TLBO to enhance predictive accuracy. The Adam optimizer further refines learning rates, addressing gradient instability during training. By classifying tweets into positive/negative sentiments, the model correlates public sentiment with stock market trends. Evaluations demonstrate the TLBO-LSTM framework’s superiority, achieving 94.73% accuracy, 95.33% precision, and 90% F-score, outperforming conventional methods. These results underscore the efficacy of combining evolutionary algorithms with deep learning for financial forecasting, particularly in handling noisy social media data. The approach bridges the gap between unstructured sentiment signals and quantitative stock predictions, offering a scalable solution for real-time market analysis. Future work could explore multilingual sentiment integration and larger datasets to bolster generalizability [2].


Sentiment Analysis of Comment Texts Based on BiLSTM
Sentiment analysis has become pivotal in the big data era, where user-generated comments on social platforms offer critical insights into public opinion. Traditional approaches, such as distributed word representations (e.g., Word2Vec, GloVe), prioritize semantic context but overlook explicit sentiment cues inherent in words, limiting their efficacy in emotion-driven classification tasks. Recent studies highlight bidirectional Long Short-Term Memory (BiLSTM) networks as superior for capturing contextual dependencies in text, outperforming unidirectional models (LSTM, RNN) and static architectures (CNN) in sequence modeling. This paper addresses the gap in sentiment-aware feature extraction by enhancing the TF-IDF algorithm to integrate sentiment weights, generating hybrid word vectors that encode both frequency and emotional salience. The weighted vectors, processed through BiLSTM, effectively model long-range contextual relationships, while a feedforward classifier maps these representations to sentiment polarities. Experimental comparisons with RNN, CNN, LSTM, and Naïve Bayes (NB) baselines demonstrate the model’s robustness, achieving higher precision, recall, and F1 scores. The success of this approach underscores the importance of combining sentiment-specific feature engineering with deep learning’s contextual prowess, particularly for noisy, opinion-rich comment data. Future work could explore dynamic sentiment weighting and cross-domain adaptability to further refine accuracy [3].





References
[1]:Divate, Manisha Satish. “Sentiment Analysis of Marathi News Using LSTM.” International Journal of Information Technology, vol. 13, no. 5, Springer Science and Business Media LLC, Aug. 2021, pp. 2069–74, https://doi.org/10.1007/s41870-021-00702-1. Accessed 22 Feb. 2025.
‌[2]:T. Swathi, et al. “An Optimal Deep Learning-Based LSTM for Stock Price Prediction Using Twitter Sentiment Analysis.” Applied Intelligence, vol. 52, no. 12, Springer Science+Business Media, Mar. 2022, pp. 13675–88, https://doi.org/10.1007/s10489-022-03175-2. Accessed 22 Feb. 2025.
[3]:Xu, Guixian, et al. “Sentiment Analysis of Comment Texts Based on BiLSTM.” IEEE Access, vol. 7, Institute of Electrical and Electronics Engineers, Jan. 2019, pp. 51522–32, https://doi.org/10.1109/access.2019.2909919. Accessed 22 Feb. 2025.
‌

Description for LSTM Implementation:
This code implements a complete sentiment analysis pipeline tailored for tweet data, focusing on detecting signs of depression. It begins by importing necessary libraries and installing the emoji package, then reads a CSV file containing tweets and their corresponding labels. The preprocessing phase includes converting text to lowercase, removing HTML tags, URLs, punctuation, and stopwords, expanding chat abbreviations, converting emojis to text, and lemmatizing the text. After cleaning, the text is tokenized and padded to prepare it for input into deep learning models. Two types of models are then constructed—a standard LSTM-based model and a Bidirectional LSTM (BiLSTM) model—where the latter is loaded with pre-trained weights. The code proceeds to train, evaluate, and visualize the performance of the models, and includes a prediction function that outputs sentiment with a confidence score.

Conclusion:
The code demonstrates an end-to-end approach for text-based sentiment analysis, particularly aimed at detecting depressive sentiment in tweets. Through comprehensive preprocessing and the application of both LSTM and BiLSTM architectures, the pipeline achieves robust sentiment classification. The evaluation metrics and visualizations further confirm that this method is effective for analyzing nuanced emotional content in social media data.

Description for BiLSTM Implementation:
This code implements a text preprocessing and sentiment classification pipeline using a Bidirectional LSTM (BiLSTM) neural network.
Key Steps in the Code:
 -Data Preprocessing:
 -Reads a CSV file containing tweets and their sentiment labels.
 -Cleans the text by:
 -Converting to lowercase.
 -Removing HTML tags, URLs, punctuation, and emojis.
 -Expanding chat abbreviations (e.g., "LOL" → "Laughing Out Loud").
 -Removing stopwords (common words like "the", "is", etc.).
 -Lemmatizing words to their base form.
 -Tokenization and Padding:
 -Converts text into sequences using Tokenizer.
 -Pads sequences to ensure uniform input size for the model.
 -BiLSTM Model:
   Uses Bidirectional LSTM layers to capture contextual dependencies in both forward and backward directions.
   Includes dropout layers to prevent overfitting.
   Uses ReLU activation for hidden layers and sigmoid activation for binary classification.
 -Model Training & Evaluation:
   Trains the BiLSTM model on the processed text data.
   Evaluates performance using accuracy and loss plots.
  This approach enables effective sentiment analysis of tweets, detecting signs of depression based on textual content.

Conclusion:
Overall, this end-to-end solution demonstrates how comprehensive text preprocessing combined with a BiLSTM network can effectively capture both forward and backward contextual dependencies in text data. This robust approach is well-suited for sentiment classification tasks, such as detecting depression indicators in tweets.










