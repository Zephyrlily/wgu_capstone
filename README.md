# wgu_capstone
My capstone project for my WGU Computer Science Bachelor's Degree program. It uses a Random Forrest classifier to train a machine learning algorithm based on a provided dataset, which is then used to predict if provided links are phishing or benign links.

<img width="975" height="422" alt="image" src="https://github.com/user-attachments/assets/65d26b5d-0ab0-44c4-a39b-8392b35f5461" />

User Guide
1.	First, download the provided zip file and extract the contents. You should have the folder “WGU CS Capstone”.
2.	Inside the folder, open the command line and run this command: pip install -r requirements.txt
<img width="975" height="37" alt="image" src="https://github.com/user-attachments/assets/083c13ae-4352-4f64-b7bf-d7587174c690" />

3.	Open “ui.py”
4.	Inside the opened program, click the “Train” button in the train tab (default should be correct, but if errors appear, try to manually select “data/csv” and “models/phish_detector.joblib” using the Browse button. Wait for training to complete, this takes a while.
5.	Congratulations, you have a joblib model! Now, let’s evaluate. Switch to the evaluate tab and click the “Evaluate” button. Again, if it doesn’t work, try using Browse to find the joblib and “eval” csv. Evaluation should be fairly quick. View the data and enjoy!
6.	You can also click on the “Predict” tab to make predictions. This is what most employees will use. Insert a single URL and click “Predict”, or select the “eval.csv” and predict it. 

https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls

https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-urls

These two datasets were used together for the program. To start, the data was normalized to be in the same format: a URL and label column, with 1s as phishes and 0s as benign. Then, a program was run to combine the data into one large data set, then split it up: 85% into a training csv, and 15% into an evaluation csv. This data was used to train and evaluate the model: the program loaded in each URL, normalized it, and extracted various features from it such as letter count, use of symbols, and more.
