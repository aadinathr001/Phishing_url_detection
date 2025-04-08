this repo has a dataset.csv file that consists of 
  4,80,000 url samples
  phishing labelled as 1
  legitimate labelled as 0
  the dataset is balanced
phishing_url_prediction/
│── app.py                  # Flask application
│── model/
│   ├── phishing_model.keras # Saved deep learning model
│   ├── tokenizer.pkl        # Saved tokenizer
│── templates/
│   ├── index.html           # Input form
│── static/
│   ├── style.css            # CSS styles
