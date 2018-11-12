# people-occupations-classifier

This project has been undertaken as a part of my undergraduate coursework, to understand generation of datasets and classification algorithms.

## Getting started

1. Clone this repo

```bash
$ git clone https://github.com/sdabhi23/people-occupations-classifier.git
```

2. Setup virtual environment (optional)

```bash
$ virtualenv .venv
# for windows
$ .venv\Scripts\activate
# for *nix
$ .venv/Scripts/activate
```

3. Installing the required libraries

```bash
$ cd scripts
$ pip install -r requirements.txt
```
4. Additional steps for nltk

```python
>>> import nltk
>>> nltk.download('stopwords')
```

## References

* Classifying Wikipedia People Into Occupations by Aleksander Gabrovski ([.pdf](http://cs229.stanford.edu/proj2014/Aleksandar%20Gabrovski,%20Classifying%20Wikipedia%20People%20Into%20Occupations.pdf))