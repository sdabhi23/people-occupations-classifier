# people-occupations-classifier

This project has been undertaken as a part of our undergraduate coursework, to understand generation of datasets and classification algorithms.

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
    $ pip install -r requirements.txt
    ```
4. Additional steps for nltk

    ```python
    >>> import nltk
    >>> nltk.download('stopwords')
    ```

5. Steps to configure ipython kernel

   > Required only if using virtual environment

   ```bash
   $ ipython kernel install --user --name=people_classifier
   ```
   Then change the kernel in the jupyter interface to **people_classifier**.

## Maintainers

* Shrey Dabhi ([sdabhi23](https://github.com/sdabhi23))
* Kartavya Soni ([kartavyasoni25](https://github.com/sdabhi23))

## References

* Classifying Wikipedia People Into Occupations by Aleksander Gabrovski ([.pdf](http://cs229.stanford.edu/proj2014/Aleksandar%20Gabrovski,%20Classifying%20Wikipedia%20People%20Into%20Occupations.pdf))
* Multi-Class Text Classification with Scikit-Learn ([article](https://datascienceplus.com/multi-class-text-classification-with-scikit-learn/))
* Using jupyter notebooks with a virtual environment ([article](https://anbasile.github.io/programming/2017/06/25/jupyter-venv/))
