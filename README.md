## ANLY-521 Final Project

#### Team Member: Zhe Zhou, Xinyao Mo



## Files  Description
There are total 5 files in this repo:
* ANLY521_FinalProject.ipynb	-	This Jupyter Notebook contain all experiments records
* ANLY521_FinalProject.py 	-	This .py file contains all the code from  Jupyter Notebook
* Sarcasm_Headlines_Dataset.json	-	Our dataset file
* ANLY521_Project_Report.pdf	-	Our project report
* README.md	-	README file. The file you are looking right now.

Since this is a experiment based project, we believe jupyter notebook is necessary. Please refer all the preprocessing and experiment records to our Jupyter Notebook.


## Code Execution Instruction:

To run this program simply navigate to this directory on terminal and type:
  ```python
  python ANLY521_FinalProject.py
  ```

**Before running the code, you have to make sure following requirments are met:**

* **Machine Learning Models:**  You should not have any problem running this part as long as you have `Scikt-Learn`and `NLTK`packages installed. 

* **LSTM Model:**  You should not have any problem running this part as long as you have `Scikt-Learn`and `Tensorflow 2.0`packages installed. If you don't have `Tensorflow 2.0`, just simply run :

  ```python
  pip install --upgrade tensorflow
  ```
  **There are two pre-trained word embedding files needed:**

  * `glove.6B.200d.txt`: this file has to be downloaded via this link http://nlp.stanford.edu/data/glove.6B.zip After you unzip it, please find `glove.6B.200d.txt` and put it in your directory.
  * `wiki-news-300d-1M.vec`: this file has to be downloaded via this link  https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip  Unzip the downloaded file, and place `wiki-news-300d-1M.vec`under your directory.

  

  Total running time for this part should be about **10 minutes**.

  

  

  
