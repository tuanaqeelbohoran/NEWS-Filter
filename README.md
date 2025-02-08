# NEWS-Filter
 The provided program is a graphical user interface (GUI) tool for filtering news articles based on semantic similarity to a user-provided query. It leverages the SentenceTransformer model to process and evaluate the content of news articles, allowing users to filter out irrelevant articles and save relevant ones.
![image](https://github.com/user-attachments/assets/63b8748d-b2e8-4f82-a69b-4828381d699b)

1. pip install the following packages:

sentence-transformers

pandas

pytz

argparse

2. Place the "nasa.jsonl", "Volkswagen.jsonl" files on the same directory

3. The experiment can be runtrough either Jupyter note-book with "News Filter.ipynb" file or 
in command line with "news_filter.py" file.

4. To run  "news_filter_CLI.py" file use the following command:

python "news_filter_CLI.py"  --path nasa.jsonl --query NASA

or 

python "news_filter_CLI.py"  --path volkswagen.jsonl --query volkswagen

5. To run  "news_filter_GUI.py" file use the following command:

python "news_filter_GUI.py"   
