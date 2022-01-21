If you want to check task1.py running, just type "python task1.py"

It will produce a file named "pipelines_example.txt", which include NLP features of some sentences.

To run the code, you have to first index the data into solr and the solr collection named "gettingstarted".

For indexing, you have to type "python indexer.py"

Wait for 5~6 minutes.

After indexing, you have to run task3, which include task2 and task3's code.

You have to type "python task3.py input_filename"

In input_filename, you have only one question per line.

And the output will be in result.csv.