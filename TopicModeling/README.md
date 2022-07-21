### Instructions to run code


1. Install Docker [(directions)](https://docs.docker.com/get-docker/)
2. Download **`TopicModeling.tar.gz`** and uncompress it
3. Copy or move the input data directory **`training_20180910`** into the **`TopicModeling/data`** directory

```bash
cd TopicModeling

# Build Docker Image
docker build -t nlp ./

# SSH into docker container (i.e. start an interactive bash session) 
./start_interactive.sh 

   # Train and evaluate topic model after SSH into docker container
   python generate_topics_and_evaluate.py

# Start Jupyter Notebook to check dev work (open: http://localhost:8888/)
./start_notebook.sh  # All notebooks are in `notebooks` directory
```


### Approach

Topic Modeling

1. Load documents and extract text from these headings: ["discharge diagnosis", "chief complaint", "history of present illness"]
2. Pre-process documents
	- Remove punctuation, whitespace, PHI
	- Tokenize the documents
	- Remove stopwords
	- Lemmatize the tokens of documents using **`ScispaCy en_core_sci_md`**
	- Filter extremes
3. Topic Modeling using Gensim
	- `AuthorTopicModel` for topic modeling
	- `CoherenceModel` for computing coherence
	- Model returns a dictionary of topic list for each medical condition in the "discharge diagnosis", "chief complaint", "history of present illness" sections of the document
4. Evaluate using NER Annotations
	- Load annotations and extract text from rows starting with **`T`** with words starting with **`Reason`** and add to a hashmap based data structure
	- Compute common topic words between model output and annotation words
	- Compute fraction of documents where topic model output matched words from documents annotation
	- Evaluation results will be displayed on terminal. Model coherence = 0.3737




