# Resume Matching with Job Descriptions Using PDF CVs

## Objective

The goal of this project is to build a model using a pretrained bert sentance transformer model to extract relevant details from CVs in PDF format and match them against job descriptions from the Hugging Face dataset. The following outline provides a general guide for accomplishing this task.

## Step 1: PDF Data Extraction

### Objective

Extract relevant details from CVs in PDF format.

### Dataset

[Kaggle Resume Dataset]([link-to-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset))

### Instructions

1. Download the Kaggle "resume dataset."
2. Build a PDF extractor using Python, leveraging libraries such as PyPDF2.
3. Extract key details from CVs, including:
   - Category 
   - Skills
   - Education

## Step 2: Job Description Data Understanding

### Objective

Fetch and comprehend job descriptions from the Hugging Face dataset.

### Dataset

[Job Descriptions from Hugging Face]([link-to-dataset](https://huggingface.co/datasets/jacob-hugging-face/job-descriptions/viewer/default/train?row=0))

### Instructions

1. Use the Hugging Face datasets library to fetch job descriptions. Extracting of 10 job descriptions has been considered for this task.

## Step 3: Candidate-Job Matching

### Objective

Match extracted CV details against the fetched job descriptions based on skills and education.

### Transformer used

The "sentence-transformers" library is an open-source Python library developed by the UKPLab (Ubiquitous Knowledge Processing Lab) at the Technical University of Darmstadt, Germany. It's used for natural language processing tasks related to sentence and text embeddings, including tasks like sentence similarity, sentence classification, and more.

You can find the "sentence-transformers" library on GitHub: https://github.com/UKPLab/sentence-transformers

This library is built on top of the popular Hugging Face Transformers library and provides an easy-to-use interface for pre-trained transformer models that can encode sentences or text into dense vector representations, making it suitable for various NLP tasks.

### Steps

1. Tokenized and preprocessed both the job descriptions and the extracted CV details from the PDFs.
2. Convert the tokenized text into embeddings using a pretrained model SentenceTransformer("bert-base-uncased").
3. Calculated the cosine similarity between each job description's embedding and the embeddings of the CVs.
4. Ranked CVs based on similarity for each job description.
5. Listed the top 5 CVs for each job description based on the highest similarity scores.

## Usage
1.Import required libraries,model and dataset on to your IDE
2.Load the jobdesccvmatchfinal.ipynb
3.Run the code


This project outlines the steps to build a CV matching system and encourages you to explore innovative approaches. Feel free to modify or enhance the process as needed.
