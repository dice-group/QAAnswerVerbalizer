# QAAnswerVerbalizer
## Introduction
There have been recent advancements in Knowledge Based Question Answering Sytems. The KBQA systems are focused on generating the answers for specific questions using Knowledge Bases. However, verbalization of the answers in the KBQA systems can present information in a more human readable form for better understanding and interaction. This gives the user a much better experience and interaction. We generate verbalizations of different questions and SPARQL queries by fine-tuning a pre-trained encoder-decoder transformer based model with conditional generation.


## Datasets
The datasets used in this scope are:
1. **GrailQA** [https://dki-lab.github.io/GrailQA/]
2. **QALD-9 Plus**
3. **VQuAnDA**

The datasets GrailQA and QALD-9 Plus have been verbalized manually.

## Requirements and Setup
Clone the repository and install requirements.txt

```bash
git clone 
cd QAAnswerVerbalizer
pip install requirements.txt
```

## Preprocessing
To preprocess the datasets run: 

```bash
python preprocess/preprocess.py --dataset qald
```
The argument --dataset can be used to select the dataset for the experiment. 

## Fine-Tuning
To fine-tune the model using the preprocessed dataset, the appropriate model and the dataset have to be selected using the arguments.

```bash
python training/train.py --dataset qald --model_name pegasus
```

## Testing
To generate the verbalizations and scores for the generations using the fine-tuned model:

```bash
python test.py --dataset qald --model_name pegasus
```  