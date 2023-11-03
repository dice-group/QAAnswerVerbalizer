# QAAnswerVerbalizer
## Introduction
There have been recent advancements in Knowledge Based Question Answering Sytems. The KBQA systems are focused on generating the answers for specific questions using Knowledge Bases. However, verbalization of the answers in the KBQA systems can present information in a more human readable form for better understanding and interaction. This gives the user a much better experience and interaction. We generate verbalizations of different questions and SPARQL queries by fine-tuning a pre-trained encoder-decoder transformer based model with conditional generation. Transfer Learning is employed to make use of previously learned weights by the transformer model. The natural language verbalizations are generated using the question, query and the answer. 


## Datasets
The datasets used in this scope are:
1. **GrailQA** [https://dki-lab.github.io/GrailQA/]
2. **QALD-9 Plus**
3. **VQuAnDA**
4. **ParaQA** 
5. **VANiLLa** [https://figshare.com/articles/dataset/Vanilla_dataset/12360743]

The datasets GrailQA and QALD-9 Plus have been verbalized manually.

## Requirements and Setup
Clone the repository and install requirements.txt

```bash
git clone 
cd QAAnswerVerbalizer
pip install requirements.txt
```

See the [args](args.py) for more customization for preprocessing, fine-tuning and testing.
## Preprocessing
The train and test datasets have to be preprocessed separately 
To preprocess the datasets run: 

```bash
python preprocess/preprocess.py --dataset qald --name train --mask_ans True
python preprocess/preprocess.py --dataset qald --name test --mask_ans True
python preprocess/preprocess.py --dataset vanilla --name train --mask_ans True
```
The argument --dataset can be used to select the dataset for the experiment. The --name argument can be used to choose the train or test dataset.

## Fine-Tuning
To fine-tune the model using the preprocessed dataset, the appropriate model and the dataset have to be selected using the arguments.

```bash
python training/train.py --dataset qald --model_name pegasus --model_path google/pegasus-xsum --tokenizer_path google/pegasus-xsum --name train

```

## Testing
To generate the verbalizations and scores for the generations using the fine-tuned model:

```bash
python test.py --dataset qald --model_name pegasus
```  