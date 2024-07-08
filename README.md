# QAAnswerVerbalizer
## Introduction
There have been recent advancements in Knowledge Based Question Answering Sytems. The KBQA systems are focused on generating the answers for specific questions using Knowledge Bases. However, verbalization of the answers in the KBQA systems can present information in a more human readable form for better understanding and interaction. This gives the user a much better experience and interaction. We generate verbalizations of different questions and SPARQL queries by fine-tuning a pre-trained encoder-decoder transformer based model with conditional generation. Transfer Learning is employed to make use of previously learned weights by the transformer model. We experiment with both SPARQL Queries and RDF Triples as input along with the natural language question. The triples are generated from SPARQL queries where they are not available. We also experiment with masking of entities and answers in the question and the verbalized answer. The natural language verbalizations are generated using the question, query/triple and the answer(s). We experiment with different metrics including BLEU, SacreBLEU, METEOR, CHRF++, TER, ROUGE.

<img src="docs/image/QQAV architecture.png" alt="QAAnswerVerbalizer" width="900"/>

## Datasets
The datasets used in this scope are:
1. **GrailQA** [https://dki-lab.github.io/GrailQA/]
2. **ParaQA** [https://figshare.com/projects/ParaQA/94010]
3. **QALD-9 Plus** [https://github.com/KGQA/QALD_9_plus/tree/main/data]
4. **VQuAnDA** [https://figshare.com/projects/VQuAnDa/72488]
5. **VANiLLa** [https://figshare.com/articles/dataset/Vanilla_dataset/12360743]

The datasets GrailQA and QALD-9 Plus have been verbalized manually.

Please move all the datasets in the [data](data) folder under their specific names.
## Models
Our finetuned models can be dowloaded from this FTP folder:
https://files.dice-research.org/projects/QA-Verbalization/SEMANTICS2024/
## Requirements and Setup
Clone the repository and install requirements.txt

```bash
git clone 
cd QAAnswerVerbalizer
pip install requirements.txt
```

See the [args](args.py) for more customization for preprocessing, fine-tuning and testing.

## Preprocessing
The train and test datasets have to be preprocessed separately. The answers can be masked using --mask_ans as shown below.
To preprocess the datasets run: 

```bash
python preprocess/preprocess.py --dataset qald --name train --mask_ans
python preprocess/preprocess.py --dataset vquanda --name test --mask_ans
python preprocess/preprocess.py --dataset vanilla --name train --mask_ans
```
The argument --dataset can be used to select the dataset for the experiment. The --name argument can be used to choose the train or test dataset.

## Fine-Tuning
To fine-tune the model using the preprocessed dataset, the appropriate model and the dataset have to be selected using the arguments. 
The supported pre-trained models that can be loaded from huggingface.co are PEGASUS, T5, BART.
The training parameters can be controlled along with directory paths. The --mode_t argument can be used to only use triples as input and is to be used along with --mode triples argument.

```bash
python training/train.py --dataset qald --model_name pegasus --model_path google/pegasus-xsum --tokenizer_path google/pegasus-xsum --mode triples --train_epochs 10
python training/train.py --dataset vquanda --model_name bart --model_path facebook/bart-base --tokenizer_path facebook/bart-base  --mode query --train_epochs 10
python training/train.py --dataset paraQA --model_name bart --model_path facebook/bart-base --tokenizer_path facebook/bart-base  --mode triples --mode_t --train_epochs 10

```

## Testing
To generate the verbalizations and scores for the generations using the fine-tuned model :

```bash
python test.py --dataset qald --model_name pegasus --checkpoint_path checkpoint-5000 --mode triples
```  
The path of the checkpoint to be tested should be given through --checkpoint_path. 


## Results

The results for the VQuAnDa dataset on different metrics following the masked approach with question and the logical form (Q+LF) as input as shown in the following table:

Model  | BLEU | METEOR | SacreBleu | CHRF++ | TER 
------------- | ------------- | ----------- | ----------- | ----------- | -----------
**PEGASUS** | 80.70  | 48.50 | 65.10 | 76.38 | 27.43 
**BART**  | 78.80  | 45.43 | 62.86 | 74.80 | 29.92 
**T5**  | 80.66  | 49.25 | 65.62 | 76.32 | 27.48



The results for the ParaQA dataset on different metrics following the masked approach with question and the logical form (Q+LF) as input as shown in the following table:

Model  | BLEU | METEOR | SacreBleu | CHRF++ | TER 
------------- | ------------- | ----------- | ----------- | ----------- | -----------
**PEGASUS** | 82.07  | 48.50 | 64.98 | 72.50 | 26.52 
**BART**  | 80.21  | 46.48 | 61.85 | 69.71 | 28.60 
**T5**  | 80.26  | 47.49 | 62.08 | 69.91 | 28.72
