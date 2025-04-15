# ProcedureT5
![The workflow of ProcedureT5](./assets/main.svg)

*Computer-aided synthesis planning (CASP) has shown strong potential to accelerate chemical 
research. However, a key challenge remains: the lack of effective automated techniques to translate 
computer-generated synthesis routes into executable experimental procedures, which still require 
extensive planning and evaluation by chemists. To address this gap, we introduce ProcedureT5, an 
approach that integrates chemistry-oriented pre-trained models with augmented multi-source datasets 
to enhance the prediction of experimental procedures across broader scenarios. Our method achieves 
state-of-the-art performance on the Pistachio dataset - a collection of reaction procedures derived 
from US patent literature, showing a 4-point increase in BLEU score and a 34% improvement in 
exact-match accuracy compared to existing methods. Additionally, we curate a small expert-
annotated dataset, Orgsyn, consisting of verified organic synthesis procedures, to assess the model’s 
performance in more diverse applications. Fine-tuning ProcedureT5 on the Orgsyn dataset 
demonstrates its adaptability, yielding a BLEU score of 41.19 and an average similarity of 50.58%. 
This work underscores the crucial role of ProcedureT5 in bridging the gap between computational 
synthesis planning and practical laboratory implementation*.

## Requirements
Create the enviroment:

```sh
conda create -n ProcedureT5 python=3.8.8
```

Install requirements:

```sh
pip install -r requirements.txt
```

## Dataset
You can find the Orgsyn dataset and the augmented Orgsyn dataset [here](./dataset). Access the Pistachio dataset requires the permission from [Vaucher et al.](https://www.nature.com/articles/s41467-021-22951-1).

## Model Train
For training and finetuning ProcedureT5, please configure shell scripts [here](./scripts/) and run them.

## Perform Predictions
Our two model variants are available on the Hugging Face Hub at the following links:

[ProcedureT5-small](https://huggingface.co/Napister/ProcedureT5-small): Derived from MolT5-small, this model was trained using the augmented Pistachio dataset.

[ProcedureT5-base](https://huggingface.co/Napister/ProcedureT5-base): The top-performing model in our study, built from MolT5-base and trained with the augmented Pistachio dataset.

## Citation