# med-graph-mae
In the field of healthcare, the adoption and integration of artificial intelligence (AI)-based systems is playing an increasingly relevant and evident role.
In this work, we attempt to place ourselves in this landscape by attempting to address the difficult challenge of predicting adverse events for patients through the use of electronic health records.
In particular, knowledge graphs are used to represent the medical history and relationships between patients. Through the use of the GraphMAE2 framework, state-of-the-art mask-autoencoding techniques will be employed to learn meaningful latent representations of the data in an unsupervised manner to be exploited for node classification tasks.

## Licence

A licence for the use of MIMIC-III data must be obtained. As the database contains detailed information regarding the clinical care of patients, it must be treated with appropriate care and respect. Researchers are required to formally request access via a process documented on the MIMIC website. There are two key steps that must be completed before access is granted:
* the researcher must complete a recognized course in protecting human research participants that includes Health Insurance Portability and Accountability Act (HIPAA) requirements.
* the researcher must sign a data use agreement, which outlines appropriate data usage and security standards, and forbids efforts to identify individual patients.
Approval requires at least a week. Once an application has been approved the researcher will receive emails containing instructions for downloading the database from PhysioNetWorks, a restricted access component of PhysioNet.
Only credentialed users who sign the DUA can access the files
Required training: https://physionet.org/content/mimiciii/view-required-training/1.4/#1
License: https://physionet.org/content/mimiciii/view-license/1.4/

## File format and folder structure

Once the licence has been obtained, pre-processed data can be requested. In particular, the folder structure in the project is as follows:
-'train/' and 'test/': Contain the training and test data respectively.

    - `dataset.arrow`: File containing the data in Arrow format.

    - `dataset_info.json`: JSON file with information about the dataset.

    - `state.json`: JSON file with the state of the dataset.

- `dataset_dict.json`: JSON file containing the dataset dictionary.

- `id2tkn.pickle`: Pickle file containing a mapping map of IDs to tokens.

## Setup

1. Copy the obtained folders to your drive or your environment (in this discussion google colaboratory was used)

2. Clone the repository https://github.com/picuslab/med-graph-mae/tree/main/GraphMAE2-main

## Method

![Image representing the proposed method to tackle the challenge of predicting events](https://github.com/picuslab/med-graph-mae/blob/main/ProposedMethod.PNG)


## Usage

1. Run the 'GM2Preparation.ipynb' (https://github.com/picuslab/med-graph-mae/blob/main/GM2Preparation.ipynb) notebook you will use the above-mentioned files to create the dataframes needed to generate the graphs and therefore to use the GraphMAE2 framework. In particular:
    - change the folders
    - uncomment lines for "GROUND TRUTH GENERATION"
    - uncomment line for "LABEL GENERATION"
    - uncomment line for "FEATURES GENERATION"
    - uncomment last line of "EDGES GENERATION" 

2. Run the 'GM2Run.ipynb' (https://github.com/picuslab/med-graph-mae/blob/main/GM2Run.ipynb) notebook to run GraphMAE2
    - You have to use a gpu to run the code
    - In google colaboratory:
        - go to "Runtime"
        - change runtime type
        - select "T4 GPU" and save 

## Baseline

The file 'Framework1_LAST.ipynb' (https://github.com/picuslab/med-graph-mae/blob/main/Framework1_LAST.ipynb) was also placed inside the folder to perform the predictions with the KMeans clustering method only, to use the code you only have to run the notebook and enter the desired parameters