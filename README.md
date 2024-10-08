# FiNER

This repository contains the weak-ner framework for the paper "[FiNER: Financial Named Entity Recognition Dataset and Weak-Supervision Model](https://arxiv.org/abs/2302.11157v1)". The dataset, available on [HuggingFace](https://huggingface.co/datasets/gtfintechlab/finer-ord), can be used as a benchmark for financial domain-specific NER and NLP tasks. FiNER consists of a manually annotated dataset of English financial news articles collected from [webz.io](https://webz.io/free-datasets/financial-news-articles/). More information is available in the [paper](https://arxiv.org/abs/2302.11157v1).

## Setup

### Clone the repo

```shell
git clone git@github.com:gtfintechlab/FiNER.git
```

### Setup virtual environment

```
conda env create -f environment.yml
```


## Adding a new entity in the pipeline
* In order to add a new entity, one needs to modify `fin_reer/enums/labels.py` file. The new entities should be appended to the existing list of entities in that file
* Entity list in this file is stored in variable: `entity_list`
  * Current list of entities include: `entity_list: List[str] = ["PER", "LOC", "ORG"]`
  
## Adding a new labeling function
* Labeling functions should be added in `fin_reer/labeling_functions/entities`folder of the repository
* One can create a separate files for adding new labeling functions or add more labling functions in the existing file `fin_reer/labeling_functions/entities/lfs.py` 
* For example, adding a new labeling function to detect person based on prefixes will require adding the following labeling function in the existing file. 

```python
@labeling_function(pre=[pre_tokenize_text])
def label_per_heuristic_prefix(x):
    titles = {"Dr", "Mr", "Mrs", "Ms", "Prof"}
    
    spans = []
    i = 0
    while i < len(tokens):
        if tokens[i][0] in titles:
            idx = i + 1
    
            while idx < len(tokens) and tokens[idx][0].isupper():
                if idx == i+1:
                    spans.append(tokens[idx][1])
                else:
                    spans[-1] = (spans[-1][0], tokens[idx][1][1])
                idx+=1
            
            i = idx
        else:
            i += 1

    return x.uuid, generate_labels(x, spans, "PER", "ENTITY")

```

## Experiment Configurations
We have simplified the process of running the experiments we added in the paper. In order to run any experiment, 
one just needs to write a JSON file containing configuration required to run the experiment. Below, we explain the config file for each experiment type. 

### Weak Supervision Configuration

```json
{
    "input_df_path": "./data/news_data/train_input_df.csv.gz",
    "unpickle_columns": [],

    "epochs": 1000,
    "log_frequency": 1000,
    "seed": 42,

    "experiment_name": "news_sigir_42",
    "experiment_version": "1.1",

    "label_matrix_save_path": "./GeneratorExperimentsResults",
    "evaluate_generated_labels": true,
    "train_gold_data_path": "./data/news_data/train_gold_data.csv.gz",
    "test_input_df_path": "./data/news_data/test_input_df.csv.gz",
    "test_gold_data_path": "./data/news_data/test_gold_data.csv.gz",

    "split_generated_data": false
}
```
* `input_df_path`: Path of input data. The input data should be in the same format as we have used in the pipeline
  * Check the sample data in `./data/news_data/train_input_df.csv.gz`
* `unpickle_columns`: If you have encoded any python objects in your data in the form of pickle dump strings, you can provide those column names in this list
  * The script will unpickle those columns before feeding it to the weak supervision pipeline
* `epochs`, `log_frequency`, `seed`: Hyper parameters to use for the Snorkel Weak Supervision aggregator model, which is used to combine the labels from labeling functions
* `experiment_name`, `experiment_version`: Identifiers for the experiment. The location of results will use these names to store the results
* `label_matrix_save_path`: This path specifies where to save the label matrix
* 'evaluate_generated_labels': If you have the gold data against which you are testing your labeling functions, you can set this parameter to True 
* `train_gold_data_path`, `test_gold_data_path`: Gold data paths if the gold data is available. If not, one can keep them empty
* `test_input_df_path`: Test data in the same format as `input_df_path`. The trained Snorkel Aggregator will be used to aggregate labeling function signals on this data
  * If one intends to do labeling from scratch using labeling functions, providing this data path does not make sense
  * This is intended if you have gold data. In that case, one can compare the performance of the weak supervision pipeline against gold data and check how well the labeling functions are labeling the raw data


### Majority Vote Configuration
* Running majority vote requires running the weak supervision pipeline first because it uses the label matrix generated from labeling function as an input 
* The Majority Vote algorithm will run on the same label matrix and generate the aggregated labels
* One needs to use the same label matrix because doing that will enable us to make the fair comparison of Snorkel based aggregation and the majority vote based aggregation 

```json
{
    "label_matrix_path": "./GeneratorExperimentsResults/news_sigir_42_1_1/label_matrix_news_sigir_42_1_1.csv.gz",
    "experiment_name": "news_majority_vote_sigir_42",
    "experiment_version": "1.1",
    "results_save_path": "./MajorityVoteExperimentsResults"
}
```

* `label_matrix_path`: Path of label matrix. It is generated by the weak supervision pipeline described above
* `experiment_name`, `experiment_version`: Identifiers for the experiment. The location of results will use these names to store the results
* "results_save_path": Location where to save the results

## How to run

* In order to run the experiments, one needs to first write the configuration file as described above
* Once done, one can run the following command to run the experiment and see the results in the GeneratorExperiments results folder

```shell
python3 fin_reer/weak_ner_pipeline.py <configuration file path>
```

For example: 

```shell
python3 fin_reer/weak_ner_pipeline.py "GeneratorExperiments/news_seed_42_sigir.json"
```

## Citation
```c
@article{shah2024finerordfinancialnamedentity,
  title={FiNER-ORD: Financial Named Entity Recognition Open Research Dataset},
  author={Agam Shah and Abhinav Gullapalli and Ruchit Vithani and Michael Galarnyk and Sudheer Chava},
  journal={arXiv preprint arXiv:2302.11157},
  year={2024}
}
```


## Contact Information: 
* Please contact Agam Shah (ashah482[at]gatech[dot]edu) or Ruchit Vithani (rvithani6[at]gatech[dot]edu) about any FiNER-related issues and questions.
* GitHub: [@shahagam4](https://github.com/shahagam4), [@ruchit2801](https://github.com/ruchit2801) 
* Website: https://shahagam4.github.io/
