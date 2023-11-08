# Readme--MCRPL

MCRPL: A Prompt Learning Paradigm for Non-overlapping Many-to-one Cross-domain Recommendation

The HVIDEO dataset used in the article includes the E domain and V domain, and the link to the HVIDEO dataset is [HVIDEO](https://drive.google.com/drive/folders/1DfQUqJTQkTm_p4mgNBQqvHmhIup0gDAR?usp=sharing).

1. `model.py` is the Python code that stores the model.
2. `main_all.py` A is the code for training the model, saving the model weight file and obtaining the result; 
3. `datasets_all.py`  is the code for loading training data. This code can choose whether to load all domain data for the first stage training or fine-tune the target domain data for the second stage based on the domain marker bit. This code can increase the offset for item numbers in other fields to ensure the uniqueness of item numbers.
4. `SASrec.py`  It is a classic sequence recommendation method, which is the first to use attention mechanism for sequence recommendation. We use the Pytorch version of the code [SASRec](https://github.com/pmixer/SASRec.pytorch). To ensure fair comparison, we have made modifications to it.
5. `utils.py` is the test function code for testing recall and MRR indicators
6. `make_weak_data.py` This code is used to generate sparse user sequence data and disrupt the order of paired users to ensure no overlapping conditions.
## Requirements  
python 3.5+  
pytorch 1.11.0  
## Usage
Using `make_weak_data.py` to process data.
`python  main_all.py test` will undergo two processes: pre-training and fine-tuning, and obtain all results.  
