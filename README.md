# ECG reconstruction in 12 leads
The project allows you to reconstruct ECG signals in 12 leads based on data from the first limb lead.
Artificial neural network models for each lead are used for signal reconstruction.
## How to use it?
1. Write the path to the PTB-XL (https://physionet.org/content/ptb-xl/1.0.3/database) on the local computer in the config.py file.
2. Run the create_db.py file to create a database of records from the PTB-XL dataset 
3. Run the create_data.py file to generate training and testing data in .pkl format 
4. The ann128.py and ann136.py files are used to train the neural networks 
5. The file ml_models.py is used to train machine learning models 