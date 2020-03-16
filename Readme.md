# similar-articles

This project trains embeddings for scientific articles
using data collected from the INSPIRE-HEP API.

The embeddings are used by the hep-recommender web application
to provide similar article recommendations in the field of 
High Energy Physics.


### Usage: 

To test the package and train the embeddings, start a virtual environment and install
the necessary dependencies using 

    pip install -r requirements.txt
    
Place the file 'test_references_data.txt' in the 'input_data' folder.  Then run

    python model_training.py

This will save the trained embeddings on the 'output_data' directory. 
Adjust the 'model_config.json' if desired.
