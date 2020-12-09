# Knowledge Graph Embeddings and Chemical Effect Prediction, 2020. 

Scripts and outputs related to paper. 

TERA snapshot is avalible [here](https://doi.org/10.5281/zenodo.4244313).

To run the codes you must install [KGE-Keras](https://github.com/NIVA-Knowledge-Graph/KGE-Keras).

To reproduce results from paper run:
```
bash run.sh
```
This will take several days, depending on hardware avalible. 3-5 days on Nvidia GTX 1070ti. Set `--MAX_TRIALS 0` to use default parameters. 

Alternativly, unzip `results.zip` and run
```
python3 analyse_results.py
```
