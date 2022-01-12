# Knowledge Graph Embeddings and Chemical Effect Prediction, 2020. 

Scripts and outputs related to the paper **Prediction of Adverse Biological Effects of Chemicals Using Knowledge Graph Embeddings**. 

A snapshot of TERA (the Toxicological Effect and Risk Assessment Knowledge Graph) is avalible as a Zenodo dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4244313.svg)](https://doi.org/10.5281/zenodo.4244313)


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

## Related publications

- Erik B. Myklebust, Ernesto Jimenez Ruiz, Jiaoyan Chen, Raoul Wolf, Knut Erik Tollefsen. **Prediction of Adverse Biological Effects of Chemicals Using Knowledge Graph Embeddings**. Accepted for publication in the Semantic Web Journal, 2021. ([arXiv](https://arxiv.org/abs/2112.04605)) ([Paper](http://semantic-web-journal.org/content/prediction-adverse-biological-effects-chemicals-using-knowledge-graph-embeddings-0)) ([REPOSITORY](https://github.com/NIVA-Knowledge-Graph/KGs_and_Effect_Prediction_2020))
