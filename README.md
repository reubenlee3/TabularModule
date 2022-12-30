# TabularModule
## Environment Setup
* Create a new conda environment using the following command<br />
`conda create -n tabular python=3.9`
* The repo uses python 3.9 
* Install requirements from the requirements.txt using the following command<br />
`pip install --no-cache-dir -r <path_requirements_file>`
* After the installation is successful, install interpret-community<br /> 
`pip install --no-cache-dir interpret-community`
* There is issue in `Jinja2` package. Open the file <br /> 
`anaconda3/envs/tabular_test/lib/python3.9/site-packages/starlette/templating.py`
and edit line `56` rename `jinja2.contextfunction` to `jinja2.pass_context`

## Training tabular module
* Configs are in `config/data_config.yaml`
* To train, cd `TabularModule` and run `python executor.py` in the terminal
* Results[Model artifacts] are kept in `results/<exp_id_in_config_file>`
* Available models in Pycaret Classification are `['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 
'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost', 'dummy']`

## Scoring/Prediction using modules
* Set the model results path in the config `paths.result`
* Make sure the model`[classification.pkl/regression.pkl]` exists in the model result's folder.
* Scored/Prediction dataframe with be saved in the new folder `<paths.result>/scoring`

## TODO's
1. Implement regression task in the module (Search: `TODO-Regression` in the module)
2. Act on both independent drift and target drift in both training and prediction mode
3. Act on Fairness reports (Decide thresholds). Post process results so that model is aligned with FEAT principles
4. Develop Custom feature explanation module for interpretability
5. Write Docstrings and Unitest for Module maintainability