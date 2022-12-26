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