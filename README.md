# CICMoD Applications

[![Build Status](https://github.com/MarcoLandtHayen/cicmod_application/workflows/Tests/badge.svg)](https://github.com/MarcoLandtHayen/cicmod_application/actions)
[![codecov](https://codecov.io/gh/MarcoLandtHayen/cicmod_application/branch/main/graph/badge.svg)](https://codecov.io/gh/MarcoLandtHayen/cicmod_application)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)[![pypi](https://img.shields.io/pypi/v/cicmod_application.svg)](https://pypi.org/project/cicmod_application)


Apply machine learning methods to Climate Index Collection based on Model Data (CICMoD).
https://github.com/MarcoLandtHayen/climate_index_collection/

## Development

For now, we're developing Docker containter with JupyterLab environment, Tensorflow and several extensions, based on jupyter/tensorflow-notebook.

To start a JupyterLab within this container, run
```shell
$ docker pull mlandthayen/py-da-tf:latest
$ docker run -p 8888:8888 --rm -it -v $PWD:/work -w /work mlandthayen/py-da-tf:latest jupyter lab --ip=0.0.0.0
```
and open the URL starting on `http://127.0.0.1...`.

Then, open a Terminal within JupyterLab and run
```shell
$ python -m pip install -e .
```
to have a local editable installation of the package.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>
