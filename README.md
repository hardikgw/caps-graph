
#### To run with docker
docker run -it -p 8888:8888 -p 6006:6006 --name=tensorflow -v $(pwd)/notebooks:/notebooks -e PASSWORD=password cithub/tensorflow

## Using venv

#### Install venv (run command below inside project dir)
```python3 -m venv env```
#### Activate venv
```source env/bin/activate```
#### Generate requirements file
```pip freeze > requirements.txt```
#### Install requirements
```pip install -r requirements.txt```

## Docker for cayley
```shell script
docker run -v $PWD/data/cayley/data:/data -p 64210:64210 -d cayleygraph/cayley
```