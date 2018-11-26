## Setup:
```
git clone https://github.com/rgerd/ar-mirror.git
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Using the right python version:
### On macOS:
* [Install python 3.6.7](https://www.python.org/ftp/python/3.6.7/python-3.6.7-macosx10.9.pkg)
```
cd .../.../ar-mirror
rm -rf venv
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```

## To-do:
* Coordinate frame stuff
* Hand tracking (?) (maybe cascades?)
* Optimize
