## Installation

### macOS
```
$ brew update
$ brew install python3
$ brew install opencv3 --with-python
$ pip3 install --upgrade pip setuptools wheel
$ pip3 install virtualenv
```

### ubuntu
```
$ sudo apt-get install python3 python3-pip
$ sudo pip3 install --upgrade pip setuptools wheel
$ sudo pip3 install virtualenv
```

### Setup Python Virtual Enviroment
```
$ virtualenv --no-site-packages -p python3 .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```
