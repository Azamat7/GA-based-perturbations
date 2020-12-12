# GA-based-perturbations

Clone the repo
```
$ git clone --origin origin https://github.com/Azamat7/GA-based-perturbations.git
$ cd GA-based-perturbations
$ git remote -v
origin  https://github.com/Azamat7/GA-based-perturbations.git (fetch)
origin  https://github.com/Azamat7/GA-based-perturbations.git (push)
```

Before writing any code, make sure local repo is up to date with origin
```
$ git fetch origin
$ git merge origin/main
```

To push
```
$ git add .
$ git commit -m "commit-message"
$ git push -u origin main
```

Better to work inside virtual env. To open `venv`
```
python3 -m venv venv
```

To activate
```
source venv/bin/activate
```

Install `requirements` after activating
```
pip3 install -r requirements.txt
```

To run
```
python GA.py cifar10 automobile10.png 0.1 30 -p 100
```