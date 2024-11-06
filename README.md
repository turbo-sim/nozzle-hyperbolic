# Nozzle-hyperbolic


## Cloning git repository and installing dependencies

Commands to clone repository, create Conda environment, and install all required packages:

```bash
git clone https://github.com/turbo-sim/nozzle-hyperbolic
cd nozzle-hyperbolic
conda env create -f environment.yaml
conda activate hyperbolic_env
poetry install
```


## Pushing to git repository

Commands to add changes and push to the remote repository

```bash
git add <filename>
git commit -m <commit message>
git push origin
```
