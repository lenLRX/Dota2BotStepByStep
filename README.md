# Dota2BotStepByStep

## requirement
python3
pytorch > 0.4
tensorboardX

## train
export $PYTHONPATH=/path/to/repo
python3 ./d2bot/test/A3CParallel.py

## test

set path of your model at ./d2bot/test/A3CEnvSpliter_load.py:line 97

python3 ./d2bot/test/A3CEnvSpliter_load.py visible

