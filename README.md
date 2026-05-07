# Env setup
```
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Training
* Edit `configs.py`, e.g., input file location and number of epochs
* Edit the main function of `Trainer.py` to invoke training, testing, and plotting for different layers