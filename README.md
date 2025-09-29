# Project:
ASR for cerebral palsy

PyTorch NN model
<img width="1025" height="205" alt="image" src="https://github.com/user-attachments/assets/38971a53-c87b-44b9-9868-183f004344c2" />


# Instructions:

First clone remote repo with:
```
git clone https://github.com/yaoruixuu/ASR.git
```

Create and activate a virtual environment to isolate depencies:
```
python3 -m venv your_venv_name
source your_venv_name/bin/activate
```

Install required libraries in requirements.txt:
```
pip3 install -r requirements.txt
```

Model is trained on 25 classification classes:
1. "I need water"
2. "I'm hungry"
3. "Please help me"
4. "Yes"
5. "No"
6. "Thank you"
7. "I'm tired",
8. "I want to go outside"
9. "Stop"
10. "Go"
11. "More"
12. "Less"
13. "I need the bathroom"
14. "I don't understand"
15. "Wait"
16. "Come here"
17. "Good morning"
18. "Good night"
19. "I'm happy",
20. "I'm sad"
21. "I need medicine"
22. "Call someone"
23. "I feel sick"
24. "I like this"
25. "I don't like this"

## How to Make an Inference
Pipeline currently supports .wav format

make an inference with the inference function in  model.py
```
inference(path)
```
