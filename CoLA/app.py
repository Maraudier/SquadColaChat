# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, render_template, request
from torch import load, tensor, device, no_grad
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from torch.utils.data import TensorDataset


def prep(input):
    MAX_LEN = 50
    
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', 
        do_lower_case=True)
    tokenize = tokenizer.encode(input, add_special_tokens = True)    
    padded = pad_sequences([tokenize], maxlen=MAX_LEN, dtype="long", value=0,truncating="post", padding="post")
    # padded = padded[0]
    att_mask = [float(token_id > 0) for token_id in padded[0]]
    tensor_data = tensor(padded[0])
    tensor_data = tensor_data.unsqueeze(0)
    tensor_mask = tensor(att_mask)
    tensor_mask = tensor_mask.unsqueeze(0)
    # tensor_label = tensor([0 for item in att_mask])
    # tensor_set = TensorDataset(tensor_data, tensor_mask, tensor_label)
    with no_grad():
        output = model(tensor_data, token_type_ids=None, attention_mask=tensor_mask)
    print(output)

    return output

app = Flask(__name__)

model = load("Colabertmodel.h5", map_location=device('cpu'))

def predict(text_in):
    model.eval()
    output = prep(text_in)
    output = list(output[0].detach().cpu().numpy()[0])
    return output.index(max(output))

@app.route("/", methods = ["GET", "POST"])
def home():
    text_in = ""
    prediction = ""
    if "input" in request.form:
        text_in = request.form["input"]
        prediction = predict(text_in)
    return render_template("index.html", fill_text = text_in, prediction=prediction)

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000, debug = True)