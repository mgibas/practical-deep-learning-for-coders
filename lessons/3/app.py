import gradio as gr
from fastai.vision.all import *

def predict(img):
    categories= ('button', 'checkbox', 'radio-button')
    learn = load_learner('../2/model.pkl')
    pred,idx,probs = learn.predict(img)
    floatProbs = map(float, probs)
    print(pred, idx, floatProbs)
    return dict(zip(categories, floatProbs))

inputs = gr.Image()
outputs = gr.Label()
textbox = gr.Textbox()
examples= ['buttons.png', 'checkbox.png', 'radio-button.png']
demo = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, examples=examples)
demo.launch()
