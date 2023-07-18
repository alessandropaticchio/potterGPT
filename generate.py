import torch
from potterGPT import GPTConfig, PotterGPT
from data import get_data
import gradio as gr

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data, val_data, vocab_size, encode, decode = get_data()

config = GPTConfig(n_layers=6, n_embd=384, block_size=256, n_head=8, vocab_size=vocab_size, dropout=0.2)
potterGPT = PotterGPT(config)
potterGPT.load_state_dict(torch.load('model/ckpt.pt', map_location=torch.device('cpu')))
potterGPT.to(device)
potterGPT.eval()


def generate(input, max_new_tokens):
    context = torch.LongTensor(encode(input), device=device).unsqueeze(0)
    return decode(potterGPT.generate(context, max_new_tokens=max_new_tokens)[0].tolist())


default_text_value = 'Harry said to Hermione:'
examples = [["Snape was staring at Harry's scar", 100], ["“Wingardium Leviosal” he shouted", 150],
            ["On Halloween morning they woke to the delicious smell", 200]]

demo = gr.Interface(
    fn=generate,
    inputs=[gr.Textbox(
        value=default_text_value, lines=2, placeholder="Input text...",
    ), gr.Slider(20, 500, value=100, label="max generated tokens", info="Choose between 20 and 500")],
    outputs=gr.components.Textbox(label="Generated Text"),
    title="PotterGPT ⚡",
    description="PotterGPT is a toy Language Model which has been trained on the 1st book of the Harry Potter saga.\n\n"
                "It's an educational project I built to learn a bit better how GPT works.\n\n"
                "Given a prompt, it is capable of generating sequence of characters that seem plausible.\n\n"
                "Since my computational resources are not enough to train a proper language model,the result don't really make "
                "much sense. Yet it has been an invaluable opportunity of learning!\n\n"
                "Have fun ⚡!",
    allow_flagging="never",
    examples=examples
)

demo.launch(share=True, server_name="0.0.0.0", server_port=9998)
