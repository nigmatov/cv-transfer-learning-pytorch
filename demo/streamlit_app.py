import streamlit as st
import torch
from PIL import Image
from torchvision import transforms as T
from src.model_cls import ClsModel

st.set_page_config(page_title="CV Demo", layout="wide")
st.title("Image Classification Demo")

ckpt = "outputs/cls.ckpt"
try:
    model = ClsModel.load_from_checkpoint(ckpt)
    model.eval()
    st.success(f"Loaded checkpoint {ckpt}")
except Exception as e:
    model = ClsModel()
    model.eval()
    st.warning("No checkpoint found. Using untrained model.")

img = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if img:
    image = Image.open(img).convert("RGB")
    tf = T.Compose([T.Resize((128,128)), T.ToTensor()])
    x = tf(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(-1).squeeze().tolist()
    st.image(image, caption="Input", width=256)
    st.write({"class0": probs[0], "class1": probs[1]})
