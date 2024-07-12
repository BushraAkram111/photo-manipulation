import streamlit as st
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import io
import requests
import torch
from torchvision import models, transforms

def main():
    st.title("Photo Manipulation Classifer")
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose an Action", ["Upload Image", "Resize", "Crop", "Rotate", "Apply Filter", "Add Text", "Draw Shapes", "Classify Image"])

    if option == "Upload Image":
        upload_image()
    elif option == "Resize":
        resize_image()
    elif option == "Crop":
        crop_image()
    elif option == "Rotate":
        rotate_image()
    elif option == "Apply Filter":
        apply_filter()
    elif option == "Add Text":
        add_text()
    elif option == "Draw Shapes":
        draw_shapes()
    elif option == "Classify Image":
        classify_image()

def upload_image():
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.image = Image.open(uploaded_file).convert("RGB")
        st.image(st.session_state.image, caption='Uploaded Image', use_column_width=True)

def resize_image():
    if 'image' not in st.session_state:
        st.error("Please upload an image first.")
        return

    st.subheader("Resize the Image")
    image = st.session_state.image
    st.image(image, caption='Current Image', use_column_width=True)
    width = st.number_input("New Width", min_value=1, value=image.width)
    height = st.number_input("New Height", min_value=1, value=image.height)
    if st.button("Resize"):
        st.session_state.image = image.resize((width, height))
        st.image(st.session_state.image, caption='Resized Image', use_column_width=True)
        st.download_button("Download Resized Image", image_to_bytes(st.session_state.image), "resized_image.jpg", "image/jpeg")

def crop_image():
    if 'image' not in st.session_state:
        st.error("Please upload an image first.")
        return

    st.subheader("Crop the Image")
    cropped_image = st_cropper(st.session_state.image, aspect_ratio=None, return_type='image')
    if st.button("Crop"):
        st.session_state.image = cropped_image
        st.image(cropped_image, caption='Cropped Image', use_column_width=True)
        st.download_button("Download Cropped Image", image_to_bytes(cropped_image), "cropped_image.jpg", "image/jpeg")

def rotate_image():
    if 'image' not in st.session_state:
        st.error("Please upload an image first.")
        return

    st.subheader("Rotate the Image")
    angle = st.slider("Rotation Angle", -360, 360, 0)
    if st.button("Rotate"):
        st.session_state.image = st.session_state.image.rotate(angle, expand=True)
        st.image(st.session_state.image, caption='Rotated Image', use_column_width=True)
        st.download_button("Download Rotated Image", image_to_bytes(st.session_state.image), "rotated_image.jpg", "image/jpeg")

def apply_filter():
    if 'image' not in st.session_state:
        st.error("Please upload an image first.")
        return

    st.subheader("Apply Filter to the Image")
    filter_type = st.selectbox("Select a Filter", ["BLUR", "CONTOUR", "DETAIL", "EDGE_ENHANCE", "SHARPEN"])
    if filter_type == "BLUR":
        radius = st.slider("Blur Radius", 0, 20, 2)
    preview_image = apply_image_filter(st.session_state.image, filter_type, radius if filter_type == "BLUR" else None)
    st.image(preview_image, caption='Preview of Filter Applied', use_column_width=True)
    if st.button("Apply Filter"):
        st.session_state.image = preview_image
        st.image(preview_image, caption='Filtered Image', use_column_width=True)
        st.download_button("Download Filtered Image", image_to_bytes(preview_image), "filtered_image.jpg", "image/jpeg")

def apply_image_filter(image, filter_type, radius=None):
    filters = {
        'BLUR': lambda: image.filter(ImageFilter.GaussianBlur(radius)),
        'CONTOUR': lambda: image.filter(ImageFilter.CONTOUR),
        'DETAIL': lambda: image.filter(ImageFilter.DETAIL),
        'EDGE_ENHANCE': lambda: image.filter(ImageFilter.EDGE_ENHANCE),
        'SHARPEN': lambda: image.filter(ImageFilter.SHARPEN)
    }
    return filters.get(filter_type, lambda: image)()

def add_text():
    if 'image' not in st.session_state:
        st.error("Please upload an image first.")
        return

    st.subheader("Add Text to the Image")
    text = st.text_input("Text to Add")
    position_x = st.number_input("Text X Position", 0, st.session_state.image.width, 50)
    position_y = st.number_input("Text Y Position", 0, st.session_state.image.height, 50)
    font_size = st.slider("Font Size", 10, 100, 30)
    if st.button("Add Text"):
        st.session_state.image = add_text_to_image(st.session_state.image, text, (position_x, position_y), font_size)
        st.image(st.session_state.image, caption='Image with Text', use_column_width=True)
        st.download_button("Download Image with Text", image_to_bytes(st.session_state.image), "text_image.jpg", "image/jpeg")

def add_text_to_image(image, text, position, font_size):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=(255, 0, 0))
    return image

def draw_shapes():
    if 'image' not in st.session_state:
        st.error("Please upload an image first.")
        return

    st.subheader("Draw Shapes on the Image")
    shape = st.selectbox("Select Shape", ["rectangle", "ellipse", "line"])
    color = st.color_picker("Select Color", "#FF0000")

    if shape in ["rectangle", "ellipse"]:
        left = st.number_input("Left", 0, st.session_state.image.width, 0)
        top = st.number_input("Top", 0, st.session_state.image.height, 0)
        right = st.number_input("Right", 0, st.session_state.image.width, st.session_state.image.width)
        bottom = st.number_input("Bottom", 0, st.session_state.image.height, st.session_state.image.height)
        position = (left, top, right, bottom)
    else:
        x1 = st.number_input("Start X", 0, st.session_state.image.width, 0)
        y1 = st.number_input("Start Y", 0, st.session_state.image.height, 0)
        x2 = st.number_input("End X", 0, st.session_state.image.width, st.session_state.image.width)
        y2 = st.number_input("End Y", 0, st.session_state.image.height, st.session_state.image.height)
        position = (x1, y1, x2, y2)

    if st.button("Draw Shape"):
        st.session_state.image = draw_shape_on_image(st.session_state.image, shape, position, color)
        st.image(st.session_state.image, caption='Image with Shape', use_column_width=True)
        st.download_button("Download Image with Shape", image_to_bytes(st.session_state.image), "shape_image.jpg", "image/jpeg")

def draw_shape_on_image(image, shape, position, color):
    draw = ImageDraw.Draw(image)
    if shape == 'rectangle':
        draw.rectangle(position, outline=color, width=3)
    elif shape == 'ellipse':
        draw.ellipse(position, outline=color, width=3)
    elif shape == 'line':
        draw.line(position, fill=color, width=3)
    return image

def classify_image():
    if 'image' not in st.session_state:
        st.error("Please upload an image first.")
        return

    st.subheader("Classify the Image")
    st.image(st.session_state.image, caption='Image to Classify', use_column_width=True)

    # Define the transformation for the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)
    model.eval()

    # Transform the image and get predictions
    input_image = preprocess(st.session_state.image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Fetch the class labels
    labels_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    labels = requests.get(labels_url).json()

    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    st.write("Top 5 Predictions:")
    for i in range(top5_prob.size(0)):
        label = labels[top5_catid[i].item()]
        probability = top5_prob[i].item()
        st.write(f"{label}: {probability:.4f}")

    # Add a download button for the classified image
    st.download_button("Download Classified Image", image_to_bytes(st.session_state.image), "classified_image.jpg", "image/jpeg")

def image_to_bytes(image):
    """Convert PIL image to bytes."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()

if __name__ == "__main__":
    main()
