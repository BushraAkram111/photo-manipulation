import streamlit as st
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import io
import requests
import torch
from torchvision import models, transforms
from streamlit_cropper import st_cropper

def main():
    st.title("Photo Manipulation and Classifer")
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
    radius = None  # Initialize radius variable
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
        font = ImageFont.load_default()  # Fallback to default font
    draw.text(position, text, font=font, fill=(255, 0, 0))
    return image

def draw_shapes():
    if 'image' not in st.session_state:
        st.error("Please upload an image first.")
        return

    st.subheader("Draw Shapes on the Image")
    shape_type = st.selectbox("Choose Shape", ["Rectangle", "Ellipse", "Line"])
    if shape_type == "Rectangle":
        x1, y1, x2, y2 = st.slider("Select Rectangle Coordinates", 0, st.session_state.image.width, (10, 10, 100, 100))
        if st.button("Draw Rectangle"):
            st.session_state.image = draw_rectangle_on_image(st.session_state.image, (x1, y1, x2, y2))
            st.image(st.session_state.image, caption='Image with Rectangle', use_column_width=True)
            st.download_button("Download Image with Rectangle", image_to_bytes(st.session_state.image), "rectangle_image.jpg", "image/jpeg")

    elif shape_type == "Ellipse":
        x1, y1, x2, y2 = st.slider("Select Ellipse Coordinates", 0, st.session_state.image.width, (10, 10, 100, 100))
        if st.button("Draw Ellipse"):
            st.session_state.image = draw_ellipse_on_image(st.session_state.image, (x1, y1, x2, y2))
            st.image(st.session_state.image, caption='Image with Ellipse', use_column_width=True)
            st.download_button("Download Image with Ellipse", image_to_bytes(st.session_state.image), "ellipse_image.jpg", "image/jpeg")

    elif shape_type == "Line":
        x1, y1, x2, y2 = st.slider("Select Line Coordinates", 0, st.session_state.image.width, (10, 10, 100, 100))
        if st.button("Draw Line"):
            st.session_state.image = draw_line_on_image(st.session_state.image, (x1, y1, x2, y2))
            st.image(st.session_state.image, caption='Image with Line', use_column_width=True)
            st.download_button("Download Image with Line", image_to_bytes(st.session_state.image), "line_image.jpg", "image/jpeg")

def draw_rectangle_on_image(image, box):
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="red", width=5)
    return image

def draw_ellipse_on_image(image, box):
    draw = ImageDraw.Draw(image)
    draw.ellipse(box, outline="blue", width=5)
    return image

def draw_line_on_image(image, box):
    draw = ImageDraw.Draw(image)
    draw.line((box[0], box[1], box[2], box[3]), fill="green", width=5)
    return image

def classify_image():
    if 'image' not in st.session_state:
        st.error("Please upload an image first.")
        return

    st.subheader("Classify the Image")
    model = models.resnet18(pretrained=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = preprocess(st.session_state.image)
    input_batch = input_image.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    st.write("Classification result:", output.argmax().item())

def image_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

if __name__ == "__main__":
    main()
