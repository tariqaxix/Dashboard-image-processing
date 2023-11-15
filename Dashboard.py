import streamlit as st
import cv2
import numpy as np
import tempfile
import scipy.stats
from skimage.exposure import match_histograms
import heapq
import collections
from bitarray import bitarray
import imghdr
import base64


with open(r'C:\Users\tariq.aziz\OneDrive - University of Central Asia\Desktop\style.css') as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


st.title("Image Processing Dashboard")
uploaded_image = st.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"])


# Function for Region Growing Segmentation

def region_growing_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    labels = output[1]
    label_count = output[0]
    largest_component = np.argmax(output[2][1:, 4]) + 1
    segmented_image = np.zeros_like(labels)
    segmented_image[labels == largest_component] = 255
    return segmented_image


# Function for Run Length Encoding
def run_length_encoding(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat_image = image.ravel()
    encoding = []
    prev_pixel = flat_image[0]
    count = 1
    for pixel in flat_image[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            encoding.append((prev_pixel, count))
            prev_pixel = pixel
            count = 1
    encoding.append((prev_pixel, count))
    return encoding

# Function for Run Length Decoding


def run_length_decoding(encoding):
    decoded_image = []
    for pixel, count in encoding:
        decoded_image.extend([pixel] * count)
    return np.array(decoded_image, dtype=np.uint8)


def encoded_data_to_string(encoding):
    return '\n'.join([f'{pixel} {count}' for pixel, count in encoding])


# Function for Huffman coding
def build_huffman_tree(data):
    frequency = collections.Counter(data)
    heap = [[weight, [char, ""]] for char, weight in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heap[0][1:], key=lambda p: (len(p[-1]), p))


def huffman_encoding(data):
    tree = build_huffman_tree(data)
    encoded_data = "".join([item[1] for item in tree])
    return encoded_data, tree


def huffman_decoding(data, tree):
    decoded_data = []
    while data:
        for item in tree:
            if data.startswith(item[1]):
                decoded_data.append(item[0])
                data = data[len(item[1]):]
                break
    return "".join(decoded_data)


def image_to_binary(image, threshold=128):
    binary_data = (image > threshold).flatten()
    return binary_data


# Function for Arithmetic Encoding
def arithmetic_encoding(data):
    frequencies = collections.Counter(data)
    total_symbols = sum(frequencies.values())
    probabilities = {}
    start = 0
    for symbol, freq in frequencies.items():
        probabilities[symbol] = (start, start + freq / total_symbols)
        start += freq / total_symbols

    low, high = 0.0, 1.0
    for symbol in data:
        symbol_range = probabilities[symbol]
        range_size = high - low
        high = low + range_size * symbol_range[1]
        low = low + range_size * symbol_range[0]

    return low, probabilities


if uploaded_image is not None:
    st.image(uploaded_image, caption="Original Image", use_column_width=True)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_image.read())
    uploaded_image.seek(0)
    image = cv2.imread(temp_file.name)

    # Image processing options
    processing_option = st.selectbox(
        "Select Image Processing Option",
        [
            "Grayscale", "Scalar Multiplication", "Min-Max Stretching", "Invert Colors",
            "Log Transformation", "Power Law Transformation",
            "Histogram Equalization",
            "Mean Filter", "Median Filter",
            "Min Filter", "Max Filter",
            "Laplacian Filter",
            "Nearest Neighbor Interpolation", "Bilinear Interpolation", "Region Growing Segmentation", "Run Length Encoding", "Huffman Encoding",
            "Arithmetic Encoding"

        ]
    )

    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Run Length Encoding
    if processing_option == "Run Length Encoding":
        if st.button("Run Length Encode"):
            encoding = run_length_encoding(image)
            encoded_string = encoded_data_to_string(encoding)
            st.text_area("Encoded Data", encoded_string, height=250)
            st.download_button(label="Download Encoded Data",
                               data=encoded_string,
                               file_name="encoded_data.txt",
                               mime="text/plain")

            if st.button("Decode"):
                decoded_image = run_length_decoding(encoding)
                decoded_image = decoded_image.reshape(image.shape)
                st.image(decoded_image, caption="Decoded Image",
                         use_column_width=True)

# Huffman Encoding
    if processing_option == "Huffman Encoding":
        if len(processed_image.shape) > 2:
            st.warning(
                "Huffman encoding can only be applied to grayscale images. Convert the image to grayscale first.")
        else:
            flattened_image = image.ravel()
            encoded_data, huffman_tree = huffman_encoding(flattened_image)
            st.write("Huffman Encoding:", encoded_data)

# Arithmetic Encoding
    if processing_option == "Arithmetic Encoding":
        if len(processed_image.shape) > 2:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

        flattened_image = processed_image.ravel().tolist()
        encoded_value, symbol_probabilities = arithmetic_encoding(
            flattened_image)

        st.write("Arithmetic Encoded Value:", encoded_value)
        st.write("Symbol Probabilities:", symbol_probabilities)


# Gray Scale
    elif processing_option == "Grayscale":
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Min-Max Stretching
    elif processing_option == "Min-Max Stretching":
        min_val = np.min(image)
        max_val = np.max(image)
        processed_image = (image - min_val) / (max_val - min_val) * 255
        processed_image = processed_image.astype(np.uint8)

# Invert Colors
    elif processing_option == "Invert Colors":
        processed_image = cv2.bitwise_not(image)

# Log Transformation
    elif processing_option == "Log Transformation":
        c = 255 / np.log(1 + np.max(image))
        processed_image = c * (np.log(image + 1))
        processed_image = processed_image.astype(np.uint8)

# Power Law Transformation
    elif processing_option == "Power Law Transformation":
        gamma = st.slider("Gamma", 0.1, 10.0, 1.0)
        processed_image = np.power(image, gamma)
        processed_image = (
            255 * (processed_image / np.max(processed_image))).astype(np.uint8)

# Histogram Equalization
    elif processing_option == "Histogram Equalization":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)
        processed_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

# Mean Filter
    elif processing_option == "Mean Filter":
        kernel = np.ones((3, 3), np.float32) / 9
        processed_image = cv2.filter2D(image, -1, kernel)

# Median Filter
    elif processing_option == "Median Filter":
        processed_image = cv2.medianBlur(image, 3)

# Min Filter
    elif processing_option == "Min Filter":
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        processed_image = cv2.erode(image, kernel, iterations=1)

# Max Filter
    elif processing_option == "Max Filter":
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        processed_image = cv2.dilate(image, kernel, iterations=1)

# Laplacian Filter
    elif processing_option == "Laplacian Filter":
        processed_image = cv2.Laplacian(image, cv2.CV_64F)
        processed_image = np.uint8(np.absolute(processed_image))

# Nearest Neighbor Interpolation
    elif processing_option == "Nearest Neighbor Interpolation":
        scale_factor = st.slider("Scale Factor", 0.1, 5.0, 2.0)
        processed_image = cv2.resize(
            image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

# Bilinear Interpolation
    elif processing_option == "Bilinear Interpolation":
        scale_factor = st.slider("Scale Factor", 0.1, 5.0, 2.0)
        processed_image = cv2.resize(
            image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Region Growing Segmentation
    elif processing_option == "Region Growing Segmentation":
        processed_image = region_growing_segmentation(image)

# Scalar Multiplication
    elif processing_option == "Scalar Multiplication":
        scalar = st.number_input("Enter Scalar Value")
        if isinstance(image, np.ndarray):
            processed_image = image * scalar
            processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)

    if processing_option != "Run Length Encoding" and processing_option != "Huffman Encoding" and processing_option != "Arithmetic Encoding":
        st.image(processed_image,
                 caption=f"{processing_option} Image", use_column_width=True)

        if st.button("Download Processed Image"):
            temp_output_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".jpg")
            cv2.imwrite(temp_output_file.name, processed_image)
            with open(temp_output_file.name, "rb") as f:
                st.download_button(label="Click to Download", data=f,
                                   key="processed_image.jpg", file_name="processed_image.jpg")
