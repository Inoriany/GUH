import streamlit as st
import keras
from PIL import Image
import numpy as np
import csv

def bounded_relu(x):

  lower_bounded = keras.backend.maximum(x, 0)
  bounded = keras.backend.minimum(x, 100)

  return bounded

im_size = 128

inp = keras.layers.Input(shape = (im_size,im_size,3))

pre_trained_model = keras.applications.EfficientNetB7(include_top = False , input_tensor=inp , weights = 'imagenet' , input_shape = (im_size,im_size,3))
for layer in pre_trained_model.layers[:-len(pre_trained_model.layers)//2]:
    layer.trainable = False

o = keras.layers.GlobalAveragePooling2D()(pre_trained_model.output)
o = keras.layers.Dense(256, activation = 'relu')(o)
o = keras.layers.BatchNormalization()(o)
o = keras.layers.Dropout(0.4)(o)
z = keras.layers.Dense(128, activation = 'relu')(o)
z = keras.layers.BatchNormalization()(z)
z = keras.layers.Dropout(0.4)(z)
z = keras.layers.Dense(64, activation = 'relu')(z)
z = keras.layers.BatchNormalization()(z)
z = keras.layers.Dense(1 , activation = bounded_relu)(z)

model = keras.models.Model(inputs=inp, outputs=z)

model.compile(loss = 'mse', optimizer= 'adam')

top_bar = """
<div style="background-color: #32612D; color: #ffffff; padding: 10px; text-align: center;">
    <h1>TimberTrack App</h1>
    <p>Shown are the stats on Deforestation based on the files provided<p>
</div>
"""

st.markdown(top_bar, unsafe_allow_html=True)

st.write("") 
st.write("""
Upload Selection of Images that show change in forest cover over time

<- Click sidebar to add files
""")

csv_file = 'Stats.csv'

# Open the CSV file
with open(csv_file, mode='r') as file:

    csv_reader = csv.reader(file)
    csv_list = list(csv_reader)
    leng = len(csv_list)
    print(leng)

    this_CSV = []
    y_values = []

    # Read the CSV file until an empty line is encountered
    for row in csv_reader:
        st.write(row)
        if list(row) == ["", 0]:  # Check if the row is empty
            st.line_chart({'y': y_values})
            this_CSV = []  # Clear this_CSV
            continue
        else:
            this_CSV.append(row) 

with st.sidebar.form("my_form"):

    st.header("Name/Coordinates of Forest Data (Images)")
    name = st.text_input("Enter Heading")
    st.write("")
    st.header('Upload file')
    uploaded_files = st.file_uploader("Choose timelapse of Images of Forest", type=['jpg'], accept_multiple_files=True)

   # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

if submitted:

    # To read file as bytes:
    new_data =[]
    keras.applications.DenseNet121()
    model.load_weights('best.h5')
    this_CSV = []
    y_values = []

    for uploaded_file in uploaded_files:
        #IMPLEMENT MODEL LOGIC HERE

        pil_image = Image.open(uploaded_file)
        
        # Convert the PIL image to a numpy array
        image_array = np.array(pil_image)
        output = model.predict(image_array.reshape(-1,128,128,3))

        if output < 0:
            st.write("INVALID INPUT")
        else:
            st.write(float(output))
    
        st.image(pil_image, caption='Uploaded Image.', use_column_width=True)

        new_data.append([name, float(output)])

        y_values.append(float(output))

    st.title(name)
    chart = st.line_chart({'Forest Cover %': y_values}, y = "Forest Cover %")

    new_data.append(["", 0])

    file_path = "Stats.csv"

    # Read the existing data
    existing_data = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        existing_data = list(reader)

    # Append the new data to the existing data
    existing_data.extend(new_data)

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(existing_data)    