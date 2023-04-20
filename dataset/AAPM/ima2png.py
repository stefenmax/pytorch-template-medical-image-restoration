import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os
# patient name
name = 'L506'

# Path to the folder containing the DICOM images
folder_path = "./IMA/Test_Image_data/low_dose" + "/" + name + "/quarter_3mm_sharp/"
output_folder = "./png_hu/Test/low_dose"
# List all the files in the folder
counter = 1
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for filename in os.listdir(folder_path):
    # Check if the file is a DICOM image with a .ima suffix
    if filename.endswith(".IMA"):
        # Load the DICOM image using pydicom
        ds = pydicom.dcmread(os.path.join(folder_path, filename))
        # convert the pixel values to Hounsfield units (HU)
        hu_image = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        # set the window level and width in Hounsfield units
        level = 40
        width = 400

        # compute the minimum and maximum HU values of the window
        min_hu = level - 0.5 * width
        max_hu = level + 0.5 * width

        # apply the window to the HU image data
        hu_image_windowed = np.clip(hu_image, min_hu, max_hu)
        hu_image_windowed = (hu_image_windowed - min_hu) / width
        # Convert the image data to a NumPy array
        image = hu_image_windowed.astype(np.float32)

        # Normalize the image data to the range [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        plt.imshow(image, cmap='gray')

        # Remove the axis and axis labels
        plt.axis('off')
        # Save the image to a PNG file with a four-digit numerical name in the output folder
        plt.savefig(os.path.join(output_folder, str(name) + "_" + str(counter).zfill(4) + ".png"), dpi=300, bbox_inches="tight", pad_inches = -0.1)

        # Clear the current figure
        plt.clf()

        # Increment the counter variable
        counter += 1