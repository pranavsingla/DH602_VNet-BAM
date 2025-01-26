import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

data = nib.load("./Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz")#("./Spleen/labelsTr/spleen_20.nii.gz") #/home/pranav/DH602_VNet-BAM/dataset/MSD-spleen/Task04_Hippocampus/imagesTr/._hippocampus_001.nii.gz
data = data.get_fdata()
imgdata = data
data = nib.load("./Task04_Hippocampus/labelsTr/hippocampus_001.nii.gz")
data = data.get_fdata()
print(data.shape, data)
label = data#.astype(np.uint8)
# label = (data == 1.0).astype(np.uint8)
# label2 = (data != 0).astype(np.uint8)
print(label.shape, label)
slice_number = 13
image = data[:, :, slice_number]  # Access the slice along the third dimension
print('n\n\n',np.unique(label))
# print(np.unique(label==label2))

plt.imshow(image, cmap='gray')  # Adjust the colormap as needed
plt.axis('off')  # Turn off axis

# Save the image to a file
plt.savefig('hippocampus_visualization_mask.png', bbox_inches='tight')  # Adjust the filename and other parameters as needed

image = imgdata[:, :, slice_number]  # Access the slice along the third dimension
print('n\n\n',np.unique(image))
# print(np.unique(label==label2))

plt.imshow(image, cmap='gray')  # Adjust the colormap as needed
plt.axis('off')  # Turn off axis

# Save the image to a file
plt.savefig('hippocampus_visualization.png', bbox_inches='tight')  # Adjust the filename and other parameters as needed
