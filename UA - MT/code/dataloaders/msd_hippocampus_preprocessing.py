import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import os
# import nrrd
import nibabel as nib
import json

output_size =[112, 112, 80]

def convert_h5(directory_path = '/dataset/MSD-spleen/Task04_Hippocampus/imagesTr'):
    listt = glob(f'{directory_path}/*.nii.gz')
    for item in tqdm(listt):
        # data = nib.load(item) #"/dataset/MSD-spleen/Task04_Hippocampus/imagesTr/spleen_20.nii.gz"
        image = nib.load(item).get_fdata()
        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        # if 'imagesTs' in directory_path:
        #     with h5py.File(item.replace('.nii.gz', '.h5'), 'w') as f:
        #         f.create_dataset('image', data=image, compression="gzip")
        #     continue
        label = np.ones(image.shape) if 'imagesTs' in directory_path else nib.load(item.replace('imagesTr', 'labelsTr')).get_fdata()
        if not os.path.isdir(directory_path.replace('images', 'labels')):
            os.makedirs(directory_path.replace('images', 'labels'))
        # pseudo labels for consistency
        # label = nib.load(item.replace('imagesTr', 'labelsTr')).get_fdata()
        # image, img_header = nrrd.read(item)
        # label, gt_header = nrrd.read(item.replace('lgemri.nrrd', 'laendo.nrrd'))
        label = label.astype(np.uint8)
        # print()
        w, h, d = label.shape

        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image = image[minx:maxx, miny:maxy]
        label = label[minx:maxx, miny:maxy]
        print('image: ', image.shape, 'label: ', label.shape, f'orginal label:({w}, {h}, {d})')
        with h5py.File(item.replace('.nii.gz', '.h5'), 'w') as f:
            f.create_dataset('image', data=image, compression="gzip")
        with h5py.File(item.replace('images', 'labels').replace('.nii.gz', '.h5'), 'w') as f:
            f.create_dataset('label', data=label, compression="gzip")
        # f.close()
    return

if __name__ == '__main__':

    base_path = '../../../dataset/MSD-spleen/Task04_Hippocampus/'
    # Load the JSON data from a file
    with open(f'{base_path}dataset.json', 'r') as f:
        data = json.load(f)

    # Extract train and test lists
    train_list = [entry['image'] for entry in data['training']]
    test_list = data['test']

    train_list += test_list
    test_list = train_list[ : int(len(train_list) * 0.2)]
    train_list = train_list[int(len(train_list) * 0.2) : ]
    # test_list = test_list[ : int((len(test_list) + len(train_list)) * 0.2) - 1]

    # Save the train list to a text file
    with open(f'{base_path}train.list', 'w') as f:
        for item in train_list:
            f.write(f"{item}\n")

    # Save the test list to a text file
    with open(f'{base_path}test.list', 'w') as f:
        for item in test_list:
            f.write(f"{item}\n")

    print("train.list and test.list have been created successfully.")

    train_path = '../../../dataset/MSD-spleen/Task04_Hippocampus/imagesTr'
    test_path = '../../../dataset/MSD-spleen/Task04_Hippocampus/imagesTs'
    convert_h5(train_path)
    convert_h5(test_path)