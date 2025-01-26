import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
# import nrrd
import nibabel as nib

output_size =[112, 112, 80]

# paths.
isles_data_dir = '/home/pranav/Downloads/ISLES-2022'
example_case = 9 

# Set images path.
dwi_path = os.path.join(isles_data_dir, 'rawdata', 'sub-strokecase{}'.format("%04d" %example_case), 'ses-0001',
                    'sub-strokecase{}_ses-0001_dwi.nii.gz'.format("%04d" % example_case))
adc_path = dwi_path.replace('dwi', 'adc')
flair_path = dwi_path.replace('dwi', 'flair')
mask_path = dwi_path.replace('rawdata', 'derivatives').replace('dwi', 'msk')

def convert_h5(directory_path = '/home/pranav/DH602_VNet-BAM/dataset/MSD-spleen/Spleen/imagesTr'):
    listt = glob(f'{directory_path}/*.nii.gz')
    for item in tqdm(listt):
        # data = nib.load(item) #"/home/pranav/DH602_VNet-BAM/dataset/MSD-spleen/Spleen/imagesTr/spleen_20.nii.gz"
        image = nib.load(item).get_fdata()
        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        # if 'imagesTs' in directory_path:
        #     with h5py.File(item.replace('.nii.gz', '.h5'), 'w') as f:
        #         f.create_dataset('image', data=image, compression="gzip")
        #     continue
        label = np.ones(image.shape) if 'imagesTs' in directory_path else nib.load(item.replace('imagesTr', 'labelsTr')).get_fdata()
        # pseudo labels for consistency
        # label = nib.load(item.replace('imagesTr', 'labelsTr')).get_fdata()
        # image, img_header = nrrd.read(item)
        # label, gt_header = nrrd.read(item.replace('lgemri.nrrd', 'laendo.nrrd'))
        label = (label != 0).astype(np.uint8)
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
        print(label.shape, f'orginal:({w}, {h}, {d})')
        with h5py.File(item.replace('.nii.gz', '.h5'), 'w') as f:
            f.create_dataset('image', data=image, compression="gzip")
        with h5py.File(item.replace('images', 'labels').replace('.nii.gz', '.h5'), 'w') as f:
            f.create_dataset('label', data=label, compression="gzip")
        # f.close()
    return

if __name__ == '__main__':
    train_path = '/home/pranav/DH602_VNet-BAM/dataset/MSD-spleen/Spleen/imagesTr'
    test_path = '/home/pranav/DH602_VNet-BAM/dataset/MSD-spleen/Spleen/imagesTs'
    convert_h5(train_path)
    convert_h5(test_path)