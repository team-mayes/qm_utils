import sys
import os
from PIL import Image
import PIL
import numpy as np

MOLECULE = 'bxyl'

# # # Directories # # #
#region
QM_1_DIR = os.path.dirname(__file__)

# root of project
QM_0_DIR = os.path.dirname(QM_1_DIR)

PROG_DATA_DIR = os.path.join(QM_0_DIR, 'pucker_prog_data')

MET_COMP_DIR = os.path.join(PROG_DATA_DIR, 'method_comparison')
SV_DIR = os.path.join(PROG_DATA_DIR, 'spherical_kmeans_voronoi')

MOL_DATA_DIR = os.path.join(MET_COMP_DIR, MOLECULE)
TS_DATA_DIR = os.path.join(MOL_DATA_DIR, 'transitions_state')
UNMERGED_DIR = os.path.join(os.path.join(TS_DATA_DIR, 'merge_img'), 'unmerged')
MERGED_DIR = os.path.join(os.path.join(TS_DATA_DIR, 'merge_img'), 'merged')
#endregion


def merge_img(img_list):
    for i in range(len(img_list)):
        img_list[i] = os.path.join(UNMERGED_DIR, img_list[i])

    images = map(Image.open, img_list)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    new_im.save(os.path.join(MERGED_DIR, (MOLECULE + img_list[0].split('-')[4])) + '.png')

def merge_img2(img_list):
    for i in range(len(img_list)):
        img_list[i] = os.path.join(UNMERGED_DIR, img_list[i])

    imgs = [PIL.Image.open(i) for i in img_list]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

    # save that beautiful picture
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save(os.path.join(MERGED_DIR, (MOLECULE + '-' + img_list[0].split('-')[4])))


def main():
    UNMERGED_LIST = os.listdir(UNMERGED_DIR)
    excluded_list = []

    for i in range(len(UNMERGED_LIST)):
        lm_group = UNMERGED_LIST[i].split('-')[4]

        img_list = []

        for j in range(len(UNMERGED_LIST)):
            if UNMERGED_LIST[j].split('-')[4] == lm_group and lm_group not in excluded_list:
                img_list.append(UNMERGED_LIST[j])

        if len(img_list) > 0:
            merge_img2(img_list)

        excluded_list.append(lm_group)

    return 0

if __name__ == '__main__':
    status = main()
    sys.exit(status)
