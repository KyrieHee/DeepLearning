import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image

def create_couple(file_name):
    folder = np.random.choice(glob.glob(file_path + "*"))
    while folder == "datalab":
        folder = np.random.choice(glob.glob(file_path + "*"))
        #print(floder)

    mat = np.zeros((480, 640), dtype = 'float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat[i][j] = float(int(val))
                j += 1
                j = j%640
            i += 1
        mat = np.asarray(mat)
    mat_small = mat[140:340, 220:420]
    mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)

    plt.imshow(mat_small)
    plt.show()

    mat2 = np.zeros((480, 640), dtype = 'float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat2[i][j] = float(int(val))
                j += 1
                j = j%640
            i += 1
        mat2 = np.asarray(mat2)
    mat2_small = mat2[140:340, 220:420]
    mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)

    plt.imshow(mat2_small)
    plt.show()
    return np.array([mat_small, mat2_small])

print(create_couple("faceid_train/"))

def create_couple_rgbd(file_path):
    folder = np.random.choice(glob.glob(file_path + "*"))
    while folder == "datalab":
        folder = np.random.choice(glob.glob(file_path + "*"))
    print(floder)

    mat = np.zeros((480, 640), dtype = 'float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))

    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val =1200
                mat[i][j] = float(int(val))
                j += 1
                j = j%640

            i += 1
        mat = np.asarray(mat)
    mat_small = mat[140:340, 220:420]
    img = Image.open(depth_file[:-5] + "c.bmp")
    img.thumbnail((640, 480))
    img = np.asarray(img)
    img = img[140:340, 220:420]
    mat_small(mat_small - np.mean(mat_small)) / np.max(mat_small)

    plt.imshow(mat_small)
    plt.show()
    plt.imshow(img)
    plt.show()

    mat2 = np.zeros((480, 640), dtype = 'float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(floder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat2[i][j] = float(int(val))
                j += 1
                j = j%640

            i += 1
        mat2 = np.asarray(mat2)
    mat2_small = mat2[140:340, 220:420]
    img2 = Image.open(depth_file[:-5] + "c.bmp")
    img2.thumbnail((640, 480))
    img2 = np.asarry(img2)
    img2 = img2[160:360, 240:440]

    plt.imshow(img2)
    plt.show()

    mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)
    plt.imshow()

    full1 = np.zeros((200, 200, 4))
    full1[:, :, :3] = img[:, :, :3]
    fill1[:, :, 3] = mat_small

    full1 = np.zeros((200, 200, 4))
    full1[:, :, :3] = img2[:, :, :3]
    fill1[:, :, 3] = mat2_small
    return np.array([fill1, full2])

create_couple_rgbd("faceid_val/")

def create_wrong(file_path):
    folder = np.random.choice(glob.glob(file_path + "*"))
    while floder == "fatalab":
        folder = np.random.choice(glob.glob(file_path + "*"))
    mat = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random_choice(glob.glob(folder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200  or int(val) == -1: val = 1200
                mat[i][j] = float(int(val))
                j += 1
                j = j%640

            i += 1
        mat = np.asarry(mat)
    mat_small = mat[140:340, 220:420]
    mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)

    plt.imshow(mat_small)
    plt.show()

    folder2 = np.random.choice(glob.glob(file_path + "*"))
    while folder==folder2 or folder2=="datalab": #it activates if it chose the same folder
        folder2=np.random.choice(glob.glob(file_path + "*"))
    mat2=np.zeros((480,640), dtype='float32')
    i=0
    j=0
    depth_file = np.random.choice(glob.glob(folder2 + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val= 1200
                mat2[i][j]=float(int(val))
                j+=1
                j=j%640

            i+=1
        mat2 = np.asarray(mat2)
    mat2_small=mat2[140:340,220:420]
    mat2_small=(mat2_small-np.mean(mat2_small))/np.max(mat2_small)
    plt.imshow(mat2_small)
    plt.show()


    return np.array([mat_small, mat2_small])

create_wrong("faceid_train/")

def create_wrong_rgbd(file_path):
    folder=np.random.choice(glob.glob(file_path + "*"))
    while folder == "datalab":
      folder=np.random.choice(glob.glob(file_path + "*"))
    mat=np.zeros((480,640), dtype='float32')
    i=0
    j=0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val= 1200
                mat[i][j]=float(int(val))
                j+=1
                j=j%640

            i+=1
        mat = np.asarray(mat)
    mat_small=mat[140:340,220:420]
    img = Image.open(depth_file[:-5] + "c.bmp")
    img.thumbnail((640,480))
    img = np.asarray(img)
    img = img[140:340,220:420]
    mat_small=(mat_small-np.mean(mat_small))/np.max(mat_small)
    plt.imshow(img)
    plt.show()
    plt.imshow(mat_small)
    plt.show()
    folder2=np.random.choice(glob.glob(file_path + "*"))
    while folder==folder2 or folder2=="datalab": #it activates if it chose the same folder
        folder2=np.random.choice(glob.glob(file_path + "*"))
    mat2=np.zeros((480,640), dtype='float32')
    i=0
    j=0
    depth_file = np.random.choice(glob.glob(folder2 + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val= 1200
                mat2[i][j]=float(int(val))
                j+=1
                j=j%640

            i+=1
        mat2 = np.asarray(mat2)
    mat2_small=mat2[140:340,220:420]
    img2 = Image.open(depth_file[:-5] + "c.bmp")
    img2.thumbnail((640,480))
    img2 = np.asarray(img2)
    img2 = img2[140:340,220:420]
    mat2_small=(mat2_small-np.mean(mat2_small))/np.max(mat2_small)
    plt.imshow(img2)
    plt.show()
    plt.imshow(mat2_small)
    plt.show()
    full1 = np.zeros((200,200,4))
    full1[:,:,:3] = img[:,:,:3]
    full1[:,:,3] = mat_small

    full2 = np.zeros((200,200,4))
    full2[:,:,:3] = img2[:,:,:3]
    full2[:,:,3] = mat2_small
    return np.array([full1, full2])

create_wrong_rgbd("faceid_val/")[0].shape
