from google.colab import files

# Install the PyDrive wrapper & import libraries.
# This only needs to be done once in a notebook.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once in a notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Create & upload a file.
uploaded = drive.CreateFile({'title': 'faceid_big_rgbd.h5'})
uploaded.SetContentFile('faceid_big_rgbd.h5')
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))

# Install the PyDrive wrapper & import libraries.
# This only needs to be done once per notebook.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Download a file based on its file ID.
#
# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
file_id = '17Lo_ZxYcKO751iYs4XRyIvVXME8Lyc75'
downloaded = drive.CreateFile({'id': file_id})
#print('Downloaded content "{}"'.format(downloaded.GetContentString()))

downloaded.GetContentFile('pesi.h5')

from keras.models import load_model

model_final.load_weights('pesi.h5')

"""# Raw output.
Here we create a model that outputs the embedding of an input face instead of the distance between two embeddings, so we can map those outputs.
"""

im_in1 = Input(shape=(200,200,4))
#im_in2 = Input(shape=(200,200,4))

feat_x1 = model_top(im_in1)
#feat_x2 = model_top(im_in2)



model_output = Model(inputs = im_in1, outputs = feat_x1)

model_output.summary()

adam = Adam(lr=0.001)

sgd = SGD(lr=0.001, momentum=0.9)

model_output.compile(optimizer=adam, loss=contrastive_loss)

cop = create_couple_rgbd("faceid_val/")
model_output.predict(cop[0].reshape((1,200,200,4)))

def create_input_rgbd(file_path):
  #  print(folder)
    mat=np.zeros((480,640), dtype='float32')
    i=0
    j=0
    depth_file = file_path
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
    plt.figure(figsize=(8,8))
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mat_small)
    plt.show()
    plt.figure(figsize=(8,8))
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()



    full1 = np.zeros((200,200,4))
    full1[:,:,:3] = img[:,:,:3]
    full1[:,:,3] = mat_small

    return np.array([full1])

"""# Data visualization.
Here we store the embeddings for all the faces in the dataset. Then, using both **t-SNE** and **PCA**, we visualize the embeddings going from 128 to 2 dimensions.
"""

outputs=[]
n=0
for folder in glob.glob('faceid_train/*'):
  i=0
  for file in glob.glob(folder + '/*.dat'):
    i+=1
    outputs.append(model_output.predict(create_input_rgbd(file).reshape((1,200,200,4))))
  print(i)
  n+=1
  print("Folder ", n, " of ", len(glob.glob('faceid_train/*')))
print(len(outputs))

outputs= np.asarray(outputs)
outputs = outputs.reshape((-1,128))
outputs.shape

import sklearn
from sklearn.manifold import TSNE

X_embedded = TSNE(2).fit_transform(outputs)
X_embedded.shape

import numpy as np
from sklearn.decomposition import PCA

X_PCA = PCA(3).fit_transform(outputs)
print(X_PCA.shape)

#X_embedded = TSNE(2).fit_transform(X_PCA)
#print(X_embedded.shape)

import matplotlib.pyplot as plt

color = 0
for i in range(len((X_embedded))):
  el = X_embedded[i]
  if i % 51 == 0 and not i==0:
    color+=1
    color=color%10
  plt.scatter(el[0], el[1], color="C" + str(color))

"""# Distance between two arbitrary RGBD pictures."""

file1 = ('faceid_train/(2012-05-16)(154211)/015_1_d.dat')
inp1 = create_input_rgbd(file1)
file1 = ('faceid_train/(2012-05-16)(154211)/011_1_d.dat')
inp2 = create_input_rgbd(file1)

model_final.predict([inp1, inp2])
