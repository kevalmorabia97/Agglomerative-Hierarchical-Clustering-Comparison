from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

###################### DATASETS ###########################
### DATASET 1 ###
# n_clusters = 10
# f = open('data/handwritten_digits.txt','r')

### DATASET 2 ###
# n_clusters = 3
# f = open('data/seeds_dataset.txt','r')

### DATASET 3 ###
n_clusters = 3
f = open('data/random_3_clusters.txt','r')

###################### READ DATA ##########################
data = []
correct_labels = []
while True:
    line = f.readline().replace("\n","")
    if line=='': break
    line = line.split(',')
    tmp = []
    # print(line)
    for i in line: tmp.append(float(i))
    data.append(tmp[:-1])
    correct_labels.append(int(tmp[-1]))
    
###################### PREPROCESS DATA ####################    
X = normalize(data,axis=0)
pca = PCA(n_components=2)
X = pca.fit_transform(X)
# print(X)

###################### PLOT DATA ##########################
no_of_subplots = 4
plt.figure(figsize=(10, no_of_subplots))
plot_no=0
for linkage in ['complete','average','ward','actual_data']:
    if linkage=='actual_data':
        labels = correct_labels
    else:
        model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
        model.fit(X)
        labels = model.labels_
    
    plot_no+=1
    plt.subplot(1, no_of_subplots, plot_no)
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title('linkage=%s' % (linkage))
    # print('linkage:',linkage)
    # print(labels,"\n")

plt.show()