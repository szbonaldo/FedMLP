import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import where
from sklearn.manifold import TSNE


color_map = ['r', 'y', 'k', 'g', 'b', 'm', 'c', 'peru'] 
# color_map = ['r', 'g']


def plot_embedding_2D(data, label, title, rnd):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(len(np.unique(label))):
        list = data[label == i]
        plt.scatter(list[:, 0], list[:, 1], marker='o', s=1, color=color_map[i], label='class:{}'.format(i))
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig('proto_fig/' + f'rnd:{rnd}' + title + '.png')
    # plt.show()
    plt.clf()
    return fig


def tnse_Visual(data, label, rnd, title):
    n_samples, n_features = data.shape

    print('Begining......')

    tsne_2D = TSNE(n_components=2, init='pca', random_state=0, perplexity=5)
    result_2D = tsne_2D.fit_transform(data)

    print('Finished......')
    fig1 = plot_embedding_2D(result_2D, label, title, rnd)  # 将二维数据用plt绘制出来


if __name__ == '__main__':
    label = torch.randint(high=2, low=0, size=(1000, ))
    print(label)
    data = torch.rand(1000, 512)
    print(data.shape)
    tnse_Visual(data, label, 1, '666')

