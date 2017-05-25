import argparse

import numpy as np
import matplotlib.pyplot as plt

import sammon as sn
import sammon_tf as sntf

argparser = argparse.ArgumentParser('Embedding points around tetrahedron.')
argparser.add_argument('framework', help='framework (theano or tensorflow)')
argparser.add_argument('--output_figurename',
                       default='embedded_tetrahedron.png',
                       help='file name of the output plot')

args = argparser.parse_args()

tetrahedron_points = [np.array([0., 0., 0.]),
                      np.array([1., 0., 0.]),
                      np.array([np.cos(np.pi/3), np.sin(np.pi/3), 0.]),
                      np.array([0.5, 0.5/np.sqrt(3), np.sqrt(2./3.)])]

sampled_points = np.concatenate([np.random.multivariate_normal(point, np.eye(3)*0.0001, 10)
                                 for point in tetrahedron_points])

init_points = np.concatenate([np.random.multivariate_normal(point[:2], np.eye(2)*0.0001, 10)
                              for point in tetrahedron_points])

if args.framework == 'theano':
    embed_points = sn.sammon_embedding(sampled_points, init_points, tol=1e-4)
else:
    embed_points = sntf.sammon_embedding(sampled_points, init_points, tol=1e-4)

X, Y = embed_points.transpose()
plt.plot(X, Y, 'x')
plt.savefig(args.output_figurename)