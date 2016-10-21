import numpy as np
import matplotlib.pyplot as plt

import sammon as sn

tetrahedron_points = [np.array([0., 0., 0.]),
                      np.array([1., 0., 0.]),
                      np.array([np.cos(np.pi/3), np.sin(np.pi/3), 0.]),
                      np.array([0.5, 0.5/np.sqrt(3), np.sqrt(2./3.)])]

sampled_points = np.concatenate([np.random.multivariate_normal(point, np.eye(3), 10)
                                 for point in tetrahedron_points])

init_points = np.concatenate([np.random.multivariate_normal(point[:2], np.eye(2), 10)
                              for point in tetrahedron_points])

embed_points = sn.sammon_embedding(sampled_points, init_points)
plt.plot(embed_points)