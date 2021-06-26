import numpy as np
from copy import deepcopy
from extmath import norm
from extmath import check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from typing import Callable, Dict
from functools import partial


def _init_pca(X, k):
    """
    To initialize centroids using PCA,
     refer https://ece.northeastern.edu/fac-ece/jdy/papers/init-kmeans.pdf
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
    k : int, n_clusters

    Returns
    -------
    array, shape(k, n_features)

    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=k).fit(X)
    return pca.components_


def _init_centroids(X, k, init, random_state=None):
    """

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
    k : int, n_clusters
    init : {'random', 'pca', array, shape(k, n_features)}
        Method to use for initialization
    random_state : int, optional
        To generate same random permutation with every execution

    Returns
    -------
    centers : array, shape(k, n_features)
    """
    random_state = check_random_state(random_state)
    if isinstance(init, str) and init.lower() == 'random':
        centers = X[random_state.permutation(X.shape[0])[:k], :]
    elif isinstance(init, str) and init.lower() == 'pca':
        centers = _init_pca(X, k)
    elif hasattr(init, '__array__'):
        # TODO check data type if array to match with X elements.
        centers = np.array(init)
    else:
        raise ValueError(f'the <init> parameter for kmeans should be: '
                         f'random or pca and or ndarray, was passed {init} (type {type(init)})')
    return centers


def _calculate_inertia(X, centers, labels, measure: Callable):
    """
    Measure the summation of distances/error of each sample in a cluster from the centroid;
    the distance measure is calculated using callable <measure>.
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
    centers : array, shape (k, n_features)
    labels : array, shape (n_samples, 1)
    measure : Callable
        callable must return a single value using any of distance measuring norms;
         input is vector <ndarray> difference of point and center.

    Returns
    -------
    inertia : int
    """
    inertia = 0
    for i in range(centers.shape[0]):
        inertia += np.sum(measure(X[labels == i] - centers[i]))
    return inertia


def _map_cluster_class_labels(y, labels) -> Dict:
    """
    Helper function to map the class(tags) to the cluster labels.
    Parameters
    ----------
    y : array (n_samples,)
    labels : array (n_samples,)

    Returns
    -------
    cl_mapping : dict [label:class]
        dict with cluster label a key and class as values, provided by <y>
    """
    if y.shape != labels.shape:
        raise AttributeError(f'shape of the input arrays must be same, provided;'
                             f' <y>: {y.shape}, <labels>: {labels.shape} ')
    unique_class = np.unique(y)
    cl_mapping = {}
    for uc in unique_class:
        cluster = np.bincount(labels[y == uc])
        cl_mapping.update({np.argmax(cluster): uc})
    return cl_mapping


def _kmeans_single(X, k, init, tol, random_state, max_iter, ord_, squared=True):
    """
    Execute kmeans clustering single run.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
    k : int
        number of clusters
    init : {'random', 'pca', array, shape(k, n_features)}
        Method to use for initialization
    tol : float
        Allowed tolerance to operate before declaring convergence
    random_state : int, RandomState or None
        To generate same random permutation with every execution
    max_iter : int
        Number of iterations allowed before declaring convergence
    ord_ :  int
        Measure of Norm applied
    squared : bool, default True
        If ord_=1 squared returns squared sum of rows

    Returns
    -------
    labels : int array, shape(n_samples,)
        Each row represents the cluster label
    inertia : float
        Summation Distance of all ponits to cluster centers
    centers : array, shape (k, n_features)
        Center of clusters
    """
    X = np.array(X)
    n = X.shape[0]

    # Generate initial centroids using <init>
    centroids = _init_centroids(X, k, init, random_state)

    centers = np.zeros(centroids.shape)
    centers_adj = deepcopy(centroids)  # centroids re-calculated after cluster element addition

    labels = np.full(n, -1, np.int)
    distances = np.zeros((n, k))

    # Measure the shift for each iteration
    delta_shift = norm(centers_adj - centers, ord_, squared).sum()

    # Exit, when exhaust max_iter or delta becomes less than tolerance
    while max_iter and delta_shift > tol:
        # Measure the distance to all centroids
        for i in range(k):
            distances[:, i] = norm(X - centers_adj[i], squared=squared, ord_=ord_)

        # Assign samples to closest centroids, cluster
        labels = np.argmin(distances, axis=1)

        centers = deepcopy(centers_adj)
        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_adj[i] = np.mean(X[labels == i], axis=0)
        delta_shift = norm(centers_adj - centers).sum()
        max_iter -= 1

    _norm = partial(norm, ord_=ord_, squared=squared)
    inertia = _calculate_inertia(X, centers, labels, _norm)

    return labels, inertia, centers


class KMeans(BaseEstimator, TransformerMixin):
    """
    KMeans cluster algorithm, read more at
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Parameters
    ----------
    n_clusters : int
        number of clusters
    init : {'random', 'pca', array, shape(k, n_features)}
        Method to use for initialization
    tol : float
        Allowed tolerance to operate before declaring convergence
    random_state : int, RandomState or None, optional
        To generate same random permutation with every execution
    max_iter : int
        Number of iterations allowed before declaring convergence
    norm_ :  int, default 2
        Distance measure to calculate distance from cluster center
        When 1, manhattan distance measure is applied
        When 2, euclidean distance measure is applied
    n_init: int, default 5
        number of job initialized when int='random'

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.

    labels_ : array, shape (n_samples,)
        Labels of each point

    inertia_ : float
        Summation distance of each point from cluster center

    Returns
    -------
    labels : int array, shape(n_samples,)
        Each row represents the cluster label
    inertia : float
        Summation Distance of all ponits to cluster centers
    centers : array, shape (k, n_features)
        Center of clusters
    """

    def __init__(self, n_clusters=4, tol=1e-4, init='random', norm_=2, max_iter=100,
                 random_state=None, n_init=5):
        self.n_clusters = n_clusters
        self.init = init
        self.tol = tol
        self.norm_ = norm_
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_init = n_init

        self._fitted = False

    def _check_init(self, X):
        n_init = self.n_init

        def _warn_set_n_init():
            warnings.warn(f'<init> parameter not set to random,'
                          f' kmeans will perform only 1 init instead of n_init={n_init}',
                          RuntimeWarning)

        if isinstance(self.init, str) and self.init.lower() == 'pca':
            if not 1 < self.n_clusters <= X.shape[1]:
                raise ValueError(f'when kmeans parameter <init> set to `pca`,'
                                 f' n_clusters must be set to range [2, n_features]')
            _warn_set_n_init()
            n_init = 1
        elif hasattr(self.init, '__array__'):
            _warn_set_n_init()
            n_init = 1

        return n_init

    def fit(self, X, y=None):
        """
        Compute k-means clustering
        Parameters
        ----------
        X :  array, shape(n_samples, n_features)
        y : None
            Ignored, kept to keep convention

        Returns
        -------
            estimator
        """

        if self.n_init <= 0:
            raise ValueError(f'value of <n_init> must be greater than zero, provided {self.n_init}')
        n_init = self._check_init(X)

        best_labels, best_inertia, best_centers = None, None, None

        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        for iter_idx in range(n_init):
            labels, inertia, centers = _kmeans_single(X, self.n_clusters, self.init, self.tol,
                                                      seeds[iter_idx], self.max_iter, self.norm_,
                                                      True)

            if best_inertia is None or inertia < best_inertia:
                best_labels, best_inertia, best_centers = labels.copy(), inertia, centers.copy()
                n_iter = iter_idx + 1

        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia

        self._fitted = True

        return self

    def _transform(self, X, y=None):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = norm(X - self.cluster_centers_[i], squared=True, ord_=self.norm_)
        return distances

    def transform(self, X, y=None):
        """
        Transform X into distance space; computes distance of each sample from each cluster center.
        Distance measure use, the one used to initialize estimator
        Parameters
        ----------
        X : array, shape(n_samples, n_features)
        y : None
            Ignored, kept to keep convention

        Returns
        -------
        array, shape(n_samples, n_clusters)
        """
        self.check_if_fitted()
        return self._transform(X)

    def map_cluster_class(self, y):
        """
        Returns class labels, reciprocates the labels_ attribute, but provide class classification
        Parameters
        ----------
        y : array, shape(n_samples,)
            Class labels for point in coordinate space;
             this will be used to determine the class:label mapping

        Returns
        -------
        list generator
            each value corresponds to the class label for the cluster label
        Dict
            dict mapping of [cluster labels: class labels]
        """
        cl_mapping = _map_cluster_class_labels(y, self.labels_)
        f_label = np.argmax(np.bincount(self.labels_))

        def _collect_class(label):
            try:
                return cl_mapping[label]
            except KeyError:
                return f_label

        return map(lambda x: _collect_class(x), self.labels_), cl_mapping

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : array, shape(n_samples, n_features)

        Returns
        -------
        labels : array, shape [n_samples,]
            Cluster label to which coordinate belong to
        """
        distances = np.full((X.shape[0], self.n_clusters), -1)
        for i in range(self.n_clusters):
            distances[:, i] = norm(X - self.cluster_centers_[i], self.norm_, squared=True)
        # Assign all training data to closest center
        return np.argmin(distances, axis=1)

    def check_if_fitted(self):
        if not self._fitted:
            raise RuntimeError(f'model has not been fit on data, call fit method first')

    def score(self, X, y=None):
        """
        Negative Inertia, keeping convention higher values are better;
        Parameters
        ----------
        X : array, shape(n_samples, n_features)
        y : None
            Ignored, keeping convention

        Returns
        -------

        """
        self.check_if_fitted()
        _norm = partial(norm, ord_=self.norm_, squared=True)
        return - _calculate_inertia(X, self.cluster_centers_, self.labels_, _norm)

    def distance_from_center(self, X, norm_, squared=True):
        """
        Regardless of norm_ provided to estimator, allows to get inertia by new norm.
        Parameters
        ----------
        X : array, shape(n_samples, n_features)
        norm_ :  int
            Distance measure to calculate distance from cluster center
            When 1, manhattan distance measure is applied
            When 2, euclidean distance measure is applied
        squared

        Returns
        -------
        inertia : float
            Summation of distance of each coordinate from cluster center
        """
        self.check_if_fitted()
        measure = partial(norm, ord_=norm_, squared=squared)
        return _calculate_inertia(X, self.cluster_centers_, self.labels_, measure)
