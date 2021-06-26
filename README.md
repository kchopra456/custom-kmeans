## Intent
- Did a test project to create custom/ improved k-means algo.
- Did not use sklearn implementation of algo.
- Create a jupyter notebook to show modeling steps in details.


## Initial Start
- This assignment was done using docker container.
- In order to play with Jupyter Notebook, launch docker container with command provided below.
- Start Jupyter container
    ```shell script
    docker run --rm --name ds-notebook -v $(pwd):~/work jupyter/scipy-notebook:6d42503c684f
    ```

## Content Explanation
- __custom_kmeans.py__: Implementation of kmeans
- __extmath.py__: Mathematical functions necessary for kmeans
- __kmeans_assignment.ipynb__: Ipython notebook contains __Deliverables__ and the use of custom kmeans algorithm.
- __kmeans_assignment.html__: contains __Deliverables__ and the use of custom kmeans algorithm (_generated from notebook_)
- __output...___: files for graphs and csv, generated part of solution
