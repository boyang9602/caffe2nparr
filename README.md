This is a simple code snippet to convert the `caffemodel` parameters to numpy arrays. 
It needs an environment with python2.7 and caffe. I would suggest to use the [caffe docker](https://github.com/BVLC/caffe/tree/master/docker). 

First, create a folder for saving the weights.  
Then run the command below:  
```bash
docker run -ti -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) bvlc/caffe:cpu python caffe2nparr.py YOURMODEL.caffemodel path_to_save_the_weights
```
Each weights blob will be saved as a single binary file with the following format:
```python
{
    'shape': numpy array
    'data': numpy array
}
```
To load a binary file, 
```python
import pickle
with open('filename.bin', 'rb') as f:
    content = pickle.load(f, encoding='latin1')
    shape = content['shape']
    data = contant['data']
    weights = data.reshape(shape)
```
