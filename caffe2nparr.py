import caffe
import pickle
import numpy as np
import sys
caffemodel = caffe.proto.caffe_pb2.NetParameter()
caffemodel.MergeFromString(open(sys.argv[1], 'rb').read())
for layer in caffemodel.layer:
    print layer.name,
    if layer.blobs:
        print(' has blobs')
        for i, blob in enumerate(layer.blobs):
            with open(sys.argv[2] + '/' + layer.name + '_' + str(i) + '.bin', 'wb') as f:
                shape = np.array(blob.shape.dim)
                data = np.array(blob.data)
                to_write = {
                    'shape': shape,
                    'data': data
                }
                pickle.dump(to_write, f)
    else:
        print(' does not have blobs')