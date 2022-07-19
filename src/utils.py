from re import S
import os
import re
import numpy as np
import glob
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal

class utils():
    def __init__(self, path):

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.x_test = x_test.reshape(10000, 784)
        self.x_train = x_train.reshape(60000, 784)
        self.y_test = y_test
        self.y_train = y_train

        self.mnist_dim =  28
        self.conv_out_dim =  26
        self.path = path
        self.kernel_size = 3
        self.dims = [128,256,10]

    def NN(self, train=False):
        num_classes = 10
        input_shape = (784)

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_test = x_test.reshape(10000, 784)
        x_train = x_train.reshape(60000, 784)
        # Scale images t the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # convert class vectors to binary class matrices
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        if train is True:
            model = tf.keras.Sequential(
                [
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
                ]
            )
    
            model.summary()
    
            batch_size = 128
            epochs = 20
    
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
            score = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])
            print(model.predict(x_test[0][np.newaxis,...] ))
    
            model.save("mnist.h5")
        else:
            model = tf.keras.models.load_model('mnist.h5')

        weights = model.get_weights()
        self.weights = weights[::2]
        self.bias = weights[1::2]

        self.layer_data = []
        data = x_test[0].flatten()
        for i in range(3):
            print(np.shape(data), np.shape(weights[2*i]), np.shape(weights[2*i+1]))
            data = data@weights[2*i] + weights[2*i+1]
            if i != 2:
                data[data<0] = 0
            else:
                pass
                # data = np.exp(data) / np.sum(np.exp(data))
            self.layer_data.append(data)
        self.res_mnist = data
        return self.weights, self.bias, self.layer_data, 


    def NN_int_quant(self):

        num_classes = 10
        input_shape = (784, 1)
          
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
          
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.reshape(60000, 784)
        # Scale images t the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
          
        # convert class vectors to binary class matrices
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)


        model = tf.keras.models.load_model('mnist.h5')

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.inference_input_type = tf.uint8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8

        # We set the following converter options to ensure our model is fully quantized.
        # An error should get thrown if there is any ops that can't be quantized.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]

        converter.experimental_new_converter=True
        converter.experimental_new_quantizer=True
        converter.allow_custom_ops = True
        # To use post training quantization we must provide some sample data that will be used to
        # calculate activation ranges for quantization. This data should be representative of the data
        # we expect to feed the model and must be provided by a generator function.
        def generate_repr_dataset():
            for i in range(100):  # 100 samples is all we should need in this example.
                yield [np.expand_dims(x_test[i], axis=0)]

        converter.representative_dataset = generate_repr_dataset
        tflite_model = converter.convert()

        interpreter = tf.lite.Interpreter(model_content=tflite_model)

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        n_test=1
        accuracy_count = 0
        num_test_images = n_test

        data = (x_test*255).astype("uint8")

        for i in range(n_test):
            interpreter.set_tensor(input_details[0]['index'], data[i][np.newaxis, ...].astype(np.uint8))
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            if np.argmax(output_data) == np.argmax(y_test[i]):
                accuracy_count += 1

            print(output_data)
        print(f"Test accuracy quantized: {accuracy_count / num_test_images:.3f}")


        tensor_details = interpreter.get_tensor_details()
        tens = []
        scal = []
        zero_p = []
        for dict in tensor_details:
            i = dict['index']
            shape = dict['shape']
            shape_signature = dict['shape_signature']
            dtype = dict['dtype']
            
            tensor_name = dict['name']
            scales = dict['quantization_parameters']['scales']
            zero_points = dict['quantization_parameters']['zero_points']
            q_dims = dict['quantization_parameters']['quantized_dimension']
            sparsity_parameters = dict['sparsity_parameters']
            tensor = interpreter.tensor(i)()

            print(i, type, tensor_name, scales.shape, zero_points.shape, tensor.shape, shape, shape_signature, q_dims, sparsity_parameters)
            tens.append(tensor)
            scal.append(scales)
            zero_p.append(zero_points)
            
            
        im = tens[0][0,:,0] 
             
        return tens, scal, zero_p

    def conv(self, input, kernel):
        input = np.reshape(input, (28,28)).astype('float32')
        self.conv_out = signal.convolve2d(input, kernel, mode='valid').astype("int32")
        self.kernel = kernel

    def MM(self, input, dim):
        self.mm_dim = dim
        self.weights_vec = np.random.randint(-2, 2, size = (dim, self.mnist_dim**2))
        print(self.weights_vec)
        # np.savetxt("weights.txt",  self.weights_vec.astype(np.int8), fmt='%i')
        self.mm_out = self.weights_vec@input


    def write_mnist(self):
        
        self.weights_int[0] = np.pad(self.weights_int[0], ((0,0),(0,128)))
        self.weights_int[2] = np.pad(self.weights_int[2], ((0,0),(0,246)))

        # self.weights_int[2] = np.pad(self.weights_int[2], ((0,0),(0,6)))
        
        w = np.array(self.weights_int[0]).flatten()
        b = np.array(self.bias_int[0]).flatten()
        res = np.array(self.out[0]).flatten()

        x = self.x_test//32
        for i in range(2):
            w = np.append(w, self.weights_int[i+1].flatten())
            b = np.append(b, self.bias_int[i+1].flatten())
            res = np.append(res, self.out[i+1].flatten())

        w1 = np.array(self.weights_int[0]).flatten().astype(np.int8)
        w2 = np.array(self.weights_int[1]).flatten().astype(np.int8)
        w3 = np.array(self.weights_int[2]).flatten().astype(np.int8)

        n_weights1 = np.shape(w1)[0]
        n_weights2 = np.shape(w2)[0]
        n_weights3 = np.shape(w3)[0]


        w = w.astype(np.int8)
        b = b.astype(np.int8)
        res = res.astype(np.int8)



        n_weights = np.shape(w)[0]
        n_bias = np.shape(b)[0]

        with open(self.path+'/mnist.h', 'w') as file:
            file.writelines('/* HW AI HLS, autogenerated File, Michael Schmid, {} */ \n \n'.format(datetime.now().strftime("%d/%m/%Y, %H:%M:%S")))
             
            file.writelines('#include <stdint.h> \n')
            file.writelines('\n')


            # file.writelines('static const int8_t weights[{}] = \n'.format(n_weights))
            # file.writelines('  {')
             
            # for j in range(n_weights):
            #     if j == n_weights -1:
            #         file.writelines('{}'.format(w[j]))
            #     else:
            #         if j % self.mnist_dim == 0:
            #             file.writelines('\n        {},'.format(w[j]))
            #         else:
            #             file.writelines('{},'.format(w[j]))
            # file.writelines('  };\n')
            # file.writelines('\n')
            
            file.writelines('static const int8_t weights1[{}] = \n'.format(n_weights1))
            file.writelines('  {')
             
            for j in range(n_weights1):
                if j == n_weights1 -1:
                    file.writelines('{}'.format(w1[j]))
                else:
                    if j % self.mnist_dim == 0:
                        file.writelines('\n        {},'.format(w1[j]))
                    else:
                        file.writelines('{},'.format(w1[j]))
            file.writelines('  };\n')
            file.writelines('\n')
            
            
            file.writelines('static const int8_t weights2[{}] = \n'.format(n_weights2))
            file.writelines('  {')
             
            for j in range(n_weights2):
                if j == n_weights2 -1:
                    file.writelines('{}'.format(w2[j]))
                else:
                    if j % self.mnist_dim == 0:
                        file.writelines('\n        {},'.format(w2[j]))
                    else:
                        file.writelines('{},'.format(w2[j]))
            file.writelines('  };\n')
            file.writelines('\n')
            
            file.writelines('static const int8_t weights3[{}] = \n'.format(n_weights3))
            file.writelines('  {')
             
            for j in range(n_weights3):
                if j == n_weights3 -1:
                    file.writelines('{}'.format(w3[j]))
                else:
                    if j % self.mnist_dim == 0:
                        file.writelines('\n        {},'.format(w3[j]))
                    else:
                        file.writelines('{},'.format(w3[j]))
            file.writelines('  };\n')
            file.writelines('\n')
            
    
            file.writelines('static const int8_t bias[{}] = \n'.format(n_bias))
            file.writelines('  {')
             
            for j in range(n_bias):
                if j == n_bias -1:
                    file.writelines('{}'.format(b[j]))
                else:
                    if j % self.mnist_dim == 0:
                        file.writelines('\n        {},'.format(b[j]))
                    else:
                        file.writelines('{},'.format(b[j]))     
            file.writelines('  };\n')
            file.writelines('\n')
            
            file.writelines('static const int8_t im[{}] = \n'.format(self.mnist_dim**2))
            file.writelines('  {')
             
            for j in range(self.mnist_dim**2):
                if j == self.mnist_dim**2 -1:
                    file.writelines('{}'.format(x[0,j]))
                else:
                    if j % self.mnist_dim == 0:
                        file.writelines('\n        {},'.format(x[0,j]))
                    else:
                        file.writelines('{},'.format(x[0,j]))
            file.writelines('  };\n')
            file.writelines('\n')
        
            file.writelines('static const int8_t res_layers[{}] = \n'.format(np.sum(self.dims)))
            file.writelines('  {')
             
            for j in range(np.sum(self.dims)):
                if j == np.sum(self.dims) -1:
                    file.writelines('{}'.format((res[j])))
                else:
                    if j % self.mnist_dim == 0:
                        file.writelines('\n        {},'.format(res[j]))
                    else:
                        file.writelines('{},'.format(res[j]))
            file.writelines('  };\n')
            file.writelines('\n')
            
            file.writelines('static const int8_t scales[{}] = \n'.format(len(self.log_scales)))
            file.writelines('  {')
             
            for j in range(3):
                if j == np.sum(np.shape(self.dims)[0]) -1:
                    file.writelines('{}'.format((self.log_scales[j])))
                else:
                    if j % self.mnist_dim == 0:
                        file.writelines('\n        {},'.format(self.log_scales[j]))
                    else:
                        file.writelines('{},'.format(self.log_scales[j]))
            file.writelines('};\n')
            file.writelines('\n')




    def rename(self):

        for i, path in enumerate(sorted(glob.glob('dump_results/dump_results_0/*.txt'))):
            filename = (os.path.basename(path))
            if filename[12].isnumeric() == False and filename[6:11] == "dense":
                filename = filename[:12] + "0_" + filename[12:]
                os.rename(path, 'dump_results/dump_results_0/' + filename)

        for i, path in enumerate(sorted(glob.glob('dump_results/dump_results_weights/*.txt'))):
            filename = (os.path.basename(path))
            if filename[12].isnumeric():
                pass
            else:
                filename = filename[:12] + "0_" + filename[12:]
                os.rename(path, 'dump_results/dump_results_weights/' + filename)

    def FullyVitis(self):

        bias = []
        weight = []
        data = []
        self.out = []
        test = []
        # scales = [1024, 256, 1024]
        scales = [512, 256, 256]

        self.log_scales = [9, 8, 8]

        layers = [784,128,256,10]
        
        for filename in sorted(glob.glob("dump_results/dump_results_weights/quant_dense_*_bias.txt")):
        # for filename in sorted(glob.glob("dump_results/dump_results_weights/quant_dense_*_bias_float.txt")):
            bias.append(np.loadtxt(filename))
        for filename in sorted(glob.glob("dump_results/dump_results_weights/quant_dense_*_kernel.txt")):
        # for filename in sorted(glob.glob("dump_results/dump_results_weights/quant_dense_*_kernel_float.txt")):
            weight.append(np.loadtxt(filename))
            
        for i in range(3):
           weight[i] = weight[i].reshape(layers[i],layers[i+1])

        self.weights_int = weight
        self.bias_int = bias

        err = 0

        for i in range(10000):
            data = self.x_test[i]//32
            # self.out.append(data)
            for j in range(3):
                test.append(data@weight[j]+bias[j])
                data = (data@weight[j]+bias[j])//scales[j]
                # data = (data@weight[j]+bias[j])/scales[j]
                if j != 2:
                    data = data*(data>0)
                self.out.append(data)
            res = np.argmax(data)
            if res != self.y_test[i]:
                err +=1
        print("acc {}".format(1-err/10000))
        
        mins= []
        maxs = []
        for i in range(30000):
            mins.append(np.min(self.out[i]))
            maxs.append(np.max(self.out[i]))

        print(np.min(mins), np.max(maxs))
        
        mins= []
        maxs = []
        for i in range(30000):
            mins.append(np.min(test[i]))
            maxs.append(np.max(test[i]))

        print(np.min(mins), np.max(maxs))

        return self.out, bias, weight, test


    def pynq_dpu(self):
        
        from tensorflow_model_optimization.quantization.keras import vitis_quantize
        
        num_classes = 10
        input_shape = (784)

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_test = x_test.reshape(10000, 784)
        x_train = x_train.reshape(60000, 784)
        # Scale images t the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # convert class vectors to binary class matrices
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)


        inputs = tf.keras.Input(shape=input_shape)
        # x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist')
        model.summary()

        batch_size = 128
        epochs = 20

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        print(model.predict(x_test[0][np.newaxis,...] ))


        quantizer = vitis_quantize.VitisQuantizer(model)
        quantized_model = quantizer.quantize_model(calib_dataset = x_test[1:1024], weight_bit=8, activation_bit=8)
        
        quantized_model.compile(loss='categorical_crossentropy', metrics=["accuracy"])
        score = quantized_model.evaluate(x_test, y_test,  verbose=0, batch_size=32)
        print(score)
        
        quantized_model.save('mnist_quant.h5')

        os.system("vai_c_tensorflow2 \
            --model ./mnist_quant.h5 \
            --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
            --output_dir comp/ \
            --net_name mnist")



# test%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    utils_obj = utils('cpp/')


    # utils_obj.conv(utils_obj.x_test[0], np.array([[1,1,1],[1,2,1],[1,1,1]]))
    # utils_obj.MM(utils_obj.x_test[0], 16)

    # utils_obj.write_c_im_array(1, True, False)

    w, b, data = utils_obj.NN(True)
    # tens, scal, zero_p = utils_obj.NN_int_quant()
    # utils_obj.write_mnist()

    # utils_obj.rename()
    # out, b, w, test = utils_obj.FullyVitis()
    # utils_obj.write_mnist()
    # print(out[2])
    # utils_obj.pynq_dpu()
