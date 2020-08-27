"""

This is a very simple demo (based on Keras) you can use to train the 2D ResANet 
or predict the fetal brain age given a single slice. 
      
author:  allard.w.shi  2019.09

"""

import os
import pickle
import numpy as np
from opts import parse_opts
from keras import backend as K
from sklearn.metrics import r2_score
from model import ran4bae_run, ran4bae_load

if __name__ == '__main__':
    
    # set fundamental parameters (adapt to your data)
    opt = parse_opts() 
    opt.input_shape = (192,192,1)
    opt.layer_name = "uncertainty_output"
    
    # seek appropriate learning rate | time to be an alchemist :)
    opt.init_lr = 1e-2

    # number of model ensembles
    M = 5
    
    opt.data_augmentation = True
    opt.choose_uncertainty = True
    
    if opt.action == 'train':
        # run RAN4BAE model 
        ran4bae_run(opt)
                
    else: 
        # load test data and label
        x_test = pickle.load(open(opt.x_test, 'rb' ) )
        y_test = pickle.load(open(opt.y_test, 'rb' ) )
        
        print(x_test.shape[0], ' test samples')
        
        y_pred = np.zeros((x_test.shape[0],M))
        y_uncertainty = np.zeros((x_test.shape[0],M))
        
        # load model 
        for i in range(M):
            
            # save the trained model in the ./models/ and name as 'demo_x.h5'
            model= ran4bae_load(input_shape=opt.input_shape,   
                                weight_path = 'models/demo_'+str(i)+'.h5',
                                uncertainty=True)
            
            print('load model '+str(i)+' success!')
            get_value = K.function(inputs=[model.input], outputs=model.get_layer(opt.layer_name).output)
            y_temp, uncertainty_temp = get_value([x_test])
            y_pred[:,i],y_uncertainty[:,i] = y_temp.reshape(-1,),uncertainty_temp.reshape(-1,)
            
        print('prediction done!')
        print('#'*30+' result '+'#'*30)
              
        # compute marker 
        y_BA = np.mean(y_pred, axis=1)
        y_uncertainty_a = np.mean(y_uncertainty, axis = 1)
        y_uncertainty_e = np.mean(y_pred**2, axis = 1) - y_BA**2
        y_uncertainty_sum = y_uncertainty_a + y_uncertainty_e

        # save result
        if not os.path.isdir('result/'):
            os.makedirs('result')

        pickle.dump(y_BA, open('result/pred_BA.p', 'wb' ))
        pickle.dump(y_uncertainty_sum, open('result/pred_uncertainty.p', 'wb' ))
   
        print('mean PAD: %.3f week'%np.mean(y_BA-y_test))
        print('mean MAE: %.3f week'%np.mean(abs(y_BA-y_test)))
        print('R2: %.3f'%r2_score(y_test,y_BA))
        
