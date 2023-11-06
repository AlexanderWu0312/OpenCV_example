from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import to_categorical

# 載入MNIST資料集
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data( )

# 建立卷積神經網路
network = Sequential( )
network.add( Conv2D( filters = 32, kernel_size = ( 3, 3 ), 
	         input_shape = ( 32, 32, 3 ), activation = 'relu', 
	         padding = 'same' ) )
network.add( MaxPooling2D( pool_size = ( 2, 2 ) ) )
network.add( Conv2D( filters = 64, kernel_size = ( 3, 3 ), 
	                 activation = 'relu', padding = 'same' ) )
network.add( MaxPooling2D( pool_size = ( 2, 2 ) ) )	
network.add( Flatten( ) )
network.add( Dense( 1024, activation = 'relu' ) )
network.add( Dense( 10, activation = 'softmax' ) )
network.compile( optimizer = 'adam', loss = 'categorical_crossentropy', 
	             metrics = ['accuracy'] )
print( network.summary() )
 
# 資料前處理
train_images = train_images.astype( 'float32' ) / 255
test_images = test_images.astype( 'float32' ) / 255
train_labels = to_categorical( train_labels )
test_labels = to_categorical( test_labels )

# 訓練階段
network.fit( train_images, train_labels, epochs = 10, batch_size = 200 )

# 測試階段
test_loss, test_acc = network.evaluate( test_images, test_labels )
print( "Test Accuracy:", test_acc )
