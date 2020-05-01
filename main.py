import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))
    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    x = x - np.max(x)
    return np.divide(np.exp(x), np.sum(np.exp(x)))

def computeLayer(X, W, b):
    return np.matmul(np.transpose(W), X) + b

def CE(target, prediction):
    return -1*np.mean(np.multiply(target, np.log(prediction)))

def gradCE(target, prediction):
    return  target - prediction

def optimize(H, gamma, alpha, epochs, X, Y):
    
    Wh = np.random.normal(0, 2/(784 + H), (784, H)) 
    bh = np.random.normal(0, 2/(784 + H), (H, 1))
    Wo = np.random.normal(0, 2/(H + 10), (H, 10))
    bo = np.random.normal(0, 2/(784 + H), (10, 1))
    
    old_vWh = vWh = old_vbh = vbh = old_vWo = vWo = old_vbo = vbo = 10**-5
    
    loss = []
    accuracy = []
    
    for i in range(epochs):
        
        h = relu(computeLayer(np.transpose(X), Wh, bh))
        o = computeLayer(h, Wo, bo)
        p = np.apply_along_axis(softmax, 0, o)
        
        loss.append(CE(np.transpose(Y), p))
        accuracy.append(np.sum(np.argmax(p, 0) == np.argmax(np.transpose(Y), 0))/len(X))
        
        gWo = np.matmul(h, np.transpose(p - np.transpose(Y)))
        gbo = np.sum(np.transpose(p - np.transpose(Y)), 0, keepdims = True)
        
        gWh = np.matmul(np.transpose(X), np.transpose((np.multiply(np.heaviside(computeLayer(np.transpose(X), Wh, bh), 0), np.matmul(Wo, p - np.transpose(Y))))))
        gbh = np.sum(np.transpose((np.multiply(np.heaviside(computeLayer(np.transpose(X), Wh, bh), 0), np.matmul(Wo, p - np.transpose(Y))))), 0, keepdims = True)
        
        vWh = gamma*old_vWh + alpha*gWh
        vbh = gamma*old_vbh + alpha*gbh
        vWo = gamma*old_vWo + alpha*gWo
        vbo = gamma*old_vbo + alpha*gbo
        
        old_vWh, old_vbh, old_vWo, old_vbo = vWh, vbh, vWo, vbo
        
        Wh = Wh - vWh
        bh = bh - np.transpose(vbh)
        Wo = Wo - vWo
        bo = bo - np.transpose(vbo)
    
    plt.plot(range(len(loss)), loss)
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    
    plt.plot(range(len(accuracy)), accuracy)
    plt.xlabel("EPOCH")
    plt.ylabel("ACCURACY")
    
    plt.legend(["Loss", "Accuracy"])
            
    return Wh, bh, Wo, bo, loss, accuracy

def main():
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)
    trainData, validData, testData = np.reshape(trainData, (len(trainData), -1)), np.reshape(validData, (len(validData), -1)), np.reshape(testData, (len(testData), -1))
    
    H = 1000
    gamma = 0.99
    alpha = 10**-6
    epochs = 200
    
    Wh, bh, Wo, bo, loss, accuracy = optimize(H, gamma, alpha, epochs, trainData, newtrain)
    
    # VALIDATION
    
    Wh, bh, Wo, bo, loss, accuracy = optimize(H, gamma, alpha, epochs, trainData, newtrain)
    
    # TEST
    
    Wh, bh, Wo, bo, loss, accuracy = optimize(H, gamma, alpha, epochs, trainData, newtrain)
   
    # NEURAL NETWORK IN TENSORFLOW 2.0
    reg = 0
    rate = 0.9
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2,2), padding='SAME'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(H, activation='relu', activity_regularizer = tf.keras.regularizers.l2(reg)),
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(10, activation='softmax', activity_regularizer = tf.keras.regularizers.l2(reg))
            ])
    
    model.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])
    
    history = model.fit(np.reshape(trainData, (len(trainData), 28, 28, 1)), trainTarget, batch_size = 35, epochs = 30)

    return -1

if __name__ == "__main__":
    main()
    
    
    


