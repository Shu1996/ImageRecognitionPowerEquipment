import numpy as np

batch_size = 600
class feed_data:
    def __init__(self):
        """
        读取.npy数据
        """
        Insulator = np.load('Insulator.npy')
        Tower = np.load('Tower.npy')
        Normal = np.load('Normal.npy')
        self.index = 0
        self.Insulator = []
        self.Tower = []
        self.Normal = []
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []


        for i in Insulator:
            self.Insulator.append([i[0],[1,0]])
        for i in Tower:
            self.Tower.append([i[0],[0,1]])
        # for i in Normal:
        #     self.Normal.append([i[0],[0,1]])

        # trainData = self.Insulator[:24720] + self.Tower[:4900] + self.Normal[:5490]
        # testData = self.Insulator[24720:] + self.Tower[4900:] + self.Normal[5490:]
        trainData = self.Insulator[:24720] + self.Tower[:4900]
        testData = self.Insulator[24720:] + self.Tower[4900:]
        np.random.shuffle(trainData)
        np.random.shuffle(testData)

        for i in trainData:
            self.x_train.append(i[0])
            self.y_train.append(i[1])
        for i in testData:
            self.x_test.append(i[0])
            self.y_test.append(i[1])

        self.x_train = np.array(self.x_train, dtype=np.float32)
        self.y_train = np.array(self.y_train, dtype=np.float32)
        self.x_test = np.array(self.x_test, dtype=np.float32)
        self.y_test = np.array(self.y_test, dtype=np.float32)

    def get_train_data(self):
        i = self.index
        x = self.x_train[batch_size * i : batch_size * (i + 1)]
        y = self.y_train[batch_size * i : batch_size * (i + 1)]
        self.index += 1
        if self.index >= (self.x_train.shape[0] / batch_size):
            np.random.shuffle(trainData)
            self.x_train = []
            self.y_train = []
            for i in trainData:
                self.x_train.append(i[0])
                self.y_train.append(i[1])
                self.x_train = np.array(self.x_train, dtype=np.float32)
                self.y_train = np.array(self.y_train, dtype=np.float32)
            self.index = 0
        return (x, y)

    def get_test_data(self):
        i = self.index
        x = self.x_test[batch_size * i : batch_size * (i + 1)]
        y = self.y_test[batch_size * i : batch_size * (i + 1)]
        self.index += 1
        if self.index >= (self.x_test.shape[0] / batch_size):
            self.index = 0
        return (x, y)
