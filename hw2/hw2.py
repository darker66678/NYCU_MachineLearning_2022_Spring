import struct
import numpy as np
import argparse
import logging



def read_file(file, type):
    if type == "idx3":
        data = open(file, 'rb').read()
        offset = 0
        fmt_header = '>iiii'
        _, num_imgs, row, col = struct.unpack_from(fmt_header, data, offset)
        img_size = row * col
        image_fmt = ">"+str(img_size)+'B'
        offset = offset + struct.calcsize(fmt_header)
        all_imgs = np.empty((num_imgs, row, col))
        for i in range(num_imgs):
            img_data = struct.unpack_from(image_fmt, data, offset)
            all_imgs[i] = np.array(img_data).reshape(row, col)
            offset = offset + struct.calcsize(image_fmt)
        return all_imgs
    elif type == "idx1":
        data = open(file, 'rb').read()
        offset = 0
        fmt_header = '>ii'
        label_fmt = '>B'
        _, num_imgs = struct.unpack_from(fmt_header, data, offset)
        offset = offset + struct.calcsize(fmt_header)
        labels = np.empty(num_imgs)
        for i in range(num_imgs):
            iabel_data = struct.unpack_from(label_fmt, data, offset)
            labels[i] = iabel_data[0]
            offset = offset + struct.calcsize(label_fmt)
        return labels


def load_data():
    file = {"train_x": "train-images-idx3-ubyte",
            "train_y": "train-labels-idx1-ubyte",
            "test_x": "t10k-images-idx3-ubyte",
            "test_y": "t10k-labels-idx1-ubyte"}
    train_x = read_file(file["train_x"], "idx3")
    train_y = read_file(file["train_y"], "idx1")
    test_x = read_file(file["test_x"], "idx3")
    test_y = read_file(file["test_y"], "idx1")
    return train_x, train_y, test_x, test_y


class Naive_Bayes():
    def __init__(self, train_x, train_y, test_x, test_y, mode):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.mode = mode
        self.label_num = 10

    def train(self):
        pseudocount = 1e-10
        amount_label = np.zeros(self.label_num)
        if self.mode == 0:
            self.train_x = self.train_x // 8
            self.test_x = self.test_x // 8
            param_pixel = np.zeros(
                (32, self.train_x.shape[1], self.train_x.shape[2], self.label_num))
            param_pixel += pseudocount
            for i in range(self.train_x.shape[0]):
                amount_label[int(self.train_y[i])] += 1
                for j in range(self.train_x.shape[1]):
                    for k in range(self.train_x.shape[2]):
                        train_bin = int(self.train_x[i][j][k])
                        param_pixel[train_bin][j][k][int(self.train_y[i])] += 1
            return amount_label, param_pixel
        elif self.mode == 1:
            param_pixel = np.zeros(
                (2, self.label_num, self.train_x.shape[1], self.train_x.shape[2]))  # 0= mean 1=sigma
            for i in range(self.label_num):
                amount_label[i] = np.where(train_y == i)[0].shape[0]
                param_pixel[0][i][:][:] = np.mean(
                    self.train_x[np.where(self.train_y == i)], axis=0) + 1
                param_pixel[1][i][:][:] = np.std(
                    self.train_x[np.where(self.train_y == i)], axis=0) + 1
            return amount_label, param_pixel

    def predict_result(self, posterior, true):
        logging.info("Posterior (in log scale):")
        for i in range(self.label_num):
            logging.info(f'{i}: {posterior[i]}')
        pred = np.argmin(posterior)
        logging.info(f'Prediction: {pred}, Ans: {true}\n')
        return pred

    def imagination(self, pixels):
        logging.info("Imagination of numbers in Bayesian classifier:\n")
        if self.mode == 0:
            shape = pixels.shape
            for i in range(shape[3]):
                logging.info(f'{i}:')
                for j in range(shape[1]):
                    row = ""
                    for k in range(shape[2]):
                        pixel_low_128 = 0
                        pixel_high_128 = 0
                        for l in range(shape[0]):
                            if l < 16:
                                pixel_low_128 += pixels[l][j][k][i]
                            else:
                                pixel_high_128 += pixels[l][j][k][i]

                        if pixel_low_128 >= pixel_high_128:  # compare
                            row += "0 "
                        else:
                            row += "1 "
                    logging.info(row)
                logging.info(" ")

        elif self.mode == 1:
            shape = pixels.shape
            for i in range(shape[1]):
                logging.info(f'{i}:')
                for j in range(shape[2]):
                    row = ""
                    for k in range(shape[3]):
                        expect = pixels[0][i][j][k]
                        if expect >= 128:
                            row += "1 "
                        else:
                            row += "0 "
                    logging.info(row)
                logging.info(" ")

    def log_gaussian(self, mean, std, x):
        pb = (1/(std*((2*np.pi)**0.5))) * np.exp(-((x-mean)**2)/(2*(std**2)))
        if(pb!=0):
            pb = np.log(pb)
        return pb

    def inference(self, amount_label, param_pixel):
        prior = np.log(amount_label/self.train_x.shape[0])
        error = 0
        if self.mode == 0:
            for i in range(self.test_x.shape[0]):
                posterior = np.zeros(self.label_num)
                for num in range(self.label_num):
                    likelihood = 0
                    for j in range(self.test_x.shape[1]):
                        for k in range(self.test_x.shape[2]):
                            test_bin = int(self.test_x[i][j][k])
                            likelihood += np.log(param_pixel[test_bin]
                                                 [j][k][num]/amount_label[num])
                    posterior[num] = prior[num]+likelihood
                posterior /= sum(posterior)
                pred = self.predict_result(posterior, int(self.test_y[i]))
                if pred != int(self.test_y[i]):
                    error += 1

        elif self.mode == 1:
            for i in range(self.test_x.shape[0]):
                posterior = np.zeros(self.label_num)
                for num in range(self.label_num):
                    likelihood = 0
                    for j in range(self.test_x.shape[1]):
                        for k in range(self.test_x.shape[2]):
                            mean = param_pixel[0][num][j][k]
                            std = param_pixel[1][num][j][k]
                            x = self.test_x[i][j][k]
                            pb = self.log_gaussian(mean, std, x)
                            likelihood += pb
                    if likelihood < -10000:
                        likelihood = -10000
                    posterior[num] = prior[num]+likelihood
                posterior /= sum(posterior)
                pred = self.predict_result(posterior, int(self.test_y[i]))
                if pred != int(self.test_y[i]):
                    error += 1

        self.imagination(param_pixel)
        error_rate = error/self.test_x.shape[0]
        logging.info(f'Error rate: {error_rate}')

    def main(self):
        amount_label, param_pixel = self.train()
        self.inference(amount_label, param_pixel)


class online_learning():
    def __init__(self, a, b, data):
        self.a = a
        self.b = b
        self.data = data

    def factorial(self, x):
        if x == 1:
            return 1
        else:
            return x * self.factorial(x-1)

    def binomial(self, N, m):
        p = m / N
        likelihood = self.factorial(
            N)/(self.factorial(m) * self.factorial(N-m)) * (p ** m) * ((1-p) ** (N-m))
        return likelihood

    def beta_bino_conjugate(self):
        for i in range(len(data)):
            N = len(data[i])
            m = data[i].count('1')
            likelihood = self.binomial(N, m)
            print(f'case {i+1}: {data[i]}')
            print(f'Likelihood: {likelihood}')
            print(f'Beta prior:     a = {self.a} b = {self.b}')
            self.a += m
            self.b += (N-m)
            print(f'Beta posterior: a = {self.a} b = {self.b}')
            print('')

    def main(self):
        self.beta_bino_conjugate()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=0, type=int)
    parser.add_argument("--type", default="online", type=str)
    args = parser.parse_args()
    if(args.mode == 0 and args.type =="bayes"):
        fh = logging.FileHandler(encoding='utf-8', mode='w', filename='Discreate.log')
    elif (args.mode == 1 and args.type =="bayes"):
        fh = logging.FileHandler(encoding='utf-8', mode='w', filename='Continue.log')
    else:
        fh = logging.FileHandler(encoding='utf-8', mode='w', filename='online_learning.log')
    logging.basicConfig(handlers=[fh], level=logging.DEBUG, format='%(message)s')

    if args.type == "bayes":
        train_x, train_y, test_x, test_y = load_data()
        bayes = Naive_Bayes(train_x, train_y, test_x, test_y, args.mode)
        bayes.main()
    elif args.type == "online":
        a = int(input("a = "))
        b = int(input("b = "))
        print("")
        data = open('testfile.txt', 'r').read().split('\n')
        learner = online_learning(a, b, data)
        learner.main()
