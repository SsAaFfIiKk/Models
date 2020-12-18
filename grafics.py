import pickle
import matplotlib.pyplot as plt


def load_eye():
    with open("eye/eye_train_los.data", "rb") as f:
        train_los_list = pickle.load(f)
    with open("eye/eye_train_acc.data", "rb") as f:
        train_acc_list = pickle.load(f)
    with open("eye/eye_test_los.data", "rb") as f:
        test_los_list = pickle.load(f)
    with open("eye/eye_test_acc.data", "rb") as f:
        test_acc_list = pickle.load(f)

    x = list(range(len(train_acc_list)))

    print(max(train_los_list))
    print(max(train_acc_list))
    print(max(test_los_list))
    print(max(test_acc_list), test_acc_list.index(max(test_acc_list)))

    plt.subplot(2, 2, 1)
    plt.plot(x, train_los_list,  label='train_los')
    plt.legend(loc='upper center')
    plt.subplot(2, 2, 3)
    plt.plot(x, train_acc_list, label='train_acc')
    plt.legend(loc='lower right')

    plt.subplot(2, 2, 2)
    plt.plot(x, test_los_list, label='test_los')
    plt.legend(loc='upper center')
    plt.subplot(2, 2, 4)
    plt.plot(x, test_acc_list, label='test_acc')
    plt.legend(loc='lower right')
    plt.show()


def load_smile():
    with open("smile/smile_train_los.data", "rb") as f:
        train_los_list = pickle.load(f)
    with open("smile/smile_train_acc.data", "rb") as f:
        train_acc_list = pickle.load(f)
    with open("smile/smile_test_los.data", "rb") as f:
        test_los_list = pickle.load(f)
    with open("smile/smile_test_acc.data", "rb") as f:
        test_acc_list = pickle.load(f)

    x = list(range(len(train_acc_list)))

    print(max(train_los_list))
    print(max(train_acc_list))
    print(max(test_los_list))
    print(max(test_acc_list), test_acc_list.index(max(test_acc_list)))

    plt.subplot(2, 2, 1)
    plt.plot(x, train_los_list, label='train_los')
    plt.legend(loc='upper center')
    plt.subplot(2, 2, 3)
    plt.plot(x, train_acc_list, label='train_acc')
    plt.legend(loc='lower right')

    plt.subplot(2, 2, 2)
    plt.plot(x, test_los_list, label='test_los')
    plt.legend(loc='upper center')
    plt.subplot(2, 2, 4)
    plt.plot(x, test_acc_list, label='test_acc')
    plt.legend(loc='lower right')
    plt.show()


ind = int(input("eye/smile?: "))
if ind == 0:
    load_eye()
elif ind == 1:
    load_smile()
