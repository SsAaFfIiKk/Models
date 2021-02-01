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


def load_emot():
    with open("emot/emot_train_los.data", "rb") as f:
        train_los_list = pickle.load(f)
    with open("emot/emot_train_acc.data", "rb") as f:
        train_acc_list = pickle.load(f)
    with open("emot/emot_test_los.data", "rb") as f:
        test_los_list = pickle.load(f)
    with open("emot/emot_test_acc.data", "rb") as f:
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


def load_res():
    with open("res18/res_train_los.data", "rb") as f:
        res18_train_los_list = pickle.load(f)
    with open("res18/res_train_acc.data", "rb") as f:
        res18_train_acc_list = pickle.load(f)
    with open("res18/res_test_los.data", "rb") as f:
        res18_test_los_list = pickle.load(f)
    with open("res18/res_test_acc.data", "rb") as f:
        res18_test_acc_list = pickle.load(f)

    with open("res50/res50_train_los.data", "rb") as f:
        res50_train_los_list = pickle.load(f)
    with open("res50/res50_train_acc.data", "rb") as f:
        res50_train_acc_list = pickle.load(f)
    with open("res50/res50_test_los.data", "rb") as f:
        res50_test_los_list = pickle.load(f)
    with open("res50/res50_test_acc.data", "rb") as f:
        res50_test_acc_list = pickle.load(f)

    x = list(range(len(res18_train_acc_list)))

    print(max(res18_train_los_list))
    print(max(res18_train_acc_list))
    print(max(res18_test_los_list))
    print(max(res18_test_acc_list), "res18", res18_test_acc_list.index(max(res18_test_acc_list)))

    print(max(res50_train_los_list))
    print(max(res50_train_acc_list))
    print(max(res50_test_los_list))
    print(max(res50_test_acc_list), "res50", res50_test_acc_list.index(max(res18_test_acc_list)))

    plt.subplot(2, 2, 1)
    plt.plot(x, res18_train_los_list,  label='res18 train_los')
    plt.plot(x, res50_train_los_list, label='res50 train_los')
    plt.legend(loc='upper center')
    plt.subplot(2, 2, 3)
    plt.plot(x, res18_train_acc_list, label='res18 train_acc')
    plt.plot(x, res50_train_acc_list, label='res50 train_acc')
    plt.legend(loc='lower right')

    plt.subplot(2, 2, 2)
    plt.plot(x, res18_test_los_list, label='res18 test_los')
    plt.plot(x, res50_test_los_list, label='res50 test_los')
    plt.legend(loc='upper center')
    plt.subplot(2, 2, 4)
    plt.plot(x, res18_test_acc_list, label='res18 test_acc')
    plt.plot(x, res50_test_acc_list, label='res50 test_acc')
    plt.legend(loc='lower right')
    plt.show()


ind = int(input("eye/smile/emot/emot_res?: "))
if ind == 0:
    load_eye()
elif ind == 1:
    load_smile()
elif ind == 2:
    load_emot()
elif ind == 3:
    load_res()

