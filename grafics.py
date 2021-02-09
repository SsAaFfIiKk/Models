import pickle
import matplotlib.pyplot as plt


def load_data(weights_folder):
    with open(weights_folder + "/" + weights_folder + "_train_los.data", "rb") as f:
        train_los_list = pickle.load(f)
    with open(weights_folder + "/" + weights_folder + "_train_acc.data", "rb") as f:
        train_acc_list = pickle.load(f)
    with open(weights_folder + "/" + weights_folder + "_test_los.data", "rb") as f:
        test_los_list = pickle.load(f)
    with open(weights_folder + "/" + weights_folder + "_test_acc.data", "rb") as f:
        test_acc_list = pickle.load(f)

    x = list(range(len(train_acc_list)))

    print(max(train_los_list))
    print(max(train_acc_list))
    print(max(test_los_list))
    print(max(test_acc_list), test_acc_list.index(max(test_acc_list)))

    plt.subplot(2, 2, 1)
    plt.plot(x, train_los_list,  label='train_los')
    plt.plot(x, test_los_list, label='test_los')
    plt.legend(loc='lower left')

    plt.subplot(2, 2, 2)
    plt.plot(x, test_acc_list, label='test_acc')
    plt.plot(x, train_acc_list, label='train_acc')
    plt.legend(loc='lower left')
    plt.show()


ind = int(input("eye/smile/emot/res18?: "))
if ind == 0:
    load_data("eye")
elif ind == 1:
    load_data("smile")
elif ind == 2:
    load_data("emot")
elif ind == 3:
    load_data("res18")
