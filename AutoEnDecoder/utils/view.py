
import matplotlib.pyplot as plt


def perform_loss_cof(y1 , y2 ):
    """Define view.
        Arguments:
            y1 -- loss
            y2 -- cof
    """
    x1 = range(0, len(y1))
    plt.subplot(1, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.xlabel('Train loss vs. epoches')
    plt.ylabel('Train loss')
    # plt.subplot(2, 1, 2)
    # plt.plot(x1, y2, 'o-')
    # plt.title('Test accuracy vs. epoches')
    # plt.ylabel('Test accuracy')
    plt.savefig('checkpoint/plot.svg', format='svg')
