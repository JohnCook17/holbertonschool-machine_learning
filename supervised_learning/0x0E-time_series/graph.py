""""""
    """
    start_of_day = int(my_data[0, 0])
    end_of_day = int(my_data[-1, 0])
    print(start_of_day, end_of_day)
    time = []
    vwap = []
    for batch in range(start_of_day, end_of_day + 1):
        my_values = my_data[my_data[:, 0] == batch]
        # print(my_values)
        if my_values.size != 0:
            print("===========================")
            print(my_values[:, -1])
            time.append(my_values[:, 0])
            vwap.append(my_values[:, -1])

    plt.plot(time, vwap)
    plt.show()"""
