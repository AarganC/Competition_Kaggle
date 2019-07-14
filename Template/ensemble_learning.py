import pandas as pd

if __name__ == "__main__":
    file1 = pd.read_csv('../result_prediction2/selection/predict_LSTM_LSTMtest2515.csv')
    file2 = pd.read_csv('../result_prediction2/selection/predict_ResNet_ResNettest137.csv')
    file3 = pd.read_csv('../result_prediction2/selection/predict_ResNet_ResNettest49.csv')
    file4 = pd.read_csv('../result_prediction2/selection/predict_ResNet_ResNettest73.csv')
    file5 = pd.read_csv('../result_prediction2/selection/predict_ResNet_ResNettest161.csv')
    print(file1["has_cactus"][1])
    print(file2["has_cactus"][1])
    print(file3["has_cactus"][1])
    print(file4["has_cactus"][1])
    print(file5["has_cactus"][1])

    i = 0
    rows_list = []
    for file_name in file1["id"]:
        # print(file_name)
        occ = 0
        tmp = []
        if file1["has_cactus"][i] == 1:
            occ += 1
        if file2["has_cactus"][i] == 1:
            occ += 1
        if file3["has_cactus"][i] == 1:
            occ += 1
        if file4["has_cactus"][i] == 1:
            occ += 1
        if file5["has_cactus"][i] == 1:
            occ += 1
        if occ > 3:
            tmp = [file_name, 1]
            rows_list.append(tmp)
            # print(tmp)
        else:
            tmp = [file_name, 0]
            rows_list.append(tmp)
            # print(tmp)
        i += 1
    columns = ['id', 'has_cactus']
    df = pd.DataFrame(rows_list, columns=list(columns))
    print(df)
    df.to_csv("ensemble_predict.csv")
