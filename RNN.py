import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, LSTM
from keras.optimizers import Adam
# parameter
session_len = 500
BATCH_SIZE = 200
fraction = 0.1
train_val_ratio = 0.8
Tri_df = pd.read_csv('./train.csv', sep=',')
# end of parameter
tr_fa = int(len(Tri_df)*fraction)
Tri_df = Tri_df[:tr_fa]
def add_impres_to_row(row):
    if row['action_type']=='clickout item':
        # row['ans'] = np.resize([0], [25])
        impress_lis = sorted(row['impressions'].split('|'))
        for i in range(len(impress_lis)):
            row['impress_' + str(i)] = impress_lis[i]
            # if row['reference'] == impress_lis[i]:
                # row['ans'][i] = 1
        return row
    else:
        pass
        # row['ans'] = np.resize([0], [25])
        return row
def add_impression_to_columns(df):
    for i in range(0,25):
        df['impress_' + str(i)] = 0
    df['ans'] = 0
    df = df.apply(add_impres_to_row, axis=1)
    return df
def drop_columns(df):
    df = df.drop(['city', 'platform', 'device', \
    'current_filters', 'impressions', 'prices', 'step'], axis=1)
    df = df[[x.isdigit() for x in df.reference.tolist()]]
    # df['reference'] = [int(x) for x in df.reference.tolist()]
    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
def data_process(df, train_val_ratio):
    id_set = df[df.action_type == 'clickout item'].reference.unique()
    id_dict = dict(zip(id_set, range(len(id_set))))
    print('id_set: ', len(id_dict))
    tt = add_impression_to_columns(df)
    print('before drop', len(tt))
    tt = drop_columns(tt)
    tt = tt.sample(frac=1).reset_index(drop=True)
    print('after drop', len(tt))
    x_train, y_train, y_ans = split_session(tt, id_dict)
    x_tr, x_val = x_train[:int(len(x_train)*train_val_ratio)], x_train[int(len(x_train)*train_val_ratio):]
    y_tr, y_val = y_train[:int(len(x_train)*train_val_ratio)], y_train[int(len(x_train)*train_val_ratio):]
    y_tr_ans, y_val_ans = y_ans[:int(len(x_train)*train_val_ratio)], y_ans[int(len(x_train)*train_val_ratio):]
    return np.array(x_tr), np.array(y_tr), np.array(y_tr_ans), np.array(x_val), np.array(y_val), np.array(y_val_ans)
def creat_NULL_df(num, labeler, labeler2, icolumns):
    tmp = pd.DataFrame(index=np.arange(num), columns=icolumns)
    tmp['user_id'] = labeler2.transform(['NULL'])[0]
    tmp['action_type'] = labeler.transform(['NULL'])[0]
    tmp['reference'] = 0
    for i in range(0,25):
        tmp['impress_' + str(i)] = 0
    return tmp
def split_session(df, id_dict):
    print(df.columns)
    df = df.append( \
        pd.DataFrame([['NULL', 'NULL','NULL', 'NULL', 'NULL', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'NULL']], columns=df.columns)\
            )
    df.reset_index(drop=True, inplace=True)
    labelencoder = LabelEncoder()
    df['session_id'] = labelencoder.fit_transform(df['session_id'])
    df['user_id'] = labelencoder.fit_transform(df['user_id'])
    # labelencoder3 = LabelEncoder()
    # df['reference'] = labelencoder3.fit_transform(df['reference'])
    labelencoder2 = LabelEncoder()
    df['action_type'] = labelencoder2.fit_transform(df['action_type'])
    df.drop(index=len(df)-1, inplace=True)
    gp_session = df.groupby('session_id')
    x_train = []
    y_train = []
    y_ans = []
    cut = 0
    longer = 0
    flag=0
    for session in df.session_id.unique():
        tmp = gp_session.get_group(session)
        tmp.reset_index(drop=True, inplace=True)
        tmp = tmp.sort_values(by=['timestamp'])
        # print(type(tmp), tmp)
        tmp = tmp.drop(['session_id', 'timestamp'], axis=1)
        if(len(tmp) > session_len):
            cut += 1
            tmp = tmp[:session_len]
        else:
            longer += 1
            tmp = tmp.append(creat_NULL_df(session_len - len(tmp), labelencoder2, labelencoder, tmp.columns))
        ytt = tmp[tmp.action_type == labelencoder2.transform(['clickout item'])[0]]
        ytt.reset_index(drop=True, inplace=True)
        if len(ytt) !=0:
            blank_index = max(tmp.index[tmp['action_type'] == labelencoder2.transform(['clickout item'])[0]].tolist())
            tmp.loc[blank_index, 'reference'] = 0
            y_train.append(creat_one_hot(id_dict, ytt.loc[len(ytt) - 1, 'reference']))
            y_ans.append(id_dict[ytt.loc[len(ytt) - 1, 'reference']])
        else:
            y_train.append(creat_one_hot(id_dict, -1))
            y_ans.append(-1)
        tmp = tmp.drop(['ans'], axis=1)
        if flag == 0:
            print(tmp.columns)
            flag=1  
        x_train.append(tmp.values)
    print('cut', cut)
    print('longer', longer)
    return x_train, y_train, y_ans
def creat_one_hot(id_dict, reference):
    if reference == -1:
        return np.resize([0], [len(id_dict)])
    else:
        one_hot = np.resize([0], [len(id_dict)])
        one_hot[id_dict[reference]] = 1
        # print(one_hot)
        return one_hot
def creat_model(inputfeature, outputdim):
    # 建立簡單的線性執行的模型
    model = Sequential()
    # 加 RNN 隱藏層(hidden layer)
    model.add(LSTM(
        # 如果後端使用tensorflow，batch_input_shape 的 batch_size 需設為 None.
        # 否則執行 model.evaluate() 會有錯誤產生.
        batch_input_shape=(None, session_len, inputfeature), 
        units= 100,
        unroll=True,
    )) 
    # 加 output 層
    model.add(Dense(units=outputdim, kernel_initializer='normal', activation='softmax'))

    # 編譯: 選擇損失函數、優化方法及成效衡量方式
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy']) 
    model.summary()
    return model
    # 一批訓練多少張圖片
def train_model(model, X_train, y_train, X_test, y_test, BATCH_INDEX=0):
    for step in range(1, 4001):
        # data shape = (batch_num, steps, inputs/outputs)
        X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
        Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
        # 逐批訓練
        loss = model.train_on_batch(X_batch, Y_batch)
        BATCH_INDEX += BATCH_SIZE
        BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

        # 每 500 批，顯示測試的準確率
        if step % 500 == 0:
            loss, accuracy = model.evaluate(X_train, y_train, batch_size=y_test.shape[0], 
                verbose=False)
            print("train loss: {}  \ttrain accuracy: {}".format(loss,accuracy))
            # 模型評估
            loss, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], 
                verbose=False)
            print("\t\t\t\t\t\t\ttest loss: {}  \ttest accuracy: {}".format(loss,accuracy))
            
    # 預測(prediction)
    X = X_test[0:10,:]
    predictions = model.predict(X)
    # get prediction result
    print(predictions)
    predictions = model.predict(X_test)
    predictions2 = model.predict(X_train)
    return predictions, predictions2
    # 模型結構存檔
    # from keras.models import model_from_json
    # json_string = model.to_json()
    # with open("SimpleRNN.config", "w") as text_file:
        # text_file.write(json_string)
        
    # 模型訓練結果存檔
    # model.save_weights("SimpleRNN.weight")

def get_id_dict(df1, df2):
    id_set = df[df.action_type == 'clickout item'].reference.unique()
    id_dict = dict(zip(id_set, range(len(id_set))))
    return id_dict
def get_user_laberler(df1, df2):
    labeler = LabelEncoder()
    print("user_id")
    print(len(df1.user_id.tolist()),len(df2.user_id.tolist()))
    user_union = list(set(df1.user_id.tolist() + df2.user_id.tolist()))
    print(len(user_union))
    labeler.fit(user_union)
    return labeler
def create_list_predict(row):
    row['list'] = []
    for i in row:
        row['list'].append(i)
    row['list'].pop()
def takefirst(ele):
    return ele[0]
def count_MRR(y_predict, y_ans):
    MRR = []
    recall = []
    for i in range(y_predict.shape[0]):
        tmplis = list(y_predict[i])
        prelis = []
        for j in range(len(tmplis)):
            prelis.append([tmplis[j], j ])
        prelis.sort(key=takefirst, reverse=True)
        for j in range(min(22, len(prelis))):
            if j == 21:
                MRR.append(0)
                recall.append(0)
                break
            if prelis[j][1] == y_ans[i]:
                # print(j, prelis[j][1], y_ans[i])
                MRR.append(1.0/(j + 1))
                recall.append(1)
                break
    print(sum(MRR)/len(MRR))
    # print(MRR)
    print(sum(recall)/len(recall))
    # print(recall)

    # predict = []
    # for i in range(len(imp_lis)):
        # predict.append([ int(y_predict.loc[i, 'list']), i])
    # imp_lis.sort(key=takesecond, reverse=True)
    # print(imp_lis)
    # for i in range(len(imp_lis)):
        # imp_lis[i] = imp_lis[i][0]
    # print(imp_lis)
    # return imp_lis
        
    pass
if __name__ == '__main__':

    x_train, y_train, x_ans, x_test, y_test, y_ans = data_process(Tri_df, train_val_ratio)

    result1 = y_train[0]

    for i in range(len(y_train)):
        result1 = result1 | y_train[i]
    print('tr_id_set: ', sum(result1))

    result2 = y_test[0]
    for i in range(len(y_test)):
        result2 = result2 | y_test[i]
    print('val_id_set: ', sum(result2))

    print('intersect', sum(result1 & result2))

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)

    model = creat_model(x_train.shape[2], y_train.shape[1])
    y_predict, x_predict = train_model(model, x_train, y_train, x_test, y_test)

    print('train_MRR_recall')
    count_MRR(x_predict, x_ans)
    print('test_MRR_recall')
    count_MRR(y_predict, y_ans)