from data_preprocessing import data1, data1_x_calc, data1_x_bin, Target, data1_x_dummy, data1_dummy, data1_x
from sklearn import model_selection
#here data1_x_calc contains 8 columns having codes as some of the columns
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target])

#data1_x_bin removes the continious varibles data for eg age and fair
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target])

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target])

# print(train1_x.shape)
# print(test1_x.shape)
# print(train1_y.shape)
# print(test1_y.shape)


# FINDING THE CORELATION OF THE OUPTUT WITH EACH OF THE COLUMNS
for x in data1_x:
    if data1[x].dtype != 'float64':
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*40)
