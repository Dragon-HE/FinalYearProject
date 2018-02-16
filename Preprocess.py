import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(style='white', color_codes=True)

train = pd.read_csv("E:\\dataset\\train.csv")


'''Drop the first feature ID'''
# train.drop(['ID'], axis=1, inplace=True)


'''Data exploration'''
print(train.delta_num_reemb_var17_1y3.value_counts())  # 把999999替换为nan后有75999个0 和 1个-1
print()
print(train['delta_num_reemb_var17_1y3'].describe())   # 这种行没有必要去手动一个一个地删除，feature selection会把它们筛掉
print()
print(train.TARGET.value_counts())
df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = df['TARGET']*100/train.shape[0]
print(df)


'''Show training set'''
print('\nTraining set info: {}'.format(train.shape))


'''Remove constant features'''
removed = []
for col in train.columns:
    if train[col].std() == 0:
        # std = 0 is a smart idea. # see https://www.kaggle.com/kobakhit/0-84-score-with-36-features-only/code
        removed.append(col)
train.drop(removed, axis=1, inplace=True)  # drop columns and return new array
print('\nAfter removing constant features: {}'.format(train.shape))


'''Remove duplicate features'''
removed = []
col = train.columns
for i in range(len(col)-1):
    val = train[col[i]].values
    for j in range(i+1, len(col)):
        val2 = train[col[j]].values
        if np.array_equal(val, val2):
            removed.append(col[j])
# print(removed)
train.drop(removed, axis=1, inplace=True)
print('\nAfter removing duplicate features: {}'.format(train.shape))


'''Now rename the remaining features. use X_i to denote the ith feature'''
col = train.columns
for i in range(len(col)):
    train.columns.values[i] = 'X_' + str(i+1)
    # see https://stackoverflow.com/questions/43759921/pandas-rename-column-by-position
print('\n''The names of the columns have been changed:')
print(train.columns)


'''All -999999 and 9999999999 have been replaced with nan. Delete samples that contain nan'''
col = train.columns
for i in range(len(col)):
    indexes = list(train[np.isnan(train[col[i]])].index)
    # isnan() returns a boolean array. # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.isnan.html
    train = train.drop(indexes)
print('\nAfter removing nan samples: {}'.format(train.shape))

# Check for # of nan
print('\nNumber of nan in the training set:')
print(sum(train.isnull().sum()))


'''Data visualisation and description'''
plt.figure(1)
plt.xlim(0.0, train.X_2.max())
# learn how to set the width of X axis
# https://stackoverflow.com/questions/17734587/why-is-set-xlim-not-setting-the-x-limits-in-my-figure
plt.hist(train['X_2'], bins=100)
plt.xlabel('X_2')
plt.ylabel('Frequency')
plt.title('Distribution of feature X_2')
# print()
# print(train.X_2.value_counts()[:20])
# print(train['X_2'].describe())
plt.show()

plt.figure(2)
# print(train['X_3'].describe())
plt.xlim(xmin=20, xmax=90)
plt.hist(train['X_3'], bins=100, color='g')
plt.xlabel('X_3')
plt.ylabel('Frequency')
plt.title('Distribution of feature X_3')
plt.show()

# grid = sns.FacetGrid(train, hue="X_305", size=6)
# grid.map(sns.kdeplot, "X_3").add_legend()
