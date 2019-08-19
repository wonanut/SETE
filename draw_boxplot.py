import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Epitope_list = ["m139", "F2", "PB1", "PA", "NP", "M38", "pp65", "BMLF", "M1", "M45"]


auc_list = [
0.84280858, 0.858130531, 0.942812255, 0.92298995, 0.90315865, 0.955397951, 0.958423913, 0.990088106, 0.969412235, 0.913253513,
0.831168028, 0.837955695, 0.945861601, 0.922450901, 0.952407048, 0.970426949, 0.935837029, 0.945124969, 0.964136166, 0.895287075,
0.868772894, 0.873277516, 0.914862915, 0.916621433, 0.909645522, 0.966964286, 0.951531729, 0.963864307, 0.981023443, 0.914286254,
0.736465324, 0.860297591, 0.949907828, 0.899214949, 0.887890458, 0.961357466, 0.942447178, 0.97390715, 0.967127635, 0.874768304,
0.818794853, 0.785682327, 0.901848267, 0.884748636, 0.928721174, 0.963691838, 0.924432396, 0.938608647, 0.957676419, 0.884881345
]

auc_list = np.array(auc_list).reshape(5, 10)
data = pd.DataFrame(auc_list, index=['f'+str(i+1) for i in range(5)], columns=Epitope_list)

# auc_list_cmp = [
# 0.819601936, 0.9772652,
# 0.843068732, 0.943705755,
# 0.931058573, 0.937703819,
# 0.909205174, 0.90302693,
# 0.916364571, 0.877826419,
# 0.963567698, 0.948864088,
# 0.942534449, 0.954083378,
# 0.962318636, 0.932694366,
# 0.96787518, 0.936167872,
# 0.896495298, 0.84611437
# ]
#
# auc_list_cmp = np.array(auc_list_cmp).reshape(10, 2)
# data_cmp = pd.DataFrame(auc_list_cmp, index=['c'+str(i+1) for i in range(10)], columns=['Without PCA', 'PCA'])

auc_list_cmp = [
0.65, 0.58, 0.82,  0.9772652,
0.81, 0.73, 0.83, 0.943705755,
0.85, 0.78, 0.89, 0.937703819,
0.87, 0.82, 0.907, 0.90302693,
0.89, 0.83, 0.92, 0.877826419,
0.892, 0.88, 0.926, 0.948864088,
0.91, 0.89, 0.935, 0.954083378,
0.92, 0.915, 0.95, 0.932694366,
0.93, 0.92, 0.96, 0.936167872,
0.935, 0.94, 0.973, 0.84611437
]

auc_list_cmp = np.array(auc_list_cmp).reshape(10, 4)
data_cmp = pd.DataFrame(auc_list_cmp, index=['c'+str(i+1) for i in range(10)], columns=['TCRdist(all)', 'TCRGP(CDR3β)', 'TCRGP(all)','SETE(CDR3β)'])

sns.boxplot(data=data_cmp)


dash_human_pca_cmp = [
0.850389775,	0.918403978,
0.792861052,	0.870512247,
0.857808807,	0.840308092,
0.824307185,	0.859158313,
0.768738856,	0.880844907
]

dash_mouse_pca_cmp = [
0.894262679,	0.906852214,
0.908140983,	0.921872752,
0.880690214,	0.898174492,
0.90488276,	    0.919115192,
0.873392793,	0.915570548
]

# cmp = np.array(dash_human_pca_cmp).reshape(5, 2)
# data_cmp = pd.DataFrame(cmp, index=['c'+str(i+1) for i in range(5)], columns=['Without PCA', 'PCA'])
# cmp2 = np.array(dash_mouse_pca_cmp).reshape(5, 2)
# data_cmp2 = pd.DataFrame(cmp2, index=['c'+str(i+1) for i in range(5)], columns=['Without PCA', 'PCA'])
#
# plt.subplot(121)
# sns.boxplot(data=data_cmp)
# plt.xlabel('Human')
# plt.ylabel('AUC')
# plt.subplot(122)
# sns.boxplot(data=data_cmp2)
# plt.xlabel('Mouse')
# plt.ylabel('AUC')
plt.xlabel('Prediction Model')
plt.ylabel('AUC')
plt.grid(True)
plt.show()