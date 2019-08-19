import csv
import ast
import numpy as np

"""
readFile：读取原始TCR-epitope数据函数
filePath:原始数据文件路径
key_fields:要存储的键值List，从前往后范围减小
value_fields:要存储的内容List，内容不分先后
delimiter:csv.DictReader参数，换行符标志
返回值：返回一个dict
"""
def readFile(filePath, key_fields, value_fields, delimiter = '\t'):
    retDict = {}
    with open(filePath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            keys = [row[k] for k in key_fields]
            values = [row[s] for s in value_fields]
            sub_dict = retDict
            for key in keys[:-1]:
                if key not in sub_dict:
                    sub_dict[key] = {}
                sub_dict = sub_dict[key]
            key = keys[-1]
            if key not in sub_dict:
                sub_dict[key] = []
            sub_dict[key].append(values)

    return retDict

"""
dataPreprocess:对数进行预处理，统计不同表位的可用数据并加以筛选
dataDict:输入的原始Dict格式数据
species:要处理的种族数据，如HomoSapiens或者MusMusculus
gene:要处理的基因类型，如TRB或TRA
miniValue:筛选epitope的最小数据量
返回值：返回一个dict
"""
def dataPreprocess(dataDict, species, gene, miniValue=50):
    retDict = {}
    for s in species:
        for g in gene:
            epitopeList = list(dataDict[s][g].keys())
            for epi in epitopeList:
                epiSize = len(dataDict[s][g][epi])
                if epiSize >= miniValue:
                    retDict[epi] = dataDict[s][g][epi]
                    for i in range(epiSize):
                        meta = ast.literal_eval(retDict[epi][i][4])
                        sub_id = meta['subject.id']
                        reference = retDict[epi][i][3]
                        retDict[epi][i][3] = reference + '_' + sub_id
                        retDict[epi][i] = retDict[epi][i][:-1]
    return retDict

def statisticsEpitope(epiDict, path='data/epiDict.csv'):
    print('{:22s} {:s}'.format('Epitope', 'Number'))
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['epitope', 'info'])
        for epi in epiDict:
            print('{:22s} {:d}'.format(epi, len(epiDict[epi])))
            for item in epiDict[epi]:
                writer.writerow([epi, item])

def removeDuplicates(epiDict):
    ret_dict = {}
    for epi in epiDict:
        item_list = []
        item_dict = set()
        for item in epiDict[epi]:
            if item[0] not in item_dict:
                item_list.append(item)
                item_dict.add(item[0])
        ret_dict[epi] = item_list
    return ret_dict

def saveEpitope(epiDict, path):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['epitope', 'cdr3'])
        for epi in epiDict:
            for item in epiDict[epi]:
                writer.writerow([epi, item[0]])

    print("Epitope-TCR pairs have been saved in: {:s}".format(path))

def splitCDR(CDRseq, k = 3):
    retDict = []
    for i in range(len(CDRseq) - k + 1):
        retDict.append(CDRseq[i:i+k])
    return retDict

def statisticsKmer(epiDict, k = 3):
    kmerDict = {}
    for epi in epiDict:
        for i in range(len(epiDict[epi])):
            splitList = splitCDR(epiDict[epi][i][0], k)
            for split in splitList:
                if split not in kmerDict:
                    kmerDict[split] = 1
                else:
                    kmerDict[split] += 1
    return kmerDict

def buildFeatures(epiDict, kmerDict, k = 3):

    counter = 0
    for epi in epiDict:
        counter += len(epiDict[epi])
    retArr = np.zeros((counter, len(kmerDict)))

    kmerList = kmerDict.keys()
    retLabel = []

    iter = 0
    epinum = 0
    for epi in epiDict:
        for cdr in range(len(epiDict[epi])):
            splitlist = splitCDR(epiDict[epi][cdr][0], k)
            retLabel.append(epinum)
            i = 0
            for kmer in kmerList:
                retArr[iter][i] = splitlist.count(kmer)
                i += 1
            iter += 1
        epinum += 1
    return np.array(retArr), np.array(retLabel)

