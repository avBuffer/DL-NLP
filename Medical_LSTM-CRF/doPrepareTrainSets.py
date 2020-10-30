#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd

filePath = 'data'
symPath = os.path.join(filePath, 'symptomMatch.csv')
disPath = os.path.join(filePath, 'diseaseMatch.csv')
disDictName = os.path.join(filePath, 'disease_new2.dic')
symDictName = os.path.join(filePath, 'symptom_new2.dic')
bodyDictName = os.path.join(filePath, 'body中文身体部位名称.dic')


# ### 加载词典
id2name = {1: 'DISEASE', 2: 'SYMPTOM', 3: 'BODY'}
print(id2name, id2name[1])
tags={'DISEASE', 'SYMPTOM', 'BODY'}
idx2tag = list(set(tags))
btag2idx = dict([(char, i) for i, char in enumerate(idx2tag)])
print(btag2idx)


def loadDict(dicName, inType):
    bodyDict = dict()
    for item in open(dicName, 'r', encoding='utf-8', errors='ignore'):
         bodyDict[item.strip().replace('\n', '')] = inType
    return bodyDict

disDict = loadDict(disDictName, 'DISEASE')
symDict = loadDict(symDictName, 'SYMPTOM')
bodyDict = loadDict(bodyDictName, 'BODY')


def dictOnly():
    disOnlyDict = loadDict(os.path.join(filePath, 'disonly'), 'dis')
    symOnlyDict = loadDict(os.path.join(filePath, 'symonly'), 'sym')
    disF = open(os.path.join(filePath, 'disease_new2.dic'), 'w')
    for dis in disDict:
        if dis in symOnlyDict:
            continue
        else:
            disF.write(dis.replace('...', '') + '\n')
    disF.flush()
    disF.close()
    disE = open(os.path.join(filePath, 'symptom_new2.dic'), 'w')

    for sym in symDict:
        if sym in disOnlyDict:
            continue
        else:
            disE.write(sym.replace('...', '') + '\n')
    disE.flush()
    disE.close()

#dictOnly()
#print(disDict['孕吐'])
#print(symDict['鼻翼扇动'])
#print(bodyDict['口'], btag2idx[(bodyDict['口'])])


# ### 加载待处理的文本
def row2ner(result, row, name,typeName):
    p = row.find(name, 0)
    while (p != -1):
        result.append(name + ' ' + str(p) + ' ' + str(p + len(name)) + ' ' + typeName)
        p = row.find(name, p + 1)

row = '阿司匹林诱发哮喘症状常见症状恶心与呕吐腹泻呼吸困难结膜充血气喘胸闷休克 阿司匹林诱发哮喘好发于中年女性，少见于儿童，' \
      '典型症状是服药30min～2h内出现结膜充血，流涕，颜面及胸部皮肤潮红，热疹，恶心，呕吐，腹泻，偶有荨麻疹，同时伴胸闷，气喘，' \
      '呼吸困难，严重者可出现休克，昏迷，呼吸停止，这类患者治疗反应较差，故一旦发作，无论症状轻重，都应引起高度重视，若鼻息肉，' \
      '阿司匹林过敏和哮喘合并存在，则称为阿司匹林哮喘三联症'
result = []
row2ner(result, row, '阿司匹林', 'DISEASE')


# ### 将检测出的实体转化成BIO格式
def ner2lable(bio, inResult, btype, itype):
    for i in range(len(inResult)):
        inStr = inResult[i]
        s = int(inStr[1])
        e = int(inStr[2])
        bio[s] = btype + '-' + inStr[3]
        for j in range(s + 1, e):
            bio[j] = itype + '-' + inStr[3]


def loadDiseaseDatasets(disPath, columnName, trainPath):
    df_dis = pd.read_csv(disPath)
    df_dis = df_dis.dropna()
    desList = df_dis[columnName].tolist()
    f = open(trainPath, 'w', encoding='utf-8')
    for i in range(len(desList)):
        des = desList[i]
        result = []
        if not des:
            continue

        #des格式化， bio初始化为O
        des = des.replace(' ', '').replace('\t', '').replace('\n', '').replace('　', '').strip()
        bio = ['O' for i in range(len(des))]

        #检索所有的疾病，记录起始位置
        typeName = 'DISEASE'
        for dis in disDict:
            row2ner(result, des, dis, typeName)

        #检索所有的症状，记录起始位置
        result1 = []
        typeName = 'SYMPTOM'
        for sym in symDict:
            row2ner(result1, des, sym, typeName)

        #检索所有的身体部位，记录起始位置
        result2 = []
        typeName = 'BODY'
        for body in bodyDict:
            row2ner(result2, des, body, typeName)
        result4 = result + result1 + result2

        #字符串转二维数组
        result5 = [[0 for i in range(5)] for j in range(len(result4))]
        for i in range(len(result4)):
            resArr = result4[i].split(' ')
            result5[i][0] = resArr[0]
            result5[i][1] = int(resArr[1])
            result5[i][2] = int(resArr[2])
            result5[i][3] = resArr[3]
            result5[i][4] = len(resArr[0])

        #按照起始位置和实体长度排序
        result5.sort(key=lambda x: (x[1], x[4]))
        #选择实体词最长的进行最大匹配
        result6 = [[0 for i in range(5)] for j in range(len(result5))]
        maxIndexNum = 0
        maxIndexAll = 0

        #迭代检索实体词，如果后面的实体词和当前实体词起始索引一致，则找最长的实体，作为当前索引的实体，下一个词的起始索引要大于最长实体的结束索引
        i= 0
        while i < (len(result5) - 1):
            indexNew = result5[i][1]
            #当前实体索引小于上一实体的结束索引，直接略过，判断下一实体
            if indexNew < maxIndexAll:
                i = i + 1
                continue
            maxIndex = i

            #训练遍历后面的实体，找到同索引的最长实体，记录实体结束索引和下一个实体的序号
            for j in range(i + 1, len(result5)):
                if result5[j][1] == indexNew:
                    maxIndex = j
                    i = maxIndex+1
                else:
                    maxIndexAll = result5[maxIndex][2]
                    i = maxIndex + 1
                    break
            result6[maxIndexNum] = result5[maxIndex]
            maxIndexNum += 1

        result6 = result6[0 : maxIndexNum]
        ner2lable(bio, result6, 'B', 'I')
        for nerIndex in range(len(bio)):
            f.write(des[nerIndex] + ' ' + bio[nerIndex] + '\n')
        f.write('\n')
    f.flush()
    f.close()


columnName = 'symptomDes'
symPath = os.path.join(filePath, 'symptomMatch.csv')
disPath = os.path.join(filePath, 'diseaseMatch.csv')
trainPathDis = os.path.join(filePath, 'ner_train_data_dis')
trainPath = os.path.join(filePath, 'ner_train_data_sym')

loadDiseaseDatasets(symPath, columnName, trainPath)
loadDiseaseDatasets(disPath, columnName, trainPathDis)
