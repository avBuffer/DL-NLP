#!/usr/bin/env python
# -*- coding: utf8 -*-
import os
import pandas as pd

# ### 加载疾病名称和描述
filePath = 'data'
pd_icd10 = pd.read_excel(os.path.join(filePath, "ICD-10.xlsx"))
print(pd_icd10.shape, pd_icd10.head(5))
icd10disList = pd_icd10['Disease'].tolist()


def loadFileDisDes(fileName, columnNameDis, columnNameDes ):
    pd_file1 = pd.read_excel(os.path.join(filePath, fileName))
    print(pd_file1.columns)

    disList = pd_file1[columnNameDis].tolist()
    desList = pd_file1[columnNameDes].tolist()
    disAliasNewList = list()
    desAliasNewList = list()
    
    for i in range(len(disList)):
        dis = disList[i]
        des = desList[i]
        disArr = dis.replace("、", ",").replace("，", ",").replace(";", ",").split(',')
        for i in range(len(disArr)):
            item = disArr[i]
            if len(item.strip()) > 0:
                disAliasNewList.append(item)
                desAliasNewList.append(des)

    print(len(disList), "名称拆分后梳理：", len(disAliasNewList))
    return disAliasNewList, desAliasNewList


# ### 加载三份数据的疾病名称和描述
def loadFiles():
    disListALL = list()
    desListALL = list()
    disList1,  desList1 = loadFileDisDes('disease汇总全部数据.xlsx', '字段1_文本', '症状')
    disListALL.extend(disList1)
    desList1_ = list()

    for des in desList1:
        des_ = des.replace('介绍分享到', '').replace('\n', '')
        desList1_.append(des_)
    desListALL.extend(desList1_)

    #名医百科
    disList2,  desList2 = loadFileDisDes('baike疾病content.xlsx', 'disease_name', '检查')
    disListALL.extend(disList2)
    desListALL.extend(desList2)

   #39健康网
    disList3,  desList3 = loadFileDisDes('jbk39jb.xlsx', '名称', '概述')
    disListALL.extend(disList3)
    desListALL.extend(desList3)
    return disListALL, desListALL

disListALL, desListALL = loadFiles()
print("疾病数量：", len(disListALL), len(desListALL))


def doComputICD(disAliasList):
    num=0
    for item in disAliasList:
        if item in icd10disList:
            num += 1
    print(num,  len(disListALL))

    numSet = 0
    numList = list()
    for item in list(set(disAliasList)):
        if item in icd10disList:
            numSet += 1
            numList.append(item)
    print(numSet, len(numList), len(numList) * 1.0 / 23658)


# ### 统计当前icd10疾病覆盖
doComputICD(disListALL)
d = {'disease': disListALL, 'des': desListALL}
df = pd.DataFrame(data=d)
df.count()
df.to_csv(os.path.join(filePath, "diseaseAll.csv"))


# ### 提取别名，作为疾病实体词典使用
print(len(disListALL), len(desListALL))
disAliasNewList = list()
desAliasNewList = list()
disAliasList_ = list()

#39健康网，提取别名
pd_file1 = pd.read_excel(os.path.join(filePath, 'jbk39jb.xlsx'))
print(pd_file1.columns)
disList = pd_file1['别名'].tolist()
desList = pd_file1['概述'].tolist()

for i in range(len(disList)):
        dis = disList[i]
        des = desList[i]
        disArr = dis.replace("、", ",").replace("，", ",").replace(";", ",").split(',')
        for item in disArr:
            if len(item) > 0:
                disAliasList_.append(item)
                disAliasNewList.append(item)
                desAliasNewList.append(des)

print("新别名词典数量：", len(disAliasList_))
print("新别名在ICD10数量：", len(disAliasNewList), len(desAliasNewList))
print(len(disAliasNewList), len(disListALL))
print(len(desAliasNewList), len(desListALL))


# ### 结果中的疾病和别名汇总, 30432
disAliasNewList.extend(disListALL)
print(len(disAliasNewList))
desAliasNewList.extend(desListALL)
print(len(desAliasNewList))
doComputICD(disAliasNewList)


# ### 最终疾病，在icd10中的，获取疾病名称和描述
diNew = list()
deNew = list()
for i in range(len(disAliasNewList)):
    di = disAliasNewList[i]
    de = desAliasNewList[i]
    if di in icd10disList:
        diNew.append(di)
        deNew.append(de)

print(len(diNew),  len(deNew))
d = {'symptom': diNew, 'symptomDes': deNew}
df = pd.DataFrame(data=d)
df.count()
df.to_csv(os.path.join(filePath, "diseaseMatch.csv"))


# ### 补充疾病名称，从症状的相关疾病中，获取疾病名称
aliasList, desList = loadFileDisDes('jbk39zz.xlsx', '名称', '相关疾病')
mayDisList = list()
for inStr in desList:
    inArr = inStr.replace("、", ",").replace("，", ",").replace(";", ",").replace(" ", ',').split(r',')
    for item in inArr:
        if len(item.strip()) > 0:
            mayDisList.append(item.replace("...", ""))
print(len(mayDisList))


#考虑相关疾病中，拿症状概述补充
doComputICD(mayDisList)
print(len(set(mayDisList)),  len(icd10disList))
disAliasNewList.extend(icd10disList)
disAliasNewList.extend(mayDisList)
print("len source=", len(disAliasNewList))
disAliasList__ = list(set(disAliasNewList))
print("len source=", len(disAliasList__))
df = pd.DataFrame(disAliasList__)
df.to_csv(os.path.join(filePath, "disease.csv"))


# ### 载入症状数据
def loadSymptomFiles():
    symptomListALL = list()
    symptomdesListALL = list()
    symList1, desList1 = loadFileDisDes('symptom症状全数据.xlsx', '症状名称', '介绍')
    symptomListALL.extend(symList1)

    desList1_ = list()
    for des in desList1:
        des_ = des.replace('介绍分享到', '').replace('\n', '')
        desList1_.append(des_)
    symptomdesListALL.extend(desList1_)

    symList2,  desList2 = loadFileDisDes('baike症状content.xls', 'disease_name', '概述')
    symptomListALL.extend(symList2)
    symptomdesListALL.extend(desList2)

    symList3,  desList3 = loadFileDisDes('jbk39zz.xlsx', '名称', '概述')
    symptomListALL.extend(symList3)
    symptomdesListALL.extend(desList3)
    return symptomListALL,  symptomdesListALL
symptomListALL, symptomdesListALL = loadSymptomFiles()


# ### 症状里面，多个症状的拆分开，规范化写，写入文件
symptomListALL_ = list()
for inStr in symptomListALL:
    inArr = inStr.replace("、", ",").replace("，", ",").replace(";", ",").replace(" ", ',').split(r',')
    for item in inArr:
        if len(item) > 0:
            symptomListALL_.append(item.replace("...", ""))

print("all=", len(symptomListALL_))
df_ = pd.DataFrame(symptomListALL_)
df_.to_csv(os.path.join(filePath, "symptom.csv"))

d = {'symptom': symptomListALL, 'symptomDes': symptomdesListALL}
df = pd.DataFrame(data=d)
df.count()
df.to_csv(os.path.join(filePath, "symptomAll.csv"))

num = 0
tmpList = list()
tmpList.extend(symptomListALL)
tmpList.extend(disListALL)
for item in tmpList:
    if item in icd10disList:
        num += 1
print(num, len(tmpList))


doComputICD(symptomListALL)
syNew = list()
desNew = list()
for i in range(len(symptomListALL)):
    sym = symptomListALL[i]
    des = symptomdesListALL[i]
    if sym in icd10disList:
        syNew.append(sym)
        desNew.append(des)

print(len(syNew), len(desNew))
d = {'symptom': syNew, 'symptomDes': desNew}
df = pd.DataFrame(data=d)
df.count()
df.to_csv(os.path.join(filePath, "symptomMatch.csv"))
