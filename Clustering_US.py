## Student Name:  Wing Au
## Student ID:    112495536

import sys
import csv
import numpy as np
from time import time
import pyspark
import json
from math import isnan

#remove some log prints
def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

#convert float
def convert_float(items):
    results = []
    for i in items:
        if(i.replace('.','', 1).isdigit()): results.append(float(i))
        else: results.append(0)
    return results

#csv line spliter
def spliter(data):
    return ['{}'.format(x) for x in list(csv.reader([data], delimiter = ',', quotechar = '"'))[0]]

#compute yearly results
def select(data, index, correct = 0, year = 0):
    if year: rs = [year]
    else: rs = []
    for i in index:
        rs.append(data[i-correct])
    return rs

#aggregate data to zip codes and year
def p_avg(data):
    rs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    yrs = len(data[1])
    for i in data[1]:
        for j in range(12):
            rs[j] += i[j]
    for i in range(12):
        rs[i] = round(rs[i]/yrs, 1)
    return rs[1:]

#save file func    
def toCSVLine(data):
    return ','.join(str(d) for d in data)

#save file func
def toCSV(RDD):
    for element in RDD:
        return ','.join(str(element))

#compute high education ratio
def agg_edu(data):
    b = [data[6], data[9], data[12], data[15], data[18]]
    sum = 0
    c = 0
    for i in b:
        if(i.replace('.','', 1).isdigit()):
            sum += float(i)
            c += 1
    if not c: return 0
    return sum/c

#compute independent living difficulty value
def lid(data):
    if not float(data[0]): return [float(0), float(0)]
    ld = (float(data[24])+float(data[50]))/float(data[0])
    nld = (float(data[26])+float(data[52]))/float(data[0])
    return [round(ld*100, 1), round(nld*100, 1)]

#age averager
def a_avg(data):
    rs = [0, 0, 0, 0]
    yrs = [len(data[1]), len(data[1]), len(data[1]), len(data[1])]
    for i in data[1]:
        for j in range(4):
            if(i[j].replace('.','', 1).isdigit()): rs[j] += float(i[j])
            else: yrs[j] -= 1
    for i in range(4):
        if not yrs[i]: rs[i] = 0
        else: rs[i] = round(rs[i]/yrs[i], 1)
    return rs[1:]

#read file spliter2
def spliter2(d):
    rs = [d[1].replace("'", '')]
    for i in d[2:]:
        rs.append(float(i))
    return (float(d[0]), tuple(rs))
    # return tuple(d)

# find max and min values
def ppmm(data):
    rs = []
    for i in data:
        rs.append(i[1][1])
    return max(rs), min(rs)

#normalize given data list
def normalize(data, max, min, c = 1, scale = 100):
    diff = max - min
    new = (data[c] - min) / diff
    rs = []
    for i in range(len(data)):
        if i == c: rs.append(round(new * scale, 1))
        else: rs.append(data[i])
    return tuple(rs)

#find cluster center by given datatset
def cps2(data, c = 5, key = False):
    results = []
    s = len(data)//c
    feats = len(data[0][1][1:])
    prs = []
    for i in range(c):
        rs = [0] * feats
        p = 0
        for j in data[i*s:(i+1)*s]:
            for k in range(feats):
                rs[k] += j[1][1:][k]
            p += j[0]
        prs.append(round(p/s, 1))
        results.append([round(x/s, 1) for x in rs])
    if key:
        sam = []
        for i in range(c):
            sam.append((prs[i], results[i]))
        return sam
    return results

#euclidean distance
def euc(data, tar):
    sum = 0
    for i in range(len(data)):
        sum += (data[i] - tar[i])**2
    return round(sum**.5, 1)

#clustering based on distance value
def clustering(data, cps):
    rs = []
    for i in cps:
        rs.append(euc(data, i))
    return np.argmin(rs)

#clustering evaluation
def eval(data, tar, top = 1000):
    c = 0
    for i in range(top):
        if(data[i][2] == tar): c += 1
    return c/top

if __name__ == "__main__":
    sc = pyspark.SparkContext()
    quiet_logs(sc) #remove spark INFO prints
    tt = time()

    #-------------------------------------------------------------------
    #1.1 Aggregate poverty data to zip codes.

    #select target features
    tar = ['S1701_C01_001E', 'S1701_C02_001E', 'S1701_C03_001E', 'S1701_C03_022E', 'S1701_C03_023E', 'S1701_C03_024E',
    'S1701_C03_025E', 'S1701_C03_026E', 'S1701_C03_027E', 'S1701_C03_028E', 'S1701_C03_031E']
    feat = ['GEOID', 'Year', 'Popu', 'Poverty', 'P%', '25%', '25NE%', '25HE%', '25CE%', '25BE%', 'L16%', 'L16E%', 'L16U%']

    print("\n1.1 ..")
    print(feat)
    dp = './poverty/ACSST5Y2015.S1701_data_with_overlays_2022-05-06T103018.csv'
    rdd = sc.textFile(dp, 32).flatMap(lambda line: line.split("\n")) #csv
    d1h = rdd.first().split(",")

    tar_index = []
    for i in range(len(d1h)):
        if d1h[i].replace('"', '') in tar: tar_index.append(i)

    #filter data
    rdd = rdd.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    #select features
    p2015 = rdd.map(lambda i: (i[0], select(convert_float(i[2:]), tar_index, 2, 2015)))
    print(p2015.first(), len(p2015.first()[1]))

    #same for 2016 - 2020
    dp = './poverty/ACSST5Y2016.S1701_data_with_overlays_2022-05-06T103018.csv'
    rdd = sc.textFile(dp, 32).flatMap(lambda line: line.split("\n")) #csv
    rdd = rdd.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    p2016 = rdd.map(lambda i: (i[0], select(convert_float(i[2:]), tar_index, 2, 2016)))
    print(p2016.first())

    dp = './poverty/ACSST5Y2017.S1701_data_with_overlays_2022-05-06T103018.csv'
    rdd = sc.textFile(dp, 32).flatMap(lambda line: line.split("\n")) #csv
    rdd = rdd.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    p2017 = rdd.map(lambda i: (i[0], select(convert_float(i[2:]), tar_index, 2, 2017)))
    print(p2017.first())

    dp = './poverty/ACSST5Y2018.S1701_data_with_overlays_2022-05-06T103018.csv'
    rdd = sc.textFile(dp, 32).flatMap(lambda line: line.split("\n")) #csv
    rdd = rdd.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    p2018 = rdd.map(lambda i: (i[0], select(convert_float(i[2:]), tar_index, 2, 2018)))
    print(p2018.first())

    dp = './poverty/ACSST5Y2019.S1701_data_with_overlays_2022-05-06T103018.csv'
    rdd = sc.textFile(dp, 32).flatMap(lambda line: line.split("\n")) #csv
    rdd = rdd.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    p2019 = rdd.map(lambda i: (i[0], select(convert_float(i[2:]), tar_index, 2, 2019)))
    print(p2019.first())
    
    dp = './poverty/ACSST5Y2020.S1701_data_with_overlays_2022-05-06T103018.csv'
    rdd = sc.textFile(dp, 32).flatMap(lambda line: line.split("\n")) #csv
    rdd = rdd.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    p2020 = rdd.map(lambda i: (i[0], select(convert_float(i[2:]), tar_index, 2, 2020)))
    print(p2020.first())

    # print("\nAfter union years")
    #union years
    pov = p2015.union(p2016).union(p2017).union(p2018).union(p2019).union(p2020)
    #group by zipcode
    pov = pov.groupByKey().mapValues(lambda x: tuple(x))
    # print(pov.first(), pov.count())

    #1.2 Aggregate poverty data to zip codes and year.
    pova = pov.map(lambda i: (i[0], p_avg(i))).sortByKey()
    print(pova.first(), pova.count())

    #1.3 filter high education in poverty
    pova = pova.map(lambda i: (i[0], [i[1][0], i[1][2], i[1][3], i[1][7], i[1][8], i[1][9], i[1][10]]))
    print(pova.first(), pova.count())

    # # #1.3 Save to file
    # # pova.saveAsTextFile('pova')
    # pova.coalesce(1).saveAsTextFile("pova")

    #1.4 Check csv file
    # print("poverty loaded")
    # pova = sc.textFile("pova")
    # print(pova.first(), pova.count())

    #-------------------------------------------------------------------
    #2.1 Education data by Liting
    # https://colab.research.google.com/drive/1uN8FSiNapJgNo10h3m8F7N_keSS2FesG?authuser=1
    print("\n2.1 ..")
    edu = sc.textFile("edu_data_comb.csv", 32).flatMap(lambda line: line.split("\n")) #csv
    #header
    eh = edu.first().split(",")
    # print(eh, "\n")

    #2.2 Aggregate data by bach_or_higher level
    edu = edu.zipWithIndex().filter(lambda i: i[1]).map(lambda x: spliter(x[0]))
    #Aggregate data by bach_or_higher in all age group
    edu = edu.map(lambda i: (i[0], agg_edu(i)))
    #Average data by year
    edu = edu.reduceByKey(lambda a, b: a + b).map(lambda i: (i[0], round(i[1]/8, 1)))
    print(edu.first(), edu.count())
    # print(agg_edu(edu.first()))

    #-------------------------------------------------------------------
    #3.1 Age 75+ with independent living difficulty by zip
    print("\n3.1 ..")
    path = './75ild/ACSDT5Y2020.B18107_data_with_overlays_2022-05-10T174351.csv'
    ild = sc.textFile(path, 32).flatMap(lambda line: line.split("\n"))

    #header
    ih = ild.first().split(",")

    #select male, female 75+ with and without independent living difficulty and total
    # tar = ['B18107_001E', 'B18107_013E', 'B18107_014E', 'B18107_026E', 'B18107_027E', 'GEO_ID']
    # tar_index = []
    # #find target features
    # for i in range(len(ih)):
    #     if ih[i].replace('"', '') in tar: tar_index.append(i)
    # print(ih, tar_index, "\n")

    #3.2 Aggregate data and computer percentage
    ild = ild.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    ild = ild.map(lambda i: (i[54], lid(i)))
    print(ild.first(), ild.count())

    #-------------------------------------------------------------------
    #4.1 Age by zip
    print("\n4.1 ..")
    path = './age/ACSST5Y2011.S0101_data_with_overlays_2022-05-11T004949.csv'
    age = sc.textFile(path, 32).flatMap(lambda line: line.split("\n"))

    #header
    ageh = age.first().split(",")
    # print(ageh)

    #select total, age 75+%, median age, Old-age dependency ratio
    tar = ['GEO_ID', 'S0101_C01_001E', 'S0101_C01_029E', 'S0101_C01_030E', 'S0101_C01_033E']
    tar_index = []
    # find target features
    for i in range(len(ageh)):
        if ageh[i].replace('"', '') in tar: tar_index.append(i)
    # print(tar_index, "\n")

    age = age.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    a2011 = age.map(lambda i: (i[0], (i[2], i[58], i[60], i[66])))
    print(a2011.first(), a2011.count())

    path = './age/ACSST5Y2012.S0101_data_with_overlays_2022-05-11T004949.csv'
    age = sc.textFile(path, 32).flatMap(lambda line: line.split("\n"))
    age = age.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    a2012 = age.map(lambda i: (i[0], (i[2], i[58], i[60], i[66])))
    print(a2012.first(), a2012.count())

    path = './age/ACSST5Y2013.S0101_data_with_overlays_2022-05-11T004949.csv'
    age = sc.textFile(path, 32).flatMap(lambda line: line.split("\n"))
    age = age.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    a2013 = age.map(lambda i: (i[0], (i[2], i[58], i[60], i[66])))
    print(a2013.first(), a2013.count())

    path = './age/ACSST5Y2014.S0101_data_with_overlays_2022-05-11T004949.csv'
    age = sc.textFile(path, 32).flatMap(lambda line: line.split("\n"))
    age = age.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    a2014 = age.map(lambda i: (i[0], (i[2], i[58], i[60], i[66])))
    print(a2014.first(), a2014.count())

    path = './age/ACSST5Y2015.S0101_data_with_overlays_2022-05-11T004949.csv'
    age = sc.textFile(path, 32).flatMap(lambda line: line.split("\n"))
    age = age.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    a2015 = age.map(lambda i: (i[0], (i[2], i[58], i[60], i[66])))
    print(a2015.first(), a2015.count())

    path = './age/ACSST5Y2016.S0101_data_with_overlays_2022-05-11T004949.csv'
    age = sc.textFile(path, 32).flatMap(lambda line: line.split("\n"))
    age = age.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    a2016 = age.map(lambda i: (i[0], (i[2], i[58], i[60], i[66])))
    print(a2016.first(), a2016.count())

    path = './age/ACSST5Y2017.S0101_data_with_overlays_2022-05-11T004949.csv'
    age = sc.textFile(path, 32).flatMap(lambda line: line.split("\n"))

    # ageh = age.first().split(",")
    # #select total, age 75+%, median age, Old-age dependency ratio
    # tar = ['GEO_ID', 'S0101_C01_001E', 'S0101_C01_031E', 'S0101_C01_032E', 'S0101_C01_035E',
    #  'S0101_C02_031E', 'S0101_C02_032E', 'S0101_C02_035E']
    # tar_index = []
    # for i in range(len(ageh)):
    #     if ageh[i].replace('"', '') in tar: tar_index.append(i)
    # print(tar_index)

    age = age.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    # a2017 = age.map(lambda i: (i[0], (i[2], i[62], i[64], i[70], i[138], i[140], i[146])))
    a2017 = age.map(lambda i: (i[0], (i[2], i[138], i[64], i[70])))
    print(a2017.first(), a2017.count())

    path = './age/ACSST5Y2018.S0101_data_with_overlays_2022-05-11T004949.csv'
    age = sc.textFile(path, 32).flatMap(lambda line: line.split("\n"))
    age = age.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    a2018 = age.map(lambda i: (i[0], (i[2], i[138], i[64], i[70])))
    print(a2018.first(), a2018.count())

    path = './age/ACSST5Y2019.S0101_data_with_overlays_2022-05-11T004949.csv'
    age = sc.textFile(path, 32).flatMap(lambda line: line.split("\n"))
    age = age.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    a2019 = age.map(lambda i: (i[0], (i[2], i[138], i[64], i[70])))
    print(a2019.first(), a2019.count())

    path = './age/ACSST5Y2020.S0101_data_with_overlays_2022-05-11T004949.csv'
    age = sc.textFile(path, 32).flatMap(lambda line: line.split("\n"))
    age = age.zipWithIndex().filter(lambda i: i[1] > 1).map(lambda x: spliter(x[0]))
    a2020 = age.map(lambda i: (i[0], (i[2], i[138], i[64], i[70])))
    print(a2020.first(), a2020.count())

    print("\nAfter union years")
    age = a2012.union(a2013).union(a2014).union(a2015).union(a2016).union(a2017).union(a2018).union(a2019).union(a2020)
    age = age.groupByKey().mapValues(lambda x: tuple(x))
    # print(age.first(), age.count())

    # 4.2 Aggregate poverty data to zip codes and year.
    age = age.map(lambda i: (i[0], a_avg(i))).sortByKey()
    print(age.first(), age.count())
    #-------------------------------------------------------------------

    print("\n5.1 ..")
    #5.1 Union poverty, education, independent living difficulty, and age data
    zd = pova.union(edu).union(ild).union(age)
    zd = zd.groupByKey().mapValues(lambda x: tuple(x))
    zd = zd.map(lambda i: (i[0][7:], i[1]))
    # print(zd.first(), zd.count())

    #poverty % as key
    zd = zd.map(lambda i: (i[1][0][1], (i[0], i[1][0][0], i[1][0][2], i[1][0][3], i[1][0][4], i[1][0][5],
     i[1][0][6], i[1][1], i[1][2][0], i[1][2][1], i[1][3][0], i[1][3][1], i[1][3][2])))
    # zd = zd.filter(lambda i: i[0]).sortByKey()

    #5.2 filter population 1k+
    zd = zd.filter(lambda i: i[1][1] > 999).sortByKey()

    #5.3 Save to file
    # zd.coalesce(1).saveAsTextFile("zd")

    #-------------------------------------------------------------------

    # #6.0 load checkpoint file
    # zd = sc.textFile("zd")
    # print("6.0 ..")
    # #header
    # print("(Poverty,  (GEOID, TTR, 25P, 25BEP, 16L, 16LE, 16LU, BE, LD, NLD, 75, MA, OD))")
    # zd = zd.map(lambda i: i.replace('(', '').replace(')', '').replace(' ', '').split(',')).map(lambda i: spliter2(i))
    # print(zd.first(), zd.count())

    max, min = ppmm(zd.collect())
    #normalize data before compute distances
    zd = zd.map(lambda i: (i[0], normalize(i[1], max, min)))
    # print(zd.first(), zd.count())

    #6.1 pick clustering anchor points
    # zdr = zd.sortByKey(False)
    #check sort by poverty rate
    # for i in zd.take(20): print(i)
    # for i in zdr.take(20): print(i)

    #pick clustering anchor points, default is 5(group)
    cp = cps2(zd.collect())
    # for i in cp: print(i)

    #6.2 clustering satisfation level by social health features
    zd = zd.map(lambda i: (i[0], i[1], clustering(i[1][1:], cp)))
    # for i in zd.take(20): print(i)
    # print("")
    # for i in zd.top(20): print(i)

    #clustering statistics
    print("\nPoverty rate vs features:")
    sample = cps2(zd.collect(), 10, True)
    for i in sample: 
        print(i[0], end = '\t')
        for i in i[1]: print(i, end = '\t')
        print()

    # c = [100, 400, 999, 2000, 4000]
    c = [1000, 2000, 4000]

    print("\n\nClustering based on Selected Social Health of Countries features:\n")
    for i in c:
        print('{:.1%}'.format(eval(zd.take(i), 0, top = i)), "of the", i,
         "lowest poverty zip were categorized as High life satisfaction(0) group")
    print()
    for i in c:
        print('{:.1%}'.format(eval(zd.top(i), 4, top = i)), "of the", i,
         "highest poverty zip were categorized as Low life satisfaction(4) group")

    g0 = zd.filter(lambda i:i[2] == 0)
    g1 = zd.filter(lambda i:i[2] == 1)
    g2 = zd.filter(lambda i:i[2] == 2)
    g3 = zd.filter(lambda i:i[2] == 3)
    g4 = zd.filter(lambda i:i[2] == 4)
   
    print('\nClustering size:', g0.count(), g1.count(), g2.count(), g3.count(), g4.count(), '\tall:', zd.count())

    #save grouping and final files
    # # g0.coalesce(1).saveAsTextFile("g0")
    # # g1.coalesce(1).saveAsTextFile("g1")
    # # g2.coalesce(1).saveAsTextFile("g2")
    # # g3.coalesce(1).saveAsTextFile("g3")
    # # g4.coalesce(1).saveAsTextFile("g4")
    # # zd.coalesce(1).saveAsTextFile("fin")

    print("\n - Done in", round(time() - tt), "s\n")

