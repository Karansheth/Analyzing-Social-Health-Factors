from unittest import result
import pyspark
import csv
import random
from pyspark import SparkContext
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sc = SparkContext()

# Loading Datasets

input_2005 = sc.textFile("Death_US/2005_data.csv", 32)
head_2005 = input_2005.first()
header_2005 = sc.broadcast(head_2005.split(','))
death_2005 = input_2005.filter(lambda row: row!=head_2005).mapPartitions(lambda line: csv.reader(line))

input_2006 = sc.textFile("Death_US/2006_data.csv", 32)
head_2006 = input_2006.first()
header_2006 = sc.broadcast(head_2006.split(','))
death_2006 = input_2006.filter(lambda row: row!=head_2006).mapPartitions(lambda line: csv.reader(line))

input_2007 = sc.textFile("Death_US/2007_data.csv", 32)
head_2007 = input_2007.first()
header_2007 = sc.broadcast(head_2007.split(','))
death_2007 = input_2007.filter(lambda row: row!=head_2007).mapPartitions(lambda line: csv.reader(line))

input_2008 = sc.textFile("Death_US/2008_data.csv", 32)
head_2008 = input_2008.first()
header_2008 = sc.broadcast(head_2008.split(','))
death_2008 = input_2008.filter(lambda row: row!=head_2008).mapPartitions(lambda line: csv.reader(line))

input_2009 = sc.textFile("Death_US/2009_data.csv", 32)
head_2009 = input_2009.first()
header_2009 = sc.broadcast(head_2009.split(','))
death_2009 = input_2009.filter(lambda row: row!=head_2009).mapPartitions(lambda line: csv.reader(line))

input_2010 = sc.textFile("Death_US/2010_data.csv", 32)
head_2010 = input_2010.first()
header_2010 = sc.broadcast(head_2010.split(','))
death_2010 = input_2010.filter(lambda row: row!=head_2010).mapPartitions(lambda line: csv.reader(line))

input_2011 = sc.textFile("Death_US/2011_data.csv", 32)
head_2011 = input_2011.first()
header_2011 = sc.broadcast(head_2011.split(','))
death_2011 = input_2011.filter(lambda row: row!=head_2011).mapPartitions(lambda line: csv.reader(line))

input_2012 = sc.textFile("Death_US/2012_data.csv", 32)
head_2012 = input_2012.first()
header_2012 = sc.broadcast(head_2012.split(','))
death_2012 = input_2012.filter(lambda row: row!=head_2012).mapPartitions(lambda line: csv.reader(line))

input_2013 = sc.textFile("Death_US/2013_data.csv", 32)
head_2013 = input_2013.first()
header_2013 = sc.broadcast(head_2013.split(','))
death_2013 = input_2013.filter(lambda row: row!=head_2013).mapPartitions(lambda line: csv.reader(line))

input_2014 = sc.textFile("Death_US/2014_data.csv", 32)
head_2014 = input_2014.first()
header_2014 = sc.broadcast(head_2014.split(','))
death_2014 = input_2014.filter(lambda row: row!=head_2014).mapPartitions(lambda line: csv.reader(line))

input_2015 = sc.textFile("Death_US/2015_data.csv", 32)
head_2015 = input_2015.first()
header_2015 = sc.broadcast(head_2015.split(','))
death_2015 = input_2015.filter(lambda row: row!=head_2015).mapPartitions(lambda line: csv.reader(line))

index_manner = header_2005.value.index('manner_of_death')
index_education_2003 = header_2005.value.index('education_2003_revision')
index_year = header_2005.value.index('current_data_year')
index_age_group = header_2005.value.index('age_recode_12')
index_sex = header_2005.value.index('sex')


# Analysis-1 (Depicts the gender distribution of causes of death in the United States)


a1_2005 = death_2005.map(lambda x: ((x[index_manner], x[index_sex]), 1)).reduceByKey(lambda x,y: x+y)
a1_2006 = death_2006.map(lambda x: ((x[index_manner], x[index_sex]), 1)).reduceByKey(lambda x,y: x+y)
a1_2007 = death_2007.map(lambda x: ((x[index_manner], x[index_sex]), 1)).reduceByKey(lambda x,y: x+y)
a1_2008 = death_2008.map(lambda x: ((x[index_manner], x[index_sex]), 1)).reduceByKey(lambda x,y: x+y)
a1_2009 = death_2009.map(lambda x: ((x[index_manner], x[index_sex]), 1)).reduceByKey(lambda x,y: x+y)
a1_2010 = death_2010.map(lambda x: ((x[index_manner], x[index_sex]), 1)).reduceByKey(lambda x,y: x+y)
a1_2011 = death_2011.map(lambda x: ((x[index_manner], x[index_sex]), 1)).reduceByKey(lambda x,y: x+y)
a1_2012 = death_2012.map(lambda x: ((x[index_manner], x[index_sex]), 1)).reduceByKey(lambda x,y: x+y)
a1_2013 = death_2013.map(lambda x: ((x[index_manner], x[index_sex]), 1)).reduceByKey(lambda x,y: x+y)
a1_2014 = death_2014.map(lambda x: ((x[index_manner], x[index_sex]), 1)).reduceByKey(lambda x,y: x+y)
a1_2015 = death_2015.map(lambda x: ((x[index_manner], x[index_sex]), 1)).reduceByKey(lambda x,y: x+y)
a1 = a1_2005.join(a1_2006).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a1_2007).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a1_2008).map(lambda x: (x[0], x[1][0] + x[1][1]))
a1 = a1.join(a1_2009).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a1_2010).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a1_2011).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a1_2012).map(lambda x: (x[0], x[1][0] + x[1][1]))
a1 = a1.join(a1_2013).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a1_2014).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a1_2015).map(lambda x: (x[0], x[1][0] + x[1][1])).sortBy(lambda x: x[0][1])
a1 = a1.map(lambda x: (x[0][0], [(x[0][1], x[1])])).reduceByKey(lambda x,y: x+y).collect()
# print(a1)

 
# Data to plot
manner_of_death = {'': 'Not Specified', '1': 'Accident', '2': 'Suicide', '3': 'Homicide', '4': 'Pending investigation', 
                    '5': 'Could not determine', '6': 'Self-Inflicted', '7': 'Natural'}
labels_a1 = []
sizes_a1 = []
labels_gender_a1 = []
sizes_gender_a1 = []
for element in a1:
    # if not element[0]:
    #     continue
    labels_a1.append(manner_of_death[element[0]])
    add = 0
    for l in element[1]:
        labels_gender_a1.append(l[0])
        sizes_gender_a1.append(l[1])
        add+=l[1]
    sizes_a1.append(add)
# print(labels_a1)
# print(sizes_a1)
# print(labels_gender_a1)
# print(sizes_gender_a1)
colors_gender = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']

plt.pie(sizes_a1, labels=labels_a1, autopct='%1.1f%%', startangle=90,frame=True)
plt.pie(sizes_gender_a1, colors=colors_gender, radius=0.75,startangle=90)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Causes of Death')
plt.axis('equal')
plt.tight_layout()
plt.savefig('Analysis_1.png', bbox_inches='tight')


# Analysis-2 (How suicide instances are distributed according to the person's gender and educational background)

a2_2005 = death_2005.map(lambda x: ((x[index_education_2003], x[index_sex], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').reduceByKey(lambda x,y: x+y)
a2_2006 = death_2006.map(lambda x: ((x[index_education_2003], x[index_sex], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').reduceByKey(lambda x,y: x+y)
a2_2007 = death_2007.map(lambda x: ((x[index_education_2003], x[index_sex], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').reduceByKey(lambda x,y: x+y)
a2_2008 = death_2008.map(lambda x: ((x[index_education_2003], x[index_sex], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').reduceByKey(lambda x,y: x+y)
a2_2009 = death_2009.map(lambda x: ((x[index_education_2003], x[index_sex], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').reduceByKey(lambda x,y: x+y)
a2_2010 = death_2010.map(lambda x: ((x[index_education_2003], x[index_sex], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').reduceByKey(lambda x,y: x+y)
a2_2011 = death_2011.map(lambda x: ((x[index_education_2003], x[index_sex], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').reduceByKey(lambda x,y: x+y)
a2_2012 = death_2012.map(lambda x: ((x[index_education_2003], x[index_sex], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').reduceByKey(lambda x,y: x+y)
a2_2013 = death_2013.map(lambda x: ((x[index_education_2003], x[index_sex], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').reduceByKey(lambda x,y: x+y)
a2_2014 = death_2014.map(lambda x: ((x[index_education_2003], x[index_sex], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').reduceByKey(lambda x,y: x+y)
a2_2015 = death_2015.map(lambda x: ((x[index_education_2003], x[index_sex], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').reduceByKey(lambda x,y: x+y)

a2 = a2_2005.join(a2_2006).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a2_2007).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a2_2008).map(lambda x: (x[0], x[1][0] + x[1][1]))
a2 = a2.join(a2_2009).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a2_2010).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a2_2011).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a2_2012).map(lambda x: (x[0], x[1][0] + x[1][1]))
a2 = a2.join(a2_2013).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a2_2014).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a2_2015).map(lambda x: (x[0], x[1][0] + x[1][1])).sortBy(lambda x: x[0][0])
a2 = a2.map(lambda x: (x[0][1], [(x[0][0], x[1])])).reduceByKey(lambda x,y: x+y).collect()
# print(a2)
education_level = {'1': '8th grade', '2': '9 - 12th grade', '3': 'High School Graduate', '4': 'College (No Degree)', 
                    '5': 'Associate degree', '6': 'Bachelors Degree', '7': 'Masters Degree', '8': 'Doctorate', '9': 'Unknown', '': 'NaN'}

result_a2 = {}
result_a2_x = []
for element in a2:
    if element[0] not in result_a2:
        result_a2[element[0]] = []
    for level in element[1]:
        if level[0] == '':
            continue
        if education_level[level[0]] not in result_a2_x:
            result_a2_x.append(education_level[level[0]])
        result_a2[element[0]].append(level[1])
# print(result_a2_x)
# print(result_a2)
df_a2 = pd.DataFrame(result_a2, index = result_a2_x)
df_a2.plot(kind='bar', stacked=True, color=['red', 'skyblue'])
plt.xlabel('Education Level')
plt.ylabel('Suicide Cases')
plt.title('Suicide Cases per Education Level')
plt.tight_layout()
plt.savefig('Analysis_2.png', bbox_inches='tight')



# Analysis-3 (How suicide incidents are distributed according to the person's age and educational background.)

def makeAgeGroup(x):
    currGroup = x[0][1]
    changedGroup = ''
    if currGroup=='01' or currGroup=='02' or currGroup=='03':
        changedGroup='1'
    elif currGroup=='04':
        changedGroup='2'
    elif currGroup=='05':
        changedGroup='3'
    elif currGroup=='06' or currGroup=='07':
        changedGroup='4'
    elif currGroup=='08' or currGroup=='09' or currGroup=='10' or currGroup=='11':
        changedGroup='5'
    return ((x[0][0], changedGroup),x[1])

a3_2005 = death_2005.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a3_2006 = death_2006.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a3_2007 = death_2007.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a3_2008 = death_2008.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a3_2009 = death_2009.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a3_2010 = death_2010.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a3_2011 = death_2011.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a3_2012 = death_2012.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a3_2013 = death_2013.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a3_2014 = death_2014.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a3_2015 = death_2015.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='2').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)

a3 = a3_2005.join(a3_2006).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a3_2007).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a3_2008).map(lambda x: (x[0], x[1][0] + x[1][1]))
a3 = a3.join(a3_2009).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a3_2010).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a3_2011).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a3_2012).map(lambda x: (x[0], x[1][0] + x[1][1]))
a3 = a3.join(a3_2013).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a3_2014).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a3_2015).map(lambda x: (x[0], x[1][0] + x[1][1])).sortBy(lambda x: x[0][0])
a3 = a3.map(lambda x: (x[0][1], [(x[0][0], x[1])])).reduceByKey(lambda x,y: x+y).collect()
# print(a3)

education_level = {'1': '8th grade', '2': '9 - 12th grade', '3': 'High School Graduate', '4': 'College (No Degree)', 
                    '5': 'Associate degree', '6': 'Bachelors Degree', '7': 'Masters Degree', '8': 'Doctorate', '9': 'Unknown', '': 'NaN'}
age_group = {'1': 'Child', '2': 'Teenager', '3': 'Young Adult', '4': 'Adult', '5': 'Senior Citizen'}
result_a3 = {}
result_a3_x = []
for element in a3:
    if element[0] == '':
        continue
    if element[0] not in result_a3:
        result_a3[age_group[element[0]]] = [0,0,0,0,0,0,0,0,0]
    for level in element[1]:
        if level[0] == '':
            continue
        if education_level[level[0]] not in result_a3_x:
            result_a3_x.append(education_level[level[0]])
        result_a3[age_group[element[0]]][int(level[0])-1] = level[1]
# print(result_a3_x)
# print(result_a3)
df_a3 = pd.DataFrame(result_a3, index = result_a3_x)
df_a3.plot(kind='bar', stacked=True, color=['red', 'skyblue', 'green', 'yellow', 'pink'])
plt.xlabel('Education Level')
plt.ylabel('Suicide Cases')
plt.title('Suicide Cases per Education Level & Age Groups')
plt.tight_layout()
plt.savefig('Analysis_3.png', bbox_inches='tight')


#Analysis-4 (How accident deaths are divided according to the person's age and educational background.)

a4_2005 = death_2005.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='1').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a4_2006 = death_2006.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='1').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a4_2007 = death_2007.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='1').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a4_2008 = death_2008.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='1').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a4_2009 = death_2009.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='1').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a4_2010 = death_2010.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='1').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a4_2011 = death_2011.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='1').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a4_2012 = death_2012.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='1').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a4_2013 = death_2013.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='1').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a4_2014 = death_2014.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='1').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)
a4_2015 = death_2015.map(lambda x: ((x[index_education_2003], x[index_age_group], x[index_manner]), 1)).filter(lambda x: x[0][2]=='1').map(lambda x: makeAgeGroup(x)).reduceByKey(lambda x,y: x+y)

a4 = a4_2005.join(a4_2006).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a4_2007).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a4_2008).map(lambda x: (x[0], x[1][0] + x[1][1]))
a4 = a4.join(a4_2009).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a4_2010).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a4_2011).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a4_2012).map(lambda x: (x[0], x[1][0] + x[1][1]))
a4 = a4.join(a4_2013).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a4_2014).map(lambda x: (x[0], x[1][0] + x[1][1])).join(a4_2015).map(lambda x: (x[0], x[1][0] + x[1][1])).sortBy(lambda x: x[0][0])
a4 = a4.map(lambda x: (x[0][1], [(x[0][0], x[1])])).reduceByKey(lambda x,y: x+y).collect()
# print(a4)

education_level = {'1': '8th grade', '2': '9 - 12th grade', '3': 'High School Graduate', '4': 'College (No Degree)', 
                    '5': 'Associate degree', '6': 'Bachelors Degree', '7': 'Masters Degree', '8': 'Doctorate', '9': 'Unknown', '': 'NaN'}
age_group = {'1': 'Child', '2': 'Teenager', '3': 'Young Adult', '4': 'Adult', '5': 'Senior Citizen'}
result_a4 = {}
result_a4_x = []
for element in a4:
    if element[0] == '':
        continue
    if element[0] not in result_a4:
        result_a4[age_group[element[0]]] = [0,0,0,0,0,0,0,0,0]
    for level in element[1]:
        if level[0] == '':
            continue
        if education_level[level[0]] not in result_a4_x:
            result_a4_x.append(education_level[level[0]])
        result_a4[age_group[element[0]]][int(level[0])-1] = level[1]
# print(result_a4_x)
# print(result_a4)
df_a4 = pd.DataFrame(result_a4, index = result_a4_x)
df_a4.plot(kind='bar', stacked=True, color=['red', 'skyblue', 'green', 'yellow', 'pink'])
plt.xlabel('Education Level')
plt.ylabel('Suicide Cases')
plt.title('Accident Cases per Education Level & Age Groups')
plt.tight_layout()
plt.savefig('Analysis_4.png', bbox_inches='tight')