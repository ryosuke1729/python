import numpy as np
import pandas as pd
import matplotlib as plt
import codecs as cd

member_map = {1:"A山B夫"}
# 三年男子→三年女子→二年男子→...の順で書く (ID=14、氏名=A山B夫の場合、14:"A山B夫"と書く)

grade_map = {1:3}
#1,2,3は学年を表す　(ID=14、一年生の場合、14:1と書く)

sex_map = {1:1}
# 1,2は男女を表す (ID=14、女子の場合、14:2と書く)

path = "kyudo.csv"
with cd.open(path, "r", "utf-8", "ignore") as file:
    df = pd.read_table(file, delimiter=",")
output = "kyudo_data.csv"

# dfのカラムは順に、"タイムスタンプ",　"ID",	"的前総矢数",	"的前的中数",	"巻藁総矢数"
df=df.drop("タイムスタンプ",axis=1)
df=df.astype(int)

p=df.groupby("ID")["的前総矢数"].sum()
q=df.groupby("ID")["巻藁総矢数"].sum()
r=df.groupby("ID")["的前的中数"].sum()

result = pd.DataFrame(columns=['総矢数(的前)', '総矢数(巻藁)','的中数', '的中率'], index=np.arange(len(member_map))+1)
result["総矢数(的前)"] = p
result["総矢数(巻藁)"] = q
result["的中数"] = r
result["的中率"] = r/p
result = result.fillna(0)
result["総矢数(的前)"] = result["総矢数(的前)"].astype(int)
result["総矢数(巻藁)"] = result["総矢数(巻藁)"].astype(int)
result["的中数"] = result["的中数"].astype(int)
result["的中率"] = result["的中率"].round(3)
result = result.rename(member_map,axis="index")

df["grade"] = df["ID"].replace(grade_map)
df["sex"] = df["ID"].replace(sex_map)
result["ID"] = np.arange(len(member_map))+1
result["grade"] = result["ID"].replace(grade_map)
result["sex"] = result["ID"].replace(sex_map)

male = result[result["sex"]==1]
female = result[result["sex"]==2]
third = result[result["grade"]==3]
second = result[result["grade"]==2]
first = result[result["grade"]==1]
day = df['ID'].value_counts()
day = day.rename(member_map,axis="index")

lis = ["男性","女性","3回生","2回生","1回生","個人練総合"]
i=0;
for df in [male,female,third,second,first,result]:
  df = df.astype(str)
  df["割合"] = df["的中数"].str.cat(df["総矢数(的前)"],sep=" / ")
  df = df.drop(["的中数", "総矢数(的前)",	"総矢数(巻藁)",	"ID",	"grade",	"sex"],axis=1)
  df = df.sort_values("的中率",ascending=False)
  print(lis[i])
  i = i + 1
  print(df.head(8))
  print()

print("的前総矢数")
matomae = result.drop(["的中率","的中数",	"総矢数(巻藁)",	"ID",	"grade",	"sex"],axis=1)
print(matomae.sort_values("総矢数(的前)",ascending=False).head(5))
print()

print("巻藁総矢数")
makiwara = result.drop(["的中率","的中数", "総矢数(的前)",	"ID",	"grade",	"sex"],axis=1)
print(makiwara.sort_values("総矢数(巻藁)",ascending=False).head(5))
print()

print("日数")
print(day.head(5))
print()

kozin_submit = pd.concat([df,makiwara,day],axis=1)
kozin_submit = kozin_submit.rename(columns={'ID':'日数'})
kozin_submit.to_csv(output,encoding='cp932')
