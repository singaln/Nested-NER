# Nested_NER

嵌套实体识别项目

## 简介
项目采用Token-Pairs的方式抽取实体，不再采用序列标注的方式。理论上该模型既可以抽取FlatEntity又可以抽取NestedEntity，主要分为两步，分解为“抽取”和“分类”两个步骤，“抽取”就是抽取出为实体的片段，“分类”则是确定每个实体的类型。

因为识别的是整个实体片段，所以不存在实体抽取过程中的解码过程，加速实体的识别效率。

## 模型
- Global模型

 `python main.py --model_type bert --mode Global --project_name car --device cuda:0,1 --do_train`

- Efficient模型

 `python main.py --model_type bert --mode Efficient --project_name car --device cuda:0,1 --do_train`

- Biaffine(双仿射)

 `python main.py --model_type bert --mode Biaffine --project_name car --device cuda:0,1 --do_train`
