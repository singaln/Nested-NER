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

## ToDo

详细看项目的小伙伴可能看到代码中有很多关于starformer的代码，起初在完成苏神的代码梳理以及双仿射的代码后，想通过starformer来进行嵌套实体识别开发的，无奈后期由于各种事情耽误，并没有进行继续开发。上传这个项目的时间距离开发时已经过去了8个月。。。

由于本人比较懒，最近又在量化交易方向上有浓厚的兴趣，可能starformer嵌套实体识别的方法要GG了，不好意思啊，有兴趣的小伙伴可以继续实现一下，个人感觉还是有一定的可实现性的。。

## 参考

- 苏剑林：[GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373)
- 苏剑林：[Efficient GlobalPointer：少点参数，多点效果](https://spaces.ac.cn/archives/8877)
- [Named Entity Recognition as Dependency Parsing](https://aclanthology.org/2020.acl-main.577.pdf)