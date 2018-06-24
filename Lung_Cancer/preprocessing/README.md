### LUNA2016文件说明
* 病人CT文件，每个病人有一份.mhd和一份.raw，里面包含单个病人的体素值和一些病人相关信息，比如说体素坐标原点、层距等（可用SimpleITK包读取数据）；

* 一份annotations.csv，内含病人ID、结节三维质心世界坐标、损伤最大程度的截面中的结节最长直径；

* 一份candidates.csv，内含病人ID、结节三维质心世界坐标、是否假阳性的flag。

* subset0.zip to subset9.zip: 10 zip files which contain all CT images
* annotations.csv: csv file that contains the annotations used as reference standard for the 'nodule detection' track
* sampleSubmission.csv: an example of a submission file in the correct format
* candidates_V2.csv: csv file that contains the candidate locations for the ‘false positive reduction’ track
