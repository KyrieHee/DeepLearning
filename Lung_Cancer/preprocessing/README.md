### LUNA2016文件说明
* 病人CT文件，每个病人有一份.mhd和一份.raw，里面包含单个病人的体素值和一些病人相关信息，比如说体素坐标原点、层距等（可用SimpleITK包读取数据）；

* 一份annotations.csv，内含病人ID、结节三维质心世界坐标、损伤最大程度的截面中的结节最长直径；

* 一份candidates.csv，内含病人ID、结节三维质心世界坐标、是否假阳性的flag。

* subset0.zip to subset9.zip: 10 zip files which contain all CT images
* annotations.csv: csv file that contains the annotations used as reference standard for the 'nodule detection' track
* sampleSubmission.csv: an example of a submission file in the correct format
* candidates_V2.csv: csv file that contains the candidate locations for the ‘false positive reduction’ track

### Dicom图像说明

* SOP Instance UID 用于唯一区分每一张dcm切片，其中Study Instance UID，Series Instance UID上面已经提过，分别用于区分检查号和一次检查对应序列号。
* Modality 表示检查模态，有MRI，CT，CR，DR等；
* Manufacturer 表示制造商，经分析共有"GE MEDICAL SYSTEMS"（最多）， "SIEMENS"， "TOSHIBA"， "Philips"四家制造商提供数据。详见：/baina/sda1/data/lidc_matrix/information.txt；
* Slice Thickness 表示z方向切片厚度，经统计有GE MEDICAL SYSTEMS：2.50， 1.25，SIEMENS：0.75，1.0， 2.0，3.0，5.0，TOSHIBA：2.0， 3.0， Philips：2.0，1.0，1.5，0.9；
* Instance Number 表示一组切片的序列号，这个可以直接用来将切面排序，在实际CT扫描时，是从胸部靠近头的一侧开始扫描，一次扫描到肺部最下，得到的instance number依次增加，对应的Image Position中的z依次减小，而对应的Slice Location是相对位置，绝大多数情况与Image Positon中的z值相同，依次减小，部分不同公司，如TOSHIBA则Slice Location可能与Image Position中的z不同，由于是相对位置，其Slice Location值为正，并且和Instance Number的变化趋势相同。为了在实际分析是不出现错误，不能仅仅采用Slice Location来对切片进行排序，而应使用Instance Number或者Image Position中的z，此次实验使用的是Instance Number。
* Image Position表示图像的左上角在空间坐标系中的x,y,z坐标，单位是毫米，如果在检查中，则指该序列中第一张影像左上角坐标；
* Slice Location为切片z轴相对位置，单位毫米，大多情况与Image Position中的z相同，但TOSHIBA公司提供的数据里面不同，所以不能仅仅根据这个值来对所有切片进行统一排序；
* Photometric Interpretation：光度计的解释,对于CT图像，用两个枚举值MONOCHROME1，MONOCHROME2.用来判断图像是否是彩色的，MONOCHROME1/2是灰度图，RGB则是真彩色图，还有其他；
* Pixel Spacing 表示像素中心间的物理间距；
* Bits Allocated表示存储每一位像素时分配位数，Bits Stored 表示存储每一位像素所用位数；
* Pixel Representation 表示像素数据的表现类型:这是一个枚举值，分别为十六进制数0000和0001，0000H = 无符号整数，0001H = 2的补码。
