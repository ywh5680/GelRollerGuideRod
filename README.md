# 智能导盲杖
## dmz_zh
文件中fgy2.py文件是导盲仗最开始能实现的整体功能代码，能够实现：
  （1）检测上下坡、上下台阶；
  （2）检测前方、左侧、右侧障碍物；
  （3）检测前方道路不平整度
最终通过蓝牙和震动实现对用户反馈

## collect_IMU+VBTS
文件中collect.py为数据收集代码，Separate.py是视频分帧成图片代码

## Force
Force文件是切向力和法向力预测代码，可以使用main_force3.py进行训练和预测。  


修改了预测力的代码，使用test.py可以直接对新采集的数据图片进行预测，以GLAS_010文件中数据为例，步骤如下：
1.上传要预测力的图片数据到session_GLAS_010文件夹中的frame文件夹。
2.在test.py文件中，将代码修改正确，然后运行即可

```python
session_name = 'session_GLAS_010'
args.DATA_PATH = args.save_path ='session_GLAS_010/'
```


需要注意的是，在test_with_force.flist文件中，文本中图片数量要与session_GLAS_010文件夹中的frame文件夹里的图片数量一致代码才不会报错。