# nmsl 抽象话转换

> 基于Transformer的抽象话翻译机

普通文本转抽象话，抽象话转普通文本。支持深度抽象模式。

交互模式下，控制台需要支持emoji显示，不然会显示为乱码。

---
## 环境需要
```
python3.6
```
## 安装依赖库：
```
pip install -r requirements.txt
```
## 运行 - 命令行形式
```python main.py -m s2e```

### 参数列表

参数|格式|含义
-|-|-
-m |str| 指定转换模式。```s2e```：普通转抽象（默认），```s2edeep```：普通转深度抽象，```e2s```：抽象转普通，```e2sdeep```：深度抽象转普通。
-i |str| 输入文件路径，为空时采用交互模式。
-o |str| 输出文件路径，当指定输入文件路径时，必须指定输出路径。

## 运行 - GUI界面
```
python gui.py
```
初次加载需要稍微等待

## 界面示例：

![im](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAhIAAAGtCAYAAABDSZ2CAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAACimSURBVHhe7d17sG13QR/wq7S1VhMtWttRaCsVL2YElKLFaLQZxrECFlAjokyx4Nw/tB3H6dCRUE1E5JWQB0EghFeE8AgmkhAiDyl1sICi7fhEJ2VaiAECSe6DQHAQVs9v3/M793d+5/dba+119j5rr3U+35mPe63fY+197sW9vnefc2+OHH3Ig5ohvv/572s+//lTRR964XmLNd969Lzm+X94V3Pfffc0x+++szl174ld6373wofsue45P/rc5oMfv7P52HsubH7o6DfvmY/OecbNzWfv+53mv357fU3Qdr0hrw8AOGNXkfjWow8GAOhULRIiIiIibVEkREREZHAUCRERERkcRUJEREQGR5EQERGRwVEkREREZHAUCRERERkcRUJEREQGR5EQERGRwVEkREREZHD2XSTe9a53N2+4/salvOI11zWXXfXK5pIrX9G88to3Nv/tOZc3r7/ujc0Xv/jF7auKiIjIFLLvIvH6697Q/Nlf/PWWD+/DXzUvu/qVzX333bd9VREREZlCVlIkTpz6TPPFL/5dpy984Qs7/u7vvrAz/rd/+7fNS19+TfO5z31u+6ql3NZcce6R5tit26e70jaX5dZjzZEjPdc2tzbHjhzb+r9tOf3c4ZqnZetvu6I5t3aNMHfuFVtXKCU897nNFflkeP3VPUnyddtf926F64uITDB739/aTTnh9f/RH/3R9lk9Yc1BfK0rKxKvveH3m19+6a3Nr7z87c1Fr3h7c/Grbml+9TVva5597U3NK254d3PffZ9rPv3pTzcf//jHF06ePNF89BPHm6c/76bmt/7Hnx1MkVjc1Mv/o6rdVG+74twzc5X9rc8bbuCVBYtrVzeXikQY2/v8xWJRKhK71lWKiojIBBPeC/tmmbWbmFgQ2spEnzWrysqKxCW/+d+bJ1z8+ubHnn1tc8HzXts86dJXN0++8prmp3/j6ua5b7qp+cypk7uKxIkToUjc0xy77Mbmtz/4v9dfJPbcSGNO35zPbbmj1m/49ZvxYs/WdffYus6txwrj28LT7Jnffu5bj5WKxe6x4rXD161IiMiME97r+maZtZuatqLQ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)